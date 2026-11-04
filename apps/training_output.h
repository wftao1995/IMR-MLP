#pragma once

#include "io/gpio.h"
#include "maxvol/active_selector.h"
#include "train/collective_error.h"
#include "train/model_output.h"
#include <filesystem>
#include <unistd.h>

namespace Training
{
inline void validate_training_output(const std::string &path, bool save_maxvol)
{
    if (save_maxvol && !GPM::extension(path))
        GPERROR("--save-maxvol-state requires a .gpm output");
}

// The callback retains each trainer's completed-result checks. A requested
// selector is built before publishing the destination, including for hybrid
// results whose effective model differs from the trainer's residual model.
template <class Save>
void publish_training_result(const std::string &path, bool save_maxvol, Save save,
                             GPSettings settings, const GPIO &input_io, const std::string &input,
                             const std::string &format)
{
    validate_training_output(path, save_maxvol);
    if (!save_maxvol)
    {
        save(path);
        return;
    }
    namespace fs = std::filesystem;
    std::string directory;
    collective_try(
        [&]
        {
            if (mpi.rank != 0)
                return;
            std::string pattern =
                (fs::absolute(path).parent_path() / ".gpml-training-XXXXXX").string();
            if (!::mkdtemp(pattern.data()))
                GPERROR("Cannot stage training output: " + path);
            directory = std::move(pattern);
        });
#ifdef IMRMLP_MPI
    int length = static_cast<int>(directory.size());
    MPI_Bcast(&length, 1, MPI_INT, 0, mpi.comm);
    directory.resize(length);
    MPI_Bcast(directory.data(), length, MPI_CHAR, 0, mpi.comm);
#endif
    struct Stage
    {
        std::string path;

        ~Stage()
        {
            if (mpi.rank == 0)
            {
                std::error_code ec;
                fs::remove_all(path, ec);
            }
        }
    } stage{directory};

    const auto temporary = (fs::path(directory) / fs::path(path).filename()).string();
    save(temporary);
    std::shared_ptr<Potentials::ModelBundle> bundle;
    std::unique_ptr<ActiveSelector> selector;
    std::vector<Atoms> pool;
    collective_try(
        [&]
        {
            bundle = Potentials::ModelBundle::load(temporary);
            GPIO io;
            io.setmap(input_io.getmap());
            if (!io.read(pool, input, format) || pool.empty())
                GPERROR("Cannot rebuild MaxVol from complete training input: " + input);
            BMatrixBuilderSettings builder;
            builder.dtype = "pot_efs";
            builder.ApplySettings(settings.ExtractSubSettings("builder"));
            auto l1 = bundle->potential().reg_L1_vector, l2 = bundle->potential().reg_L2_vector;
            auto rows = make_builder(builder, settings, bundle->potential());
            bundle->potential().reg_L1_vector = std::move(l1);
            bundle->potential().reg_L2_vector = std::move(l2);
            if (!rows)
                GPERROR("Cannot build training-output MaxVol rows");
            selector = std::make_unique<ActiveSelector>(rows.release());
            selector->ApplySettings(settings.ExtractSubSettings("selector"));
        });
    if (selector->select_mode == "full")
    {
        collective_try(
            [&]
            {
                selector->add_candidates_parallel(pool);
            });
        collective_try(
            [&]
            {
                selector->select();
            });
    }
    else
    {
        collective_try(
            [&]
            {
                selector->add_candidates_unbuilt(pool);
            });
        if (selector->select_mode == "stream" && mpi.size > 1)
            selector->select_streaming_mpi(mpi.comm);
        else
            collective_try(
                [&]
                {
                    selector->select();
                });
    }
    collective_try(
        [&]
        {
            if (mpi.rank != 0)
                return;
            bundle->attach_selector(*selector);
            bundle->save(temporary, Potentials::ModelBundle::Profile::Resume);
            Potentials::ModelBundle::validate(temporary);
            fs::rename(temporary, path);
        });
}
}
