#include "potentials/mtp_initialization.h"
#include "potentials/mtp_adjustment.h"
#include "utils/temporary_directory.h"
#include "potentials/potential_wrapper.h"
#include "potentials/potential_mtp_m.h"
#include "potentials/model_bundle.h"
#include "train/collective_error.h"
#include <catch2/catch_all.hpp>
#include "potentials/potential_hybrid.h"
#include <cstdlib>
#include <numeric>

using namespace Potentials;

namespace
{
struct CorpusCase
{
    std::string name;
    std::filesystem::path path;
};

std::vector<CorpusCase> cases()
{
    std::vector<CorpusCase> result;
    if (const char *external = std::getenv("MTP_BASIS_CORPUS_ROOT"))
    {
        const std::filesystem::path root(external);
        if (!std::filesystem::is_directory(root))
            throw std::runtime_error("MTP_BASIS_CORPUS_ROOT is not a directory");
        for (const auto &entry : std::filesystem::recursive_directory_iterator(root))
            if (entry.is_regular_file() && entry.path().filename() == "potential.mtp_0")
                result.push_back(
                    {std::filesystem::relative(entry.path(), root).generic_string(), entry.path()});
        if (result.empty())
            throw std::runtime_error("No potential.mtp_0 files in MTP_BASIS_CORPUS_ROOT");
    }
    else
    {
        const std::filesystem::path root = "unitest/test-data/mtp-basis-corpus";
        for (const char *name :
             {"shapeev-02.mtp", "shapeev-06.mtp", "shapeev-16.mtp", "shapeev-20.mtp",
              "shapeev-20-t.mtp", "ps1-7b-517.mtp", "ps1-5b-5323.mtp", "ps2-7b-4464.mtp"})
            result.push_back({name, root / name});
    }
    std::sort(result.begin(), result.end(),
              [](const auto &a, const auto &b)
              {
                  return a.name < b.name;
              });
    return result;
}

std::vector<PotMTP *> leaves(AnyPotential &model)
{
    if (auto *mtp = dynamic_cast<PotMTP *>(&model))
        return {mtp};
    auto *hybrid = dynamic_cast<PotentialHybrid *>(&model);
    REQUIRE(hybrid);
    std::vector<PotMTP *> result;
    for (auto &child : hybrid->ppots)
        if (auto *mtp = dynamic_cast<PotMTP *>(child.get()))
            result.push_back(mtp);
    REQUIRE(result.size() == 1);
    return result;
}

std::vector<Atoms> geometries()
{
    Atoms a;
    a.cell_ = 16. * Eigen::Matrix3d::Identity();
    a.pbc_ = {false, false, false};
    a.pos_ = {{2, 2, 2}, {4.2, 2.3, 2.1}, {2.2, 4.3, 2.4}, {2.4, 2.2, 4.6}, {4.3, 4.6, 3.8}};
    a.chemmap_ = {0, 1, 0, 1, 0};
    auto b = a;
    b.cell_ = 6.4 * Eigen::Matrix3d::Identity();
    b.pbc_ = {true, true, true};
    auto c = b;
    Eigen::Matrix3d deformation = Eigen::Matrix3d::Identity();
    deformation(0, 0) = 1.07;
    deformation(1, 1) = .96;
    deformation(0, 1) = .04;
    c.cell_ *= deformation;
    for (auto &position : c.pos_)
        position = position * deformation;
    return {a, b, c};
}

std::unique_ptr<AnyPotential> initialize(const CorpusCase &fixture, const std::string &kind,
                                         Utils::TemporaryDirectory &stage)
{
    const auto input = (stage.path / "input.mtp").string(),
               output = (stage.path / "initial.gpm").string();
    std::filesystem::copy_file(fixture.path, input);
    GPSettings config;
    config.Modify("pot:pottype", kind);
    if (kind == "mtp_m")
    {
        config.Modify("pot:extmodel", "mtp_m_n2");
        config.Modify("pot:n_input", "2");
        config.Modify("pot:n_neuron", "3");
        config.Modify("pot:n_neuron2", "2");
    }
    else if (kind == "hybrid")
    {
        config.Modify("pot:npot", "2");
        config.Modify("pot:pot1:pottype", "mtp");
        config.Modify("pot:pot1:potfile", input);
        config.Modify("pot:pot1:train", "true");
        config.Modify("pot:pot2:pottype", "zbl");
        config.Modify("pot:pot2:r0", ".6");
        config.Modify("pot:pot2:r1", "3");
    }
    RNG.mtd.seed(71329);
    Potentials::save_inference_model(
        *Potentials::MTP::initialize_model(config, {.input = input,
                                                    .type = "",
                                                    .elements = {"Al", "Ni"},
                                                    .reset = false,
                                                    .randomize = true,
                                                    .lower = -.08,
                                                    .upper = .12,
                                                    .scale = 1.,
                                                    .index = "",
                                                    .explicit_index = false,
                                                    .dataset = "",
                                                    .format = "",
                                                    .fit = false,
                                                    .normalize = false,
                                                    .offset_target = ""}),
        output);
    auto model = ModelBundle::load(output)->release_potential();
    for (auto *mtp : leaves(*model))
    {
        mtp->set_scale(1.);
        for (int i = 0; i < mtp->radial_coeff_size; ++i)
            mtp->radial_coeffs[i] = .04 + .025 * std::sin(.73 * (i + 1));
        const int count = mtp->alpha_scalar_moments;
        for (int c = 0; c < mtp->moment_coeff_size / count; ++c)
            for (int i = 0; i < count; ++i)
                mtp->moment_coeffs[c * count + i] = .0001 * (i + 1) / count * (1. + .37 * c);
        if (kind == "mtp_m")
        {
            int nr = 0, ns = 0, nm = 0, nh = 0;
            mtp->get_n_coeffs(nr, ns, nm, nh);
            double *radial = nullptr, *species = nullptr, *moments = nullptr, *head = nullptr;
            mtp->get_pointers(radial, species, moments, head);
            for (int i = 0; i < nh; ++i)
                head[i] = .15 + .04 * std::sin(.37 * (i + 1));
            mtp->init_coeffs();
        }
    }
    std::filesystem::remove(input);
    return model;
}

using Features = std::vector<std::vector<Eigen::VectorXd>>;

Features features(PotMTP &model, const std::vector<Atoms> &atoms)
{
    Features result;
    for (const auto &frame : atoms)
    {
        Neighborhoods neighborhoods(frame, model.cutoff);
        std::vector<Eigen::VectorXd> sites;
        for (int i = 0; i < frame.size(); ++i)
        {
            model.CalcSiteBasis(neighborhoods[i]);
            Eigen::VectorXd value(model.alpha_scalar_moments);
            for (int k = 0; k < value.size(); ++k)
                value[k] = model.moment_vals[model.alpha_moment_mapping[k]];
            REQUIRE(value.allFinite());
            REQUIRE(value.cwiseAbs().maxCoeff() > 1e-6);
            sites.push_back(std::move(value));
        }
        result.push_back(std::move(sites));
    }
    return result;
}

void close(const Eigen::VectorXd &actual, const Eigen::VectorXd &expected, double absolute = 1e-11,
           double relative = 2e-9)
{
    REQUIRE(actual.size() == expected.size());
    REQUIRE(actual.allFinite());
    REQUIRE(expected.allFinite());
    REQUIRE(((actual - expected).cwiseAbs().array() /
             (absolute + relative * expected.cwiseAbs().array()))
                .maxCoeff() <= 1.);
}

std::vector<Atoms> evaluate(AnyPotential &model, std::vector<Atoms> frames)
{
    for (auto &frame : frames)
        model.getEFS(frame);
    return frames;
}

void observables(AnyPotential &model, const std::vector<Atoms> &frames,
                 const std::vector<Atoms> &expected)
{
    const auto actual = evaluate(model, frames);
    for (size_t f = 0; f < actual.size(); ++f)
    {
        Eigen::VectorXd a(1 + 3 * actual[f].size() + 9), b = a;
        int k = 0;
        a[k] = actual[f].energy_;
        b[k++] = expected[f].energy_;
        for (size_t i = 0; i < actual[f].size(); ++i)
            for (int d = 0; d < 3; ++d)
            {
                a[k] = actual[f].forces_[i][d];
                b[k++] = expected[f].forces_[i][d];
            }
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
            {
                a[k] = actual[f].stress_(i, j);
                b[k++] = expected[f].stress_(i, j);
            }
        close(a, b);
    }
}

void graph_levels(PotMTP &model)
{
    Potentials::MTP::validate_graph(model);
    std::vector<int> levels(model.alpha_moments_count, 0);
    for (int i = 0; i < model.alpha_index_times_count; ++i)
    {
        auto row = model.alpha_index_times.row(i);
        levels[row[3]] = std::max({levels[row[3]], levels[row[0]] + 1, levels[row[1]] + 1});
    }
    const int maximum = *std::max_element(levels.begin(), levels.end());
    REQUIRE(model.max_level == maximum);
    std::vector<int> counts(maximum + 1, 0);
    for (int i = 0; i < model.alpha_scalar_moments; ++i)
        ++counts[levels[model.alpha_moment_mapping[i]]];
    REQUIRE(model.lev_scalar_count == counts);
    REQUIRE(std::accumulate(model.lev_times_count.begin(), model.lev_times_count.end(), 0) ==
            model.alpha_index_times_count);
    for (int level = 1; level <= maximum; ++level)
    {
        int first = model.alpha_moments_count;
        for (int id = 0; id < model.alpha_moments_count; ++id)
            if (levels[id] >= level)
                first = std::min(first, id);
        REQUIRE(model.lev_scalar_nodes[level] == first);
    }
}

void mapped_features(PotMTP &model, const std::vector<Atoms> &frames, const Features &expected,
                     const std::vector<double> &original)
{
    const int b = model.alpha_scalar_moments;
    std::map<double, int> by_coefficient;
    for (int i = 0; i < b; ++i)
        REQUIRE(by_coefficient.emplace(original[i], i).second);
    std::vector<int> permutation;
    for (int i = 0; i < b; ++i)
    {
        auto it = by_coefficient.find(model.moment_coeffs[i]);
        REQUIRE(it != by_coefficient.end());
        permutation.push_back(it->second);
    }
    for (int c = 0; c < model.moment_coeff_size / b; ++c)
        for (int i = 0; i < b; ++i)
            REQUIRE(model.moment_coeffs[c * b + i] == original[c * b + permutation[i]]);
    const auto actual = features(model, frames);
    for (size_t f = 0; f < actual.size(); ++f)
        for (size_t site = 0; site < actual[f].size(); ++site)
        {
            Eigen::VectorXd sorted(b);
            for (int i = 0; i < b; ++i)
                sorted[i] = expected[f][site][permutation[i]];
            close(actual[f][site], sorted);
        }
}
}

TEST_CASE("Real MTM corpus retains scalar columns and EFS through sorting and migration",
          "[mtp][basis-corpus]")
{
    const auto fixture = GENERATE_COPY(Catch::Generators::from_range(cases()));
    const std::string kind = GENERATE("mtp", "mtp_m", "hybrid");
    INFO(fixture.name << " / " << kind);
    Utils::TemporaryDirectory stage;
    auto model = initialize(fixture, kind, stage);
    auto frames = geometries();
    const auto expected = evaluate(*model, frames);
    auto *leaf = leaves(*model).front();
    for (const auto &frame : expected)
    {
        double force = 0.;
        for (const auto &value : frame.forces_)
            force = std::max(force, value.norm());
        REQUIRE(force > 1e-12);
    }
    const auto original_features = features(*leaf, frames);
    const std::vector<double> coefficients(leaf->moment_coeffs,
                                           leaf->moment_coeffs + leaf->moment_coeff_size);
    for (const char *order :
         {"level", "level", "level-group", "level-group", "level", "level-group"})
    {
        INFO(order);
        if (kind == "mtp")
        { // Also inspect the transient caches before GPM rebuilds the model.
            PotMTP direct(*leaves(*model).front());
            if (std::string(order) == "level")
                direct.ReorderByLevel();
            else
                direct.ReorderByLevelGroup();
            graph_levels(direct);
            mapped_features(direct, frames, original_features, coefficients);
        }
        Potentials::MTP::Adjustment op;
        op.basis = order;
        model = Potentials::MTP::adjust_tree(*model, op);
        observables(*model, frames, expected);
        mapped_features(*leaves(*model).front(), frames, original_features, coefficients);
    }
    Potentials::MTP::Adjustment extend;
    extend.appended = {"Cu"};
    extend.sources = {1};
    extend.initialization = "clone";
    extend.order = {2, 0, 1};
    extend.basis = "level-group";
    model = Potentials::MTP::adjust_tree(*model, extend);
    for (auto &frame : frames)
        for (int &id : frame.chemmap_)
            ++id;
    observables(*model, frames, expected);
    mapped_features(*leaves(*model).front(), frames, original_features, coefficients);
    const auto gpm = (stage.path / "continued.gpm").string();
    Potentials::save_inference_model(*model, gpm);
    model = ModelBundle::load(gpm)->release_potential();
    observables(*model, frames, expected);
    mapped_features(*leaves(*model).front(), frames, original_features, coefficients);
    const auto native = (stage.path / "native.model").string();
    Potentials::save_inference_model(*model, native);
    GPSettings settings;
    settings.Modify("potfile", native);
    settings.Modify("pottype", kind);
    settings.Modify("typemap", "");
    if (kind == "mtp_m")
    {
        auto meta = leaves(*model).front()->model_metadata();
        settings.Modify("extmodel", GPM::string(meta.at("model")));
        settings.Modify("n_input", "2");
        settings.Modify("n_neuron", "3");
        settings.Modify("n_neuron2", "2");
    }
    potential_wrapper reload(settings);
    observables(*reload.pot, frames, expected);
    mapped_features(*leaves(*reload.pot).front(), frames, original_features, coefficients);
}

TEST_CASE("Alternative real contraction graphs represent the same scalar basis",
          "[mtp][basis-corpus]")
{
    std::vector<std::pair<CorpusCase, CorpusCase>> pairs;
    if (const char *external = std::getenv("MTP_BASIS_CORPUS_ROOT"))
        for (const char *level : {"20", "30"})
        {
            const std::filesystem::path root = std::filesystem::path(external) / "shapeev" / level;
            REQUIRE(std::filesystem::exists(root / "potential.mtp_0"));
            REQUIRE(std::filesystem::exists(root / "t/potential.mtp_0"));
            pairs.push_back({{std::string(level), root / "potential.mtp_0"},
                             {std::string(level) + "/t", root / "t/potential.mtp_0"}});
        }
    else
        pairs.push_back({{"20", "unitest/test-data/mtp-basis-corpus/shapeev-20.mtp"},
                         {"20/t", "unitest/test-data/mtp-basis-corpus/shapeev-20-t.mtp"}});
    for (const auto &[left, right] : pairs)
    {
        INFO(left.name << " and " << right.name);
        Utils::TemporaryDirectory a_stage, b_stage;
        auto a = initialize(left, "mtp", a_stage), b = initialize(right, "mtp", b_stage);
        auto *ma = leaves(*a).front(), *mb = leaves(*b).front();
        REQUIRE(ma->alpha_scalar_moments == mb->alpha_scalar_moments);
        std::fill_n(ma->moment_coeffs, ma->moment_coeff_size, 1. / ma->alpha_scalar_moments);
        std::fill_n(mb->moment_coeffs, mb->moment_coeff_size, 1. / mb->alpha_scalar_moments);
        const auto frames = geometries();
        auto fa = features(*ma, frames), fb = features(*mb, frames);
        for (size_t f = 0; f < fa.size(); ++f)
            for (size_t site = 0; site < fa[f].size(); ++site)
            {
                std::sort(fa[f][site].data(), fa[f][site].data() + fa[f][site].size());
                std::sort(fb[f][site].data(), fb[f][site].data() + fb[f][site].size());
                close(fb[f][site], fa[f][site]);
            }
        observables(*b, frames, evaluate(*a, frames));
    }
}
