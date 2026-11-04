#include "potentials/mtp_initialization.h"
#include "potentials/mtp_adjustment.h"
#include "utils/temporary_directory.h"
#include "potentials/potential_wrapper.h"
#include "potentials/potential_mtp_m.h"
#include "potentials/model_bundle.h"
#include "train/collective_error.h"
#include <catch2/catch_all.hpp>
#include "apps/apps.h"

#if defined(GPML_MPI) || defined(IMRMLP_MPI)
// The oracle is process-local; distributed publication/error propagation is
// exercised separately through the CLI integration tests.
int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    mpi.InitComm(MPI_COMM_SELF);
    mpi.fnm_ending.clear();
    const int result = Catch::Session().run(argc, argv);
    MPI_Finalize();
    return result;
}
#endif

namespace
{
struct Options
{
    CMDOption::CommandLineOptions saved = CMDOption::global_cmdopts;

    ~Options()
    {
        CMDOption::global_cmdopts = saved;
    }

    int run(std::vector<std::string> args)
    {
        CMDOption::global_cmdopts.reset();
        std::vector<char *> argv;
        for (auto &arg : args)
            argv.push_back(arg.data());
        return command_maps.at(args.front()).first(argv.size(), argv.data());
    }
};
}

TEST_CASE("MTP commands apply settings-only paths and model options", "[mtp][cli][settings]")
{
    using namespace Potentials;
    const std::string kind = GENERATE("mtp", "mtp_m");
    Utils::TemporaryDirectory stage;
    Options command;
    const auto input = (stage.path / "template.mtp").string(),
               output = (stage.path / "initial.gpm").string();
    std::filesystem::copy_file("unitest/test-data/mtp-basis-corpus/shapeev-06.mtp", input);
    const auto init = (stage.path / "init.ini").string();
    {
        std::ofstream settings(init);
        settings << "input " << input << "\noutput " << output
                 << "\nelements H He\nrandomize false\n";
        if (kind == "mtp")
            settings << "type mtp\n";
        else
            settings
                << "pot:pottype mtp_m\npot:extmodel mtp_m_n2\npot:n_input 2\npot:n_neuron 3\npot:n_neuron2 2\n";
    }
    REQUIRE(command.run({"initmtpm", "--settings=" + init}) == 0);
    auto model = ModelBundle::load(output);
    REQUIRE(ModelBundle::type(model->potential()) == kind);
    REQUIRE(model->potential().gpm_elements == std::map<int, std::string>{{0, "H"}, {1, "He"}});
    const auto adjusted = (stage.path / "adjusted.gpm").string(),
               settings = (stage.path / "adjust.ini").string();
    std::ofstream(settings) << "input " << output << "\noutput " << adjusted
                            << "\nneworder 1 0\nbasis-order level-group\n";
    REQUIRE(command.run({"adjustmtp", "--settings=" + settings}) == 0);
    auto reloaded = ModelBundle::load(adjusted);
    REQUIRE(ModelBundle::type(reloaded->potential()) == kind);
    REQUIRE(reloaded->potential().gpm_elements == std::map<int, std::string>{{0, "He"}, {1, "H"}});
    REQUIRE(model->potential().gpm_elements == std::map<int, std::string>{{0, "H"}, {1, "He"}});
    // The same deck may explicitly replace its input file.
    std::ofstream(settings) << "input " << adjusted << "\noutput " << adjusted
                            << "\nneworder 1 0\n";
    REQUIRE(command.run({"adjustmtp", "--settings=" + settings}) == 0);
    auto inplace = ModelBundle::load(adjusted);
    REQUIRE(inplace->potential().gpm_elements == std::map<int, std::string>{{0, "H"}, {1, "He"}});
    // A CLI output path must win over the deck's explicit path.
    const auto override = (stage.path / "override.gpm").string();
    REQUIRE(command.run({"adjustmtp", "--settings=" + settings, "--output=" + override}) == 0);
    REQUIRE(ModelBundle::load(override)->potential().gpm_elements ==
            std::map<int, std::string>{{0, "He"}, {1, "H"}});
    REQUIRE(ModelBundle::load(adjusted)->potential().gpm_elements ==
            std::map<int, std::string>{{0, "H"}, {1, "He"}});
}

TEST_CASE("MTP leaf initialization keeps the last repeated head setting", "[mtp][settings]")
{
    GPSettings settings;
    settings.Modify("pot:pottype", "mtp_m");
    settings.Modify("pot:extmodel", "mtp_m_n2");
    settings.settings.emplace("pot:n_input", "1");
    settings.settings.emplace("pot:n_input", "2");
    settings.Modify("pot:n_neuron", "3");
    settings.Modify("pot:n_neuron2", "2");
    auto model = Potentials::MTP::initialize_model(
        settings, {.input = "unitest/test-data/mtp-basis-corpus/shapeev-06.mtp",
                   .elements = {"H", "He"},
                   .randomize = false});
    const auto metadata = dynamic_cast<Potentials::PotMTP &>(*model).model_metadata();
    REQUIRE(metadata.at("q").get<double>() == 2.);
    REQUIRE(settings.count("pot:n_input") == 2);
}
