#include "potentials/mtp_initialization.h"
#include "potentials/mtp_adjustment.h"
#include "utils/temporary_directory.h"
#include "potentials/potential_wrapper.h"
#include "potentials/potential_mtp_m.h"
#include "potentials/model_bundle.h"
#include "train/collective_error.h"
#include <catch2/catch_all.hpp>
#include "gpm_fixture.h"
#include "potentials/potential_hybrid_scale.h"
#include "potentials/potential_eam_poly.h"
using namespace Potentials;

namespace
{
void compare(AnyPotential &model, Atoms atoms, const Atoms &expected)
{
    model.getEFS(atoms);
    REQUIRE(atoms.energy_ == Catch::Approx(expected.energy_).margin(1e-8));
    REQUIRE((atoms.stress_ - expected.stress_).norm() < 1e-7);
    for (size_t i = 0; i < atoms.size(); ++i)
        REQUIRE((atoms.forces_[i] - expected.forces_[i]).norm() < 1e-8);
}
}

TEST_CASE("MTP tree adapters migrate split bases and full heads", "[mtp][workflow][tree]")
{
    const auto kind = GENERATE("mtp_les", "mtp_m", "mtpnn1", "hybrid", "hybrid_scale", "hmtp",
                               "descriptor_linear");
    CAPTURE(kind);
    Utils::TemporaryDirectory stage;
    auto file = [&](const char *name)
    {
        return (stage.path / name).string();
    };
    PotMTP source;
    load_swap_fixture(source, 2, true);
    source.set_element_map({{0, "Al"}, {1, "Ni"}});
    source.Save(file("source.mtp"));
    GPSettings config;
    config.Modify("pot:pottype", kind);
    config.Modify("pot:extmodel", "mtp_m_n2");
    config.Modify("pot:n_input", "2");
    config.Modify("pot:n_neuron", "3");
    config.Modify("pot:n_neuron2", "2");
    config.Modify("pot:nnlay", "2");
    config.Modify("pot:nnhidden", "3");
    config.Modify("pot:nnhidden2", "2");
    if (std::string(kind) == "mtp_m" || std::string(kind) == "mtp_les")
    {
        config.Modify("pot:lr_les", "true");
        config.Modify("pot:les_kernel", "sog");
        config.Modify("pot:sog_channels", "2");
        config.Modify("pot:sog_weights", "1 2 1.5 .5");
        config.Modify("pot:sog_log_bandwidths", "-.5 .2 -.2 .3");
    }
    config.Modify("pot:npot", "2");
    config.Modify("pot:feature", "min_distance");
    config.Modify("pot:boundary1", "1 4");
    for (int i = 1; i <= 2; ++i)
    {
        auto prefix = "pot:pot" + std::to_string(i) + ":";
        config.Modify(prefix + "pottype", "mtp");
        config.Modify(prefix + "potfile", file("source.mtp"));
    }
    Potentials::save_inference_model(
        *Potentials::MTP::initialize_model(config, {.input = file("source.mtp"),
                                                    .type = "",
                                                    .elements = {},
                                                    .reset = false,
                                                    .randomize = true,
                                                    .lower = .01,
                                                    .upper = .02,
                                                    .scale = 1.,
                                                    .index = "",
                                                    .explicit_index = false,
                                                    .dataset = "",
                                                    .format = "",
                                                    .fit = false,
                                                    .normalize = false,
                                                    .offset_target = ""}),
        file("tree.gpm"));
    auto bundle = ModelBundle::load(file("tree.gpm"));
    auto &model = bundle->potential();
    // Rebuild from owned children without supplying their original filenames.
    GPSettings rebuild;
    const std::string family = kind;
    if (family == "hybrid" || family == "hybrid_scale" || family == "hmtp")
    {
        rebuild.Modify("pot:npot", "2");
        auto reused = MTP::initialize_model(rebuild, {.input = file("tree.gpm")});
        std::vector<double> before(model.get_NL_coeffs_count()),
            after(reused->get_NL_coeffs_count());
        model.nl_get_coeffs(before.data());
        reused->nl_get_coeffs(after.data());
        REQUIRE(after == before);
    }
    auto reset_model = MTP::initialize_model(
        GPSettings(), {.input = file("tree.gpm"), .reset = true, .randomize = false});
    REQUIRE(reset_model->species_count == model.species_count);
    REQUIRE(ModelBundle::type(*reset_model) == family);
    model.reg_L1_vector.resize(model.get_NL_coeffs_count());
    model.reg_L2_vector.resize(model.get_NL_coeffs_count());
    for (size_t i = 0; i < model.reg_L1_vector.size(); ++i)
    {
        model.reg_L1_vector[i] = .3 + .001 * i;
        model.reg_L2_vector[i] = .4 + .002 * i;
    }
    // Nonuniform penalties expose misplaced coordinates that uniform weights hide.
    const auto penalty = [](AnyPotential &potential)
    {
        std::vector<double> x(potential.get_NL_coeffs_count());
        potential.nl_get_coeffs(x.data());
        double result = 0.;
        for (size_t i = 0; i < x.size(); ++i)
            result += potential.reg_L1_vector[i] * std::abs(x[i]) +
                      potential.reg_L2_vector[i] * x[i] * x[i];
        return result;
    };
    const auto original = ModelBundle::fingerprint(model);
    const double original_penalty = penalty(model);
    for (const auto *order : {"level", "level-group"})
    {
        auto reordered = Potentials::MTP::adjust_tree(model, {.order = {1, 0}, .basis = order});
        REQUIRE(penalty(*reordered) == Catch::Approx(original_penalty).margin(1e-10));
        REQUIRE(ModelBundle::fingerprint(model) == original);
    }
    auto atoms = swap_lattice(2, 2), expected = atoms;
    model.getEFS(expected);
    Potentials::MTP::Adjustment op;
    op.appended = {"Cu"};
    op.sources = {1};
    op.initialization = "clone";
    op.order = {2, 0, 1};
    op.basis = "level-group";
    auto transformed = Potentials::MTP::adjust_tree(model, op);
    for (int &id : atoms.chemmap_)
        id = id == 0 ? 1 : 2;
    compare(*transformed, atoms, expected);
    REQUIRE(transformed->gpm_elements ==
            std::map<int, std::string>{{0, "Cu"}, {1, "Al"}, {2, "Ni"}});
    REQUIRE(transformed->reg_L1_vector.size() == size_t(transformed->get_NL_coeffs_count()));
    for (double penalty : transformed->reg_L1_vector)
        REQUIRE((penalty == 0. || std::find(model.reg_L1_vector.begin(), model.reg_L1_vector.end(),
                                            penalty) != model.reg_L1_vector.end()));
    Potentials::save_inference_model(*transformed, file("adjusted.gpm"));
    auto reloaded = ModelBundle::load(file("adjusted.gpm"));
    compare(reloaded->potential(), atoms, expected);
}

TEST_CASE("Grouped basis sorting preserves interleaved accumulation targets",
          "[mtp][workflow][tree]")
{
    PotMTP source;
    load_swap_fixture(source, 2, false);
    source.set_element_map({{0, "Al"}, {1, "Ni"}});
    source.alpha_index_times << 0, 0, 1, 4, 2, 2, 1, 4, 1, 1, 1, 5, 3, 3, 1, 5;
    auto atoms = swap_lattice(2, 2), expected = atoms;
    source.getEFS(expected);
    Potentials::MTP::Adjustment op;
    op.basis = "level-group";
    auto adjusted = Potentials::MTP::adjust_tree(source, op);
    Potentials::MTP::validate_graph(dynamic_cast<PotMTP &>(*adjusted));
    compare(*adjusted, atoms, expected);
    Utils::TemporaryDirectory stage;
    const auto output = (stage.path / "grouped.gpm").string();
    Potentials::save_inference_model(*adjusted, output);
    auto reloaded = ModelBundle::load(output);
    compare(reloaded->potential(), atoms, expected);
}

TEST_CASE("Basis levels follow graph dependencies across scalar coordinate gaps",
          "[mtp][workflow][tree]")
{
    const std::string order = GENERATE("level", "level-group");
    const bool derived = GENERATE(false, true);
    std::ostringstream text;
    text
        << "MTP\nversion = 1.1.0\npotential_name = MTP1m\nscaling = 1\nspecies_count = 1\npotential_tag =\n"
        << "radial_basis_type = RBChebyshev\nmin_dist = 0.5\nmax_dist = 5\nradial_basis_size = 2\nradial_funcs_count = 1\nradial_coeffs\n0-0\n{0.03, 0.01}\n"
        << "alpha_moments_count = 5\nalpha_index_basic_count = 3\nalpha_index_basic = {{0,0,0,0}, {0,0,0,0}, {0,0,0,0}}\n"
        << "alpha_index_times_count = 2\nalpha_index_times = {{0,0,1,3}, {1,1,1,4}}\n"
        << "alpha_scalar_moments = " << (derived ? 3 : 2)
        << "\nalpha_moment_mapping = " << (derived ? "{0,2,4}" : "{0,2}")
        << "\nspecies_coeffs = {-0.2}\nmoment_coeffs = "
        << (derived ? "{0.01,0.02,0.03}" : "{0.01,0.02}") << '\n';
    std::istringstream input(text.str());
    PotMTP source;
    source.Load(input);
    auto atoms = swap_lattice(2, 1), expected = atoms;
    source.getEFS(expected);
    for (int repeat = 0; repeat < 2; ++repeat)
    {
        if (order == "level")
            source.ReorderByLevel();
        else
            source.ReorderByLevelGroup();
        Potentials::MTP::validate_graph(source);
        compare(source, atoms, expected);
        REQUIRE(source.lev_scalar_count == std::vector<int>{2, derived ? 1 : 0});
        REQUIRE(source.lev_scalar_nodes == std::vector<int>{0, 3});
        REQUIRE(source.lev_times_count == std::vector<int>{0, 2});
    }
    Utils::TemporaryDirectory stage;
    const auto output = (stage.path / "sparse.gpm").string();
    Potentials::save_inference_model(source, output);
    auto reloaded = ModelBundle::load(output);
    compare(reloaded->potential(), atoms, expected);
}

TEST_CASE("MTP adjustment maps masked and physical regularization layouts", "[mtp][workflow][tree]")
{
    const bool physical = GENERATE(false, true);
    const bool random = GENERATE(false, true);
    Utils::TemporaryDirectory stage;
    const auto source_path = (stage.path / "source.mtp").string();
    PotMTP source;
    load_swap_fixture(source, 2, true);
    source.set_element_map({{0, "Al"}, {1, "Ni"}});
    source.Save(source_path);
    GPSettings settings;
    settings.Modify("pot:pottype", "mtp_m");
    settings.Modify("pot:extmodel", "mtp_m_n2");
    settings.Modify("pot:n_input", "2");
    settings.Modify("pot:n_neuron", "3");
    settings.Modify("pot:n_neuron2", "2");
    settings.Modify("pot:req_radical", "false");
    settings.Modify("pot:req_species", "false");
    auto model = Potentials::MTP::initialize_model(settings, {.input = source_path});
    auto &mtp = dynamic_cast<PotMTP &>(*model);
    const int count = physical ? mtp.reg_coeff_size : model->get_NL_coeffs_count();
    model->reg_L1_vector.resize(count);
    for (int i = 0; i < count; ++i)
        model->reg_L1_vector[i] = i + 1;
    const auto before = ModelBundle::fingerprint(*model);
    Potentials::MTP::Adjustment op;
    op.appended = {"Cu"};
    op.sources = {1};
    op.order = {2, 0, 1};
    op.initialization = random ? "random" : "clone";
    op.basis = "level-group";
    auto adjusted = Potentials::MTP::adjust_tree(*model, op);
    REQUIRE(ModelBundle::fingerprint(*model) == before);
    REQUIRE(adjusted->CurrentSettings()["req_radical"] == model->CurrentSettings()["req_radical"]);
    REQUIRE(adjusted->CurrentSettings()["req_species"] == model->CurrentSettings()["req_species"]);
    const auto &weights = adjusted->reg_L1_vector;
    for (int i = 0; i < count; ++i)
        REQUIRE(std::find(weights.begin(), weights.end(), i + 1.) != weights.end());
    REQUIRE((std::find(weights.begin(), weights.end(), 0.) != weights.end()) == random);
}

TEST_CASE("Scalar sorting reports the coefficient permutation directly", "[mtp][workflow][tree]")
{
    const bool grouped = GENERATE(false, true);
    PotMTP source;
    load_swap_fixture(source, 2, false);
    std::vector<double> previous(source.moment_coeffs,
                                 source.moment_coeffs + source.alpha_scalar_moments);
    std::vector<int> permutation;
    if (grouped)
        source.ReorderByLevelGroup(permutation);
    else
        source.ReorderByLevel(permutation);
    REQUIRE(permutation.size() == previous.size());
    auto sorted = permutation;
    std::sort(sorted.begin(), sorted.end());
    for (size_t i = 0; i < previous.size(); ++i)
    {
        REQUIRE(sorted[i] == int(i));
        REQUIRE(source.moment_coeffs[i] == previous[permutation[i]]);
    }
}

TEST_CASE("Composite adjustment retains child request masks before packing parameters",
          "[mtp][workflow][tree]")
{
    Utils::TemporaryDirectory stage;
    const auto input = (stage.path / "source.mtp").string();
    PotMTP source;
    load_swap_fixture(source, 2, true);
    source.set_element_map({{0, "Al"}, {1, "Ni"}});
    source.Save(input);
    GPSettings settings;
    settings.Modify("pot:pottype", "mtp_m");
    settings.Modify("pot:extmodel", "mtp_m_li");
    settings.Modify("pot:n_input", "2");
    settings.Modify("pot:req_radical", "false");
    settings.Modify("pot:req_species", "false");
    std::vector<std::unique_ptr<AnyPotential>> children;
    children.push_back(Potentials::MTP::initialize_model(settings, {.input = input}));
    settings.Modify("pot:req_moment", "false");
    children.push_back(Potentials::MTP::initialize_model(settings, {.input = input}));
    const int parameters = children[0]->get_NL_coeffs_count() + children[1]->get_NL_coeffs_count();
    GPSettings config;
    config.Modify("npot", "2");
    config.Modify("feature", "min_distance");
    config.Modify("boundary1", "1 4");
    config.Modify("pot1:pottype", "mtp_m");
    config.Modify("pot2:pottype", "mtp_m");
    PotentialHybridScale model(config, std::move(children));
    model.reg_L2_vector.resize(parameters);
    for (int i = 0; i < parameters; ++i)
        model.reg_L2_vector[i] = 1. + i;
    auto penalty = [](AnyPotential &p)
    {
        std::vector<double> x(p.get_NL_coeffs_count());
        p.nl_get_coeffs(x.data());
        double sum = 0.;
        for (size_t i = 0; i < x.size(); ++i)
            sum += p.reg_L2_vector[i] * x[i] * x[i];
        return sum;
    };
    auto adjusted = Potentials::MTP::adjust_tree(model, {.order = {1, 0}, .basis = "level"});
    REQUIRE(adjusted->get_NL_coeffs_count() == parameters);
    REQUIRE(penalty(*adjusted) == Catch::Approx(penalty(model)).margin(1e-10));
    auto atoms = swap_lattice(2, 2), expected = atoms;
    model.getEFS(expected);
    for (auto &species : atoms.chemmap_)
        species = 1 - species;
    compare(*adjusted, atoms, expected);
}

TEST_CASE("Unchanged EAM branches own their copied parameters", "[mtp][workflow][tree]")
{
    GPSettings settings;
    settings.Modify("nelems", "2");
    PotentialEAMPoly source(settings);
    source.set_element_map({{0, "Al"}, {1, "Ni"}});
    for (size_t i = 0; i < source.regression_coeffs.size(); ++i)
        source.regression_coeffs[i] = .01 + .001 * i;
    const auto coefficients = source.regression_coeffs;
    auto copy = MTP::adjust_tree(source, {});
    auto atoms = swap_lattice(2, 2), expected = atoms;
    static_cast<AnyPotential &>(source).getEFS(expected);
    compare(*copy, atoms, expected);
    std::vector<double> changed(copy->get_NL_coeffs_count(), .02);
    copy->nl_set_coeffs(changed.data());
    REQUIRE(source.regression_coeffs == coefficients);
}

TEST_CASE("MTP adjustment retains fitted polynomial bases", "[mtp][workflow][tree]")
{
    const bool cosine = GENERATE(false, true);
    PotMTP source;
    load_swap_fixture(source, 2, true);
    source.set_element_map({{0, "Al"}, {1, "Ni"}});
    for (auto *&basis : source.p_radical_basis_pair)
    {
        auto fitted = [&](auto *poly) -> AnyBasis *
        {
            poly->coeffs(0, 1) += .25;
            poly->invNalpha[1] *= .8;
            poly->dinv[1] = poly->invNalpha[1] * 4;
            poly->ddinv[1] = poly->invNalpha[1] * 12;
            return poly;
        };
        auto *next =
            cosine ? fitted(new RBasis_PolyBCos(basis->min_val, basis->max_val, basis->basis_size))
                   : fitted(new RBasis_PolyB(basis->min_val, basis->max_val, basis->basis_size));
        delete basis;
        basis = next;
    }
    auto atoms = swap_lattice(2, 2), expected = atoms;
    source.getEFS(expected);
    auto adjusted = MTP::adjust_tree(source, {.order = {1, 0}, .basis = "level-group"});
    for (int &id : atoms.chemmap_)
        id = 1 - id;
    compare(*adjusted, atoms, expected);
    Utils::TemporaryDirectory stage;
    const auto path = (stage.path / "polynomial.gpm").string();
    save_inference_model(*adjusted, path);
    auto reloaded = ModelBundle::load(path);
    compare(reloaded->potential(), atoms, expected);
}
