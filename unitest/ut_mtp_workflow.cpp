// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#include "potentials/mtp_initialization.h"
#include "potentials/mtp_adjustment.h"
#include "utils/temporary_directory.h"
#include "potentials/potential_wrapper.h"
#include "potentials/potential_mtp_m.h"
#include "potentials/model_bundle.h"
#include "train/collective_error.h"
#include <catch2/catch_all.hpp>
#include "gpm_fixture.h"
using namespace Potentials;

TEST_CASE("MTP workflow initialization and migration preserve old species", "[mtp][workflow]")
{
    Utils::TemporaryDirectory stage;
    auto file = [&](const char *name)
    {
        return (stage.path / name).string();
    };
    PotMTP source;
    load_swap_fixture(source, 2, false);
    source.set_element_map({{0, "Al"}, {1, "Ni"}});
    source.Save(file("source.mtp"));
    GPSettings settings;
    settings.Modify("pot:pottype", "mtp_m");
    settings.Modify("pot:extmodel",
                    GENERATE("mtp_m_li", "mtp_m_q2", "mtp_m_fs", "mtp_m_n1", "mtp_m_n2"));
    settings.Modify("pot:n_input", "2");
    settings.Modify("pot:n_neuron", "3");
    settings.Modify("pot:n_neuron2", "2");
    Potentials::save_inference_model(
        *Potentials::MTP::initialize_model(settings, {.input = file("source.mtp"),
                                                      .type = "",
                                                      .elements = {},
                                                      .reset = false,
                                                      .randomize = true,
                                                      .lower = -.1,
                                                      .upper = .1,
                                                      .index = "",
                                                      .dataset = "",
                                                      .format = "",
                                                      .fit = false,
                                                      .normalize = false}),
        file("initial.gpm"));
    auto model = ModelBundle::load(file("initial.gpm"));
    Potentials::save_inference_model(
        *Potentials::MTP::initialize_model(GPSettings(), {.input = file("initial.gpm"),
                                                          .type = "",
                                                          .elements = {},
                                                          .reset = false,
                                                          .randomize = true,
                                                          .lower = -.1,
                                                          .upper = .1,
                                                          .index = "",
                                                          .dataset = "",
                                                          .format = "",
                                                          .fit = false,
                                                          .normalize = false}),
        file("reused.gpm"));
    auto reused = ModelBundle::load(file("reused.gpm"));
    REQUIRE(ModelBundle::fingerprint(model->potential()) ==
            ModelBundle::fingerprint(reused->potential()));
    auto atoms = swap_lattice(2, 2), before = atoms;
    model->potential().getEFS(before);
    auto &mtp = dynamic_cast<PotMTP &>(model->potential());
    mtp.append_species({1}, {"Cu"});
    auto after = atoms;
    mtp.getEFS(after);
    REQUIRE(after.energy_ == Catch::Approx(before.energy_).margin(1e-10));
    for (size_t i = 0; i < after.size(); ++i)
        REQUIRE((after.forces_[i] - before.forces_[i]).norm() < 1e-9);
    REQUIRE((after.stress_ - before.stress_).norm() < 1e-8);
    auto permuted = Potentials::MTP::adjust_tree(mtp, {.order = {2, 0, 1}});
    for (auto &id : atoms.chemmap_)
        id = id ? 2 : 1;
    permuted->getEFS(atoms);
    REQUIRE(atoms.energy_ == Catch::Approx(before.energy_).margin(1e-10));
    permuted = Potentials::MTP::adjust_tree(*permuted, {.basis = "level"});
    auto reordered = atoms;
    permuted->getEFS(reordered);
    REQUIRE(reordered.energy_ == Catch::Approx(atoms.energy_).margin(1e-10));
    for (size_t i = 0; i < atoms.size(); ++i)
        REQUIRE((reordered.forces_[i] - atoms.forces_[i]).norm() < 1e-9);
    REQUIRE((reordered.stress_ - atoms.stress_).norm() < 1e-8);
}
