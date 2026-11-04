// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#include <catch2/catch_all.hpp>
#include "apps/mtp_workflow.h"
#include "gpm_fixture.h"
using namespace Potentials;

TEST_CASE("MTP workflow initialization and migration preserve old species", "[mtp][workflow]")
{
    MTPWorkflow::Stage stage;
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
    MTPWorkflow::initialize(settings, file("source.mtp"), file("initial.gpm"), "", {}, false, true,
                            -.1, .1, "", "", "", false, false);
    auto model = ModelBundle::load(file("initial.gpm"));
    MTPWorkflow::initialize(GPSettings(), file("initial.gpm"), file("reused.gpm"), "", {}, false,
                            true, -.1, .1, "", "", "", false, false);
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
    auto permuted = MTPWorkflow::permute(mtp, {2, 0, 1});
    for (auto &id : atoms.chemmap_)
        id = id ? 2 : 1;
    permuted->getEFS(atoms);
    REQUIRE(atoms.energy_ == Catch::Approx(before.energy_).margin(1e-10));
    MTPWorkflow::reorder_basis(dynamic_cast<PotMTP &>(*permuted), "level");
    auto reordered = atoms;
    permuted->getEFS(reordered);
    REQUIRE(reordered.energy_ == Catch::Approx(atoms.energy_).margin(1e-10));
    for (size_t i = 0; i < atoms.size(); ++i)
        REQUIRE((reordered.forces_[i] - atoms.forces_[i]).norm() < 1e-9);
    REQUIRE((reordered.stress_ - atoms.stress_).norm() < 1e-8);
}
