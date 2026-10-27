// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#pragma once

#include "mindriver/relax_new.h"
#include <string>

namespace AL { class ALPotential; }  // fwd

/// @brief Active-learning adapter for RelaxationDrvV2.
///
/// Thin bridge between RelaxationDrvV2's eval() pipeline and
/// AL::ALPotential::check().  Holds a raw pointer to ALPotential
/// (which owns all scorer state); does NOT own it.
///
/// NOT inserted into modifiers_list.  Instead RelaxationDrvV2 has a
/// dedicated ALModifier* field and calls forward() explicitly at the
/// end of eval(), avoiding ownership conflicts with free_modifiers().
///
/// Usage in app code:
///   auto al = AL::ALPotential::create(sett.ExtractSubSettings("al"));
///   AL::ALModifier al_mod(al.get());     // stack-allocated bridge
///   relaxdrv.al_modifier_ = &al_mod;
///   relaxdrv.init(pot, at, &constraints);
///   relaxdrv.run();
///   al->finalize();
class ALModifier : public AnyModifier {
public:
    explicit ALModifier(AL::ALPotential *al_pot = nullptr)
        : al_pot_(al_pot) { name = "ALModifier"; type = FILTER; }

    void set_al_pot(AL::ALPotential *p) { al_pot_ = p; }

    void forward(RelaxationDrvV2 *drv) override;

private:
    AL::ALPotential *al_pot_ = nullptr;
};
