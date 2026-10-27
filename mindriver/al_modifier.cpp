// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#include "mindriver/al_modifier.h"
#include "active_learn/al_potential.h"
#include <stdexcept>

void ALModifier::forward(RelaxationDrvV2 *drv)
{
    if (al_pot_ && al_pot_->check(drv->atom))
        throw std::runtime_error("AL: threshold_break exceeded — "
                                 "aborting relaxation");
}
