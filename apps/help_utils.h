// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

/**
 * @file help_utils.h
 * @brief Helper macros and utilities for consistent --help output across commands.
 *
 * Provides convenience macros for registering ConnectSettings-based help groups
 * in the CommandLineOptions framework, reducing boilerplate in command files.
 */

#pragma once
#include <ostream>
#include <functional>
#include "../utils/cmdopts.h"
#include "../utils/gputils.h"

namespace HelpUtils {

/**
 * @brief Register a single ConnectSettings subclass as a help group.
 *
 * Creates a temporary object of @p Type and calls its ListSettings(os, ns)
 * when --help is displayed.
 *
 * Usage in a command function:
 * @code
 *   REGISTER_SETTINGS_HELP(cmdopts, "sro:", "SRO analysis", ComputeSRO);
 *   REGISTER_SETTINGS_HELP(cmdopts, "ha_index:", "HA indices", ComputeHA);
 * @endcode
 */
#define REGISTER_SETTINGS_HELP(cmdopts, ns, title, Type) \
    (cmdopts).attach_settings(ns, title, \
        [](std::ostream &os) { Type().ListSettings(os, ns); })

/**
 * @brief Unified --help handler.
 *
 * Call this after phrase_options() when you only need to show
 * standard help + registered settings groups and exit.
 */
inline void show_command_help(CMDOption::CommandLineOptions &cmdopts) {
    cmdopts.showhelp(std::cout);
    std::exit(0);
}

} // namespace HelpUtils
