// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

/**
 * @file apps.h
 * @brief Command registration and dispatch for the imrmlp CLI.
 * @ingroup apps
 */

#pragma once

#include <map>
#include <string>
#include <functional>

#define MAKE_COMMAND(a) int a(int, char *[])
MAKE_COMMAND(model_convert);
MAKE_COMMAND(model_info);
MAKE_COMMAND(model_validate);
MAKE_COMMAND(convstr);
MAKE_COMMAND(calc_efs);
MAKE_COMMAND(calc_err);
MAKE_COMMAND(calc_eos);
MAKE_COMMAND(calc_fc);
MAKE_COMMAND(ti_fit_fc);
MAKE_COMMAND(calc_elastic);
MAKE_COMMAND(split_set);
MAKE_COMMAND(calc_ferr);
MAKE_COMMAND(calc_neb);
MAKE_COMMAND(calc_dimer);
MAKE_COMMAND(nroptimize);
MAKE_COMMAND(mtp2spline);
MAKE_COMMAND(adjustmtp);
MAKE_COMMAND(testpot);
MAKE_COMMAND(phonopyfc);
MAKE_COMMAND(nnmtp);
MAKE_COMMAND(calc_prop);
MAKE_COMMAND(calc_descriptors);
MAKE_COMMAND(mtpbasis);
MAKE_COMMAND(calc_trjavg);
MAKE_COMMAND(relaxstrv2);
MAKE_COMMAND(mtmbasis);
MAKE_COMMAND(modefollowing);
MAKE_COMMAND(smooth_traj);
MAKE_COMMAND(remove_duplicate_atoms);
MAKE_COMMAND(remove_duplicate_structures);
MAKE_COMMAND(trainmtp);
MAKE_COMMAND(trainnlpot);
MAKE_COMMAND(train_linear);
MAKE_COMMAND(initmtpm);
MAKE_COMMAND(mk_orth);
MAKE_COMMAND(mk_lower);
MAKE_COMMAND(mk_super);
MAKE_COMMAND(trans_reduce);
MAKE_COMMAND(primi_reduce);
MAKE_COMMAND(swap_axis);
MAKE_COMMAND(sameq);
MAKE_COMMAND(fastfit);
MAKE_COMMAND(mk_orth_search);
MAKE_COMMAND(findint);
MAKE_COMMAND(maxvol_select);
MAKE_COMMAND(maxvol_sample);
MAKE_COMMAND(maxvol_extend);
MAKE_COMMAND(diverse_select);
MAKE_COMMAND(pot2table);
MAKE_COMMAND(attach_hessian);
MAKE_COMMAND(find_mindist);
MAKE_COMMAND(calc_crowding);
#undef MAKE_COMMAND

extern std::map<std::string, std::pair<std::function<int(int, char *[])>, std::string>> command_maps;
