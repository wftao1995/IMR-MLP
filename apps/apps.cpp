// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#include "apps.h"

#define REG_COMMAND(a, desc) {#a, std::make_pair(a, desc)}
#define REG_NICKNAME(a, b) {a, std::make_pair(b, "")}
#define REG_NICKCMD(a, b, desc) {a, std::make_pair(b, desc)}
/**
 * Command status:
 * transmatch: unfinished matching implementation; registration remains disabled.
 * sameq: uses the shared structure-comparison implementation.
 * calc_neb: optional variableimage mode inserts/removes images during optimization.
 * calc_descriptors: descriptor dispatch and CLI regression coverage are implemented.
 *
 * Need improve: 
 * nroptimize: make algorithm more stable
 * mtp2spline: need more variant support
 * nnmtp: need PotMTP_M type nn mtp initialize, and add torch support
 * 
 */

std::map<std::string, std::pair<std::function<int(int, char *[])>, std::string>> command_maps = {
    REG_NICKNAME("rmdups", remove_duplicate_atoms),
    REG_NICKNAME("unique", remove_duplicate_structures),
    REG_NICKNAME("mlpcalccellhess", calc_elastic),
    REG_NICKNAME("mlpcalcfc", calc_fc),
    REG_NICKNAME("efs", calc_efs),
    REG_NICKNAME("err", calc_err),
    REG_NICKNAME("ev", calc_eos),
    REG_NICKNAME("fc", calc_fc),
    REG_NICKNAME("elastic", calc_elastic),
    REG_NICKNAME("mlpphdisp", phonopyfc),
    REG_NICKNAME("calc_neb", calc_neb),
    REG_NICKNAME("calc_dimer", calc_dimer),
    REG_NICKCMD("dimer", calc_dimer, "Dimer saddle-point search"),
    REG_COMMAND(nroptimize, "Newton-Raphson stationary-point optimization"),
    REG_NICKCMD("relax", relaxstrv2, "Relax structures using various optimization methods and constraints"),
    REG_NICKCMD("neb", calc_neb, "Calculate NEB path between two structures"),
    REG_COMMAND(model_convert, "Convert potentials and active learning state to GPM"),
    REG_COMMAND(model_info, "Inspect GPM metadata without reading numerical blocks"),
    REG_COMMAND(model_validate, "Validate all GPM model and active learning state"),
    REG_COMMAND(convstr, "Convert structures between formats"),
    REG_COMMAND(calc_efs, "Calculate energy, force and stress tensors"),
    REG_COMMAND(calc_err, "Calculate errors relative to reference data"),
    REG_COMMAND(calc_eos, "Calculate E-V or P-V curves and optionally fit equation of state (use --curve ev|pv)"),
    REG_COMMAND(calc_fc, "Calculate and export force constants, or diagnose lowest modes using analytic HVPs"),
    REG_COMMAND(ti_fit_fc, "Fit a positive finite-temperature harmonic reference from MD or self-consistent ensembles"),
    REG_COMMAND(calc_elastic, "Calculate elastic constants using analytical Hessian or numerical strain method (use --method numerical)"),
    REG_COMMAND(split_set, "Split structure datasets into multiple files"),
    REG_COMMAND(calc_ferr, "Estimate free energy errors"),
    REG_COMMAND(mtp2spline, "Convert MTP typed potentials to spline representation"),
    REG_COMMAND(adjustmtp, "Reorder MTP potential"),
    REG_COMMAND(testpot, "Test potential by checking forces and stress tensors"),
    REG_COMMAND(phonopyfc, "Calculate FORCE_SETS according to phonopy_disp.yaml"),
    REG_COMMAND(nnmtp, "Create NNMTP potential template by MTP template"),
    REG_COMMAND(calc_prop, "Calculate structural properties: bond angles, dihedrals, RDF, etc."),
    REG_COMMAND(calc_descriptors, "Calculate descriptors for machine learning potentials"),
    REG_COMMAND(mtpbasis, "Compare MTP bases and safely remap linear parameters"),
    REG_COMMAND(calc_trjavg, "Compute average structure from trajectory data"),
    REG_COMMAND(mtmbasis, "Generate moment tensor basis for MTP potentials"),
    REG_COMMAND(modefollowing, "Mode following algorithm"),
    REG_COMMAND(smooth_traj, "Smooth trajectories using low-pass filtering"),
    REG_COMMAND(remove_duplicate_atoms, "Remove duplicate atoms based on distance criteria"),
    REG_COMMAND(remove_duplicate_structures, "Remove duplicate structures based on symmetry equivalance"),
    REG_COMMAND(trainmtp, "Train MTP potential using reference data"),
    REG_COMMAND(trainnlpot, "Train general nonlinear potential using reference data"),
    REG_COMMAND(train_linear, "Train linear descriptor-based potential using least-squares regression"),
    REG_COMMAND(initmtpm, "Initialize MTP_M potential with random coefficients"),
    // cell utils
    REG_COMMAND(mk_orth, "Find an (nearly) orthogonal cell for a given structure, lattice vector a will be kept"),
    REG_COMMAND(mk_orth_search, "Find an (nearly) orthogonal cell for a given structure, search all possible orientations"),
    REG_COMMAND(trans_reduce, "Reduce translational multiplicity of a structure along lattice directions, keep orientation"),
    REG_COMMAND(primi_reduce, "Reduce to primitive cell (60 deg < all angles < 120 deg), keep orientation"),
    REG_COMMAND(mk_lower, "Make cell lattice lower triangular"),
    REG_COMMAND(mk_super, "Make supercell, accept 3 or 9 (left, rowwise) integers."),
    REG_COMMAND(swap_axis, "Reorder a/b/c axis of a structure"),
    REG_COMMAND(sameq, "Check if two structures are equivalent"),
    REG_COMMAND(fastfit, "Fast MTP fitting using basis selection (CUR/SA/NSGA-II) and weighted least-squares"),
    REG_COMMAND(findint, "Find interstitial sites in crystal structures using Voronoi tessellation"),
    REG_COMMAND(maxvol_select, "Build a maximally-independent training set from a structure pool using MaxVol"),
    REG_COMMAND(maxvol_sample, "Grade structures against an existing MaxVol active set (extrapolation detection)"),
    REG_COMMAND(maxvol_extend, "Incrementally expand active set from candidates — supports --state_file for iterative selection"),
    REG_COMMAND(diverse_select, "Diversity-based representative selection (FPS or CUR) in descriptor space"),
    REG_NICKNAME("fps_select", diverse_select),
    REG_COMMAND(pot2table, "Export potential as tabulated interaction tables (PairTable, ThreebodyTable, LAMMPS format)"),
    REG_NICKCMD("hessian2xyz", attach_hessian, "Attach DFPT/FD Hessian to an extxyz structure"),
    REG_COMMAND(attach_hessian, "Attach DFPT vasprun.xml or FD FORCE_CONSTANTS Hessian to extxyz"),
    REG_COMMAND(find_mindist, "Find minimum distances between atoms in a structure"),
    REG_COMMAND(calc_crowding, "Compute per-atom crowding (descriptor-space density) weights as atomic_fitness")
};

#undef REG_COMMAND
#undef REG_NICKNAME
#undef REG_NICKCMD
