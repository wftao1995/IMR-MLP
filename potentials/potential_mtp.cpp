// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

/**
 * @file potential_mtp.cpp
 * @author Jiantao Wang (wftao1995@gmail.com)
 * @brief 
 * @version 0.1
 * @date 2024-06-03
 * 
 * @copyright Copyright (c) 2024
 * 
 * 
 * Species-pair cutoffs and optional separate radial bases are supported.
 */

#include "potential_mtp.h"
#include <fstream>
#include "base/anyneighborhoods.h"
#include "base/neighborhood.h"
#include "basis/basis.h"
#include "tools/tools.h"
#include "utils/gputils.h"
#include <iomanip>
#include <algorithm>
#include <iterator>
#include <numeric>
#include "external/spglib_driver.h"
#include "utils/compressed_stream.h"
#include <fmt/format.h>
#include <string>
#include <ranges>
#include <sstream>
// #define USE_OMP
#ifdef USE_OMP
#include <omp.h>
#endif

using namespace Potentials;

namespace {

AnyBasis *clone_basis(AnyBasis &basis) {
    std::stringstream serialized;
    serialized << std::setprecision(std::numeric_limits<double>::max_digits10);
    basis.SaveToStream(serialized);

    std::string tag;
    std::string equals;
    std::string basis_name;
    if (!(serialized >> tag >> equals >> basis_name) ||
        tag != "radial_basis_type" || equals != "=") {
        GPERROR("Unable to clone radial basis");
    }
    return create_basis_by_name(basis_name, serialized, basis.scaling);
}

} // namespace

void PotMTP::clear_loaded_model() noexcept {
    delete[] alpha_moment_mapping;
    alpha_moment_mapping = nullptr;
    delete[] nu_arr_list;
    nu_arr_list = nullptr;
    if (!coeffs_handled_external) delete[] regression_coeffs;
    regression_coeffs = nullptr;
    loading_radial_coeffs_.reset();
    radial_coeffs = linear_coeffs = species_coeffs = moment_coeffs = nullptr;
    coeffs_handled_external = false;
    delete p_radical_basis;
    p_radical_basis = nullptr;
    for (auto *basis : p_radical_basis_pair) delete basis;
    p_radical_basis_pair.clear();
    radial_basis_split = false;
    species_count = reg_coeff_size = radial_coeff_size = lin_coeff_size = 0;
    species_coeff_size = moment_coeff_size = 0;
    alpha_moments_count = alpha_index_basic_count = alpha_index_times_count = alpha_scalar_moments = 0;
    rb_size = radial_func_count = max_alpha_index_basic = max_mu = max_level = 0;
    alpha_index_basic.resize(0, 4);
    alpha_index_times.resize(0, 4);
    cutoff = cutoffsq = 0;
    cutoffs.resize(0, 0);
    cutoffsqs.resize(0, 0);
    r_mins.resize(0, 0);
    potential_tag.fields.clear();
    potential_tag.legacy.clear();
    lev_scalar_nodes.clear();
    lev_scalar_count.clear();
    lev_times_nodes.clear();
    lev_times_count.clear();
    reg_L1_vector.clear();
    reg_L2_vector.clear();
    omp_buffers_list.clear();
    omp_inited = nl_inited = false;
    empty = radial_empty = linear_empty = true;
}

void PotMTP::allocate_loaded_coefficients() {
    auto coeffs = std::make_unique<double[]>(reg_coeff_size);
    if (radial_coeff_size > 0)
        std::copy_n(radial_coeffs, radial_coeff_size, coeffs.get());
    regression_coeffs = coeffs.release();
    loading_radial_coeffs_.reset();
    radial_coeffs = regression_coeffs;
}

void Potentials::PotMTP::Load(std::istream& ifs) try
{
    clear_loaded_model();
    std::string tmpstr;
    char tmpline[1000];
    std::string rbasis_type;

    ifs.getline(tmpline, 1000);
    int len = (int) ((std::string) tmpline).length();
    if (len > 0 && tmpline[len - 1] == '\r') // Ensures compatibility between Linux and Windows line endings
        tmpline[len - 1] = '\0';

    if ((std::string) tmpline != "MTP") {
        GPERROR("Can only read MTP format potentials");
    }

    // version reading block
    ifs.getline(tmpline, 1000);
    len = (int) ((std::string) tmpline).length();
    if (len > 0 && tmpline[len - 1] == '\r') // Ensures compatibility between Linux and Windows line endings
        tmpline[len - 1] = '\0';
    if ((std::string) tmpline != "version = 1.1.0")
        GPERROR("MTP file must have version \"1.1.0\"");

    // name/description reading block
    ifs >> tmpstr;
    if (tmpstr == "potential_name") // optional
    {
        ifs.ignore(2);
        ifs >> tmpstr;
        ifs >> tmpstr;
    }
    double scaling = 0.01;
    if (tmpstr == "scaling") // optional
    {
        ifs.ignore(2);
        ifs >> scaling;
        ifs >> tmpstr;
    }

    if (tmpstr != "species_count")
        GPERROR("Error reading .mtp file, no species_count tag found");
    ifs.ignore(2);
    ifs >> species_count;
    if (!ifs || species_count <= 0) GPERROR("Error reading .mtp file, invalid species_count");

    //@ADD@
    ifs >> tmpstr;

    potential_tag = {};
    if (tmpstr == "potential_tag") {
        getline(ifs, tmpstr);
        auto equals=tmpstr.find('=');
        if(equals==std::string::npos)GPERROR("Invalid potential_tag assignment");
        potential_tag=MTPMetadata::parse_tag(tmpstr.substr(equals+1),species_count);
        if(metadata_model_type()=="MTP") MTPMetadata::require_model(potential_tag);
        const auto declared=potential_tag.text("type");
        if(!declared.empty() && declared!="MTP" && declared!=metadata_model_type()) GPERROR("Potential tag model type conflicts with selected potential type");
        ifs >> tmpstr;
    }

    // if split-basis
    if (tmpstr == "radial_split" || tmpstr == "radial_basis_split") {
        ifs.ignore(2);
        int split_tag = 0;
        ifs >> split_tag;
        if (split_tag) {
            radial_basis_split = true;
        } else {
            radial_basis_split = false;
            ifs >> tmpstr;
        }
    }

    if (radial_basis_split) {
        /**
         * In this case:
         * - each species pair has its own radial basis, which can be different from other pairs
         * potential_tag = 
         * pair 0 1
         * radial_basis_type = rbtype1
         * pair 0 2
         * radial_basis_type = rbtype2
         * pair 1 0
         * radial_basis_type = rbtype1
         * ...
         */
        p_radical_basis_pair.resize(species_count * species_count);
        int ci, cj;
        int rb_last = 0;
        double cutmax = 0;
        cutoffs.resize(species_count, species_count);
        r_mins.resize(species_count, species_count);
        cutoffsqs.resize(species_count, species_count);
        for (int i = 0; i < species_count; ++i) {
            for (int j = 0; j < species_count; ++j) {
                ifs >> tmpstr;
                if (tmpstr != "pair") GPERROR("Error reading .mtp file, incorrect pair setting");
                ifs >> ci >> cj;
                if (!ifs || ci < 0 || ci >= species_count || cj < 0 || cj >= species_count)
                    GPERROR("Error reading .mtp file, invalid radial basis pair");
                if (p_radical_basis_pair[ci * species_count + cj])
                    GPERROR("Error reading .mtp file, duplicate radial basis pair");
                ifs >> tmpstr;
                if (tmpstr != "radial_basis_type") GPERROR("Error reading .mtp file, incorrect pair setting");
                ifs.ignore(2);
                ifs >> rbasis_type;
                btype = basis_name_to_code(rbasis_type);
                p_radical_basis_pair[ci * species_count + cj] = create_basis_by_code(btype, ifs, scaling);
                if (p_radical_basis_pair[ci * species_count + cj] == nullptr) {
                    GPERROR("Error reading .mtp file, FALILED TO CREATE BASIS");
                }
                const int pair_rb_size =
                    p_radical_basis_pair[ci * species_count + cj]->basis_size;
                if (rb_last == 0) {
                    rb_last = pair_rb_size;
                } else if (pair_rb_size != rb_last) {
                    GPERROR("Error reading .mtp file, split radial bases must have the same radial_basis_size");
                }
                double cut_this = p_radical_basis_pair[ci * species_count + cj]->max_val;
                if (cutmax < cut_this) cutmax = cut_this;
                cutoffs(ci, cj) = cut_this;
                cutoffsqs(ci, cj) = cut_this * cut_this;
                r_mins(ci, cj) = p_radical_basis_pair[ci * species_count + cj]->min_val;
            }
        }
        cutoff = cutmax;
        cutoffsq = cutmax * cutmax;
        rb_size = rb_last;
    } else {
        if (tmpstr != "radial_basis_type")
            GPERROR("Error reading .mtp file");
        ifs.ignore(2);
        ifs >> rbasis_type;
        // read rbasis type
        btype = basis_name_to_code(rbasis_type);
        p_radical_basis = create_basis_by_code(btype, ifs, scaling);
        if (p_radical_basis == nullptr) {
            GPERROR("Error reading .mtp file, FALILED TO CREATE BASIS");
        }
        cutoff = p_radical_basis->max_val;
        cutoffsq = cutoff * cutoff;
        cutoffs.resize(species_count, species_count);
        cutoffsqs.resize(species_count, species_count);
        r_mins.resize(species_count, species_count);
        cutoffs.setConstant(cutoff);
        cutoffsqs.setConstant(cutoffsq);
        r_mins.setConstant(p_radical_basis->min_val);
        rb_size = p_radical_basis->basis_size;
    }


    ifs >> tmpstr;
    if (tmpstr != "radial_funcs_count")
        GPERROR("Error reading .mtp file");
    ifs.ignore(2);
    ifs >> radial_func_count;
    if (!ifs || radial_func_count < 0 || rb_size <= 0)
        GPERROR("Error reading .mtp file, invalid radial basis dimensions");

    // Radial coeffs initialization
    int pairs_count = species_count * species_count; // number of species pairs

    char foo = ' ';

    loading_radial_coeffs_ = std::make_unique<double[]>(pairs_count * radial_func_count * rb_size);
    radial_coeffs = loading_radial_coeffs_.get();

    empty = false;

    ifs >> tmpstr;
    if (tmpstr == "radial_coeffs") {
        for (int s1 = 0; s1 < species_count; s1++)
            for (int s2 = 0; s2 < species_count; s2++) {
                int ts1, ts2;
                ifs >> ts1 >> foo >> ts2;
                if (!ifs || ts1 < 0 || ts1 >= species_count || ts2 < 0 || ts2 >= species_count)
                    GPERROR("Error reading .mtp file, invalid radial coefficient pair");

                double t;

                for (int i = 0; i < radial_func_count; ++i) {
                    ifs >> foo;
                    for (int j = 0; j < rb_size; ++j) {
                        if (!(ifs >> t >> foo)) GPERROR("Error reading .mtp file, missing radial coefficient");
                        radial_coeffs[(ts1 * species_count + ts2) * radial_func_count * rb_size +
                                      i * rb_size + j] = t;
                    }
                }
            }

        ifs >> tmpstr;
    } else {
        empty = true;
        for (int i = 0; i < pairs_count; ++i) {
            for (int k = 0; k < radial_func_count; ++k) {
                for (int l = 0; l < rb_size; ++l) {
                    radial_coeffs[i * radial_func_count * rb_size + k * rb_size + l] = 1e-5;
                }
                radial_coeffs[i * radial_func_count * rb_size + k * rb_size + std::min(k, rb_size - 1)] = 1e-3 + k;
            }
        }
    }

    if (tmpstr != "alpha_moments_count") {
        _DEBUG_OUTPUT_(tmpstr);
        GPERROR("Error reading .mtp file, except: alpha_moments_count");
    }
    ifs.ignore(2);
    ifs >> alpha_moments_count;
    if (ifs.fail())
        GPERROR("Error reading .mtp file");

    ifs >> tmpstr;
    if (tmpstr != "alpha_index_basic_count")
        GPERROR("Error reading .mtp file");
    ifs.ignore(2);
    ifs >> alpha_index_basic_count;
    if (ifs.fail())
        GPERROR("Error reading .mtp file");

    ifs >> tmpstr;
    if (tmpstr != "alpha_index_basic")
        GPERROR("Error reading .mtp file");
    ifs.ignore(4);

    alpha_index_basic = Eigen::MatrixXi(alpha_index_basic_count, 4);
    nu_arr_list = new int[alpha_index_basic_count]();

    int radial_func_max = -1;
    for (int i = 0; i < alpha_index_basic_count; ++i) {
        char tmpch;
        ifs.ignore(1000, '{');
        ifs >> alpha_index_basic(i, 0) >> tmpch >> alpha_index_basic(i, 1) >> tmpch >> alpha_index_basic(i, 2) >> tmpch
            >> alpha_index_basic(i, 3);
        nu_arr_list[i] = alpha_index_basic(i, 1) + alpha_index_basic(i, 2) + alpha_index_basic(i, 3);
        if (ifs.fail())
            GPERROR("Error reading .mtp file");

        if (alpha_index_basic(i, 0) > radial_func_max)
            radial_func_max = alpha_index_basic(i, 0);
    }

    if (radial_func_max != radial_func_count - 1)
        GPERROR("Wrong number of radial functions specified");

    ifs.ignore(1000, '\n');

    ifs >> tmpstr;
    if (tmpstr != "alpha_index_times_count")
        GPERROR("Error reading .mtp file");
    ifs.ignore(2);
    ifs >> alpha_index_times_count;
    if (ifs.fail())
        GPERROR("Error reading .mtp file");

    ifs >> tmpstr;
    if (tmpstr != "alpha_index_times")
        GPERROR("Error reading .mtp file");
    ifs.ignore(4);

    alpha_index_times = Eigen::MatrixXi(alpha_index_times_count, 4);
    for (int i = 0; i < alpha_index_times_count; ++i) {
        char tmpch;
        ifs.ignore(1000, '{');
        ifs >> alpha_index_times(i, 0) >> tmpch >> alpha_index_times(i, 1) >> tmpch >> alpha_index_times(i, 2) >> tmpch
            >> alpha_index_times(i, 3);
        if (ifs.fail())
            GPERROR("Error reading .mtp file");
    }

    ifs.ignore(1000, '\n');

    ifs >> tmpstr;
    if (tmpstr != "alpha_scalar_moments")
        GPERROR("Error reading .mtp file");
    ifs.ignore(2);
    ifs >> alpha_scalar_moments;
    if (alpha_index_times_count < 0)
        GPERROR("Error reading .mtp file");
    alpha_moment_mapping = new int[alpha_scalar_moments];
    ifs >> tmpstr;
    if (tmpstr != "alpha_moment_mapping")
        GPERROR("Error reading .mtp file");
    ifs.ignore(4);
    for (int i = 0; i < alpha_scalar_moments; ++i) {
        char tmpch = ' ';
        ifs >> alpha_moment_mapping[i] >> tmpch;
        if (ifs.fail())
            GPERROR("Error reading .mtp file");
    }
    ifs.ignore(1000, '\n');

    // Reading linear coeffs
    radial_coeff_size = pairs_count * radial_func_count * rb_size;
    lin_coeff_size = alpha_scalar_moments + species_count;
    reg_coeff_size = radial_coeff_size + lin_coeff_size;
    species_coeff_size = species_count;
    moment_coeff_size = alpha_scalar_moments;
    if (read_flag_early_exit) {
        nl_inited = false;
        empty = true;
        return;
    }
    allocate_loaded_coefficients();
    species_coeffs = regression_coeffs + radial_coeff_size;
    linear_coeffs = species_coeffs;
    moment_coeffs = linear_coeffs + species_count;
    ifs >> tmpstr;
    for (int i = 0; i < lin_coeff_size; ++i) {
        linear_coeffs[i] = 1e-3;
    }
    // case1. no species_coeffs and moment_coeffs
    // case2. have both
    // case3. have none 
    // todo: remove empty flag
    if (tmpstr == "species_coeffs") {
        ifs.ignore(4);
        for (int i = 0; i < species_count; ++i) {
            if (!(ifs >> species_coeffs[i] >> foo)) GPERROR("Error reading .mtp file, missing species coefficient");
        }
        ifs.ignore(1000, '\n');
        ifs >> tmpstr;
    }
    if (tmpstr == "moment_coeffs") {
        empty = false;
        ifs.ignore(2);
        ifs.ignore(10, '{');
        for (int i = 0; i < alpha_scalar_moments; ++i) {
            if (!(ifs >> moment_coeffs[i] >> foo)) GPERROR("Error reading .mtp file, missing moment coefficient");
        }
        ifs.ignore(1000, '\n');
        if (!coeffs_handled_external)
            ifs >> tmpstr;
    } else {
        empty = true;
    }

    nl_inited = true; // just for compability


    /*** level specific
    *    max_level = ?
    *    lev_scalar_nodes = {?, ?, ?}
    *    lev_scalar_count = {?, ?, ?}
    ***/
    if (!read_flag_immediate_return && tmpstr == "max_level") {
        ifs.ignore(2);
        ifs >> max_level;
        ifs.ignore(1000, '\n');
        ifs >> tmpstr;
        if (tmpstr != "lev_scalar_nodes") {
            GPERROR("no lev_scalar_nodes tag provided");
        }
        ifs.ignore(2);
        ifs.ignore(10, '{');
        lev_scalar_nodes.resize(max_level + 1);
        for (int i = 0; i < max_level + 1; ++i) {
            ifs >> lev_scalar_nodes[i] >> foo;
        }
        ifs.ignore(1000, '\n');
        ifs >> tmpstr;
        if (tmpstr != "lev_scalar_count") {
            GPERROR("no lev_scalar_count tag provided");
        }
        ifs.ignore(2);
        ifs.ignore(10, '{');
        lev_scalar_count.resize(max_level + 1);
        for (int i = 0; i < max_level + 1; ++i) {
            ifs >> lev_scalar_count[i] >> foo;
        }
        ifs.ignore(1000, '\n');
        ifs >> tmpstr;
        if (tmpstr != "lev_times_count") {
            std::cerr << "Error: lev_times_count not found" << std::endl;
        }
        ifs.ignore(2);
        ifs.ignore(10, '{');
        lev_times_count.resize(max_level + 1);
        for (int i = 0; i < max_level + 1; ++i) {
            ifs >> lev_times_count[i] >> foo;
        }
        lev_times_nodes.resize(max_level + 1);
        lev_times_nodes[0] = alpha_index_basic_count;
        for (int i = 1; i  < max_level + 1; ++i) {
            lev_times_nodes[i] = lev_times_nodes[i-1] + lev_times_count[i-1];
        }
    }


    init_basic();


    // Model deserialization is deliberately not an MPI collective. Library
    // users may load a model on one rank or within a subcommunicator.
}
catch (...) {
    clear_failed_load();
    throw;
}

void PotMTP::init_basic() {
    max_alpha_index_basic = 0;
    max_mu = -1;
    omp_inited = false;
    for (int i = 0; i < alpha_index_basic_count; ++i) {
        max_alpha_index_basic =
                std::max(max_alpha_index_basic,
                         alpha_index_basic(i, 1) + alpha_index_basic(i, 2) + alpha_index_basic(i, 3));
        max_mu = std::max(max_mu, alpha_index_basic(i, 0));
    }
    max_alpha_index_basic++;
    max_mu++;
    // allocate
    coords_powers.resize(max_alpha_index_basic + 1, 3); // {rij_x^k, rij_y^k, rij_z^k}
    coords_powers(0, 0) = coords_powers(0, 1) = coords_powers(0, 2) = 0;
    coords_powers(0 + 1, 0) = coords_powers(0 + 1, 1) = coords_powers(0 + 1, 2) = 1;
    // +6 for moment_hessian
    dist_powers.resize(max_alpha_index_basic + 6); // rij^k
    moment_vals.resize(alpha_moments_count, 0);
    rb_vals.resize(rb_size, 0);
    rb_ders.resize(rb_size, 0);
    rb_derders.resize(rb_size, 0);
    val_mu.resize(max_mu, 0); // basis function value
    der_mu.resize(max_mu, 0); // basis function der wrt r
    derder_mu.resize(max_mu, 0);
    site_energy_ders_wrt_moments_.resize(alpha_moments_count);
}

/**
 * @brief Loading potential form file
 *
 * @param potfile potential file name
 */
void PotMTP::Load(const std::string &potfile) try {
    if(load_gpm_if_requested(*this,potfile))return;
    auto ifs = open_ifile(potfile);
    if (!ifs) {
        GPERROR((std::string) "ERROR: can not open potential file " + potfile);
    }
    Load(*ifs);
}
catch (...) {
    clear_failed_load();
    throw;
}

/**
 * @brief Allocate memorys before training.
 * 
 */
void PotMTP::train_init() {
    moment_jacobian_coeffs.resize(alpha_index_basic_count, species_count, radial_func_count * rb_size);
    moment_jacobian_coeff_coord.resize(alpha_index_basic_count, species_count, radial_func_count * rb_size);
    dloss_dmom.resize(alpha_moments_count, 0);
    dloss_dsenders.resize(alpha_moments_count, 0);
    reg_L1_vector.assign(reg_coeff_size, 0);
    reg_L2_vector.assign(reg_coeff_size, 0);
}

void PotMTP::write_header(std::ostream &ofs, bool include_radial_split) {
    require_loaded_model();
    ofs << std::scientific;
    ofs << std::setprecision(std::numeric_limits<double>::max_digits10);
    ofs << "MTP\n";
    ofs << "version = 1.1.0\n";
    ofs << "potential_name = MTP1m\n";
    if (radial_basis_split) {
        ofs << "scaling = " << p_radical_basis_pair[0]->scaling << '\n';
    } else {
        ofs << "scaling = " << p_radical_basis->scaling << '\n';
    }
    ofs << "species_count = " << species_count << '\n';
    ofs << "potential_tag = " << MTPMetadata::Value(potential_tag.output(model_metadata(),species_count)).serialize() << '\n';
    if (radial_basis_split && include_radial_split) {
        ofs << "radial_basis_split = 1\n";
    }
}

void PotMTP::write_radial_info(std::ostream &ofs) {
    if (radial_basis_split) {
        for (int i = 0; i < species_count; ++i) {
            for (int j = 0; j < species_count; ++j) {
                ofs << "pair " << i << " " << j << "\n";
                p_radical_basis_pair[i * species_count + j]->SaveToStream(ofs);
            }
        }
    } else {
        p_radical_basis->SaveToStream(ofs);
    }
    ofs << "radial_funcs_count = " << radial_func_count << '\n';
}

void PotMTP::write_radial_coeff(std::ostream &ofs) {
        ofs << "radial_coeffs\n";
        int m = 0;
        for (int i = 0; i < species_count; ++i) {
            for (int j = 0; j < species_count; ++j) {
                ofs << "\t" << i << '-' << j << '\n';
                for (int k = 0; k < radial_func_count; ++k) {
                    ofs << "\t\t{";
                    for (int l = 0; l < rb_size; ++l) {
                        ofs << radial_coeffs[m++];
                        if (l != rb_size - 1)
                            ofs << ", ";
                        else
                            ofs << "}\n";
                    }
                }
            }
        }
}

void PotMTP::write_rules(std::ostream &ofs) {
    ofs << "alpha_moments_count = " << alpha_moments_count << '\n';
    ofs << "alpha_index_basic_count = " << alpha_index_basic_count << '\n';
    ofs << "alpha_index_basic = {";
    for (int i = 0; i < alpha_index_basic_count; ++i) {
        ofs << '{'
            << alpha_index_basic(i, 0) << ", "
            << alpha_index_basic(i, 1) << ", "
            << alpha_index_basic(i, 2) << ", "
            << alpha_index_basic(i, 3) << "}";
        if (i < alpha_index_basic_count - 1)
            ofs << ", ";
    }
    ofs << "}\n";
    ofs << "alpha_index_times_count = " << alpha_index_times_count << '\n';
    ofs << "alpha_index_times = {";
    for (int i = 0; i < alpha_index_times_count; i++) {
        ofs << '{'
            << alpha_index_times(i, 0) << ", "
            << alpha_index_times(i, 1) << ", "
            << alpha_index_times(i, 2) << ", "
            << alpha_index_times(i, 3) << "}";
        if (i < alpha_index_times_count - 1)
            ofs << ", ";
    }
    ofs << "}\n";

    ofs << "alpha_scalar_moments = " << alpha_scalar_moments << '\n';

    ofs << "alpha_moment_mapping = {";
    for (int i = 0; i < alpha_scalar_moments; ++i) {
        if (i > 0) ofs << ", ";
        ofs << alpha_moment_mapping[i];
    }
    ofs << "}\n";
}

void PotMTP::write_linear_coeff(std::ostream &ofs) {
        ofs << "species_coeffs = {";
        if (species_coeffs) {
            for (int i = 0; i < species_coeff_size; i++) {
                if (i != 0)
                    ofs << ", ";
                ofs << species_coeffs[i];
            }
        } else {
            for (int i = 0; i < species_coeff_size; ++i) {
                if (i != 0)
                    ofs << ", ";
                ofs << 0.0;
            }
        }
        ofs << "}\n";

        ofs << "moment_coeffs = {";
        for (int i = 0; i < moment_coeff_size; i++) {
            if (i != 0)
                ofs << ", ";
            ofs << moment_coeffs[i];
        }
        ofs << "}\n";
}
// FORWARD GROUP AND BACKWARD GROUP
void Potentials::PotMTP::ReorderByLevelGroup() {
    struct alpha_index_basic_t {
        int mu, b1, b2, b3;
    };
    struct alpha_times_t {
        int l1, l2, c, id;
    };
    typedef std::pair<int, alpha_index_basic_t> basic_t;


    max_level = 0;
    lev_scalar_count.clear();
    lev_scalar_nodes.clear();
    lev_times_count.clear();

    // level record for each b
    std::map<int, int> bidlev;
    for (int i = 0; i < alpha_moments_count; ++i) {
        bidlev[i] = -1;
    }

    // 1. sort alpha_index_basic by mu and b1, b2, b3
    std::vector<basic_t> basic_index(alpha_index_basic_count);
    for (int i = 0; i < alpha_index_basic_count; ++i) {
        alpha_index_basic_t tid;
        tid.mu = alpha_index_basic(i, 0);
        tid.b1 = alpha_index_basic(i, 1);
        tid.b2 = alpha_index_basic(i, 2);
        tid.b3 = alpha_index_basic(i, 3);
        basic_index[i] = std::make_pair(i, tid);
        bidlev[i] = 0;
    }
    std::stable_sort(basic_index.begin(), basic_index.end(), [](const basic_t &a, const basic_t &b) {
        if (a.second.b1 + a.second.b2 + a.second.b3 < b.second.b1 + b.second.b2 + b.second.b3) return true; 
        if (a.second.b1 + a.second.b2 + a.second.b3 > b.second.b1 + b.second.b2 + b.second.b3) return false; 
        if (a.second.mu < b.second.mu) return true; 
        return false;
    });
    for (int i = 0; i < alpha_index_basic_count; ++i) {
        alpha_index_basic(i, 0) = basic_index[i].second.mu;
        alpha_index_basic(i, 1) = basic_index[i].second.b1;
        alpha_index_basic(i, 2) = basic_index[i].second.b2;
        alpha_index_basic(i, 3) = basic_index[i].second.b3;
        nu_arr_list[i] = basic_index[i].second.b1 + basic_index[i].second.b2 + basic_index[i].second.b3;
    }
    // 2. change alpha_index_times and alpha_moment_mapping
    std::map<int, int> index_basic_old_to_new;
    for (int i = 0; i < alpha_index_basic_count; ++i) {
        index_basic_old_to_new[basic_index[i].first] = i;
    }
    for (int i = 0; i < alpha_index_times_count; ++i) {
        int l1 = alpha_index_times(i, 0);
        int l2 = alpha_index_times(i, 1);
        if (l1 < alpha_index_basic_count)
            alpha_index_times(i, 0) = index_basic_old_to_new[l1];
        if (l2 < alpha_index_basic_count)
            alpha_index_times(i, 1) = index_basic_old_to_new[l2];
    }
    for (int i = 0; i < alpha_scalar_moments; ++i) {
        int id = alpha_moment_mapping[i];
        if (id < alpha_index_basic_count) {
            alpha_moment_mapping[i] = index_basic_old_to_new[id];
        }
    }
    // 3. index times 
    typedef std::pair<alpha_times_t, std::array<int, 5>> times_t; // id(will switch to current id in following loop), bid, blev, bdup, bisscalar
    std::map<int, int> bdup; // how many times bid is present in alpha_index_times list
    std::map<int, int> isscalar;
    for (int i = 0; i < alpha_scalar_moments; ++i) {
        isscalar[alpha_moment_mapping[i]] = 1;
    }
    std::vector<times_t> alpha_index_times_pair(alpha_index_times_count);
    for (int i = 0; i < alpha_index_times_count; ++i) {
        alpha_times_t ttimes;
        ttimes.l1 = alpha_index_times(i, 0);
        ttimes.l2 = alpha_index_times(i, 1);
        ttimes.c  = alpha_index_times(i, 2);
        ttimes.id = alpha_index_times(i, 3);
        std::array<int, 5> ntmp;
        if (bidlev[ttimes.l1] == -1 || bidlev[ttimes.l2] == -1) {
            std::cerr << "Error: incorrect order ";
            std::cerr << ttimes.l1 << " " << ttimes.l2 << " " << ttimes.c << " " << ttimes.id;
            std::cerr << std::endl;
            exit(1);
        }
        if (bidlev[ttimes.id] == -1)
            bdup[ttimes.id] = 0;
        ++bdup[ttimes.id];
        int lev = std::max(bidlev[ttimes.l1], bidlev[ttimes.l2]) + 1;
        lev = std::max(lev, bidlev[ttimes.id]);
        max_level = std::max(lev, max_level);
        bidlev[ttimes.id] = lev;
        ntmp[0] = i;
        ntmp[1] = ttimes.id;
        ntmp[2] = lev;
        ntmp[3] = bdup[ttimes.id];
        ntmp[4] = isscalar[ttimes.id];
        alpha_index_times_pair[i] = std::make_pair(ttimes, ntmp);
    }
    for (int i = 0; i < alpha_index_times_count; ++i) {
        int bid = alpha_index_times_pair[i].second[1];
        alpha_index_times_pair[i].second[2] = bidlev[bid];
    }
    // 3.2 sort index times by level and is_scalar
    stable_sort(alpha_index_times_pair.begin(), alpha_index_times_pair.end(), [](const times_t &a, const times_t &b) {
        if (a.second[2] < b.second[2]) return true;
        if (a.second[2] > b.second[2]) return false;
        if (a.first.l1 < b.first.l1) return true;
        if (a.first.l1 > b.first.l1) return false;
        if (a.first.l2 < b.first.l2) return true;
        if (a.first.l2 > b.first.l2) return false; // make sure {a+b ->c, a+b->d} are adjuscent
        if (a.second[4] > b.second[4]) return true; // leading terms are scalar ones
        return false;
    });
    int bidprev = 0;
    int id_curr = alpha_index_basic_count - 1;
    for (int i = 0; i < alpha_index_times_count; ++i) {
        int bid = alpha_index_times_pair[i].second[1];
        if (bidprev != bid) {
            bidprev = bid;
            ++id_curr;
        }
        alpha_index_times_pair[i].second[0] = id_curr;
    }
    // 3.3 re-index alpha_index_times array
    std::map<int, int> index_nonbasic_old_to_new;
    for (int i = 0; i < alpha_index_times_count; ++i) {
        index_nonbasic_old_to_new[alpha_index_times_pair[i].second[1]] = alpha_index_times_pair[i].second[0];
    }
    lev_times_count.resize(max_level + 1, 0);
    for (int i = 0; i < alpha_index_times_count; ++i) {
        int l1 = alpha_index_times_pair[i].first.l1;
        int l2 = alpha_index_times_pair[i].first.l2;
        int id = alpha_index_times_pair[i].first.id;
        int lev = alpha_index_times_pair[i].second[2];
        lev_times_count[lev]++;
        if (l1 >= alpha_index_basic_count)
            alpha_index_times_pair[i].first.l1 = index_nonbasic_old_to_new[l1];
        if (l2 >= alpha_index_basic_count)
            alpha_index_times_pair[i].first.l2 = index_nonbasic_old_to_new[l2];
        alpha_index_times_pair[i].first.id = index_nonbasic_old_to_new[id];
        alpha_index_times(i, 0) = alpha_index_times_pair[i].first.l1;
        alpha_index_times(i, 1) = alpha_index_times_pair[i].first.l2;
        alpha_index_times(i, 2) = alpha_index_times_pair[i].first.c;
        alpha_index_times(i, 3) = alpha_index_times_pair[i].first.id;
    }
    for (int i = 0; i < alpha_scalar_moments; ++i) {
        if (alpha_moment_mapping[i] >= alpha_index_basic_count) {
            alpha_moment_mapping[i] = index_nonbasic_old_to_new[alpha_moment_mapping[i]];
        }
    }
    //4. resort alpha_moment_mapping and adjust coeffs
    std::vector<std::pair<int, double>> tmp_moment_coeffs(alpha_scalar_moments);
    for (int i = 0; i < alpha_scalar_moments; ++i) {
        tmp_moment_coeffs[i] = std::make_pair(alpha_moment_mapping[i], moment_coeffs[i]);
    }
    std::ranges::sort(tmp_moment_coeffs, 
        [](const std::pair<int, double>&a, const std::pair<int, double>&b) {
            if (a.first < b.first) return true;
            return false;
        });
    int mom_prev = 0;
    int levcurr = 0;
    lev_scalar_count.resize(max_level + 1, 0);
    for (int i = 0; i < alpha_scalar_moments; ++i) {
        alpha_moment_mapping[i] = tmp_moment_coeffs[i].first;
        moment_coeffs[i] = tmp_moment_coeffs[i].second;
        if (alpha_moment_mapping[i] == mom_prev++) {
            ++lev_scalar_count[levcurr];
        } else {
            ++levcurr;
            mom_prev = alpha_moment_mapping[i] + 1;
            ++lev_scalar_count[levcurr];
        }
    }
    int tmp_accu = 0;
    lev_scalar_nodes.resize(max_level+1, 0);
    for (int i = 1; i <= max_level; ++i) {
        tmp_accu += lev_scalar_count[i - 1];
        lev_scalar_nodes[i] = alpha_moment_mapping[tmp_accu];
    }
}

void PotMTP::Save(const std::string &potf) {
    if (save_gpm_if_requested(*this, potf)) return;
    auto ofs = open_ofile(potf);
    if (!ofs) {
        throw std::runtime_error("Failed to open file: " + potf);
    }
    try {
        Save(*ofs);
    } catch (const std::exception &error) {
        throw std::runtime_error("Failed to write " + potf + ": " + error.what());
    }
    if (!close_ofile(ofs))
        throw std::runtime_error("Failed to close output file: " + potf);
}

/**
 * @brief Save potential to file
 *
 * @param potfile potential file name
 */
void PotMTP::Save(std::ostream &os) {
    write_header(os);
    write_radial_info(os);
    write_radial_coeff(os);
    write_rules(os);
    write_linear_coeff(os);
    if (!os) throw std::runtime_error("Failed to write MTP potential");
}

void PotMTP::SaveSpline(std::string potfile, double rmin, double dr) {
    auto ofs = open_ofile(potfile);
    if (!ofs) throw std::runtime_error("Failed to open file: " + potfile);
    // Spline tables already store each ordered species pair independently.
    write_header(*ofs, false);
    // radial info
    *ofs << "radial_basis_type = RBSpline\n";
    *ofs << "\t\tmin_dist = " << 0.0 << '\n';
    *ofs << "\t\tmax_dist = " << cutoff << '\n';
    // spline info
    *ofs << "\tradial_splines\n";
    // first line: Nr r
    // pair i-j
    // radial values
    std::vector<double> rs;
    int Nr;
    double rg = cutoff;
    Nr = ceil(rg / dr);
    dr = rg / (Nr-1);
    rs.resize(Nr);
    for (int i = 0; i < Nr; ++i) {
        rs[i] = i * dr;
    }
    *ofs << "\t\t" << Nr << " " << dr << " " << radial_func_count << "\n\t\t";
    for (int i = 0; i < Nr; ++i) {
        *ofs << rs[i] << " ";
    }
    *ofs << '\n';
    // spline values
    Eigen::Tensor<double, 4> rbv;
    rbv.resize(species_count, species_count, radial_func_count, Nr);
    for (int i = 0; i < species_count; ++i) {
        for (int j = 0; j < species_count; ++j) {
            const double inner_radius = rmin < 0 ? r_mins(i, j) : rmin;
            for (int k = 0; k < Nr; ++k) {
                const double r = std::max(rs[k], inner_radius);
                if (r > cutoffs(i, j))
                    std::fill(rb_vals.begin(), rb_vals.end(), 0.0);
                else
                    rb_calcval(i, j, r);
                for (int mu = 0; mu < radial_func_count; ++mu) {
                    double value = 0;
                    int idxoffset = (i * species_count + j) * radial_func_count * rb_size + mu * rb_size;
                    for (int xi = 0; xi < rb_size; ++xi) {
                        value += radial_coeffs[idxoffset + xi] * rb_vals[xi];
                    }
                    rbv(i, j, mu, k) = value;
                }
            }
        }
    }
    for (int i = 0; i < species_count; ++i) {
        for (int j = 0; j < species_count; ++j) {
            for (int mu = 0; mu < radial_func_count; ++mu) {
                *ofs << "\t\t" << mu << " " << i << " " << j << " ";
                for (int k = 0; k < Nr; ++k) {
                    *ofs << rbv(i, j, mu, k) << " ";
                }
                *ofs << '\n';
            }
        }
    }    

    write_rules(*ofs);
    write_linear_coeff(*ofs);
    if (!close_ofile(ofs))
        throw std::runtime_error("Failed to write output file: " + potfile);
}

/**
 * @brief Save potential to file, scalar moments will be reordered
 * 
 * @param potfile Potential file name
 */
void PotMTP::SaveOrdered(std::string potfile) {
    auto ofs = open_ofile(potfile);
    if (!ofs) throw std::runtime_error("Failed to open file: " + potfile);
    write_header(*ofs);
    write_radial_info(*ofs);
    if (!empty) write_radial_coeff(*ofs);
    write_rules(*ofs);
    if (!empty) write_linear_coeff(*ofs);

    *ofs << "max_level = " << max_level << "\n";
    *ofs << "lev_scalar_nodes = {";
    for (int i = 0; i < max_level + 1; ++i) {
        if (i != 0) *ofs << ", ";
        *ofs << lev_scalar_nodes[i];
    }
    *ofs << "}\n";
    *ofs << "lev_scalar_count = {";
    for (int i = 0; i < max_level + 1; ++i) {
        if (i != 0) *ofs << ", ";
        *ofs << lev_scalar_count[i];
    }
    *ofs << "}\n";
    *ofs << "lev_times_count = {";
    for (int i = 0; i < max_level + 1; ++i) {
        if (i != 0) *ofs << ", ";
        *ofs << lev_times_count[i];
    }
    *ofs << "}\n";

    // create files for ladder fitting
    if (!close_ofile(ofs))
        throw std::runtime_error("Failed to write output file: " + potfile);
    int times_count_prev = lev_times_count[0];
    int scalar_count_prev = lev_scalar_count[0];
    for (int i = 0; i < max_level; ++i) {
        const auto ladderfile = potfile + ".ladder" + std::to_string(i+2);
        auto ofsladder = open_ofile(ladderfile);
        if (!ofsladder) throw std::runtime_error("Failed to open file: " + ladderfile);
        write_header(*ofsladder);
        write_radial_info(*ofsladder);
        if (!empty) write_radial_coeff(*ofsladder);
        *ofsladder << "alpha_moments_count = " << lev_scalar_nodes[i + 1]  << '\n';
        *ofsladder << "alpha_index_basic_count = " << alpha_index_basic_count << '\n';
        *ofsladder << "alpha_index_basic = {";
        for (int i = 0; i < alpha_index_basic_count; ++i) {
            *ofsladder << '{'
                << alpha_index_basic(i, 0) << ", "
                << alpha_index_basic(i, 1) << ", "
                << alpha_index_basic(i, 2) << ", "
                << alpha_index_basic(i, 3) << "}";
            if (i < alpha_index_basic_count - 1)
                *ofsladder << ", ";
        }
        *ofsladder << "}\n";
        int maxmomid = alpha_moment_mapping[scalar_count_prev - 1];
        int upto = 0;
        for (; upto < alpha_index_times_count; ++upto) {
            if (alpha_index_times(upto, 3) > maxmomid) break;
        }
        *ofsladder << "alpha_index_times_count = " << upto << '\n';
        *ofsladder << "alpha_index_times = {";
        for (int i = 0; i < upto; i++) {
            *ofsladder << '{'
                << alpha_index_times(i, 0) << ", "
                << alpha_index_times(i, 1) << ", "
                << alpha_index_times(i, 2) << ", "
                << alpha_index_times(i, 3) << "}";
            if (i < upto - 1)
                *ofsladder << ", ";
        }
        *ofsladder << "}\n";
        times_count_prev += lev_times_count[i+1];

        *ofsladder << "alpha_scalar_moments = " << scalar_count_prev << '\n';

        *ofsladder << "alpha_moment_mapping = {";
        for (int i = 0; i < scalar_count_prev - 1; i++)
            *ofsladder << alpha_moment_mapping[i] << ", ";
        *ofsladder << alpha_moment_mapping[scalar_count_prev - 1] << "}\n";

        if (!empty) {
            *ofsladder << "species_coeffs = {";
            for (int species = 0; species < species_count; ++species) {
                if (species != 0) *ofsladder << ", ";
                *ofsladder << species_coeffs[species];
            }
            *ofsladder << "}\n";

            *ofsladder << "moment_coeffs = {";
            for (int scalar = 0; scalar < scalar_count_prev; ++scalar) {
                if (scalar != 0) *ofsladder << ", ";
                *ofsladder << moment_coeffs[scalar];
            }
            *ofsladder << "}\n";
        }

        *ofsladder << "max_level = " << i << "\n";
        *ofsladder << "lev_scalar_nodes = {";
        for (int level = 0; level <= i; ++level) {
            if (level != 0) *ofsladder << ", ";
            *ofsladder << lev_scalar_nodes[level];
        }
        *ofsladder << "}\n";
        *ofsladder << "lev_scalar_count = {";
        for (int level = 0; level <= i; ++level) {
            if (level != 0) *ofsladder << ", ";
            *ofsladder << lev_scalar_count[level];
        }
        *ofsladder << "}\n";
        *ofsladder << "lev_times_count = {";
        for (int level = 0; level <= i; ++level) {
            if (level != 0) *ofsladder << ", ";
            *ofsladder << lev_times_count[level];
        }
        *ofsladder << "}\n";
        scalar_count_prev += lev_scalar_count[i+1];
        if (!close_ofile(ofsladder))
            throw std::runtime_error("Failed to write output file: " + ladderfile);
    }
}
AnyPotential::SpeciesAppendResult PotMTP::append_species(
    const std::vector<int>& clone_sources,const std::vector<std::string>& new_elements) {
    if(!supports_species_append()) GPERROR("Species append requires an ordinary owned MTP model");
    require_loaded_model();
    const int old_count=species_count,old_parameters=reg_coeff_size;
    for(int source:clone_sources)
        if(source<0 || source>=old_count) GPERROR("Species clone source must be an existing species ID");
    if(!new_elements.empty() && new_elements.size()!=clone_sources.size())
        GPERROR("Appended element declarations must match clone sources");
    auto new_metadata=potential_tag;
    auto names=potential_tag.elements();
    if(!clone_sources.empty() && !names.empty()) {
        if(new_elements.size()!=clone_sources.size())
            GPERROR("Appending to an element-tagged MTP requires explicit new element names");
        names.insert(names.end(),new_elements.begin(),new_elements.end());
        MTPMetadata::Array elements;for(const auto& name:names) elements.emplace_back(name);
        new_metadata.fields["elements"]=MTPMetadata::Value(elements);
        new_metadata.validate(old_count+clone_sources.size());
    } else if(!new_elements.empty()) {
        GPERROR("Declare the existing element map before appending named elements");
    }
    SpeciesAppendResult result;
    result.old_to_new_parameters.resize(old_parameters);
    if(clone_sources.empty()) {
        std::iota(result.old_to_new_parameters.begin(),result.old_to_new_parameters.end(),0);
        return result;
    }
    const size_t new_count_size=size_t(old_count)+clone_sources.size();
    const size_t stride=size_t(radial_func_count)*rb_size;
    if(new_count_size>size_t(std::numeric_limits<int>::max()) ||
        new_count_size*new_count_size>(size_t(std::numeric_limits<int>::max())-new_count_size-alpha_scalar_moments)/stride)
        GPERROR("Appended MTP parameter layout is too large");
    const int new_count=new_count_size,new_radial=new_count*new_count*stride;
    const int new_parameters=new_radial+new_count+alpha_scalar_moments;
    std::vector<int> sources(new_count);std::iota(sources.begin(),sources.begin()+old_count,0);
    std::copy(clone_sources.begin(),clone_sources.end(),sources.begin()+old_count);
    auto coefficients=std::make_unique<double[]>(new_parameters);
    std::vector<int> source_parameter(new_parameters);
    for(int i=0;i<new_count;++i) for(int j=0;j<new_count;++j)
        for(size_t a=0;a<stride;++a) {
            const int dst=(i*new_count+j)*stride+a;
            const int src=(sources[i]*old_count+sources[j])*stride+a;
            source_parameter[dst]=src;coefficients[dst]=regression_coeffs[src];
            if(i<old_count && j<old_count) result.old_to_new_parameters[src]=dst;
        }
    for(int i=0;i<new_count;++i) {
        const int dst=new_radial+i,src=radial_coeff_size+sources[i];
        source_parameter[dst]=src;coefficients[dst]=regression_coeffs[src];
        if(i<old_count) result.old_to_new_parameters[src]=dst;
    }
    for(int a=0;a<alpha_scalar_moments;++a) {
        const int src=radial_coeff_size+old_count+a,dst=new_radial+new_count+a;
        source_parameter[dst]=src;coefficients[dst]=regression_coeffs[src];
        result.old_to_new_parameters[src]=dst;
    }
    // Prepare every allocation and validate potential_tag before replacing model data.
    std::vector<std::unique_ptr<AnyBasis>> appended_bases;
    std::vector<AnyBasis*> bases;
    if(radial_basis_split) {
        bases.resize(new_count*new_count);
        for(int i=0;i<new_count;++i) for(int j=0;j<new_count;++j) {
            auto* source=p_radical_basis_pair[sources[i]*old_count+sources[j]];
            if(i<old_count && j<old_count) bases[i*new_count+j]=source;
            else {
                appended_bases.emplace_back(clone_basis(*source));
                bases[i*new_count+j]=appended_bases.back().get();
            }
        }
    }
    Eigen::MatrixXd new_cutoffs(new_count,new_count),new_cutoffsqs(new_count,new_count),new_rmins(new_count,new_count);
    for(int i=0;i<new_count;++i) for(int j=0;j<new_count;++j) {
        new_cutoffs(i,j)=cutoffs(sources[i],sources[j]);
        new_cutoffsqs(i,j)=cutoffsqs(sources[i],sources[j]);
        new_rmins(i,j)=r_mins(sources[i],sources[j]);
    }
    auto remap_regularization=[&](const std::vector<double>& old) {
        // Legacy trainers own a linear/subset-layout penalty vector. Its
        // indexing cannot be inferred here; discard those weights and let
        // trainer reinitialization rebuild them for the appended model.
        // Empty vectors also become full zeros for safe direct penalty calls.
        std::vector<double> values(new_parameters,0.);
        if(old.size()==size_t(old_parameters))
            for(int i=0;i<new_parameters;++i) values[i]=old[source_parameter[i]];
        return values;
    };
    auto l1=remap_regularization(reg_L1_vector),l2=remap_regularization(reg_L2_vector);
    delete[] regression_coeffs;regression_coeffs=coefficients.release();
    species_count=new_count;radial_coeff_size=new_radial;reg_coeff_size=new_parameters;
    species_coeff_size=new_count;moment_coeff_size=alpha_scalar_moments;
    lin_coeff_size=new_count+alpha_scalar_moments;
    radial_coeffs=regression_coeffs;linear_coeffs=species_coeffs=radial_coeffs+radial_coeff_size;
    moment_coeffs=species_coeffs+species_count;
    if(radial_basis_split) {p_radical_basis_pair=std::move(bases);for(auto& basis:appended_bases) basis.release();}
    cutoffs=std::move(new_cutoffs);cutoffsqs=std::move(new_cutoffsqs);r_mins=std::move(new_rmins);
    potential_tag=std::move(new_metadata);
    omp_buffers_list.clear();omp_inited=false;
    train_init();reg_L1_vector=std::move(l1);reg_L2_vector=std::move(l2);
    return result;
}

void PotMTP::adjust_species(const std::vector<int> &newsp) {
    const int sp_count_old = species_count;
    const int sp_count_new = static_cast<int>(newsp.size());
    if (sp_count_new < sp_count_old) {
        GPERROR("New species order cannot contain fewer species than the source potential");
    }

    std::vector<bool> seen(sp_count_new, false);
    for (int source : newsp) {
        if (source < 0 || source >= sp_count_new) {
            GPERROR("New species order contains an out-of-range species index");
        }
        if (seen[source]) {
            GPERROR("New species order must be a permutation without duplicates");
        }
        seen[source] = true;
    }

    auto labels=element_symbols();
    if(!labels.empty()&&sp_count_new>sp_count_old)GPERROR("Expanding a tagged model requires an explicit new element declaration");
    if(!labels.empty()){
        std::vector<std::string> reordered;for(int source:newsp)reordered.push_back(labels[source]);
        MTPMetadata::Array values;for(const auto &name:reordered)values.emplace_back(name);
        potential_tag.fields["elements"]=MTPMetadata::Value(values);
    }
    const int pair_stride = radial_func_count * rb_size;
    const int radial_coeff_size_new = sp_count_new * sp_count_new * pair_stride;
    const int lin_coeff_size_new = sp_count_new + alpha_scalar_moments;
    const int reg_coeff_size_new = radial_coeff_size_new + lin_coeff_size_new;
    std::vector<double> coeffs_new(reg_coeff_size_new, 0.0);
    std::vector<bool> new_pair(sp_count_new * sp_count_new, false);
    std::uniform_real_distribution<double> uniform(random_lower, random_upper);

    for (int i = 0; i < sp_count_new; ++i) {
        const int source_i = newsp[i];
        coeffs_new[radial_coeff_size_new + i] =
            source_i < sp_count_old ? species_coeffs[source_i] : uniform(RNG.mtd);
        for (int j = 0; j < sp_count_new; ++j) {
            const int source_j = newsp[j];
            const int dst_pair = i * sp_count_new + j;
            const int dst_offset = dst_pair * pair_stride;
            if (source_i < sp_count_old && source_j < sp_count_old) {
                const int src_offset =
                    (source_i * sp_count_old + source_j) * pair_stride;
                std::copy_n(radial_coeffs + src_offset, pair_stride,
                            coeffs_new.begin() + dst_offset);
            } else {
                new_pair[dst_pair] = true;
                std::generate_n(coeffs_new.begin() + dst_offset, pair_stride,
                                [&]() { return uniform(RNG.mtd); });
            }
        }
    }
    std::copy_n(moment_coeffs, alpha_scalar_moments,
                coeffs_new.begin() + radial_coeff_size_new + sp_count_new);

    if (sp_count_new > sp_count_old && !radial_mode) {
        for (int k = 0; k < radial_func_count; ++k) {
            for (int k2 = 0; k2 < k; ++k2) {
                double denominator = 0.0;
                double projection = 0.0;
                for (int pair = 0; pair < sp_count_new * sp_count_new; ++pair) {
                    if (!new_pair[pair]) continue;
                    for (int l = 0; l < rb_size; ++l) {
                        const int idx = pair * pair_stride + k * rb_size + l;
                        const int idx2 = pair * pair_stride + k2 * rb_size + l;
                        projection += coeffs_new[idx] * coeffs_new[idx2];
                        denominator += coeffs_new[idx2] * coeffs_new[idx2];
                    }
                }
                if (denominator > std::numeric_limits<double>::epsilon()) {
                    const double scale = projection / denominator;
                    for (int pair = 0; pair < sp_count_new * sp_count_new; ++pair) {
                        if (!new_pair[pair]) continue;
                        for (int l = 0; l < rb_size; ++l) {
                            const int idx = pair * pair_stride + k * rb_size + l;
                            const int idx2 = pair * pair_stride + k2 * rb_size + l;
                            coeffs_new[idx] -= scale * coeffs_new[idx2];
                        }
                    }
                }
            }

            double norm = 0.0;
            for (int pair = 0; pair < sp_count_new * sp_count_new; ++pair) {
                if (!new_pair[pair]) continue;
                for (int l = 0; l < rb_size; ++l) {
                    const int idx = pair * pair_stride + k * rb_size + l;
                    norm += coeffs_new[idx] * coeffs_new[idx];
                }
            }
            norm = std::sqrt(norm);
            if (norm > std::numeric_limits<double>::epsilon()) {
                for (int pair = 0; pair < sp_count_new * sp_count_new; ++pair) {
                    if (!new_pair[pair]) continue;
                    for (int l = 0; l < rb_size; ++l) {
                        const int idx = pair * pair_stride + k * rb_size + l;
                        coeffs_new[idx] /= norm;
                    }
                }
            }
        }
    }

    if (radial_basis_split) {
        std::vector<AnyBasis *> old_bases = std::move(p_radical_basis_pair);
        AnyBasis *default_basis = old_bases.at(0);
        std::vector<AnyBasis *> bases_new(sp_count_new * sp_count_new, nullptr);
        for (int i = 0; i < sp_count_new; ++i) {
            for (int j = 0; j < sp_count_new; ++j) {
                const int source_i = newsp[i];
                const int source_j = newsp[j];
                AnyBasis *&dst = bases_new[i * sp_count_new + j];
                if (source_i < sp_count_old && source_j < sp_count_old) {
                    AnyBasis *&src = old_bases[source_i * sp_count_old + source_j];
                    dst = src;
                    src = nullptr;
                } else {
                    dst = clone_basis(*default_basis);
                }
            }
        }
        for (AnyBasis *basis : old_bases) delete basis;
        p_radical_basis_pair = std::move(bases_new);
    }

    delete[] regression_coeffs;
    regression_coeffs = new double[reg_coeff_size_new];
    std::copy(coeffs_new.begin(), coeffs_new.end(), regression_coeffs);
    species_count = sp_count_new;
    radial_coeff_size = radial_coeff_size_new;
    lin_coeff_size = lin_coeff_size_new;
    reg_coeff_size = reg_coeff_size_new;
    species_coeff_size = sp_count_new;
    moment_coeff_size = alpha_scalar_moments;
    radial_coeffs = regression_coeffs;
    species_coeffs = regression_coeffs + radial_coeff_size;
    linear_coeffs = species_coeffs;
    moment_coeffs = linear_coeffs + species_count;

    cutoffs.resize(species_count, species_count);
    cutoffsqs.resize(species_count, species_count);
    r_mins.resize(species_count, species_count);
    if (radial_basis_split) {
        cutoff = 0.0;
        for (int i = 0; i < species_count; ++i) {
            for (int j = 0; j < species_count; ++j) {
                const AnyBasis *basis = p_radical_basis_pair[i * species_count + j];
                cutoffs(i, j) = basis->max_val;
                cutoffsqs(i, j) = basis->max_val * basis->max_val;
                r_mins(i, j) = basis->min_val;
                cutoff = std::max(cutoff, basis->max_val);
            }
        }
        cutoffsq = cutoff * cutoff;
    } else {
        cutoffs.setConstant(cutoff);
        cutoffsqs.setConstant(cutoffsq);
        r_mins.setConstant(p_radical_basis->min_val);
    }
    reg_L1_vector.clear();
    reg_L2_vector.clear();
    omp_inited = false;
}

/**
 * @brief Reorder alpha_index_basic, alpha_index_times and alpha_scalar_moments
 * 
 * @note 
 * alpha_index_basic is sorted by \mu, b1, b2, b3.
 * alpha_index_times is sorted by its level (alpha index of size one tensors as level 0, etc.) and weather it computes a scalar 
 * Finally, alpha_moment_mapping will collect all same level scalars together with adjecent alpha_moment_index, so that 
 * the memory will be continues.
 */
void PotMTP::ReorderByLevel() {
    struct alpha_index_basic_t {
        int mu, b1, b2, b3;
    };
    struct alpha_times_t {
        int l1, l2, c, id;
    };
    typedef std::pair<int, alpha_index_basic_t> basic_t;


    max_level = 0;
    lev_scalar_count.clear();
    lev_scalar_nodes.clear();
    lev_times_count.clear();

    // level record for each b
    std::map<int, int> bidlev;
    for (int i = 0; i < alpha_moments_count; ++i) {
        bidlev[i] = -1;
    }

    // 1. sort alpha_index_basic by mu and b1, b2, b3
    std::vector<basic_t> basic_index(alpha_index_basic_count);
    for (int i = 0; i < alpha_index_basic_count; ++i) {
        alpha_index_basic_t tid;
        tid.mu = alpha_index_basic(i, 0);
        tid.b1 = alpha_index_basic(i, 1);
        tid.b2 = alpha_index_basic(i, 2);
        tid.b3 = alpha_index_basic(i, 3);
        basic_index[i] = std::make_pair(i, tid);
        bidlev[i] = 0;
    }
    std::stable_sort(basic_index.begin(), basic_index.end(), [](const basic_t &a, const basic_t &b) {
        if (a.second.b1 + a.second.b2 + a.second.b3 < b.second.b1 + b.second.b2 + b.second.b3) return true; 
        if (a.second.b1 + a.second.b2 + a.second.b3 > b.second.b1 + b.second.b2 + b.second.b3) return false; 
        if (a.second.mu < b.second.mu) return true; 
        return false;
    });
    for (int i = 0; i < alpha_index_basic_count; ++i) {
        alpha_index_basic(i, 0) = basic_index[i].second.mu;
        alpha_index_basic(i, 1) = basic_index[i].second.b1;
        alpha_index_basic(i, 2) = basic_index[i].second.b2;
        alpha_index_basic(i, 3) = basic_index[i].second.b3;
        nu_arr_list[i] = basic_index[i].second.b1 + basic_index[i].second.b2 + basic_index[i].second.b3;
    }
    // 2. change alpha_index_times and alpha_moment_mapping
    std::map<int, int> index_basic_old_to_new;
    for (int i = 0; i < alpha_index_basic_count; ++i) {
        index_basic_old_to_new[basic_index[i].first] = i;
    }
    for (int i = 0; i < alpha_index_times_count; ++i) {
        int l1 = alpha_index_times(i, 0);
        int l2 = alpha_index_times(i, 1);
        if (l1 < alpha_index_basic_count)
            alpha_index_times(i, 0) = index_basic_old_to_new[l1];
        if (l2 < alpha_index_basic_count)
            alpha_index_times(i, 1) = index_basic_old_to_new[l2];
    }
    for (int i = 0; i < alpha_scalar_moments; ++i) {
        int id = alpha_moment_mapping[i];
        if (id < alpha_index_basic_count) {
            alpha_moment_mapping[i] = index_basic_old_to_new[id];
        }
    }
    // 3. index times 
    typedef std::pair<alpha_times_t, std::array<int, 5>> times_t; // id(will switch to current id in following loop), bid, blev, bdup, bisscalar
    std::map<int, int> bdup; // how many times bid is present in alpha_index_times list
    std::map<int, int> isscalar;
    for (int i = 0; i < alpha_scalar_moments; ++i) {
        isscalar[alpha_moment_mapping[i]] = 1;
    }
    std::vector<times_t> alpha_index_times_pair(alpha_index_times_count);
    for (int i = 0; i < alpha_index_times_count; ++i) {
        alpha_times_t ttimes;
        ttimes.l1 = alpha_index_times(i, 0);
        ttimes.l2 = alpha_index_times(i, 1);
        ttimes.c  = alpha_index_times(i, 2);
        ttimes.id = alpha_index_times(i, 3);
        std::array<int, 5> ntmp;
        if (bidlev[ttimes.l1] == -1 || bidlev[ttimes.l2] == -1) {
            std::cerr << "Error: incorrect order ";
            std::cerr << ttimes.l1 << " " << ttimes.l2 << " " << ttimes.c << " " << ttimes.id;
            std::cerr << std::endl;
            exit(1);
        }
        if (bidlev[ttimes.id] == -1)
            bdup[ttimes.id] = 0;
        ++bdup[ttimes.id];
        int lev = std::max(bidlev[ttimes.l1], bidlev[ttimes.l2]) + 1;
        lev = std::max(lev, bidlev[ttimes.id]);
        max_level = std::max(lev, max_level);
        bidlev[ttimes.id] = lev;
        ntmp[0] = i;
        ntmp[1] = ttimes.id;
        ntmp[2] = lev;
        ntmp[3] = bdup[ttimes.id];
        ntmp[4] = isscalar[ttimes.id];
        alpha_index_times_pair[i] = std::make_pair(ttimes, ntmp);
    }
    for (int i = 0; i < alpha_index_times_count; ++i) {
        int bid = alpha_index_times_pair[i].second[1];
        alpha_index_times_pair[i].second[2] = bidlev[bid];
    }
    // 3.2 sort index times by level and is_scalar
    stable_sort(alpha_index_times_pair.begin(), alpha_index_times_pair.end(), [](const times_t &a, const times_t &b) {
        if (a.second[2] < b.second[2]) return true;
        if (a.second[2] > b.second[2]) return false;
        if (a.second[4] > b.second[4]) return true; // leading terms are scalar ones
        return false;
    });
    int bidprev = 0;
    int id_curr = alpha_index_basic_count - 1;
    for (int i = 0; i < alpha_index_times_count; ++i) {
        int bid = alpha_index_times_pair[i].second[1];
        if (bidprev != bid) {
            bidprev = bid;
            ++id_curr;
        }
        alpha_index_times_pair[i].second[0] = id_curr;
    }
    // 3.3 re-index alpha_index_times array
    std::map<int, int> index_nonbasic_old_to_new;
    for (int i = 0; i < alpha_index_times_count; ++i) {
        index_nonbasic_old_to_new[alpha_index_times_pair[i].second[1]] = alpha_index_times_pair[i].second[0];
    }
    lev_times_count.resize(max_level + 1, 0);
    for (int i = 0; i < alpha_index_times_count; ++i) {
        int l1 = alpha_index_times_pair[i].first.l1;
        int l2 = alpha_index_times_pair[i].first.l2;
        int id = alpha_index_times_pair[i].first.id;
        int lev = alpha_index_times_pair[i].second[2];
        lev_times_count[lev]++;
        if (l1 >= alpha_index_basic_count)
            alpha_index_times_pair[i].first.l1 = index_nonbasic_old_to_new[l1];
        if (l2 >= alpha_index_basic_count)
            alpha_index_times_pair[i].first.l2 = index_nonbasic_old_to_new[l2];
        alpha_index_times_pair[i].first.id = index_nonbasic_old_to_new[id];
        alpha_index_times(i, 0) = alpha_index_times_pair[i].first.l1;
        alpha_index_times(i, 1) = alpha_index_times_pair[i].first.l2;
        alpha_index_times(i, 2) = alpha_index_times_pair[i].first.c;
        alpha_index_times(i, 3) = alpha_index_times_pair[i].first.id;
    }
    for (int i = 0; i < alpha_scalar_moments; ++i) {
        if (alpha_moment_mapping[i] >= alpha_index_basic_count) {
            alpha_moment_mapping[i] = index_nonbasic_old_to_new[alpha_moment_mapping[i]];
        }
    }
    //4. resort alpha_moment_mapping and adjust coeffs
    std::vector<std::pair<int, double>> tmp_moment_coeffs(alpha_scalar_moments);
    for (int i = 0; i < alpha_scalar_moments; ++i) {
        tmp_moment_coeffs[i] = std::make_pair(alpha_moment_mapping[i], moment_coeffs[i]);
    }
    std::ranges::sort(tmp_moment_coeffs, 
        [](const std::pair<int, double>&a, const std::pair<int, double>&b) {
            if (a.first < b.first) return true;
            return false;
        });
    int mom_prev = 0;
    int levcurr = 0;
    lev_scalar_count.resize(max_level + 1, 0);
    for (int i = 0; i < alpha_scalar_moments; ++i) {
        alpha_moment_mapping[i] = tmp_moment_coeffs[i].first;
        moment_coeffs[i] = tmp_moment_coeffs[i].second;
        if (alpha_moment_mapping[i] == mom_prev++) {
            ++lev_scalar_count[levcurr];
        } else {
            ++levcurr;
            mom_prev = alpha_moment_mapping[i] + 1;
            ++lev_scalar_count[levcurr];
        }
    }
    int tmp_accu = 0;
    lev_scalar_nodes.resize(max_level+1, 0);
    for (int i = 1; i <= max_level; ++i) {
        tmp_accu += lev_scalar_count[i - 1];
        lev_scalar_nodes[i] = alpha_moment_mapping[tmp_accu];
    }
}

void PotMTP::getEFS(Atoms &atom) {
    require_loaded_model();
    Neighborhoods nbhs(atom, cutoff);
    getEFS(atom, nbhs);
}

void PotMTP::getEFS(Atoms &atom, AnyNeighborhoods &nbhs)
{
    require_loaded_model();
    if (full_symmetry_reuse && try_full_symmetry_efs(atom, nbhs)) return;
    if (!add_efs) atom.resetEFS(); else atom.setEFSflag();

    for (int ii = 0; ii < nbhs.size(); ii++) {
        Neighborhood &nbh = nbhs[ii];
        buff_site_energy = 0;
        CalcSiteEFS(nbh, buff_site_energy, buff_site_energy_ders);
        // extract energy, force, virual, etc
        atom.energy_ += buff_site_energy;
        atom.site_energies_[ii] += buff_site_energy;

        for (int i = 0; i < nbh.nbhcount_; ++i) {
            // dE/dxj = \sum_{i!=j}dVi/d(xj-xi) - \sum_{i}dVj/d(xi-xj)
            atom.forces_[ii] += buff_site_energy_ders.row(i);
            // for each local configuration n_j, r is acutally r_ii - r_j, thus we have negative terms.
            atom.forces_[nbh.indices_[i]] -= buff_site_energy_ders.row(i);
            for (int a = 0; a < 3; a++) {
                for (int b = 0; b < 3; b++) {
                    atom.stress_(a, b) -= buff_site_energy_ders(i, a) * nbh.pos_[i](b);
                }
            }
        }
    }
}

void Potentials::PotMTP::getEFSFluxStress(Atoms &atom)
{
    Neighborhoods nbhs(atom, cutoff);
    atom.alloc_lvd("fluxstress", 9);
    getEFSFluxStress(atom, nbhs);
}

void Potentials::PotMTP::getEFSFluxStress(Atoms &atom, AnyNeighborhoods &nbhs)
{
    if (!add_efs) atom.resetEFS(); else atom.setEFSflag();
    if (!add_efs) atom.local_attribs_v_["fluxstress"].setZero();

    for (int ii = 0; ii < nbhs.size(); ii++) {
        Neighborhood &nbh = nbhs[ii];
        buff_site_energy = 0;
        CalcSiteEFS(nbh, buff_site_energy, buff_site_energy_ders);
        // extract energy, force, virual, etc
        atom.energy_ += buff_site_energy;
        atom.site_energies_[ii] += buff_site_energy;

        for (int i = 0; i < nbh.nbhcount_; ++i) {
            atom.forces_[ii] += buff_site_energy_ders.row(i);
            atom.forces_[nbh.indices_[i]] -= buff_site_energy_ders.row(i);
            for (int a = 0; a < 3; a++) {
                for (int b = 0; b < 3; b++) {
                    atom.stress_(a, b) -= buff_site_energy_ders(i, a) * nbh.pos_[i](b);
                }
            }
            // flux stress
            auto &flux = atom.local_attribs_v_["fluxstress"];
            flux(0, nbh.indices_[i]) += -nbh.pos_[i][0] * buff_site_energy_ders(i, 0);
            flux(1, nbh.indices_[i]) += -nbh.pos_[i][1] * buff_site_energy_ders(i, 1);
            flux(2, nbh.indices_[i]) += -nbh.pos_[i][2] * buff_site_energy_ders(i, 2);
            // 
            flux(3, nbh.indices_[i]) += -nbh.pos_[i][0] * buff_site_energy_ders(i, 1);
            flux(4, nbh.indices_[i]) += -nbh.pos_[i][0] * buff_site_energy_ders(i, 2);
            flux(5, nbh.indices_[i]) += -nbh.pos_[i][1] * buff_site_energy_ders(i, 2);
            flux(6, nbh.indices_[i]) += -nbh.pos_[i][1] * buff_site_energy_ders(i, 0);
            flux(7, nbh.indices_[i]) += -nbh.pos_[i][2] * buff_site_energy_ders(i, 0);
            flux(8, nbh.indices_[i]) += -nbh.pos_[i][2] * buff_site_energy_ders(i, 1);
        }
    }
}

/**
 * @brief
 * In MPI mode:
 *  1. Do not store full neighborhood info on each node
 *  2. Do not sture full atom info on each node, which means, some program may need a mpi varient. TODO: avoid this by assinging an op in GPIO
 *  3. In case of 2, Atoms class may need modify
 * Maybe let lammps handle this.
 * @param atom
 * @param nbhs
 */
void Potentials::PotMTP::getEFSmpi(Atoms &atom, AnyNeighborhoods &nbhs) {
    if (!add_efs) atom.resetEFS(); else atom.setEFSflag();

    /**
     * @brief 
     * Code here
     */

    #ifdef IMRMLP_MPI
    // energy
    MPI_Allreduce(MPI_IN_PLACE, &atom.energy_, 1, MPI_DOUBLE, MPI_SUM, mpi.comm);

    #endif
}

void Potentials::PotMTP::getEFSOMPadd(Atoms &atom, AnyNeighborhoods &nbhs) {
    omp_init();
    int actual_threads = 1;

#ifdef USE_OMP
#pragma omp parallel num_threads(omp_buffers_list.size())
#endif 
{
#ifdef USE_OMP
    int ithread = omp_get_thread_num();
    if (ithread == 0) actual_threads = omp_get_num_threads();
#else 
    int ithread = 0;
#endif
    ompbuffers &buffer = omp_buffers_list[ithread];
    buffer.energy = 0;
    buffer.force.assign(atom.size(), Eigen::RowVector3d::Zero());
    buffer.siteenergy.assign(atom.size(), 0.0);
    buffer.stress = Eigen::Matrix3d::Zero();
#ifdef USE_OMP
#pragma omp for schedule(static)
#endif
    for (int ii = 0; ii < nbhs.size(); ii++) {
        Neighborhood &nbh = nbhs[ii];
        buffer.buff_site_energy = 0;
        CalcSiteEFSOMP(nbh, buffer);
        // extract energy, force, virual, etc
        buffer.energy += buffer.buff_site_energy;
        atom.site_energies_[ii] += buffer.buff_site_energy;
        for (int i = 0; i < nbh.nbhcount_; ++i) {
            buffer.force[ii] += buffer.buff_site_energy_ders.row(i);
            buffer.force[nbh.indices_[i]] -= buffer.buff_site_energy_ders.row(i);
            for (int a = 0; a < 3; a++) {
                for (int b = 0; b < 3; b++) {
                    buffer.stress(a, b) -= buffer.buff_site_energy_ders(i, a) * nbh.pos_[i](b);
                }
            }
        }
    }
}
    // merge
    for (int i = 0; i < actual_threads; ++i) {
        auto &buffer = omp_buffers_list[i];
        atom.energy_ += buffer.energy;
        atom.stress_ += buffer.stress;
        for (int iat = 0; iat < atom.size(); ++iat) {
            atom.forces_[iat] += buffer.force[iat];
        }
    }
}

void Potentials::PotMTP::getEFSOMPadd(Atoms &atom, AnyNeighborhoods &nbhs, double sf) {
    omp_init();
    int actual_threads = 1;

#ifdef USE_OMP
#pragma omp parallel num_threads(omp_buffers_list.size())
#endif 
{
#ifdef USE_OMP
    int ithread = omp_get_thread_num();
    if (ithread == 0) actual_threads = omp_get_num_threads();
#else 
    int ithread = 0;
#endif
    ompbuffers &buffer = omp_buffers_list[ithread];
    buffer.energy = 0;
    buffer.force.assign(atom.size(), Eigen::RowVector3d::Zero());
    buffer.siteenergy.assign(atom.size(), 0.0);
    buffer.stress = Eigen::Matrix3d::Zero();
#ifdef USE_OMP
#pragma omp for schedule(static)
#endif
    for (int ii = 0; ii < nbhs.size(); ii++) {
        Neighborhood &nbh = nbhs[ii];
        buffer.buff_site_energy = 0;
        CalcSiteEFSOMP(nbh, buffer);
        // apply scale
        buffer.buff_site_energy *= sf;
        buffer.buff_site_energy_ders *= sf;
        // extract energy, force, virual, etc
        buffer.energy += buffer.buff_site_energy;
        atom.site_energies_[ii] += buffer.buff_site_energy;
        for (int i = 0; i < nbh.nbhcount_; ++i) {
            // dE/dxj = \sum_{i!=j}dVi/d(xj-xi) - \sum_{i}dVj/d(xi-xj)
            buffer.force[ii](0) += buffer.buff_site_energy_ders(i, 0);
            buffer.force[ii](1) += buffer.buff_site_energy_ders(i, 1);
            buffer.force[ii](2) += buffer.buff_site_energy_ders(i, 2);
            // for each local configuration n_j, r is acutally r_ii - r_j, thus we have negative terms.
            buffer.force[nbh.indices_[i]](0) -= buffer.buff_site_energy_ders(i, 0);
            buffer.force[nbh.indices_[i]](1) -= buffer.buff_site_energy_ders(i, 1);
            buffer.force[nbh.indices_[i]](2) -= buffer.buff_site_energy_ders(i, 2);

            for (int a = 0; a < 3; a++) {
                for (int b = 0; b < 3; b++) {
                    buffer.stress(a, b) -= buffer.buff_site_energy_ders(i, a) * nbh.pos_[i](b);
                }
            }
        }
    }
}
    // merge
    for (int i = 0; i < actual_threads; ++i) {
        auto &buffer = omp_buffers_list[i];
        atom.energy_ += buffer.energy;
        atom.stress_ += buffer.stress;
        for (int iat = 0; iat < atom.size(); ++iat) {
            atom.forces_[iat] += buffer.force[iat];
        }
    }
}

// without init
// todo: convert to safe version, sightly lower effciency but much less coding.
void Potentials::PotMTP::getEFSadd(Atoms &atom, AnyNeighborhoods &nbhs) {
    for (int ii = 0; ii < nbhs.size(); ii++) {
        Neighborhood &nbh = nbhs[ii];
        buff_site_energy = 0;
        CalcSiteEFS(nbh, buff_site_energy, buff_site_energy_ders);
        // extract energy, force, virual, etc
        atom.energy_ += buff_site_energy;
        atom.site_energies_[ii] += buff_site_energy;

        for (int i = 0; i < nbh.nbhcount_; ++i) {
            // dE/dxj = \sum_{i!=j}dVi/d(xj-xi) - \sum_{i}dVj/d(xi-xj)
            atom.forces_[ii](0) += buff_site_energy_ders(i, 0);
            atom.forces_[ii](1) += buff_site_energy_ders(i, 1);
            atom.forces_[ii](2) += buff_site_energy_ders(i, 2);
            // for each local configuration n_j, r is acutally r_ii - r_j, thus we have negative terms.
            atom.forces_[nbh.indices_[i]](0) -= buff_site_energy_ders(i, 0);
            atom.forces_[nbh.indices_[i]](1) -= buff_site_energy_ders(i, 1);
            atom.forces_[nbh.indices_[i]](2) -= buff_site_energy_ders(i, 2);

            for (int a = 0; a < 3; a++) {
                for (int b = 0; b < 3; b++) {
                    atom.stress_(a, b) -= buff_site_energy_ders(i, a) * nbh.pos_[i](b);
                }
            }
        }
    }
}

void Potentials::PotMTP::getEFSadd(Atoms &atom, AnyNeighborhoods &nbhs, double sf) {
    for (int ii = 0; ii < nbhs.size(); ii++) {
        Neighborhood &nbh = nbhs[ii];
        buff_site_energy = 0;
        CalcSiteEFS(nbh, buff_site_energy, buff_site_energy_ders);
        // apply scaling
        buff_site_energy *= sf;
        buff_site_energy_ders *= sf;
        // extract energy, force, virual, etc
        atom.energy_ += buff_site_energy;
        atom.site_energies_[ii] += buff_site_energy;

        for (int i = 0; i < nbh.nbhcount_; ++i) {
            // dE/dxj = \sum_{i!=j}dVi/d(xj-xi) - \sum_{i}dVj/d(xi-xj)
            atom.forces_[ii](0) += buff_site_energy_ders(i, 0);
            atom.forces_[ii](1) += buff_site_energy_ders(i, 1);
            atom.forces_[ii](2) += buff_site_energy_ders(i, 2);
            // for each local configuration n_j, r is acutally r_ii - r_j, thus we have negative terms.
            atom.forces_[nbh.indices_[i]](0) -= buff_site_energy_ders(i, 0);
            atom.forces_[nbh.indices_[i]](1) -= buff_site_energy_ders(i, 1);
            atom.forces_[nbh.indices_[i]](2) -= buff_site_energy_ders(i, 2);

            for (int a = 0; a < 3; a++) {
                for (int b = 0; b < 3; b++) {
                    atom.stress_(a, b) -= buff_site_energy_ders(i, a) * nbh.pos_[i](b);
                }
            }
        }
    }
}

void PotMTP::getEFSSymmOMP(Atoms &atom, double symprec) {
    if (full_symmetry_reuse) {
        Neighborhoods nbhs(atom, cutoff);
        if (!try_full_symmetry_efs(atom, nbhs, symprec)) {
            // Mapping was not exact enough to reuse a site. The ordinary
            // path below the opt-in dispatch evaluates every site.
            const bool saved = full_symmetry_reuse;
            full_symmetry_reuse = false;
            try { getEFSOMP(atom, nbhs); }
            catch (...) { full_symmetry_reuse = saved; throw; }
            full_symmetry_reuse = saved;
        }
        return;
    }
    omp_init();
    int actual_threads = 1;

    if (!add_efs) atom.resetEFS(); else atom.setEFSflag();
    Neighborhoods nbhs(atom, cutoff);

    // 1. find permitive cell and mapping relations, only translation symmetry
    auto dataset = Symmetry3D::get_sym_dataset(atom, symprec);
    auto &mapping_to_primitive = dataset.mapping_to_primitive;
    std::set<int> nonequat;
    std::vector<int> nonequatid;
    std::vector<int> nonequind;
    for (int i = 0; i < mapping_to_primitive.size(); ++i) {
        int primid = mapping_to_primitive[i];
        if (nonequat.count(primid) == 0) {
            nonequind.push_back(i);
            nonequat.insert(primid);
            nonequatid.push_back(primid);
        }
    }
    // 2. build neighbor list
    std::vector<std::pair<int, Eigen::RowVector3d>> sortind;
    for (int i = 0; i < nbhs.size(); ++i) {
        auto &nbh = nbhs[i];
        // sort
        sortind.resize(nbh.nbhcount_);
        for (int j = 0; j < nbh.nbhcount_; ++j) {
            sortind[j].first = j;
            sortind[j].second = nbh.pos_[j];
        }
        std::ranges::sort(sortind, [&symprec](const std::pair<int, Eigen::RowVector3d> &a, const std::pair<int, Eigen::RowVector3d> &b) {
            if (a.second(0) < b.second(0) - symprec) return true;
            if (a.second(0) > b.second(0) + symprec) return false;
            
            if (a.second(1) < b.second(1) - symprec) return true;
            if (a.second(1) > b.second(1) + symprec) return false;
            
            if (a.second(2) < b.second(2) - symprec) return true;
            if (a.second(2) > b.second(2) + symprec) return false;

            return false;
        });
        // reorder
        auto nbhbck = nbh;
        for (int j = 0; j < nbh.nbhcount_; ++j) {
            nbh.chemmap_[j] = nbhbck.chemmap_[sortind[j].first];
            nbh.distances_[j] = nbhbck.distances_[sortind[j].first];
            nbh.indices_[j] = nbhbck.indices_[sortind[j].first];
            nbh.pos_[j] = nbhbck.pos_[sortind[j].first];
        }
    }

    // 3. compute
#ifdef USE_OMP
#pragma omp parallel num_threads(omp_buffers_list.size())
#endif 
{
#ifdef USE_OMP
    int ithread = omp_get_thread_num();
    if (ithread == 0) actual_threads = omp_get_num_threads();
#else 
    int ithread = 0;
#endif
    ompbuffers &buffer = omp_buffers_list[ithread];
    buffer.energy = 0;
    buffer.force.assign(atom.size(), Eigen::RowVector3d::Zero());
    buffer.siteenergy.assign(atom.size(), 0.0);
    buffer.stress = Eigen::Matrix3d::Zero();
#ifdef USE_OMP
#pragma omp for schedule(static)
#endif
    for (int in = 0; in < nonequind.size(); ++in) {
        auto &nbhp = nbhs[nonequind[in]];
        int aid = nonequind[in];
        Message(fmt::format("Currently on atom {} at proc {}, {} in total", in, ithread, nonequind.size()));
        CalcSiteEFSOMP(nbhp, buffer);
        
        // translational eqv atoms
        for (int ii = 0; ii < nbhs.size(); ++ii) {
            if (mapping_to_primitive[ii] != nonequatid[in]) continue;
            auto &nbh = nbhs[ii];
            // energy
            buffer.siteenergy[ii] = buffer.buff_site_energy;
            buffer.energy += buffer.buff_site_energy;
            // force
            for (int i = 0; i < nbh.nbhcount_; ++i) {
                buffer.force[ii](0) += buffer.buff_site_energy_ders(i, 0);
                buffer.force[ii](1) += buffer.buff_site_energy_ders(i, 1);
                buffer.force[ii](2) += buffer.buff_site_energy_ders(i, 2);
                buffer.force[nbh.indices_[i]](0) -= buffer.buff_site_energy_ders(i, 0);
                buffer.force[nbh.indices_[i]](1) -= buffer.buff_site_energy_ders(i, 1);
                buffer.force[nbh.indices_[i]](2) -= buffer.buff_site_energy_ders(i, 2);
                // stress
                for (int a = 0; a < 3; a++) {
                    for (int b = 0; b < 3; b++) {
                        buffer.stress(a, b) -= buffer.buff_site_energy_ders(i, a) * nbh.pos_[i](b);
                    }
                }
            }
        }
    }
}
    // merge
    for (int i = 0; i < actual_threads; ++i) {
        auto &buffer = omp_buffers_list[i];
        atom.energy_ += buffer.energy;
        atom.stress_ += buffer.stress;
        for (int iat = 0; iat < atom.size(); ++iat) {
            atom.forces_[iat] += buffer.force[iat];
            atom.site_energies_[iat] += buffer.siteenergy[iat];
        }
    }

}

/**
 * @brief Construct a new Pot M T P R::ompbuffers::ompbuffers object, allocate memorys
 * 
 * @param max_alpha_index_basic as it is
 * @param alpha_moments_count as it is
 * @param rb_size as it is
 * @param max_mu as it is
 */
PotMTP::ompbuffers::ompbuffers(int max_alpha_index_basic, int alpha_moments_count, int rb_size, int max_mu) {
    coords_powers.resize(max_alpha_index_basic + 1, 3); // {rij_x^k, rij_y^k, rij_z^k}
    coords_powers(0, 0) = coords_powers(0, 1) = coords_powers(0, 2) = 0;
    coords_powers(0 + 1, 0) = coords_powers(0 + 1, 1) = coords_powers(0 + 1, 2) = 1;
    // +6 for moment_hessian
    dist_powers.resize(max_alpha_index_basic + 6); // rij^k
    moment_vals.resize(alpha_moments_count, 0);
    rb_vals.resize(rb_size, 0);
    rb_ders.resize(rb_size, 0);
    rb_derders.resize(rb_size, 0);
    val_mu.resize(max_mu, 0); // basis function value
    der_mu.resize(max_mu, 0); // basis function der wrt r
    derder_mu.resize(max_mu, 0);
    site_energy_ders_wrt_moments_.resize(alpha_moments_count);
}

/**
 * @brief Predict energy, force and stress, OPENMP enabled. 
 * 
 * @param atom 
 * @param nbhs 
 * @note  
 * Each openmp thread use its own buffer to ensure thread safity.
 * Local environment contribution of each atom is parallelized.
 */
void PotMTP::getEFSOMP(Atoms &atom, AnyNeighborhoods &nbhs) {
    if (full_symmetry_reuse && try_full_symmetry_efs(atom, nbhs)) return;
    omp_init();
    int actual_threads = 1;

    if (!add_efs) atom.resetEFS(); else atom.setEFSflag();
#ifdef USE_OMP
#pragma omp parallel num_threads(omp_buffers_list.size())
#endif 
{
#ifdef USE_OMP
    int ithread = omp_get_thread_num();
    if (ithread == 0) actual_threads = omp_get_num_threads();
#else 
    int ithread = 0;
#endif
    ompbuffers &buffer = omp_buffers_list[ithread];
    buffer.energy = 0;
    buffer.force.assign(atom.size(), Eigen::RowVector3d::Zero());
    buffer.siteenergy.assign(atom.size(), 0.0);
    buffer.stress = Eigen::Matrix3d::Zero();
#ifdef USE_OMP
#pragma omp for schedule(static)
#endif
    for (int ii = 0; ii < nbhs.size(); ii++) {
        Neighborhood &nbh = nbhs[ii];
        buffer.buff_site_energy = 0;
        CalcSiteEFSOMP(nbh, buffer);
        // extract energy, force, virual, etc
        buffer.energy += buffer.buff_site_energy;
        atom.site_energies_[ii] += buffer.buff_site_energy;
        for (int i = 0; i < nbh.nbhcount_; ++i) {
            // dE/dxj = \sum_{i!=j}dVi/d(xj-xi) - \sum_{i}dVj/d(xi-xj)
            buffer.force[ii](0) += buffer.buff_site_energy_ders(i, 0);
            buffer.force[ii](1) += buffer.buff_site_energy_ders(i, 1);
            buffer.force[ii](2) += buffer.buff_site_energy_ders(i, 2);
            // for each local configuration n_j, r is acutally r_ii - r_j, thus we have negative terms.
            buffer.force[nbh.indices_[i]](0) -= buffer.buff_site_energy_ders(i, 0);
            buffer.force[nbh.indices_[i]](1) -= buffer.buff_site_energy_ders(i, 1);
            buffer.force[nbh.indices_[i]](2) -= buffer.buff_site_energy_ders(i, 2);

            for (int a = 0; a < 3; a++) {
                for (int b = 0; b < 3; b++) {
                    buffer.stress(a, b) -= buffer.buff_site_energy_ders(i, a) * nbh.pos_[i](b);
                }
            }
        }
    }
}
    // merge
    for (int i = 0; i < actual_threads; ++i) {
        auto &buffer = omp_buffers_list[i];
        atom.energy_ += buffer.energy;
        atom.stress_ += buffer.stress;
        for (int iat = 0; iat < atom.size(); ++iat) {
            atom.forces_[iat] += buffer.force[iat];
        }
    }
}

void PotMTP::getEFSOMP(Atoms &atom) {
    require_loaded_model();
    Neighborhoods nbhs(atom, cutoff);
    getEFSOMP(atom, nbhs);
}

void PotMTP::getEFSOMPthread(Atoms &atom) {
    require_loaded_model();
    Neighborhoods nbhs(atom, cutoff);
    getEFSOMPthread(atom, nbhs);
}

/**
 * @brief Predict energy, force and stress, OPENMP enabled. 
 * 
 * @param atom 
 * @param nbhs 
 * @note  
 * Each openmp thread use its own buffer to ensure thread safity.
 * For outer loop using OpenMP, avoid thread conflict
 */
void PotMTP::getEFSOMPthread(Atoms &atom, AnyNeighborhoods &nbhs) {
    require_loaded_model();
    if (!add_efs) atom.resetEFS(); else atom.setEFSflag();
    // This entry point is called concurrently by an outer team (or native
    // threads). Its scratch must not depend on team IDs or resize shared state.
    ompbuffers buffer(max_alpha_index_basic, alpha_moments_count, rb_size, max_mu);

    for (int ii = 0; ii < nbhs.size(); ii++) {
        Neighborhood &nbh = nbhs[ii];
        buffer.buff_site_energy = 0;
        CalcSiteEFSOMP(nbh, buffer);
        // extract energy, force, virual, etc
        atom.energy_ += buffer.buff_site_energy;
        atom.site_energies_[ii] += buffer.buff_site_energy;
        for (int i = 0; i < nbh.nbhcount_; ++i) {
            // dE/dxj = \sum_{i!=j}dVi/d(xj-xi) - \sum_{i}dVj/d(xi-xj)
            atom.forces_[ii](0) += buffer.buff_site_energy_ders(i, 0);
            atom.forces_[ii](1) += buffer.buff_site_energy_ders(i, 1);
            atom.forces_[ii](2) += buffer.buff_site_energy_ders(i, 2);
            // for each local configuration n_j, r is acutally r_ii - r_j, thus we have negative terms.
            atom.forces_[nbh.indices_[i]](0) -= buffer.buff_site_energy_ders(i, 0);
            atom.forces_[nbh.indices_[i]](1) -= buffer.buff_site_energy_ders(i, 1);
            atom.forces_[nbh.indices_[i]](2) -= buffer.buff_site_energy_ders(i, 2);

            for (int a = 0; a < 3; a++) {
                for (int b = 0; b < 3; b++) {
                    atom.stress_(a, b) -= buffer.buff_site_energy_ders(i, a) * nbh.pos_[i](b);
                }
            }
        }
    }
}


void PotMTP::getEOMP(Atoms &atom, AnyNeighborhoods &nbhs) {
    require_loaded_model();
    omp_init();
    int actual_threads = 1;

    if (!add_efs) atom.resetE(); else atom.setEflag();
#ifdef USE_OMP
#pragma omp parallel num_threads(omp_buffers_list.size())
#endif 
{
#ifdef USE_OMP
    int ithread = omp_get_thread_num();
    if (ithread == 0) actual_threads = omp_get_num_threads();
#else 
    int ithread = 0;
#endif
    ompbuffers &buffer = omp_buffers_list[ithread];
    buffer.energy = 0;
    buffer.siteenergy.assign(atom.size(), 0.0);
#ifdef USE_OMP
#pragma omp for schedule(static)
#endif
    for (int ii = 0; ii < nbhs.size(); ii++) {
        Neighborhood &nbh = nbhs[ii];
        buffer.buff_site_energy = 0;
        CalcSiteEOMP(nbh, buffer);
        // extract energy, force, virual, etc
        buffer.energy += buffer.buff_site_energy;
        atom.site_energies_[ii] += buffer.buff_site_energy;
    }
}
    // merge
    for (int i = 0; i < actual_threads; ++i) {
        auto &buffer = omp_buffers_list[i];
        atom.energy_ += buffer.energy;
    }
}

void Potentials::PotMTP::getE(Atoms &atom) {
    require_loaded_model();
    Neighborhoods nbhs(atom, cutoff);
    getE(atom, nbhs);
}

void PotMTP::getE(Atoms &atom, AnyNeighborhoods &nbhs) {
    require_loaded_model();
    if (!add_efs) atom.resetE(); else atom.setEflag();

    for (int ii = 0; ii < nbhs.size(); ii++) {
        Neighborhood &nbh = nbhs[ii];
        buff_site_energy = 0;
        CalcSiteE(nbh, buff_site_energy);
        // extract energy, force, virual, etc
        atom.energy_ += buff_site_energy;
        atom.site_energies_[ii] += buff_site_energy;
    }
}

void PotMTP::getEOMP(Atoms &atom) {
    require_loaded_model();
    Neighborhoods nbhs(atom, cutoff);
    getEOMP(atom, nbhs);
}

/**
 * @brief Compute energy, force and stress compoments of each scalar basis.
 * 
 * @param atom 
 * @param nbhs 
 * @param energy_cmpts vector, output of energy compoments
 * @param force_cmpts 
 * @param stress_cmpts 
 * @note 
 *  For linear fit, very expensive when scalar basis grow large.
 */
void PotMTP::CalcEFSCmpt(Atoms &atom, AnyNeighborhoods &nbhs,
                          std::vector<double> &energy_cmpts,
                          Eigen::Tensor<double, 3> &force_cmpts,
                          Eigen::Tensor<double, 3> &stress_cmpts) {
    energy_cmpts.clear();
    energy_cmpts.assign(lin_coeff_size, 0);
    force_cmpts.resize(atom.size(), lin_coeff_size, 3);
    force_cmpts.setZero();
    stress_cmpts.resize(lin_coeff_size, 3, 3);
    stress_cmpts.setZero();
    for (int id = 0; id < atom.size(); ++id) {
        CalcSiteBasisDers(nbhs[id]);
        energy_cmpts[nbhs[id].centertype] += basis_vals[0];
        for (int k = species_count; k < lin_coeff_size; ++k) {
            int i = k - species_count + 1;
            energy_cmpts[k] += basis_vals[i];
            for (int j = 0; j < nbhs[id].nbhcount_; ++j) {
                for (int a = 0; a < 3; ++a) {
                    force_cmpts(id, k, a) += basis_ders(i, j, a);
                    force_cmpts(nbhs[id].indices_[j], k, a) -= basis_ders(i, j, a);
                }
            }
            for (int j = 0; j < nbhs[id].nbhcount_; ++j)
                for (int a = 0; a < 3; ++a)
                    for (int b = 0; b < 3; ++b)
                        stress_cmpts(k, a, b) -= basis_ders(i, j, a) * nbhs[id].pos_[j][b];
        }
    }
}

void Potentials::PotMTP::CalcEFSCmpt_R(Atoms &atom, AnyNeighborhoods &nbhs, std::vector<double> &energy_cmpts, Eigen::Tensor<double, 3, Eigen::RowMajor> &force_cmpts, Eigen::Tensor<double, 3, Eigen::RowMajor> &stress_cmpts)
{
    energy_cmpts.assign(lin_coeff_size, 0);
    force_cmpts.resize(atom.size(), 3, lin_coeff_size);
    force_cmpts.setZero();
    stress_cmpts.resize(3, 3, lin_coeff_size);
    stress_cmpts.setZero();
    for (int id = 0; id < atom.size(); ++id) {
        CalcSiteBasisDers(nbhs[id]);
        energy_cmpts[nbhs[id].centertype] += basis_vals[0];
        for (int k = species_count; k < lin_coeff_size; ++k) {
            int i = k - species_count + 1;
            energy_cmpts[k] += basis_vals[i];
            for (int j = 0; j < nbhs[id].nbhcount_; ++j) {
                for (int a = 0; a < 3; ++a) {
                    force_cmpts(id, a, k) += basis_ders(i, j, a);
                    force_cmpts(nbhs[id].indices_[j], a, k) -= basis_ders(i, j, a);
                }
            }
            for (int j = 0; j < nbhs[id].nbhcount_; ++j)
                for (int a = 0; a < 3; ++a)
                    for (int b = 0; b < 3; ++b)
                        stress_cmpts(a, b, k) -= basis_ders(i, j, a) * nbhs[id].pos_[j][b];
        }
    }
}


//         // stress go second


void PotMTP::CalcEFSCmpt(Atoms &atom, AnyNeighborhoods &nbhs,
                          std::vector<double> &energy_cmpts,
                          Eigen::Tensor<double, 3> &force_cmpts,
                          Eigen::Tensor<double, 3> &stress_cmpts,
                          int offset) {
    for (int id = 0; id < atom.size(); ++id) {
        CalcSiteBasisDers(nbhs[id]);
        for (int k = 0; k < alpha_scalar_moments; ++k) {
            int i = k + 1;
            energy_cmpts[offset + k] += basis_vals[i];
            for (int j = 0; j < nbhs[id].nbhcount_; ++j) {
                for (int a = 0; a < 3; ++a) {
                    force_cmpts(id, k + offset, a) += basis_ders(i, j, a);
                    force_cmpts(nbhs[id].indices_[j], k + offset, a) -= basis_ders(i, j, a);
                }
            }
            for (int j = 0; j < nbhs[id].nbhcount_; ++j)
                for (int a = 0; a < 3; ++a)
                    for (int b = 0; b < 3; ++b)
                        stress_cmpts(k + offset, a, b) -= basis_ders(i, j, a) * nbhs[id].pos_[j][b];
        }
    }
}

void Potentials::PotMTP::CalcEFSCmptNSp(Atoms &atom, AnyNeighborhoods &nbhs, std::vector<double> &energy_cmpts, Eigen::Tensor<double, 3> &force_cmpts, Eigen::Tensor<double, 3> &stress_cmpts, int offset) {
    for (int id = 0; id < atom.size(); ++id) {
        CalcSiteBasisDers(nbhs[id]);
        for (int k = 0; k < alpha_scalar_moments; ++k) {
            int i = k + 1;
            for (int j = 0; j < nbhs[id].nbhcount_; ++j) {
                for (int a = 0; a < 3; ++a) {
                    force_cmpts(id, k + offset, a) += basis_ders(i, j, a);
                    force_cmpts(nbhs[id].indices_[j], k + offset, a) -= basis_ders(i, j, a);
                }
            }
            for (int j = 0; j < nbhs[id].nbhcount_; ++j)
                for (int a = 0; a < 3; ++a)
                    for (int b = 0; b < 3; ++b)
                        stress_cmpts(k + offset, a, b) -= basis_ders(i, j, a) * nbhs[id].pos_[j][b];
        }
    }
}

void PotMTP::CalcECmpt(Atoms &atom, AnyNeighborhoods &nbhs, 
                         std::vector<double> &energy_cmpts) {
    energy_cmpts.clear();
    energy_cmpts.assign(lin_coeff_size, 0);
    for (int id = 0; id < atom.size(); ++id) {
        CalcSiteBasis(nbhs[id]);
        energy_cmpts[nbhs[id].centertype] += basis_vals[0];
        for (int k = species_count; k < lin_coeff_size; ++k) {
            int i = k - species_count + 1;
            energy_cmpts[k] += basis_vals[i];
        }
    }
}

void PotMTP::CalcECmpt(Atoms &atom, AnyNeighborhoods &nbhs, 
                         std::vector<double> &energy_cmpts, int offset) {
    for (int id = 0; id < atom.size(); ++id) {
        CalcSiteBasis(nbhs[id]);
        for (int k = 0; k < alpha_scalar_moments; ++k) {
            energy_cmpts[k + offset] += basis_vals[k+1];
        }
    }
}
// todo remove this
void Potentials::PotMTP::CalcECmptNSp(Atoms &atom, AnyNeighborhoods &nbhs, std::vector<double> &energy_cmpts, int offset){
    for (int id = 0; id < atom.size(); ++id) {
        CalcSiteBasis(nbhs[id]);
        for (int k = 0; k < alpha_scalar_moments; ++k) {
            int i = k + 1;
            energy_cmpts[k + offset] += basis_vals[i];
        }
    }
}

// ── Unified descriptor + basic Jacobian (serial / OMP) ───────────────────
// Template parameter IsOMP selects member buffers (false) or ompbuffers (true).
// Ternary references are resolved at compile time → zero runtime overhead.
// Fills: moment_vals (all, forward product rules) + moment_jacobian (basic only).
template<bool IsOMP>
void PotMTP::CalcSiteDescriptorsImpl(Neighborhood &nbh, ompbuffers &buf) {
    auto& mv = IsOMP ? buf.moment_vals     : moment_vals;
    auto& rv = IsOMP ? buf.rb_vals         : rb_vals;
    auto& rd = IsOMP ? buf.rb_ders         : rb_ders;
    auto& vm = IsOMP ? buf.val_mu          : val_mu;
    auto& dm = IsOMP ? buf.der_mu          : der_mu;
    auto& mj = IsOMP ? buf.moment_jacobian : moment_jacobian;
    auto& cp = IsOMP ? buf.coords_powers   : coords_powers;
    auto& dp = IsOMP ? buf.dist_powers     : dist_powers;
    auto& cache = IsOMP ? buf.radial_cache : radial_cache;
    if (radial_local_cache) cache.begin_site(max_mu, max_alpha_index_basic);

    mj.resize(alpha_index_basic_count, nbh.nbhcount_, 3);
    mj.setZero();

    std::ranges::fill(mv, 0);
    std::ranges::fill(rv, 0);
    std::ranges::fill(rd, 0);
    std::ranges::fill(vm, 0);
    std::ranges::fill(dm, 0);

    const int &itype = nbh.centertype;

    for (int jj = 0; jj < nbh.nbhcount_; ++jj) {
        const double &rij = nbh.distances_[jj];
        const int &jtype = nbh.chemmap_[jj];
        if (rij > cutoffs(itype, jtype)) continue;
        const double rijinv = 1.0 / rij;
        if constexpr (IsOMP) {
            rb_calcder_omp(itype, jtype, rij, buf);
        } else {
            rb_calcder(itype, jtype, rij);
        }

        dp[0] = 1;
        for (int k = 1; k < max_alpha_index_basic; ++k) {
            dp[k] = dp[k - 1] * rijinv;
            for (int a = 0; a < 3; a++) {
                cp(k + 1, a) = cp(k, a) * nbh.pos_[jj][a];
            }
        }

        for (int mu = 0; mu < max_mu; ++mu) {
            vm[mu] = 0;
            dm[mu] = 0;
            int idxoffset =
                    (nbh.centertype * species_count + nbh.chemmap_[jj]) * radial_func_count * rb_size + mu * rb_size;
            for (int xi = 0; xi < rb_size; ++xi) {
                vm[mu] += radial_coeffs[idxoffset + xi] * rv[xi];
                dm[mu] += radial_coeffs[idxoffset + xi] * rd[xi];
            }
        }

        if (radial_local_cache) cache.begin_neighbor();
        for (int i = 0; i < alpha_index_basic_count; ++i) {
            int mu = alpha_index_basic(i, 0);
            double val = vm[mu];
            double der = dm[mu];
            int k = nu_arr_list[i];
            double &powk = dp[k];
            if (radial_local_cache) {
                const auto normalized = cache.get<true>(mu, k, val, der, powk, rijinv);
                val = normalized.first;
                der = normalized.second;
            } else {
                val *= powk;
                der = der * powk - k * val / rij;
            }
            double pow0 = cp(alpha_index_basic(i, 1) + 1, 0);
            double pow1 = cp(alpha_index_basic(i, 2) + 1, 1);
            double pow2 = cp(alpha_index_basic(i, 3) + 1, 2);

            double mult0 = pow0 * pow1 * pow2;
            mv[i] += val * mult0;
            mult0 *= der / rij;
            mj(i, jj, 0) += mult0 * nbh.pos_[jj][0];
            mj(i, jj, 1) += mult0 * nbh.pos_[jj][1];
            mj(i, jj, 2) += mult0 * nbh.pos_[jj][2];
            mj(i, jj, 0) +=
                    alpha_index_basic(i, 1) * val *
                    cp(alpha_index_basic(i, 1), 0) *
                    pow1 * pow2;
            mj(i, jj, 1) +=
                    alpha_index_basic(i, 2) * val *
                    cp(alpha_index_basic(i, 2), 1) *
                    pow0 * pow2;
            mj(i, jj, 2) +=
                    alpha_index_basic(i, 3) * val *
                    cp(alpha_index_basic(i, 3), 2) *
                    pow0 * pow1;
        }
    }

    // Forward product rules for moment_vals only (no Jacobian propagation)
    for (int i = 0; i < alpha_index_times_count; ++i) {
        double val0 = mv[alpha_index_times(i, 0)];
        double val1 = mv[alpha_index_times(i, 1)];
        int val2 = alpha_index_times(i, 2);
        mv[alpha_index_times(i, 3)] += val2 * val0 * val1;
    }
}

void PotMTP::CalcSiteDescriptors(Neighborhood &nbh) {
    ompbuffers unused;
    CalcSiteDescriptorsImpl<false>(nbh, unused);
}

void PotMTP::CalcSiteDescriptorsOMP(Neighborhood &nbh, ompbuffers &buffer) {
    CalcSiteDescriptorsImpl<true>(nbh, buffer);
}

void PotMTP::CalcSiteEFS(Neighborhood &nbh, double &site_energy,
                          Eigen::Matrix<double, -1, 3, Eigen::RowMajor> &site_energy_ders) {
    // Delegate descriptor + basic Jacobian to lightweight shared implementation.
    CalcSiteDescriptors(nbh);

    // Compute site energy (linear combination of scalar invariants)
    site_energy += species_coeffs[nbh.centertype];
    for (int i = 0; i < alpha_scalar_moments; ++i) {
        site_energy += moment_coeffs[i] * moment_vals[alpha_moment_mapping[i]];
    }

    // Backpropagation step 1: dE/dB for scalar moments = moment_coeffs
    std::ranges::fill(site_energy_ders_wrt_moments_, 0);
    for (int i = 0; i < alpha_scalar_moments; ++i) {
        site_energy_ders_wrt_moments_[alpha_moment_mapping[i]] = moment_coeffs[i];
    }

    // Backpropagation step 2: reverse traversal through product rules
    backpropagate_moments();

    // Backpropagation step 3: contract with Jacobian → site_energy_ders
    site_energy_ders.resize(nbh.nbhcount_, 3);
    site_energy_ders.setZero();
    for (int i = 0; i < alpha_index_basic_count; ++i)
        for (int j = 0; j < nbh.nbhcount_; ++j)
            for (int a = 0; a < 3; a++) {
                site_energy_ders(j, a) += site_energy_ders_wrt_moments_[i] * moment_jacobian(i, j, a);
            }
}

void PotMTP::CalcSiteEFSOMP(Neighborhood &nbh, ompbuffers &buffer) {
    // Delegate descriptor + basic Jacobian to lightweight OMP implementation
    CalcSiteDescriptorsOMP(nbh, buffer);

    // convolving with coefficients
    buffer.buff_site_energy += species_coeffs[nbh.centertype];
    for (int i = 0; i < alpha_scalar_moments; ++i) {
        buffer.buff_site_energy += moment_coeffs[i] * buffer.moment_vals[alpha_moment_mapping[i]];
    }

    // Backpropagation step 1: dE/dB for scalar moments = moment_coeffs
    std::ranges::fill(buffer.site_energy_ders_wrt_moments_, 0);
    for (int i = 0; i < alpha_scalar_moments; ++i) {
        buffer.site_energy_ders_wrt_moments_[alpha_moment_mapping[i]] = moment_coeffs[i];
    }

    // Backpropagation step 2: reverse traversal through product rules
    for (int i = alpha_index_times_count - 1; i >= 0; i--) {
        double val0 = buffer.moment_vals[alpha_index_times(i, 0)];
        double val1 = buffer.moment_vals[alpha_index_times(i, 1)];
        int val2 = alpha_index_times(i, 2);

        buffer.site_energy_ders_wrt_moments_[alpha_index_times(i, 1)] +=
                buffer.site_energy_ders_wrt_moments_[alpha_index_times(i, 3)] * val2 * val0;
        buffer.site_energy_ders_wrt_moments_[alpha_index_times(i, 0)] +=
                buffer.site_energy_ders_wrt_moments_[alpha_index_times(i, 3)] * val2 * val1;
    }

    // Backpropagation step 3: contract with Jacobian → site_energy_ders
    buffer.buff_site_energy_ders.resize(nbh.nbhcount_, 3);
    buffer.buff_site_energy_ders.setZero();
    for (int i = 0; i < alpha_index_basic_count; ++i)
        for (int j = 0; j < nbh.nbhcount_; ++j)
            for (int a = 0; a < 3; a++) {
                buffer.buff_site_energy_ders(j, a) += buffer.site_energy_ders_wrt_moments_[i] * buffer.moment_jacobian(i, j, a);
            }
}

// ── Unified energy-only descriptor (serial / OMP) ───────────────────────
// Template parameter IsOMP selects member buffers (false) or ompbuffers (true).
template<bool IsOMP>
void PotMTP::CalcSiteBasisImpl(Neighborhood &nbh, ompbuffers &buf) {
    auto& mv = IsOMP ? buf.moment_vals   : moment_vals;
    auto& rv = IsOMP ? buf.rb_vals       : rb_vals;
    auto& vm = IsOMP ? buf.val_mu        : val_mu;
    auto& cp = IsOMP ? buf.coords_powers : coords_powers;
    auto& dp = IsOMP ? buf.dist_powers   : dist_powers;
    auto& cache = IsOMP ? buf.radial_cache : radial_cache;
    if (radial_local_cache) cache.begin_site(max_mu, max_alpha_index_basic);

    // basis_vals is only on the member (CalcEFSCmpt needs it); the OMP
    // energy-only path never reads basis_vals, but computing it is cheap.
    if constexpr (!IsOMP) {
        basis_vals.resize(alpha_scalar_moments + 1, 0);
    }

    std::ranges::fill(mv, 0);
    std::ranges::fill(rv, 0);
    std::ranges::fill(vm, 0);

    const int& itype = nbh.centertype;

    for (int jj = 0; jj < nbh.nbhcount_; ++jj) {
        const double &rij = nbh.distances_[jj];
        const int &jtype = nbh.chemmap_[jj];
        if (rij > cutoffs(itype, jtype)) continue;
        const double rijinv = 1.0 / rij;
        if constexpr (IsOMP) {
            rb_calcval_omp(itype, jtype, rij, buf);
        } else {
            rb_calcval(itype, jtype, rij);
        }

        dp[0] = 1;
        for (int k = 1; k < max_alpha_index_basic; ++k) {
            dp[k] = dp[k - 1] * rijinv;
            for (int a = 0; a < 3; a++) {
                cp(k + 1, a) = cp(k, a) * nbh.pos_[jj][a];
            }
        }

        for (int mu = 0; mu < max_mu; ++mu) {
            vm[mu] = 0;
            int idxoffset =
                    (nbh.centertype * species_count + nbh.chemmap_[jj]) * radial_func_count * rb_size + mu * rb_size;
            for (int xi = 0; xi < rb_size; xi++) {
                vm[mu] += radial_coeffs[idxoffset + xi] * rv[xi];
            }
        }

        if (radial_local_cache) cache.begin_neighbor();
        for (int i = 0; i < alpha_index_basic_count; ++i) {
            int mu = alpha_index_basic(i, 0);
            double val = vm[mu];
            int k = nu_arr_list[i];
            double &powk = dp[k];
            if (radial_local_cache)
                val = cache.get<false>(mu, k, val, 0., powk, rijinv).first;
            else
                val *= powk;
            double pow0 = cp(alpha_index_basic(i, 1) + 1, 0);
            double pow1 = cp(alpha_index_basic(i, 2) + 1, 1);
            double pow2 = cp(alpha_index_basic(i, 3) + 1, 2);
            double mult0 = pow0 * pow1 * pow2;
            mv[i] += val * mult0;
        }
    }

    // non-elementary b_i (forward product rules)
    for (int i = 0; i < alpha_index_times_count; ++i) {
        double val0 = mv[alpha_index_times(i, 0)];
        double val1 = mv[alpha_index_times(i, 1)];
        int val2 = alpha_index_times(i, 2);
        mv[alpha_index_times(i, 3)] += val2 * val0 * val1;
    }

    // Extract scalar invariants → basis_vals (only needed by serial path)
    if constexpr (!IsOMP) {
        basis_vals[0] = 1.0;
        for (int i = 0; i < alpha_scalar_moments; ++i)
            basis_vals[i + 1] = mv[alpha_moment_mapping[i]];
    }
}

void PotMTP::CalcSiteBasis(Neighborhood &nbh) {
    ompbuffers unused;
    CalcSiteBasisImpl<false>(nbh, unused);
}

void PotMTP::CalcSiteBasisOMP(Neighborhood &nbh, ompbuffers &buffer) {
    CalcSiteBasisImpl<true>(nbh, buffer);
}



void PotMTP::CalcSiteEOMP(Neighborhood &nbh, ompbuffers &buffer) {
    // Delegate descriptor computation to shared OMP implementation
    CalcSiteBasisOMP(nbh, buffer);

    // convolving with coefficients
    buffer.buff_site_energy += species_coeffs[nbh.centertype];
    for (int i = 0; i < alpha_scalar_moments; ++i) {
        buffer.buff_site_energy += moment_coeffs[i] * buffer.moment_vals[alpha_moment_mapping[i]];
    }
    buffer.buff_site_energy_ders.resize(nbh.nbhcount_, 3);
    buffer.buff_site_energy_ders.setZero();
}

void Potentials::PotMTP::CalcSiteE(Neighborhood &nbh, double &site_energy) {
    // Delegate descriptor computation to shared implementation.
    CalcSiteBasis(nbh);

    // Compute site energy (linear combination of scalar invariants)
    site_energy += species_coeffs[nbh.centertype];
    for (int i = 0; i < alpha_scalar_moments; ++i) {
        site_energy += moment_coeffs[i] * moment_vals[alpha_moment_mapping[i]];
    }
}

void PotMTP::CalcSiteBasisDers(Neighborhood &nbh) {
    int ns = alpha_scalar_moments + species_count;
    basis_vals.resize(alpha_scalar_moments + 1, 0);
    basis_ders.resize(alpha_scalar_moments + 1, nbh.nbhcount_, 3);
    basis_ders.setZero();
    buff_site_energy_ders.resize(nbh.nbhcount_, 3);
    moment_jacobian.resize(alpha_moments_count, nbh.nbhcount_, 3);
    moment_jacobian.setZero();

    std::ranges::fill(site_energy_ders_wrt_moments_, 0);
    std::ranges::fill(moment_vals, 0);
    std::ranges::fill(rb_vals, 0);
    std::ranges::fill(rb_ders, 0);
    std::ranges::fill(val_mu, 0);
    std::ranges::fill(der_mu, 0);

    const int &itype = nbh.centertype;

    for (int jj = 0; jj < nbh.nbhcount_; ++jj) {
        const double &rij = nbh.distances_[jj];
        const int &jtype = nbh.chemmap_[jj];
        if (rij > cutoffs(itype, jtype)) continue;
        const double rijinv = 1.0 / rij;
        rb_calcder(itype, jtype, rij);

        dist_powers[0] = 1;
        for (int k = 1; k < max_alpha_index_basic; ++k) {
            dist_powers[k] = dist_powers[k - 1] * rijinv;
            for (int a = 0; a < 3; a++) {
                coords_powers(k + 1, a) = coords_powers(k, a) * nbh.pos_[jj][a];
            }
        }

        for (int mu = 0; mu < max_mu; ++mu) {
            val_mu[mu] = 0;
            der_mu[mu] = 0;
            int idxoffset =
                    (nbh.centertype * species_count + nbh.chemmap_[jj]) * radial_func_count * rb_size + mu * rb_size;
            for (int xi = 0; xi < rb_size; xi++) {
                val_mu[mu] += radial_coeffs[idxoffset + xi] * rb_vals[xi];
                der_mu[mu] += radial_coeffs[idxoffset + xi] * rb_ders[xi];
            }
        }

        for (int i = 0; i < alpha_index_basic_count; ++i) {
            int mu = alpha_index_basic(i, 0);
            double val = val_mu[mu];
            double der = der_mu[mu];

            int k = nu_arr_list[i];
            double &powk = dist_powers[k];
            // TODO: test weather move this out can accelearte computation by defining val_mu_nv[:] and der_mu_nv[:]
            // since some (mu, nv) pair may not exist.
            val *= powk; // normalize, all tenser element is devided by rij^\nv to aviod extremely large values
            der = der * powk - k * val / rij;
            double pow0 = coords_powers(alpha_index_basic(i, 1) + 1, 0);
            double pow1 = coords_powers(alpha_index_basic(i, 2) + 1, 1);
            double pow2 = coords_powers(alpha_index_basic(i, 3) + 1, 2);

            double mult0 = pow0 * pow1 * pow2; // element of tensor r ^ ("x" nv)

            moment_vals[i] += val * mult0;
            mult0 *= der / rij;
            moment_jacobian(i, jj, 0) += mult0 * nbh.pos_[jj][0];
            moment_jacobian(i, jj, 1) += mult0 * nbh.pos_[jj][1];
            moment_jacobian(i, jj, 2) += mult0 * nbh.pos_[jj][2];

            moment_jacobian(i, jj, 0) +=
                    val * alpha_index_basic(i, 1) *
                    coords_powers(alpha_index_basic(i, 1), 0) *
                    pow1 * pow2;
            moment_jacobian(i, jj, 1) +=
                    val * alpha_index_basic(i, 2) *
                    coords_powers(alpha_index_basic(i, 2), 1) *
                    pow0 * pow2;
            moment_jacobian(i, jj, 2) +=
                    val * alpha_index_basic(i, 3) *
                    coords_powers(alpha_index_basic(i, 3), 2) *
                    pow0 * pow1;
        }
    }

    // non-elementary b_i
    for (int i = 0; i < alpha_index_times_count; ++i) {
        double val0 = moment_vals[alpha_index_times(i, 0)];
        double val1 = moment_vals[alpha_index_times(i, 1)];
        int val2 = alpha_index_times(i, 2);
        moment_vals[alpha_index_times(i, 3)] += val2 * val0 * val1;
        for (int j = 0; j < nbh.nbhcount_; ++j) {
            for (int a = 0; a < 3; ++a)
                moment_jacobian(alpha_index_times(i, 3), j, a) +=
                        val2 * ((moment_jacobian(alpha_index_times(i, 0), j, a)) * val1 +
                        val0 * (moment_jacobian(alpha_index_times(i, 1), j, a)));
        }
    }

    // convolving with coefficients
    // setting constant term as zero
    basis_vals[0] = 1.0;
    for (int i = 0; i < alpha_scalar_moments; ++i) {
        basis_vals[i + 1] = moment_vals[alpha_moment_mapping[i]];
        memcpy(&basis_ders(i + 1, 0, 0), &moment_jacobian(alpha_moment_mapping[i], 0, 0),
               3 * nbh.nbhcount_ * sizeof(double));
    }
}

// when using non-linear model, radial coeff part does not need any modification, only change dE/dB to model value rather than linear coeff
// for model coeffs {t}, we need add another BP cycle, need d2Ui/dBdt given by model then compute d2Ui/dbdt by BP, need n_model_coeffs BP cycle
// we can see that linear fit is fast, really fast compared to non-linear one, especally for large coefficies in non-linear model
// ── Shared backpropagation helper ─────────────────────────────────────────
// Performs step 2 of force backpropagation: reverse traversal through the
// product-rule tree (alpha_index_times) to express dE/d(non-elementary moment)
// as dE/d(basic moment).  Caller must fill the scalar-moment entries of
// site_energy_ders_wrt_moments_ (step 1) before calling.
// Reads: moment_vals, alpha_index_times, alpha_index_times_count
// Updates: site_energy_ders_wrt_moments_ in-place
void PotMTP::backpropagate_moments() {
    for (int i = alpha_index_times_count - 1; i >= 0; i--) {
        double val0 = moment_vals[alpha_index_times(i, 0)];
        double val1 = moment_vals[alpha_index_times(i, 1)];
        int val2 = alpha_index_times(i, 2);

        site_energy_ders_wrt_moments_[alpha_index_times(i, 1)] +=
                site_energy_ders_wrt_moments_[alpha_index_times(i, 3)] * val2 * val0;
        site_energy_ders_wrt_moments_[alpha_index_times(i, 0)] +=
                site_energy_ders_wrt_moments_[alpha_index_times(i, 3)] * val2 * val1;
    }
}

// In non-linear model case, we still need compute db/dr for all b
// In summary, first we need to set scalar part of site_energy_ders_wrt_moments_ to model value of dU/dB, then radial coeffs part is just fine
// Second deal with model parameters, fill scalar part of d2U/dBdt, when do BP to get all d2U/dbdt
//
// Additional input: dU/dB (vector), d2U/dBdt (matrix) (t is model parameter), dU/dt (vector)
void PotMTP::AccumulateFPGard(Neighborhood &nbh,
                               double *out_gard_accumulator,
                               double se_weight,
                               Eigen::Matrix<double, -1, 3, Eigen::RowMajor> *f_weight) {
    moment_jacobian.resize(alpha_index_basic_count, nbh.nbhcount_, 3);
    moment_jacobian.setZero();
    moment_jacobian_coeff_coord.setZero();
    moment_jacobian_coeffs.setZero();

    std::ranges::fill(moment_vals, 0);
    std::ranges::fill(rb_vals, 0);
    std::ranges::fill(rb_ders, 0);
    std::ranges::fill(val_mu, 0);
    std::ranges::fill(der_mu, 0);
    std::fill(dloss_dmom.begin(), dloss_dmom.end(), 0);
    std::fill(dloss_dsenders.begin(), dloss_dsenders.end(), 0);
    std::fill(site_energy_ders_wrt_moments_.begin(), site_energy_ders_wrt_moments_.end(), 0);
    const int& itype = nbh.centertype;
    for (int jj = 0; jj < nbh.nbhcount_; ++jj) {
        const double &rij = nbh.distances_[jj];
        const int &jtype = nbh.chemmap_[jj];
        if (rij > cutoffs(itype, jtype)) continue;
        const double rijinv = 1.0 / rij;
        rb_calcder(itype, jtype, rij);

        dist_powers[0] = 1;
        for (int k = 1; k < max_alpha_index_basic; ++k) {
            dist_powers[k] = dist_powers[k - 1] * rijinv;
            for (int a = 0; a < 3; a++) {
                coords_powers(k + 1, a) = coords_powers(k, a) * nbh.pos_[jj][a]; // off set by 1 to aviod if later
            }
        }

        for (int mu = 0; mu < max_mu; ++mu) {
            val_mu[mu] = 0;
            der_mu[mu] = 0;
            int idxoffset =
                    (nbh.centertype * species_count + nbh.chemmap_[jj]) * radial_func_count * rb_size + mu * rb_size;
            for (int xi = 0; xi < rb_size; ++xi) {
                val_mu[mu] += radial_coeffs[idxoffset + xi] * rb_vals[xi];
                der_mu[mu] += radial_coeffs[idxoffset + xi] * rb_ders[xi];
            }
        }

        for (int i = 0; i < alpha_index_basic_count; ++i) {
            // @OPTIMIZE: init mu-array while loading potentials, do not need offset search
            int mu = alpha_index_basic(i, 0);
            double val = val_mu[mu];
            double der = der_mu[mu];
            int k = nu_arr_list[i];
            double &powk = dist_powers[k];
            // TODO: test weather move this out can accelearte computation by defining val_mu_nv[:] and der_mu_nv[:]
            // since some (mu, nv) pair may not exist.
            val *= powk; // normalize, all tenser element is devided by rij^\nv to avoid extremely large values
            der = der * powk - k * val / rij;
            // offset by 1
            double pow0 = coords_powers(alpha_index_basic(i, 1) + 1, 0); // x
            double pow1 = coords_powers(alpha_index_basic(i, 2) + 1, 1); // y
            double pow2 = coords_powers(alpha_index_basic(i, 3) + 1, 2); // z

            double mult0 = pow0 * pow1 * pow2; // element of tensor r ^ ("x" nv)

            moment_vals[i] += val * mult0;
            for (int xi = 0; xi < rb_size; xi++) {
                // dbm / dc
                moment_jacobian_coeffs(i, nbh.chemmap_[jj], mu * rb_size + xi) +=
                        rb_vals[xi] * powk * mult0;
                // force related
                if (f_weight != nullptr) {
                    // here, derx = d2bm / dxdc
                    double tail = (rb_ders[xi] * powk * mult0 - rb_vals[xi] * k * powk * mult0 / nbh.distances_[jj]);
                    double derx = (nbh.pos_[jj][0] / nbh.distances_[jj]) * tail;
                    double dery = (nbh.pos_[jj][1] / nbh.distances_[jj]) * tail;
                    double derz = (nbh.pos_[jj][2] / nbh.distances_[jj]) * tail;
                    derx += rb_vals[xi] * powk *
                            alpha_index_basic(i, 1) *
                            coords_powers(alpha_index_basic(i, 1), 0) *
                            pow1 * pow2;
                    dery += rb_vals[xi] * powk *
                            alpha_index_basic(i, 2) *
                            coords_powers(alpha_index_basic(i, 2), 1) *
                            pow0 * pow2;
                    derz += rb_vals[xi] * powk *
                            alpha_index_basic(i, 3) *
                            coords_powers(alpha_index_basic(i, 3), 2) *
                            pow0 * pow1;
                    // this is w_i \cdot d2bm/drdc
                    moment_jacobian_coeff_coord(i, nbh.chemmap_[jj], mu * rb_size + xi) +=
                            (*f_weight)(jj, 0) * derx + (*f_weight)(jj, 1) * dery + (*f_weight)(jj, 2) * derz;
                }
            }
            mult0 *= der / rij;
            moment_jacobian(i, jj, 0) += mult0 * nbh.pos_[jj][0];
            moment_jacobian(i, jj, 1) += mult0 * nbh.pos_[jj][1];
            moment_jacobian(i, jj, 2) += mult0 * nbh.pos_[jj][2];
            // coords_powers is shifted by 3 to avoid if statement
            // note that 0 * Inf = NaN
            moment_jacobian(i, jj, 0) +=
                    alpha_index_basic(i, 1) * val *
                    coords_powers(alpha_index_basic(i, 1), 0) *
                    pow1 * pow2;
            moment_jacobian(i, jj, 1) +=
                    alpha_index_basic(i, 2) * val *
                    coords_powers(alpha_index_basic(i, 2), 1) *
                    pow0 * pow2;
            moment_jacobian(i, jj, 2) +=
                    alpha_index_basic(i, 3) * val *
                    coords_powers(alpha_index_basic(i, 3), 2) *
                    pow0 * pow1;
        }
    }

    // non-elementary b_i
    for (int i = 0; i < alpha_index_times_count; ++i) {
        double val0 = moment_vals[alpha_index_times(i, 0)];
        double val1 = moment_vals[alpha_index_times(i, 1)];
        double val2 = alpha_index_times(i, 2);
        moment_vals[alpha_index_times(i, 3)] += val2 * val0 * val1;
    }


    // Backpropagation step 1: site energy derivative is the corresponding linear combination
    for (int i = 0; i < alpha_scalar_moments; ++i)
        site_energy_ders_wrt_moments_[alpha_moment_mapping[i]] = moment_coeffs[i]; // this is dE/dbi

    // Backpropagation step 2: expressing through basic moments:
    for (int i = alpha_index_times_count - 1; i >= 0; --i) {
        double val0 = moment_vals[alpha_index_times(i, 0)];
        double val1 = moment_vals[alpha_index_times(i, 1)];
        double val2 = alpha_index_times(i, 2);

        site_energy_ders_wrt_moments_[alpha_index_times(i, 1)] +=
                site_energy_ders_wrt_moments_[alpha_index_times(i, 3)] * val2 * val0;
        site_energy_ders_wrt_moments_[alpha_index_times(i, 0)] +=
                site_energy_ders_wrt_moments_[alpha_index_times(i, 3)] * val2 * val1;
    }

    // Backpropagation step 3: multiply by the Jacobian:
    for (int i = 0; i < alpha_index_basic_count; ++i)
        for (int j = 0; j < nbh.nbhcount_; ++j)
            for (int a = 0; a < 3; a++) {

                if (f_weight != nullptr) {
                    dloss_dsenders[i] += (*f_weight)(j, a) * moment_jacobian(i, j, a); // this is wgt_fceij \cdot dbm/drij
                }
            }
    for (int i = 0; i < alpha_index_basic_count; ++i)
        dloss_dmom[i] = se_weight * site_energy_ders_wrt_moments_[i]; // currently this is weight_eng * dUi/dbm, these basic term won't be inkoved during BP process later if force is requesed
    if (f_weight != nullptr) {
        // f_weight * db_j/dr
        for (int i = 0; i < alpha_index_times_count; ++i) {
            double val2 = alpha_index_times(i, 2);
            double val0 = moment_vals[alpha_index_times(i, 0)];
            double val1 = moment_vals[alpha_index_times(i, 1)];
            dloss_dsenders[alpha_index_times(i, 3)] +=
                    dloss_dsenders[alpha_index_times(i, 0)] * val2 * val1 +
                    dloss_dsenders[alpha_index_times(i, 1)] * val2 * val0;
        }

        // d^2E/db_jdr,
        /**
         * Note1. Backpropagation only ADD things to basic moments index terms, thus dloss_dmom still valid for energy der components.
         * Note2. This loop
         */
        // for local energy U_i, we have dU/dbj += c * bi * dU/dbk over rule {i, j, c, k}
        // then we have d/dr(dU/dbj) = c dbi/dr dU/dbk + c bi d/dr(dU/dbk), this loop add the first term 
        for (int i = 0; i < alpha_index_times_count; ++i) {
            double val2 = alpha_index_times(i, 2);

            dloss_dmom[alpha_index_times(i, 1)] += dloss_dsenders[alpha_index_times(i, 0)] *
                                                   site_energy_ders_wrt_moments_[alpha_index_times(i, 3)] * val2;
            dloss_dmom[alpha_index_times(i, 0)] += dloss_dsenders[alpha_index_times(i, 1)] *
                                                   site_energy_ders_wrt_moments_[alpha_index_times(i, 3)] * val2;
        }

        // Backpropagation, calculate contribution of d/dr(dU/dbm), bm is basic 
        for (int i = alpha_index_times_count - 1; i >= 0; --i) {
            double val0 = moment_vals[alpha_index_times(i, 0)];
            double val1 = moment_vals[alpha_index_times(i, 1)];
            double val2 = alpha_index_times(i, 2);

            dloss_dmom[alpha_index_times(i, 0)] += dloss_dmom[alpha_index_times(i, 3)] * val2 * val1;
            dloss_dmom[alpha_index_times(i, 1)] += dloss_dmom[alpha_index_times(i, 3)] * val2 * val0;
        }
    }
    // radial coeffs
    for (int i = 0; i < alpha_index_basic_count; ++i) {
        for (int j = 0; j < species_count; ++j) {
            for (int k = 0; k < radial_func_count * rb_size; ++k) {
                int idx = (nbh.centertype * species_count + j) * rb_size * radial_func_count + k;
                out_gard_accumulator[idx] += dloss_dmom[i] * moment_jacobian_coeffs(i, j, k);
                if (f_weight != nullptr) {
                    // dE/db_i*d^2b/(dxdc) term, the d^2E/(db_mdrij)*db_m/dc term is included in the dloss_dmom array
                    out_gard_accumulator[idx] +=
                            site_energy_ders_wrt_moments_[i] * moment_jacobian_coeff_coord(i, j, k);
                }
            }
        }
    }
    // species coeffs
    out_gard_accumulator[radial_coeff_size + nbh.centertype] += se_weight;
    // linear coeffs, when swiching to model coeffs, this part need rewrite. Energy part is simple, just replace B to dU/dt, force part need BP
    for (int i = 0; i < alpha_scalar_moments; ++i) {
        out_gard_accumulator[i + radial_coeff_size + species_count] +=
                se_weight * moment_vals[alpha_moment_mapping[i]];
        if (f_weight != nullptr)
            out_gard_accumulator[i + radial_coeff_size +
                                 species_count] += dloss_dsenders[alpha_moment_mapping[i]];
    }
}

void Potentials::PotMTP::AccumulateFPGard(Neighborhood &nbh, double *out_gard_accumulator, double se_weight, Eigen::Matrix<double, -1, 3, Eigen::RowMajor> *f_weight, int roffset, int loffset)
{
    moment_jacobian.resize(alpha_index_basic_count, nbh.nbhcount_, 3);
    moment_jacobian.setZero();
    moment_jacobian_coeff_coord.setZero();
    moment_jacobian_coeffs.setZero();

    std::ranges::fill(moment_vals, 0);
    std::ranges::fill(rb_vals, 0);
    std::ranges::fill(rb_ders, 0);
    std::ranges::fill(val_mu, 0);
    std::ranges::fill(der_mu, 0);
    std::fill(dloss_dmom.begin(), dloss_dmom.end(), 0);
    std::fill(dloss_dsenders.begin(), dloss_dsenders.end(), 0);
    std::fill(site_energy_ders_wrt_moments_.begin(), site_energy_ders_wrt_moments_.end(), 0);
    const int&itype = nbh.centertype;
    for (int jj = 0; jj < nbh.nbhcount_; ++jj) {
        const double &rij = nbh.distances_[jj];
        const int &jtype = nbh.chemmap_[jj];
        if (rij > cutoffs(itype, jtype)) continue;
        const double rijinv = 1.0 / rij;
        rb_calcder(itype, jtype, rij);

        dist_powers[0] = 1;
        for (int k = 1; k < max_alpha_index_basic; ++k) {
            dist_powers[k] = dist_powers[k - 1] * rijinv;
            for (int a = 0; a < 3; a++) {
                coords_powers(k + 1, a) = coords_powers(k, a) * nbh.pos_[jj][a]; // off set by 1 to aviod if later
            }
        }

        for (int mu = 0; mu < max_mu; ++mu) {
            val_mu[mu] = 0;
            der_mu[mu] = 0;
            int idxoffset =
                    (nbh.centertype * species_count + nbh.chemmap_[jj]) * radial_func_count * rb_size + mu * rb_size;
            for (int xi = 0; xi < rb_size; ++xi) {
                val_mu[mu] += radial_coeffs[idxoffset + xi] * rb_vals[xi];
                der_mu[mu] += radial_coeffs[idxoffset + xi] * rb_ders[xi];
            }
        }

        for (int i = 0; i < alpha_index_basic_count; ++i) {
            // @OPTIMIZE: init mu-array while loading potentials, do not need offset search
            int mu = alpha_index_basic(i, 0);
            double val = val_mu[mu];
            double der = der_mu[mu];
            int k = nu_arr_list[i];
            double &powk = dist_powers[k];
            // TODO: test weather move this out can accelearte computation by defining val_mu_nv[:] and der_mu_nv[:]
            // since some (mu, nv) pair may not exist.
            val *= powk; // normalize, all tenser element is devided by rij^\nv to avoid extremely large values
            der = der * powk - k * val / rij;
            // offset by 1
            double pow0 = coords_powers(alpha_index_basic(i, 1) + 1, 0); // x
            double pow1 = coords_powers(alpha_index_basic(i, 2) + 1, 1); // y
            double pow2 = coords_powers(alpha_index_basic(i, 3) + 1, 2); // z

            double mult0 = pow0 * pow1 * pow2; // element of tensor r ^ ("x" nv)

            moment_vals[i] += val * mult0;
            for (int xi = 0; xi < rb_size; xi++) {
                moment_jacobian_coeffs(i, nbh.chemmap_[jj], mu * rb_size + xi) +=
                        rb_vals[xi] * powk * mult0;
                // force related
                if (f_weight != nullptr) {
                    double tail = (rb_ders[xi] * powk * mult0 - rb_vals[xi] * k * powk * mult0 / nbh.distances_[jj]);
                    double derx = (nbh.pos_[jj][0] / nbh.distances_[jj]) * tail;
                    double dery = (nbh.pos_[jj][1] / nbh.distances_[jj]) * tail;
                    double derz = (nbh.pos_[jj][2] / nbh.distances_[jj]) * tail;
                    derx += rb_vals[xi] * powk *
                            alpha_index_basic(i, 1) *
                            coords_powers(alpha_index_basic(i, 1), 0) *
                            pow1 * pow2;
                    dery += rb_vals[xi] * powk *
                            alpha_index_basic(i, 2) *
                            coords_powers(alpha_index_basic(i, 2), 1) *
                            pow0 * pow2;
                    derz += rb_vals[xi] * powk *
                            alpha_index_basic(i, 3) *
                            coords_powers(alpha_index_basic(i, 3), 2) *
                            pow0 * pow1;

                    moment_jacobian_coeff_coord(i, nbh.chemmap_[jj], mu * rb_size + xi) +=
                            (*f_weight)(jj, 0) * derx + (*f_weight)(jj, 1) * dery + (*f_weight)(jj, 2) * derz;
                }
            }
            mult0 *= der / rij;
            moment_jacobian(i, jj, 0) += mult0 * nbh.pos_[jj][0];
            moment_jacobian(i, jj, 1) += mult0 * nbh.pos_[jj][1];
            moment_jacobian(i, jj, 2) += mult0 * nbh.pos_[jj][2];
            // coords_powers is shifted by 3 to avoid if statement
            // note that 0 * Inf = NaN
            moment_jacobian(i, jj, 0) +=
                    alpha_index_basic(i, 1) * val *
                    coords_powers(alpha_index_basic(i, 1), 0) *
                    pow1 * pow2;
            moment_jacobian(i, jj, 1) +=
                    alpha_index_basic(i, 2) * val *
                    coords_powers(alpha_index_basic(i, 2), 1) *
                    pow0 * pow2;
            moment_jacobian(i, jj, 2) +=
                    alpha_index_basic(i, 3) * val *
                    coords_powers(alpha_index_basic(i, 3), 2) *
                    pow0 * pow1;
        }
    }

    // non-elementary b_i
    for (int i = 0; i < alpha_index_times_count; ++i) {
        double val0 = moment_vals[alpha_index_times(i, 0)];
        double val1 = moment_vals[alpha_index_times(i, 1)];
        double val2 = alpha_index_times(i, 2);
        moment_vals[alpha_index_times(i, 3)] += val2 * val0 * val1;
    }


    // Backpropagation step 1: site energy derivative is the corresponding linear combination
    for (int i = 0; i < alpha_scalar_moments; ++i)
        site_energy_ders_wrt_moments_[alpha_moment_mapping[i]] = moment_coeffs[i];

    // Backpropagation step 2: expressing through basic moments:
    for (int i = alpha_index_times_count - 1; i >= 0; --i) {
        double val0 = moment_vals[alpha_index_times(i, 0)];
        double val1 = moment_vals[alpha_index_times(i, 1)];
        double val2 = alpha_index_times(i, 2);

        site_energy_ders_wrt_moments_[alpha_index_times(i, 1)] +=
                site_energy_ders_wrt_moments_[alpha_index_times(i, 3)] * val2 * val0;
        site_energy_ders_wrt_moments_[alpha_index_times(i, 0)] +=
                site_energy_ders_wrt_moments_[alpha_index_times(i, 3)] * val2 * val1;
    }

    // Backpropagation step 3: multiply by the Jacobian:
    for (int i = 0; i < alpha_index_basic_count; ++i)
        for (int j = 0; j < nbh.nbhcount_; ++j)
            for (int a = 0; a < 3; a++) {

                if (f_weight != nullptr) {
                    dloss_dsenders[i] += (*f_weight)(j, a) * moment_jacobian(i, j, a);
                }
            }
    for (int i = 0; i < alpha_index_basic_count; ++i)
        dloss_dmom[i] = se_weight * site_energy_ders_wrt_moments_[i];
    if (f_weight != nullptr) {
        // db_j/dx
        for (int i = 0; i < alpha_index_times_count; ++i) {
            double val2 = alpha_index_times(i, 2);
            double val0 = moment_vals[alpha_index_times(i, 0)];
            double val1 = moment_vals[alpha_index_times(i, 1)];
            dloss_dsenders[alpha_index_times(i, 3)] +=
                    dloss_dsenders[alpha_index_times(i, 0)] * val2 * val1 +
                    dloss_dsenders[alpha_index_times(i, 1)] * val2 * val0;
        }

        // d^2E/db_jdx,
        /**
         * Note1. Backpropagation only ADD things to basic moments index terms, thus dloss_dmom still valid for energy der components.
         * Note2. This loop
         */
        for (int i = 0; i < alpha_index_times_count; ++i) {
            double val2 = alpha_index_times(i, 2);

            dloss_dmom[alpha_index_times(i, 1)] += dloss_dsenders[alpha_index_times(i, 0)] *
                                                   site_energy_ders_wrt_moments_[alpha_index_times(i, 3)] * val2;
            dloss_dmom[alpha_index_times(i, 0)] += dloss_dsenders[alpha_index_times(i, 1)] *
                                                   site_energy_ders_wrt_moments_[alpha_index_times(i, 3)] * val2;
        }

        // Backpropagation to get all basic terms, d/db_j (dE/dx)
        for (int i = alpha_index_times_count - 1; i >= 0; --i) {
            double val0 = moment_vals[alpha_index_times(i, 0)];
            double val1 = moment_vals[alpha_index_times(i, 1)];
            double val2 = alpha_index_times(i, 2);

            dloss_dmom[alpha_index_times(i, 0)] += dloss_dmom[alpha_index_times(i, 3)] * val2 * val1;
            dloss_dmom[alpha_index_times(i, 1)] += dloss_dmom[alpha_index_times(i, 3)] * val2 * val0;
        }
    }
    // radial coeffs
    for (int i = 0; i < alpha_index_basic_count; ++i) {
        for (int j = 0; j < species_count; ++j) {
            for (int k = 0; k < radial_func_count * rb_size; ++k) {
                int idx = (nbh.centertype * species_count + j) * rb_size * radial_func_count + k;
                out_gard_accumulator[idx + roffset] += dloss_dmom[i] * moment_jacobian_coeffs(i, j, k);
                if (f_weight != nullptr) {
                    // dE/db_i*d^2b/(dxdc) term, the d^2E/(db_idc)*db_i/dc term is included in the dloss_dmom array
                    out_gard_accumulator[idx + roffset] +=
                            site_energy_ders_wrt_moments_[i] * moment_jacobian_coeff_coord(i, j, k);
                }
            }
        }
    }
    for (int i = 0; i < alpha_scalar_moments; ++i) {
        out_gard_accumulator[i + loffset] +=
                se_weight * moment_vals[alpha_moment_mapping[i]];
        if (f_weight != nullptr)
            out_gard_accumulator[i + loffset] += dloss_dsenders[alpha_moment_mapping[i]];
    }
}

void PotMTP::AccumulateFPGardRonly(Neighborhood &nbh,
                               double *out_gard_accumulator,
                               double se_weight,
                               Eigen::Matrix<double, -1, 3, Eigen::RowMajor> *f_weight) {
    moment_jacobian.resize(alpha_index_basic_count, nbh.nbhcount_, 3);
    moment_jacobian.setZero();
    moment_jacobian_coeff_coord.setZero();
    moment_jacobian_coeffs.setZero();

    std::ranges::fill(moment_vals, 0);
    std::ranges::fill(rb_vals, 0);
    std::ranges::fill(rb_ders, 0);
    std::ranges::fill(val_mu, 0);
    std::ranges::fill(der_mu, 0);
    std::fill(dloss_dmom.begin(), dloss_dmom.end(), 0);
    std::fill(dloss_dsenders.begin(), dloss_dsenders.end(), 0);
    std::fill(site_energy_ders_wrt_moments_.begin(), site_energy_ders_wrt_moments_.end(), 0);
    const int& itype = nbh.centertype;
    for (int jj = 0; jj < nbh.nbhcount_; ++jj) {
        const double &rij = nbh.distances_[jj];
        const int &jtype = nbh.chemmap_[jj];
        const double rijinv = 1.0 / rij;
        if (rij > cutoffs(itype, jtype)) continue;
        rb_calcder(itype, jtype, rij);

        dist_powers[0] = 1;
        for (int k = 1; k < max_alpha_index_basic; ++k) {
            dist_powers[k] = dist_powers[k - 1] * rijinv;
            for (int a = 0; a < 3; a++) {
                coords_powers(k + 1, a) = coords_powers(k, a) * nbh.pos_[jj][a]; // off set by 1 to aviod if later
            }
        }

        for (int mu = 0; mu < max_mu; ++mu) {
            val_mu[mu] = 0;
            der_mu[mu] = 0;
            int idxoffset =
                    (nbh.centertype * species_count + nbh.chemmap_[jj]) * radial_func_count * rb_size + mu * rb_size;
            for (int xi = 0; xi < rb_size; ++xi) {
                val_mu[mu] += radial_coeffs[idxoffset + xi] * rb_vals[xi];
                der_mu[mu] += radial_coeffs[idxoffset + xi] * rb_ders[xi];
            }
        }

        for (int i = 0; i < alpha_index_basic_count; ++i) {
            // @OPTIMIZE: init mu-array while loading potentials, do not need offset search
            int mu = alpha_index_basic(i, 0);
            double val = val_mu[mu];
            double der = der_mu[mu];
            int k = nu_arr_list[i];
            double &powk = dist_powers[k];
            // TODO: test weather move this out can accelearte computation by defining val_mu_nv[:] and der_mu_nv[:]
            // since some (mu, nv) pair may not exist.
            val *= powk; // normalize, all tenser element is devided by rij^\nv to avoid extremely large values
            der = der * powk - k * val / rij;
            // offset by 1
            double pow0 = coords_powers(alpha_index_basic(i, 1) + 1, 0); // x
            double pow1 = coords_powers(alpha_index_basic(i, 2) + 1, 1); // y
            double pow2 = coords_powers(alpha_index_basic(i, 3) + 1, 2); // z

            double mult0 = pow0 * pow1 * pow2; // element of tensor r ^ ("x" nv)

            moment_vals[i] += val * mult0;
            for (int xi = 0; xi < rb_size; xi++) {
                moment_jacobian_coeffs(i, nbh.chemmap_[jj], mu * rb_size + xi) +=
                        rb_vals[xi] * powk * mult0;
                // force related
                if (f_weight != nullptr) {
                    double tail = (rb_ders[xi] * powk * mult0 - rb_vals[xi] * k * powk * mult0 / nbh.distances_[jj]);
                    double derx = (nbh.pos_[jj][0] / nbh.distances_[jj]) * tail;
                    double dery = (nbh.pos_[jj][1] / nbh.distances_[jj]) * tail;
                    double derz = (nbh.pos_[jj][2] / nbh.distances_[jj]) * tail;
                    derx += rb_vals[xi] * powk *
                            alpha_index_basic(i, 1) *
                            coords_powers(alpha_index_basic(i, 1), 0) *
                            pow1 * pow2;
                    dery += rb_vals[xi] * powk *
                            alpha_index_basic(i, 2) *
                            coords_powers(alpha_index_basic(i, 2), 1) *
                            pow0 * pow2;
                    derz += rb_vals[xi] * powk *
                            alpha_index_basic(i, 3) *
                            coords_powers(alpha_index_basic(i, 3), 2) *
                            pow0 * pow1;

                    moment_jacobian_coeff_coord(i, nbh.chemmap_[jj], mu * rb_size + xi) +=
                            (*f_weight)(jj, 0) * derx + (*f_weight)(jj, 1) * dery + (*f_weight)(jj, 2) * derz;
                }
            }
            mult0 *= der / rij;
            moment_jacobian(i, jj, 0) += mult0 * nbh.pos_[jj][0];
            moment_jacobian(i, jj, 1) += mult0 * nbh.pos_[jj][1];
            moment_jacobian(i, jj, 2) += mult0 * nbh.pos_[jj][2];
            // coords_powers is shifted by 3 to avoid if statement
            // note that 0 * Inf = NaN
            moment_jacobian(i, jj, 0) +=
                    alpha_index_basic(i, 1) * val *
                    coords_powers(alpha_index_basic(i, 1), 0) *
                    pow1 * pow2;
            moment_jacobian(i, jj, 1) +=
                    alpha_index_basic(i, 2) * val *
                    coords_powers(alpha_index_basic(i, 2), 1) *
                    pow0 * pow2;
            moment_jacobian(i, jj, 2) +=
                    alpha_index_basic(i, 3) * val *
                    coords_powers(alpha_index_basic(i, 3), 2) *
                    pow0 * pow1;
        }
    }

    // non-elementary b_i
    for (int i = 0; i < alpha_index_times_count; ++i) {
        double val0 = moment_vals[alpha_index_times(i, 0)];
        double val1 = moment_vals[alpha_index_times(i, 1)];
        double val2 = alpha_index_times(i, 2);
        moment_vals[alpha_index_times(i, 3)] += val2 * val0 * val1;
    }


    // Backpropagation step 1: site energy derivative is the corresponding linear combination
    for (int i = 0; i < alpha_scalar_moments; ++i)
        site_energy_ders_wrt_moments_[alpha_moment_mapping[i]] = moment_coeffs[i];

    // Backpropagation step 2: expressing through basic moments:
    for (int i = alpha_index_times_count - 1; i >= 0; --i) {
        double val0 = moment_vals[alpha_index_times(i, 0)];
        double val1 = moment_vals[alpha_index_times(i, 1)];
        double val2 = alpha_index_times(i, 2);

        site_energy_ders_wrt_moments_[alpha_index_times(i, 1)] +=
                site_energy_ders_wrt_moments_[alpha_index_times(i, 3)] * val2 * val0;
        site_energy_ders_wrt_moments_[alpha_index_times(i, 0)] +=
                site_energy_ders_wrt_moments_[alpha_index_times(i, 3)] * val2 * val1;
    }

    // Backpropagation step 3: multiply by the Jacobian:
    for (int i = 0; i < alpha_index_basic_count; ++i)
        for (int j = 0; j < nbh.nbhcount_; ++j)
            for (int a = 0; a < 3; a++) {

                if (f_weight != nullptr) {
                    dloss_dsenders[i] += (*f_weight)(j, a) * moment_jacobian(i, j, a);
                }
            }
    for (int i = 0; i < alpha_index_basic_count; ++i)
        dloss_dmom[i] = se_weight * site_energy_ders_wrt_moments_[i];
    if (f_weight != nullptr) {
        // db_j/dx
        for (int i = 0; i < alpha_index_times_count; ++i) {
            double val2 = alpha_index_times(i, 2);
            double val0 = moment_vals[alpha_index_times(i, 0)];
            double val1 = moment_vals[alpha_index_times(i, 1)];
            dloss_dsenders[alpha_index_times(i, 3)] +=
                    dloss_dsenders[alpha_index_times(i, 0)] * val2 * val1 +
                    dloss_dsenders[alpha_index_times(i, 1)] * val2 * val0;
        }

        // d^2E/db_jdx,
        /**
         * Note1. Backpropagation only ADD things to basic moments index terms, thus dloss_dmom still valid for energy der components.
         * Note2. This loop
         * TODO: what's this?
         */
        for (int i = 0; i < alpha_index_times_count; ++i) {
            double val2 = alpha_index_times(i, 2);

            dloss_dmom[alpha_index_times(i, 1)] += dloss_dsenders[alpha_index_times(i, 0)] *
                                                   site_energy_ders_wrt_moments_[alpha_index_times(i, 3)] * val2;
            dloss_dmom[alpha_index_times(i, 0)] += dloss_dsenders[alpha_index_times(i, 1)] *
                                                   site_energy_ders_wrt_moments_[alpha_index_times(i, 3)] * val2;
        }

        // Backpropagation to get all basic terms, d/db_j (dE/dx)
        // TODO: what's this?
        for (int i = alpha_index_times_count - 1; i >= 0; --i) {
            double val0 = moment_vals[alpha_index_times(i, 0)];
            double val1 = moment_vals[alpha_index_times(i, 1)];
            double val2 = alpha_index_times(i, 2);

            dloss_dmom[alpha_index_times(i, 0)] += dloss_dmom[alpha_index_times(i, 3)] * val2 * val1;
            dloss_dmom[alpha_index_times(i, 1)] += dloss_dmom[alpha_index_times(i, 3)] * val2 * val0;
        }
    }
    // radial coeffs
    for (int i = 0; i < alpha_index_basic_count; ++i) {
        for (int j = 0; j < species_count; ++j) {
            for (int k = 0; k < radial_func_count * rb_size; ++k) {
                int idx = (nbh.centertype * species_count + j) * rb_size * radial_func_count + k;
                out_gard_accumulator[idx] += dloss_dmom[i] * moment_jacobian_coeffs(i, j, k);
                if (f_weight != nullptr) {
                    // dE/db_i*d^2b/(dxdc) term, the d^2E/(db_idc)*db_i/dc term is included in the dloss_dmom array
                    out_gard_accumulator[idx] +=
                            site_energy_ders_wrt_moments_[i] * moment_jacobian_coeff_coord(i, j, k);
                }
            }
        }
    }
}

void Potentials::PotMTP::AccumulateFPGardRonly(Neighborhood &nbh, double *out_gard_accumulator, double se_weight, Eigen::Matrix<double, -1, 3, Eigen::RowMajor> *f_weight, int coffset)
{
    moment_jacobian.resize(alpha_index_basic_count, nbh.nbhcount_, 3);
    moment_jacobian.setZero();
    moment_jacobian_coeff_coord.setZero();
    moment_jacobian_coeffs.setZero();

    std::ranges::fill(moment_vals, 0);
    std::ranges::fill(rb_vals, 0);
    std::ranges::fill(rb_ders, 0);
    std::ranges::fill(val_mu, 0);
    std::ranges::fill(der_mu, 0);
    std::ranges::fill(dloss_dmom, 0);
    std::fill(dloss_dsenders.begin(), dloss_dsenders.end(), 0);
    std::fill(site_energy_ders_wrt_moments_.begin(), site_energy_ders_wrt_moments_.end(), 0);
    const int& itype = nbh.centertype;
    for (int jj = 0; jj < nbh.nbhcount_; ++jj) {
        const double &rij = nbh.distances_[jj];
        const int& jtype = nbh.chemmap_[jj];
        if (rij > cutoffs(itype, jtype)) continue;
        const double rijinv = 1.0 / rij;
        rb_calcder(itype, jtype, rij);

        dist_powers[0] = 1;
        for (int k = 1; k < max_alpha_index_basic; ++k) {
            dist_powers[k] = dist_powers[k - 1] * rijinv;
            for (int a = 0; a < 3; a++) {
                coords_powers(k + 1, a) = coords_powers(k, a) * nbh.pos_[jj][a]; // off set by 1 to aviod if later
            }
        }

        for (int mu = 0; mu < max_mu; ++mu) {
            val_mu[mu] = 0;
            der_mu[mu] = 0;
            int idxoffset =
                    (nbh.centertype * species_count + nbh.chemmap_[jj]) * radial_func_count * rb_size + mu * rb_size;
            for (int xi = 0; xi < rb_size; ++xi) {
                val_mu[mu] += radial_coeffs[idxoffset + xi] * rb_vals[xi];
                der_mu[mu] += radial_coeffs[idxoffset + xi] * rb_ders[xi];
            }
        }

        for (int i = 0; i < alpha_index_basic_count; ++i) {
            // @OPTIMIZE: init mu-array while loading potentials, do not need offset search
            int mu = alpha_index_basic(i, 0);
            double val = val_mu[mu];
            double der = der_mu[mu];
            int k = nu_arr_list[i];
            double &powk = dist_powers[k];
            // TODO: test weather move this out can accelearte computation by defining val_mu_nv[:] and der_mu_nv[:]
            // since some (mu, nv) pair may not exist.
            val *= powk; // normalize, all tenser element is devided by rij^\nv to avoid extremely large values
            der = der * powk - k * val / rij;
            // offset by 1
            double pow0 = coords_powers(alpha_index_basic(i, 1) + 1, 0); // x
            double pow1 = coords_powers(alpha_index_basic(i, 2) + 1, 1); // y
            double pow2 = coords_powers(alpha_index_basic(i, 3) + 1, 2); // z

            double mult0 = pow0 * pow1 * pow2; // element of tensor r ^ ("x" nv)

            moment_vals[i] += val * mult0;
            for (int xi = 0; xi < rb_size; xi++) {
                moment_jacobian_coeffs(i, nbh.chemmap_[jj], mu * rb_size + xi) +=
                        rb_vals[xi] * powk * mult0;
                // force related
                if (f_weight != nullptr) {
                    double tail = (rb_ders[xi] * powk * mult0 - rb_vals[xi] * k * powk * mult0 / nbh.distances_[jj]);
                    double derx = (nbh.pos_[jj][0] / nbh.distances_[jj]) * tail;
                    double dery = (nbh.pos_[jj][1] / nbh.distances_[jj]) * tail;
                    double derz = (nbh.pos_[jj][2] / nbh.distances_[jj]) * tail;
                    derx += rb_vals[xi] * powk *
                            alpha_index_basic(i, 1) *
                            coords_powers(alpha_index_basic(i, 1), 0) *
                            pow1 * pow2;
                    dery += rb_vals[xi] * powk *
                            alpha_index_basic(i, 2) *
                            coords_powers(alpha_index_basic(i, 2), 1) *
                            pow0 * pow2;
                    derz += rb_vals[xi] * powk *
                            alpha_index_basic(i, 3) *
                            coords_powers(alpha_index_basic(i, 3), 2) *
                            pow0 * pow1;

                    moment_jacobian_coeff_coord(i, nbh.chemmap_[jj], mu * rb_size + xi) +=
                            (*f_weight)(jj, 0) * derx + (*f_weight)(jj, 1) * dery + (*f_weight)(jj, 2) * derz;
                }
            }
            mult0 *= der / rij;
            moment_jacobian(i, jj, 0) += mult0 * nbh.pos_[jj][0];
            moment_jacobian(i, jj, 1) += mult0 * nbh.pos_[jj][1];
            moment_jacobian(i, jj, 2) += mult0 * nbh.pos_[jj][2];
            // coords_powers is shifted by 3 to avoid if statement
            // note that 0 * Inf = NaN
            moment_jacobian(i, jj, 0) +=
                    alpha_index_basic(i, 1) * val *
                    coords_powers(alpha_index_basic(i, 1), 0) *
                    pow1 * pow2;
            moment_jacobian(i, jj, 1) +=
                    alpha_index_basic(i, 2) * val *
                    coords_powers(alpha_index_basic(i, 2), 1) *
                    pow0 * pow2;
            moment_jacobian(i, jj, 2) +=
                    alpha_index_basic(i, 3) * val *
                    coords_powers(alpha_index_basic(i, 3), 2) *
                    pow0 * pow1;
        }
    }

    // non-elementary b_i
    for (int i = 0; i < alpha_index_times_count; ++i) {
        double val0 = moment_vals[alpha_index_times(i, 0)];
        double val1 = moment_vals[alpha_index_times(i, 1)];
        double val2 = alpha_index_times(i, 2);
        moment_vals[alpha_index_times(i, 3)] += val2 * val0 * val1;
    }


    // Backpropagation step 1: site energy derivative is the corresponding linear combination
    for (int i = 0; i < alpha_scalar_moments; ++i)
        site_energy_ders_wrt_moments_[alpha_moment_mapping[i]] = moment_coeffs[i];

    // Backpropagation step 2: expressing through basic moments:
    for (int i = alpha_index_times_count - 1; i >= 0; --i) {
        double val0 = moment_vals[alpha_index_times(i, 0)];
        double val1 = moment_vals[alpha_index_times(i, 1)];
        double val2 = alpha_index_times(i, 2);

        site_energy_ders_wrt_moments_[alpha_index_times(i, 1)] +=
                site_energy_ders_wrt_moments_[alpha_index_times(i, 3)] * val2 * val0;
        site_energy_ders_wrt_moments_[alpha_index_times(i, 0)] +=
                site_energy_ders_wrt_moments_[alpha_index_times(i, 3)] * val2 * val1;
    }

    // Backpropagation step 3: multiply by the Jacobian:
    for (int i = 0; i < alpha_index_basic_count; ++i)
        for (int j = 0; j < nbh.nbhcount_; ++j)
            for (int a = 0; a < 3; a++) {

                if (f_weight != nullptr) {
                    dloss_dsenders[i] += (*f_weight)(j, a) * moment_jacobian(i, j, a);
                }
            }
    for (int i = 0; i < alpha_index_basic_count; ++i)
        dloss_dmom[i] = se_weight * site_energy_ders_wrt_moments_[i];
    if (f_weight != nullptr) {
        // db_j/dx
        for (int i = 0; i < alpha_index_times_count; ++i) {
            double val2 = alpha_index_times(i, 2);
            double val0 = moment_vals[alpha_index_times(i, 0)];
            double val1 = moment_vals[alpha_index_times(i, 1)];
            dloss_dsenders[alpha_index_times(i, 3)] +=
                    dloss_dsenders[alpha_index_times(i, 0)] * val2 * val1 +
                    dloss_dsenders[alpha_index_times(i, 1)] * val2 * val0;
        }

        // d^2E/db_jdx,
        /**
         * Note1. Backpropagation only ADD things to basic moments index terms, thus dloss_dmom still valid for energy der components.
         * Note2. This loop
         * TODO: what's this?
         */
        for (int i = 0; i < alpha_index_times_count; ++i) {
            double val2 = alpha_index_times(i, 2);

            dloss_dmom[alpha_index_times(i, 1)] += dloss_dsenders[alpha_index_times(i, 0)] *
                                                   site_energy_ders_wrt_moments_[alpha_index_times(i, 3)] * val2;
            dloss_dmom[alpha_index_times(i, 0)] += dloss_dsenders[alpha_index_times(i, 1)] *
                                                   site_energy_ders_wrt_moments_[alpha_index_times(i, 3)] * val2;
        }

        // Backpropagation to get all basic terms, d/db_j (dE/dx)
        // TODO: what's this?
        for (int i = alpha_index_times_count - 1; i >= 0; --i) {
            double val0 = moment_vals[alpha_index_times(i, 0)];
            double val1 = moment_vals[alpha_index_times(i, 1)];
            double val2 = alpha_index_times(i, 2);

            dloss_dmom[alpha_index_times(i, 0)] += dloss_dmom[alpha_index_times(i, 3)] * val2 * val1;
            dloss_dmom[alpha_index_times(i, 1)] += dloss_dmom[alpha_index_times(i, 3)] * val2 * val0;
        }
    }
    // radial coeffs
    for (int i = 0; i < alpha_index_basic_count; ++i) {
        for (int j = 0; j < species_count; ++j) {
            for (int k = 0; k < radial_func_count * rb_size; ++k) {
                int idx = (nbh.centertype * species_count + j) * rb_size * radial_func_count + k;
                out_gard_accumulator[idx + coffset] += dloss_dmom[i] * moment_jacobian_coeffs(i, j, k);
                if (f_weight != nullptr) {
                    // dE/db_i*d^2b/(dxdc) term, the d^2E/(db_idc)*db_i/dc term is included in the dloss_dmom array
                    out_gard_accumulator[idx + coffset] +=
                            site_energy_ders_wrt_moments_[i] * moment_jacobian_coeff_coord(i, j, k);
                }
            }
        }
    }
}

void PotMTP::AccumulateFPGardLonly(Neighborhood &nbh,
                               double *out_gard_accumulator,
                               double se_weight,
                               Eigen::Matrix<double, -1, 3, Eigen::RowMajor> *f_weight) {
    const bool need_derivs = (f_weight != nullptr);

    if (need_derivs) {
        moment_jacobian.resize(alpha_index_basic_count, nbh.nbhcount_, 3);
        moment_jacobian.setZero();
    }

    std::ranges::fill(moment_vals, 0);
    std::ranges::fill(rb_vals, 0);
    std::ranges::fill(val_mu, 0);
    if (need_derivs) {
        std::ranges::fill(rb_ders, 0);
        std::ranges::fill(der_mu, 0);
        std::fill(dloss_dmom.begin(), dloss_dmom.end(), 0);
        std::fill(dloss_dsenders.begin(), dloss_dsenders.end(), 0);
        std::fill(site_energy_ders_wrt_moments_.begin(), site_energy_ders_wrt_moments_.end(), 0);
    }
    const int &itype = nbh.centertype;
    for (int jj = 0; jj < nbh.nbhcount_; ++jj) {
        const double &rij = nbh.distances_[jj];
        const int &jtype = nbh.chemmap_[jj];
        if (rij > cutoffs(itype, jtype)) continue;
        const double rijinv = 1.0 / rij;
        if (need_derivs)
            rb_calcder(itype, jtype, rij);
        else
            rb_calcval(itype, jtype, rij);

        dist_powers[0] = 1;
        for (int k = 1; k < max_alpha_index_basic; ++k) {
            dist_powers[k] = dist_powers[k - 1] * rijinv;
            for (int a = 0; a < 3; a++) {
                coords_powers(k + 1, a) = coords_powers(k, a) * nbh.pos_[jj][a]; // off set by 1 to aviod if later
            }
        }

        for (int mu = 0; mu < max_mu; ++mu) {
            val_mu[mu] = 0;
            int idxoffset =
                    (nbh.centertype * species_count + nbh.chemmap_[jj]) * radial_func_count * rb_size + mu * rb_size;
            for (int xi = 0; xi < rb_size; ++xi) {
                val_mu[mu] += radial_coeffs[idxoffset + xi] * rb_vals[xi];
            }
            if (need_derivs) {
                der_mu[mu] = 0;
                for (int xi = 0; xi < rb_size; ++xi) {
                    der_mu[mu] += radial_coeffs[idxoffset + xi] * rb_ders[xi];
                }
            }
        }

        for (int i = 0; i < alpha_index_basic_count; ++i) {
            // @OPTIMIZE: init mu-array while loading potentials, do not need offset search
            int mu = alpha_index_basic(i, 0);
            double val = val_mu[mu];
            int k = nu_arr_list[i];
            double &powk = dist_powers[k];
            val *= powk; // normalize, all tenser element is devided by rij^\nv to avoid extremely large values
            // offset by 1
            double pow0 = coords_powers(alpha_index_basic(i, 1) + 1, 0); // x
            double pow1 = coords_powers(alpha_index_basic(i, 2) + 1, 1); // y
            double pow2 = coords_powers(alpha_index_basic(i, 3) + 1, 2); // z

            double mult0 = pow0 * pow1 * pow2; // element of tensor r ^ ("x" nv)

            moment_vals[i] += val * mult0;

            if (need_derivs) {
                double der = der_mu[mu];
                der = der * powk - k * val / rij;
                mult0 *= der / rij;
                moment_jacobian(i, jj, 0) += mult0 * nbh.pos_[jj][0];
                moment_jacobian(i, jj, 1) += mult0 * nbh.pos_[jj][1];
                moment_jacobian(i, jj, 2) += mult0 * nbh.pos_[jj][2];
                // coords_powers is shifted by 3 to avoid if statement
                // note that 0 * Inf = NaN
                moment_jacobian(i, jj, 0) +=
                        alpha_index_basic(i, 1) * val *
                        coords_powers(alpha_index_basic(i, 1), 0) *
                        pow1 * pow2;
                moment_jacobian(i, jj, 1) +=
                        alpha_index_basic(i, 2) * val *
                        coords_powers(alpha_index_basic(i, 2), 1) *
                        pow0 * pow2;
                moment_jacobian(i, jj, 2) +=
                        alpha_index_basic(i, 3) * val *
                        coords_powers(alpha_index_basic(i, 3), 2) *
                        pow0 * pow1;
            }
        }
    }

    // non-elementary b_i
    for (int i = 0; i < alpha_index_times_count; ++i) {
        double val0 = moment_vals[alpha_index_times(i, 0)];
        double val1 = moment_vals[alpha_index_times(i, 1)];
        double val2 = alpha_index_times(i, 2);
        moment_vals[alpha_index_times(i, 3)] += val2 * val0 * val1;
    }

    if (need_derivs) {
        // Backpropagation step 3: multiply by the Jacobian:
        for (int i = 0; i < alpha_index_basic_count; ++i)
            for (int j = 0; j < nbh.nbhcount_; ++j)
                for (int a = 0; a < 3; a++) {
                    dloss_dsenders[i] += (*f_weight)(j, a) * moment_jacobian(i, j, a);
                }
        // db_j/dx
        for (int i = 0; i < alpha_index_times_count; ++i) {
            double val2 = alpha_index_times(i, 2);
            double val0 = moment_vals[alpha_index_times(i, 0)];
            double val1 = moment_vals[alpha_index_times(i, 1)];
            dloss_dsenders[alpha_index_times(i, 3)] +=
                    dloss_dsenders[alpha_index_times(i, 0)] * val2 * val1 +
                    dloss_dsenders[alpha_index_times(i, 1)] * val2 * val0;
        }
    }
    // species coeffs
    out_gard_accumulator[nbh.centertype] += se_weight;
    // linear coeffs
    for (int i = 0; i < alpha_scalar_moments; ++i) {
        out_gard_accumulator[i + species_count] +=
                se_weight * moment_vals[alpha_moment_mapping[i]];
        if (need_derivs)
            out_gard_accumulator[i + species_count] += dloss_dsenders[alpha_moment_mapping[i]];
    }
}

void Potentials::PotMTP::AccumulateFPGardLonly(Neighborhood &nbh, double *out_gard_accumulator, double se_weight, Eigen::Matrix<double, -1, 3, Eigen::RowMajor> *f_weight, int coffset)
{
    const bool need_derivs = (f_weight != nullptr);

    if (need_derivs) {
        moment_jacobian.resize(alpha_index_basic_count, nbh.nbhcount_, 3);
        moment_jacobian.setZero();
    }

    std::ranges::fill(moment_vals, 0);
    std::ranges::fill(rb_vals, 0);
    std::ranges::fill(val_mu, 0);
    if (need_derivs) {
        std::ranges::fill(rb_ders, 0);
        std::ranges::fill(der_mu, 0);
        std::ranges::fill(dloss_dmom, 0);
        std::ranges::fill(dloss_dsenders, 0);
        std::ranges::fill(site_energy_ders_wrt_moments_, 0);
    }
    const int& itype = nbh.centertype;
    for (int jj = 0; jj < nbh.nbhcount_; ++jj) {
        const double &rij = nbh.distances_[jj];
        const int& jtype = nbh.chemmap_[jj];
        if (rij > cutoffs(itype, jtype)) continue;
        const double rijinv = 1.0 / rij;
        if (need_derivs)
            rb_calcder(itype, jtype, rij);
        else
            rb_calcval(itype, jtype, rij);

        dist_powers[0] = 1;
        for (int k = 1; k < max_alpha_index_basic; ++k) {
            dist_powers[k] = dist_powers[k - 1] * rijinv;
            for (int a = 0; a < 3; a++) {
                coords_powers(k + 1, a) = coords_powers(k, a) * nbh.pos_[jj][a]; // off set by 1 to aviod if later
            }
        }

        for (int mu = 0; mu < max_mu; ++mu) {
            val_mu[mu] = 0;
            int idxoffset =
                    (nbh.centertype * species_count + nbh.chemmap_[jj]) * radial_func_count * rb_size + mu * rb_size;
            for (int xi = 0; xi < rb_size; ++xi) {
                val_mu[mu] += radial_coeffs[idxoffset + xi] * rb_vals[xi];
            }
            if (need_derivs) {
                der_mu[mu] = 0;
                for (int xi = 0; xi < rb_size; ++xi) {
                    der_mu[mu] += radial_coeffs[idxoffset + xi] * rb_ders[xi];
                }
            }
        }

        for (int i = 0; i < alpha_index_basic_count; ++i) {
            // @OPTIMIZE: init mu-array while loading potentials, do not need offset search
            int mu = alpha_index_basic(i, 0);
            double val = val_mu[mu];
            int k = nu_arr_list[i];
            double &powk = dist_powers[k];
            val *= powk; // normalize, all tenser element is devided by rij^\nv to avoid extremely large values
            // offset by 1
            double pow0 = coords_powers(alpha_index_basic(i, 1) + 1, 0); // x
            double pow1 = coords_powers(alpha_index_basic(i, 2) + 1, 1); // y
            double pow2 = coords_powers(alpha_index_basic(i, 3) + 1, 2); // z

            double mult0 = pow0 * pow1 * pow2; // element of tensor r ^ ("x" nv)

            moment_vals[i] += val * mult0;

            if (need_derivs) {
                double der = der_mu[mu];
                der = der * powk - k * val / rij;
                mult0 *= der / rij;
                moment_jacobian(i, jj, 0) += mult0 * nbh.pos_[jj][0];
                moment_jacobian(i, jj, 1) += mult0 * nbh.pos_[jj][1];
                moment_jacobian(i, jj, 2) += mult0 * nbh.pos_[jj][2];
                // coords_powers is shifted by 3 to avoid if statement
                // note that 0 * Inf = NaN
                moment_jacobian(i, jj, 0) +=
                        alpha_index_basic(i, 1) * val *
                        coords_powers(alpha_index_basic(i, 1), 0) *
                        pow1 * pow2;
                moment_jacobian(i, jj, 1) +=
                        alpha_index_basic(i, 2) * val *
                        coords_powers(alpha_index_basic(i, 2), 1) *
                        pow0 * pow2;
                moment_jacobian(i, jj, 2) +=
                        alpha_index_basic(i, 3) * val *
                        coords_powers(alpha_index_basic(i, 3), 2) *
                        pow0 * pow1;
            }
        }
    }

    // non-elementary b_i
    for (int i = 0; i < alpha_index_times_count; ++i) {
        double val0 = moment_vals[alpha_index_times(i, 0)];
        double val1 = moment_vals[alpha_index_times(i, 1)];
        double val2 = alpha_index_times(i, 2);
        moment_vals[alpha_index_times(i, 3)] += val2 * val0 * val1;
    }

    if (need_derivs) {
        // Backpropagation step 3: multiply by the Jacobian:
        for (int i = 0; i < alpha_index_basic_count; ++i)
            for (int j = 0; j < nbh.nbhcount_; ++j)
                for (int a = 0; a < 3; a++) {
                    dloss_dsenders[i] += (*f_weight)(j, a) * moment_jacobian(i, j, a);
                }
        // db_j/dx
        for (int i = 0; i < alpha_index_times_count; ++i) {
            double val2 = alpha_index_times(i, 2);
            double val0 = moment_vals[alpha_index_times(i, 0)];
            double val1 = moment_vals[alpha_index_times(i, 1)];
            dloss_dsenders[alpha_index_times(i, 3)] +=
                    dloss_dsenders[alpha_index_times(i, 0)] * val2 * val1 +
                    dloss_dsenders[alpha_index_times(i, 1)] * val2 * val0;
        }
    }
    for (int i = 0; i < alpha_scalar_moments; ++i) {
        out_gard_accumulator[i + coffset] +=
                se_weight * moment_vals[alpha_moment_mapping[i]];
        if (need_derivs)
            out_gard_accumulator[i + coffset] += dloss_dsenders[alpha_moment_mapping[i]];
    }
}
// Copies own independent radial bases and coefficient storage.
PotMTP::PotMTP(PotMTP &opt) {
    cutoff = opt.cutoff;
    cutoffsq = opt.cutoffsq;
    cutoffs = opt.cutoffs;
    cutoffsqs = opt.cutoffsqs;
    r_mins = opt.r_mins;
    species_count = opt.species_count;
    potential_tag = opt.potential_tag;
    reg_coeff_size = opt.reg_coeff_size;
    lin_coeff_size = opt.lin_coeff_size;
    radial_coeff_size = opt.radial_coeff_size;
    species_coeff_size = opt.species_coeff_size;
    moment_coeff_size = opt.moment_coeff_size;
    alpha_moments_count = opt.alpha_moments_count;
    alpha_index_basic_count = opt.alpha_index_basic_count;
    alpha_index_times_count = opt.alpha_index_times_count;
    alpha_scalar_moments = opt.alpha_scalar_moments;
    alpha_moment_mapping = new int[alpha_scalar_moments];
    memcpy(alpha_moment_mapping, opt.alpha_moment_mapping, sizeof(int) * alpha_scalar_moments);
    alpha_index_basic = opt.alpha_index_basic;
    alpha_index_times = opt.alpha_index_times;
    btype = opt.btype;
    rb_size = opt.rb_size;
    radial_func_count = opt.radial_func_count;
    p_radical_basis = opt.p_radical_basis ? clone_basis(*opt.p_radical_basis) : nullptr;
    for (AnyBasis *basis : opt.p_radical_basis_pair)
        p_radical_basis_pair.push_back(basis ? clone_basis(*basis) : nullptr);
    regression_coeffs = new double[reg_coeff_size];
    memcpy(regression_coeffs, opt.regression_coeffs, reg_coeff_size * sizeof(double));
    radial_coeffs = regression_coeffs;
    species_coeffs = regression_coeffs + radial_coeff_size;
    linear_coeffs = species_coeffs;
    moment_coeffs = linear_coeffs + species_count;
    nu_arr_list = new int[alpha_index_basic_count]();
    memcpy(nu_arr_list, opt.nu_arr_list, alpha_index_basic_count * sizeof(int));
    max_alpha_index_basic = opt.max_alpha_index_basic;
    max_mu = opt.max_mu;
    empty = opt.empty;
    radial_empty = opt.radial_empty;
    linear_empty = opt.linear_empty;
    hc_correction = opt.hc_correction;
    rR_correction = opt.rR_correction;
    radial_basis_split = opt.radial_basis_split;
    radial_local_cache = opt.radial_local_cache;
    full_symmetry_reuse = opt.full_symmetry_reuse;
    max_level = opt.max_level;
    lev_scalar_nodes = opt.lev_scalar_nodes;
    lev_scalar_count = opt.lev_scalar_count;
    lev_times_nodes = opt.lev_times_nodes;
    lev_times_count = opt.lev_times_count;
    radial_mode = opt.radial_mode;
    reg_L1_vector = opt.reg_L1_vector;
    reg_L2_vector = opt.reg_L2_vector;
    omp_inited = false; // Reset OMP state for new object

    coords_powers.resize(max_alpha_index_basic + 1, 3); // {rij_x^k, rij_y^k, rij_z^k}
    coords_powers(0, 0) = coords_powers(0, 1) = coords_powers(0, 2) = 0;
    coords_powers(0 + 1, 0) = coords_powers(0 + 1, 1) = coords_powers(0 + 1, 2) = 1;
    // +4 for moment_hessian
    dist_powers.resize(max_alpha_index_basic + 4); // rij^k
    moment_vals.resize(alpha_moments_count, 0);
    rb_vals.resize(rb_size, 0);
    rb_ders.resize(rb_size, 0);
    rb_derders.resize(rb_size, 0);
    val_mu.resize(max_mu, 0); // basis function value
    der_mu.resize(max_mu, 0); // basis function der wrt r
    derder_mu.resize(max_mu, 0);
    site_energy_ders_wrt_moments_.resize(alpha_moments_count);
}
// without species coeffs
void PotMTP::random_init_linear() {
    Message("Random initialize linear coeffs");
    // init radical coeffs
    std::uniform_real_distribution<double> uniform(random_lower, random_upper); // -1e-3 - 1e-3 ??
    for (int i = 0; i < lin_coeff_size; ++i) {
        linear_coeffs[i] = uniform(RNG.mtd);
    }
    #ifdef IMRMLP_MPI
    MPI_Bcast(linear_coeffs, lin_coeff_size, MPI_DOUBLE, 0, mpi.comm);
    #endif
    empty = false;
}

void PotMTP::random_init_radial() {
    Message("Random initialize radial coeffs");
    // init radical coeffs
    std::uniform_real_distribution<double> uniform(random_lower, random_upper); // -1e-3 - 1e-3 ??
    for (int i = 0; i < radial_coeff_size; ++i) {
        radial_coeffs[i] = uniform(RNG.mtd);
    }
    #ifdef IMRMLP_MPI
    MPI_Bcast(radial_coeffs, reg_coeff_size, MPI_DOUBLE, 0, mpi.comm);
    #endif
    empty = false;
}

void Potentials::PotMTP::random_init_moments() {
    Message("Random initialize moment coeffs");
    // init radical coeffs
    std::uniform_real_distribution<double> uniform(random_lower, random_upper); // -1e-3 - 1e-3 ??
    for (int i = 0; i < alpha_scalar_moments; ++i) {
        moment_coeffs[i] = uniform(RNG.mtd);
    }
    #ifdef IMRMLP_MPI
    MPI_Bcast(moment_coeffs, alpha_scalar_moments, MPI_DOUBLE, 0, mpi.comm);
    #endif
    empty = false;
}

void PotMTP::random_init() {
    Message("Random initialize");
    // init radical coeffs
    std::uniform_real_distribution<double> uniform(random_lower, random_upper); // -1e-3 - 1e-3 ??
    for (int i = 0; i < reg_coeff_size; ++i) {
        regression_coeffs[i] = uniform(RNG.mtd);
    }
    // modify radial coeffs so that it is rational
    for (int k = 0; k < radial_func_count; ++k) {
        double norm = 0;
        for (int i = 0; i < species_count; i++)
            for (int j = 0; j < species_count; j++)
                for (int l = 0; l < rb_size; l++) {
                    int idx = (i * species_count + j) * radial_func_count * rb_size + k * rb_size + l;
                    norm += regression_coeffs[idx] * regression_coeffs[idx] + 1e-10;
                }
        norm = sqrt(norm);
        for (int i = 0; i < species_count; i++) {
            for (int j = 0; j < species_count; j++) {
                for (int l = 0; l < rb_size; l++) {
                    int idx = (i * species_count + j) * radial_func_count * rb_size + k * rb_size + l;
                    regression_coeffs[idx] /= norm;
                }
            }
        }
    }
    #ifdef IMRMLP_MPI
    MPI_Bcast(regression_coeffs, reg_coeff_size, MPI_DOUBLE, 0, mpi.comm);
    #endif
    empty = false;
}

// note that radial basis is already orthogonalized
// Gram–Schmidt process
// if radial_mode is 1, skip this process
void PotMTP::Orthogonalize() {
    if (radial_mode) return;
    for (int k = 0; k < radial_func_count; ++k) {
        for (int k2 = 0; k2 < k; ++k2) {
            double norm = 0;

            for (int i = 0; i < species_count; ++i) {
                for (int j = 0; j < species_count; ++j) {
                    for (int l = 0; l < rb_size; ++l) {
                        int idx = (i * species_count + j) * radial_func_count * rb_size + k * rb_size + l;
                        norm += radial_coeffs[idx] * radial_coeffs[idx];
                    }
                }
            }

            double scal = 0;
            for (int i = 0; i < species_count; ++i) {
                for (int j = 0; j < species_count; ++j) {
                    for (int l = 0; l < rb_size; ++l) {
                        int idx1 = (i * species_count + j) * radial_func_count * rb_size + k * rb_size + l;
                        int idx2 = (i * species_count + j) * radial_func_count * rb_size + k2 * rb_size + l;
                        scal += radial_coeffs[idx1] * radial_coeffs[idx2];
                    }
                }
            }

            for (int i = 0; i < species_count; ++i) {
                for (int j = 0; j < species_count; ++j) {
                    for (int l = 0; l < rb_size; ++l) {
                        int idx1 = (i * species_count + j) * radial_func_count * rb_size + k * rb_size + l;
                        int idx2 = (i * species_count + j) * radial_func_count * rb_size + k2 * rb_size + l;
                        radial_coeffs[idx1] -= radial_coeffs[idx2] * scal / norm;
                    }
                }
            }
        }

        // normalize
        double norm = 0;

        for (int i = 0; i < species_count; i++)
            for (int j = 0; j < species_count; j++)
                for (int l = 0; l < rb_size; l++) {
                    int idx = (i * species_count + j) * radial_func_count * rb_size + k * rb_size + l;
                    norm += radial_coeffs[idx] * radial_coeffs[idx] + 1e-10;
                }

        norm = sqrt(norm);
        for (int i = 0; i < species_count; i++) {
            for (int j = 0; j < species_count; j++) {
                for (int l = 0; l < rb_size; l++) {
                    int idx = (i * species_count + j) * radial_func_count * rb_size + k * rb_size + l;
                    radial_coeffs[idx] /= norm;
                }
            }
        }
    }
}

void PotMTP::CalcLocalDescriptors(Atoms &atom, Eigen::MatrixXd &desc)
{
    omp_init();
    int natoms = atom.size();
    Neighborhoods nbhs(atom, cutoff);
    desc.resize(alpha_scalar_moments, natoms);
    #ifdef USE_OMP
    #pragma omp parallel num_threads(omp_buffers_list.size())
    {
    #endif
        int ithread = 0;
        #ifdef USE_OMP
        ithread = omp_get_thread_num();
        auto &buffer = omp_buffers_list[ithread];
        #pragma omp for schedule(static)
        #else 
        auto &buffer = omp_buffers_list[ithread];
        #endif
        for (int iat = 0; iat < natoms; ++iat) {
            auto nbh = nbhs[iat];
            std::ranges::fill(buffer.moment_vals, 0);
            std::ranges::fill(buffer.rb_vals, 0);
            std::ranges::fill(buffer.val_mu, 0);
            const int& itype = nbh.centertype;
            for (int jj = 0; jj < nbh.nbhcount_; ++jj) {
                const double &rij = nbh.distances_[jj];
                const int &jtype = nbh.chemmap_[jj];
                if (rij > cutoffs(itype, jtype)) continue;
                const double rijinv = 1.0 / rij;
                rb_calcval_omp(itype, jtype, rij, buffer);

                buffer.dist_powers[0] = 1;
                for (int k = 1; k < max_alpha_index_basic; ++k) {
                    buffer.dist_powers[k] = buffer.dist_powers[k - 1] * rijinv;
                    for (int a = 0; a < 3; a++) {
                        buffer.coords_powers(k + 1, a) = buffer.coords_powers(k, a) * nbh.pos_[jj][a];
                    }
                }

                for (int mu = 0; mu < max_mu; ++mu) {
                    buffer.val_mu[mu] = 0;
                    int idxoffset =
                            (nbh.centertype * species_count + nbh.chemmap_[jj]) * radial_func_count * rb_size + mu * rb_size;
                    for (int xi = 0; xi < rb_size; xi++) {
                        buffer.val_mu[mu] += radial_coeffs[idxoffset + xi] * buffer.rb_vals[xi];
                    }
                }

                for (int i = 0; i < alpha_index_basic_count; ++i) {
                    int mu = alpha_index_basic(i, 0);
                    double val = buffer.val_mu[mu];

                    int k = nu_arr_list[i];
                    double &powk = buffer.dist_powers[k];
                    // TODO: test weather move this out can accelearte computation by defining val_mu_nv[:] and der_mu_nv[:]
                    // since some (mu, nv) pair may not exist.
                    val *= powk; // normalize, all tenser element is devided by rij^\nv to aviod extremely large values
                    double pow0 = buffer.coords_powers(alpha_index_basic(i, 1) + 1, 0);
                    double pow1 = buffer.coords_powers(alpha_index_basic(i, 2) + 1, 1);
                    double pow2 = buffer.coords_powers(alpha_index_basic(i, 3) + 1, 2);
                    double mult0 = pow0 * pow1 * pow2; // element of tensor r ^ ("x" nv)
                    buffer.moment_vals[i] += val * mult0;
                }
            }

            // non-elementary b_i
            for (int i = 0; i < alpha_index_times_count; ++i) {
                double val0 = buffer.moment_vals[alpha_index_times(i, 0)];
                double val1 = buffer.moment_vals[alpha_index_times(i, 1)];
                int val2 = alpha_index_times(i, 2);
                buffer.moment_vals[alpha_index_times(i, 3)] += val2 * val0 * val1;
            }

            // convolving with coefficients
            // setting constant term as zero
            for (int i = 0; i < alpha_scalar_moments; ++i) {
                desc(i, iat) = buffer.moment_vals[alpha_moment_mapping[i]];
            }
        }
    #ifdef USE_OMP
    }
    #endif
}

void PotMTP::compute_descriptors(Neighborhood &nbh, std::vector<double> &basis_vals) {
    int ns = alpha_scalar_moments + species_count;
    basis_vals.resize(alpha_scalar_moments + 1, 0);

    std::ranges::fill(moment_vals, 0);
    std::ranges::fill(rb_vals, 0);
    std::ranges::fill(val_mu, 0);
    const int& itype = nbh.centertype;
    for (int jj = 0; jj < nbh.nbhcount_; ++jj) {
        const double &rij = nbh.distances_[jj];
        const int& jtype = nbh.chemmap_[jj];
        if (rij > cutoffs(itype, jtype)) continue;
        const double rijinv = 1.0 / rij;
        rb_calcval(itype, jtype, rij);

        dist_powers[0] = 1;
        for (int k = 1; k < max_alpha_index_basic; ++k) {
            dist_powers[k] = dist_powers[k-1] * rijinv;
            for (int a = 0; a < 3; a++) {
                coords_powers(k + 1, a) = coords_powers(k, a) * nbh.pos_[jj][a];
            }
        }

        for (int mu = 0; mu < max_mu; ++mu) {
            val_mu[mu] = 0;
            int idxoffset =
                    (nbh.centertype * species_count + nbh.chemmap_[jj]) * radial_func_count * rb_size + mu * rb_size;
            for (int xi = 0; xi < rb_size; xi++)
                val_mu[mu] += radial_coeffs[idxoffset + xi] * rb_vals[xi];
        }

        for (int i = 0; i < alpha_index_basic_count; ++i) {
            int mu = alpha_index_basic(i, 0);
            double val = val_mu[mu];

            int k = nu_arr_list[i];
            double &powk = dist_powers[k];
            val *= powk; // normalize, all tenser element is devided by rij^\nv to aviod extremely large values
            double pow0 = coords_powers(alpha_index_basic(i, 1) + 1, 0);
            double pow1 = coords_powers(alpha_index_basic(i, 2) + 1, 1);
            double pow2 = coords_powers(alpha_index_basic(i, 3) + 1, 2);
            moment_vals[i] += val * pow0 * pow1 * pow2;
        }
    }

    // non-elementary b_i
    for (int i = 0; i < alpha_index_times_count; ++i) {
        double val0 = moment_vals[alpha_index_times(i, 0)];
        double val1 = moment_vals[alpha_index_times(i, 1)];
        int val2 = alpha_index_times(i, 2);
        moment_vals[alpha_index_times(i, 3)] += val2 * val0 * val1;
    }

    // convolving with coefficients
    // setting constant term as zero
    basis_vals[0] = 1.0;
    for (int i = 0; i < alpha_scalar_moments; ++i) {
        basis_vals[i + 1] = moment_vals[alpha_moment_mapping[i]];
    }
}

void PotMTP::compute_local_ecmpt(Neighborhood &nbh, std::vector<double> &energy_cmpts, bool fuse) {
    int ns = alpha_scalar_moments + species_count;
    energy_cmpts.clear();
    if (fuse)
        energy_cmpts.resize(alpha_scalar_moments + 1, 0);
    else
        energy_cmpts.resize(ns, 0);

    std::vector<double> basis_vals;
    compute_descriptors(nbh, basis_vals);
    if (fuse) {
        energy_cmpts[0] = basis_vals[0] * species_coeffs[nbh.centertype];
        for (int k = species_count; k < ns; ++k) {
            int i = k - species_count + 1;
            energy_cmpts[i] = basis_vals[i] * moment_coeffs[i - 1];
        }
    } else {
        energy_cmpts[nbh.centertype] = basis_vals[0] * species_coeffs[nbh.centertype];
        for (int k = species_count; k < ns; ++k) {
            int i = k - species_count + 1;
            energy_cmpts[k] = basis_vals[i] * moment_coeffs[i - 1];
        }
    }
}

PotMTP::~PotMTP() {
    clear_loaded_model();
}

/**
 * @brief Set linear coeffs, including species coeffs and moment coeffs, according to *val
 * 
 * @param val data pointer
 * @note  
 *  For training.
 */
void PotMTP::set_linear_coeff(double *val) {
    for (int i = 0; i < lin_coeff_size; ++i) {
        regression_coeffs[radial_coeff_size + i] = val[i];
    }
}

void PotMTP::set_moment_coeff(double *val) {
    for (int i = 0; i < lin_coeff_size - species_count; ++i) {
        moment_coeffs[i] = val[i];
    }
}

void PotMTP::get_coeffs(Eigen::VectorXd &_x) const {
    _x = Eigen::Map<Eigen::VectorXd>(regression_coeffs, reg_coeff_size);
}

void Potentials::PotMTP::get_coeffs(double *_x) const
{
    for (int i = 0; i < reg_coeff_size; ++i) {
        _x[i] = regression_coeffs[i];
    }
}

void Potentials::PotMTP::model_get_coeffs(double *_x) const
{
    for (int i = 0; i < radial_coeff_size + species_count; ++i) {
        _x[i] = regression_coeffs[i];
    }
}

void PotMTP::get_coeffs(Eigen::VectorXd &_x, bool fix_radial, bool fix_linear) const
{
    if (fix_radial) {
        _x = Eigen::Map<Eigen::VectorXd>(linear_coeffs, lin_coeff_size);
    } else if (fix_linear) {
        _x = Eigen::Map<Eigen::VectorXd>(radial_coeffs, radial_coeff_size);
    } else {
        _x = Eigen::Map<Eigen::VectorXd>(regression_coeffs, reg_coeff_size);
    }
}

void PotMTP::set_coeffs(const Eigen::VectorXd &_x) {
    memcpy(regression_coeffs, _x.data(), reg_coeff_size * sizeof(double));
}

void PotMTP::set_coeffs(const Eigen::VectorXd &_x, bool fix_radial, bool fix_linear) {
    if (fix_radial) {
        memcpy(linear_coeffs, _x.data(), lin_coeff_size * sizeof(double));
    } else if (fix_linear) {
        memcpy(radial_coeffs, _x.data(), radial_coeff_size * sizeof(double));
    } else {
        memcpy(regression_coeffs, _x.data(), reg_coeff_size * sizeof(double));
    }
}

void PotMTP::set_coeffs(const double *x) {
    memcpy(regression_coeffs, x, reg_coeff_size * sizeof(double));
}

void Potentials::PotMTP::model_set_coeffs(const double *_x)
{
    for (int i = 0; i < radial_coeff_size + species_count; ++i) {
        regression_coeffs[i] = _x[i];
    }
}

void PotMTP::set_coeffs(const double *x, bool fix_radial, bool fix_linear)
{
    if (fix_radial) {
        memcpy(linear_coeffs, x, lin_coeff_size * sizeof(double));
    } else if (fix_linear) {
        memcpy(radial_coeffs, x, radial_coeff_size * sizeof(double));
    } else {
        memcpy(regression_coeffs, x, reg_coeff_size * sizeof(double));
    }
}

void Potentials::PotMTP::omp_init()
{
    require_loaded_model();
    const int requested_threads = get_omp_max_threads();
    if (!omp_inited) {
        // A reload can change the descriptor dimensions even at the same
        // thread count. Discard all old scratch when the model is invalidated.
        omp_buffers_list.clear();
        omp_inited = true;
    }
    if (omp_buffers_list.size() != static_cast<size_t>(requested_threads)) {
        omp_buffers_list.resize(requested_threads,
            ompbuffers(max_alpha_index_basic, alpha_moments_count, rb_size, max_mu));
    }
}

void PotMTP::ae_set_coeffs(const alglib::real_1d_array &_x) {
    for (int i = 0; i < reg_coeff_size; ++i)
        regression_coeffs[i] = _x[i];
}

void PotMTP::ae_set_coeffs(const alglib::real_1d_array &_x, bool fix_radial, bool fix_linear) {
    if (fix_radial) {
        for (int i = 0; i < lin_coeff_size; ++i) {
            linear_coeffs[i] = _x[i];
        }
    } else if (fix_linear) {
        for (int i = 0; i < radial_coeff_size; ++i) {
            radial_coeffs[i] = _x[i];
        }
    } else {
        for (int i = 0; i < reg_coeff_size; ++i) {
            regression_coeffs[i] = _x[i];
        }
    }
}

void PotMTP::ae_get_coeffs(alglib::real_1d_array &_x) const { 
    for (int i = 0; i < reg_coeff_size; ++i)
        _x[i] = regression_coeffs[i];
}
// Do not set fix_radial and fix_linear to true the same time
void PotMTP::ae_get_coeffs(alglib::real_1d_array &_x, bool fix_radial, bool fix_linear) const {
    if (fix_radial) {
        for (int i = 0; i < lin_coeff_size; ++i) {
            _x[i] = linear_coeffs[i];
        }
    } else if (fix_linear) {
        for (int i = 0; i < radial_coeff_size; ++i) {
            _x[i] = radial_coeffs[i];
        }
    } else {
        for (int i = 0; i < reg_coeff_size; ++i)
            _x[i] = regression_coeffs[i];
    }
}

/**
 * @brief Compute site hessian for each local environment, very expensive
 * 
 * @param nbh 
 * 
 */
void PotMTP::apply_site_hvp(Neighborhood &nbh, const SiteCoordinateMatrix &direction,
    double &energy, SiteCoordinateMatrix &gradient, SiteCoordinateMatrix &product) {
    if (!supports_site_hvp())
        AnyPotential::apply_site_hvp(nbh, direction, energy, gradient, product);
    require_loaded_model();
    if (direction.rows() != nbh.nbhcount_ || !direction.allFinite())
        GPERROR("Invalid site HVP direction: expected finite neighborhood rows");
    // Retain the input even when a caller aliases direction with an output.
    const SiteCoordinateMatrix input = direction;
    CalcSiteHessianVectorProduct(nbh, input, product);
    energy = buff_site_energy;
    gradient = buff_site_energy_ders;
}

void PotMTP::evaluate_site_hessian(Neighborhood &nbh, double &energy,
    SiteCoordinateMatrix &gradient, HessianMatrix &hessian) {
    if (!supports_site_hessian())
        AnyPotential::evaluate_site_hessian(nbh, energy, gradient, hessian);
    require_loaded_model();
    CalcSiteHessLowMemory(nbh);
    energy = buff_site_energy;
    gradient = buff_site_energy_ders;
    hessian = buff_site_hessian;
}



void PotMTP::CalcSiteHess(Neighborhood &nbh) {
    int ns = alpha_scalar_moments + species_count;
    moment_jacobian.resize(alpha_moments_count, nbh.nbhcount_, 3);
    moment_jacobian.setZero();
    // b i alpha j beta
    moment_hessian.resize(alpha_moments_count);
    for (int i = 0; i < moment_hessian.size(); ++i) {
        moment_hessian[i].resize(nbh.nbhcount_);
    }

    std::fill(site_energy_ders_wrt_moments_.begin(),
              site_energy_ders_wrt_moments_.end(), 0);
    std::ranges::fill(moment_vals, 0);
    std::ranges::fill(rb_vals, 0);
    std::ranges::fill(rb_ders, 0);
    std::fill(rb_derders.begin(), rb_derders.end(), 0);
    std::ranges::fill(val_mu, 0);
    std::ranges::fill(der_mu, 0);
    const int& itype = nbh.centertype;
    for (int jj = 0; jj < nbh.nbhcount_; ++jj) {
        const double &rij = nbh.distances_[jj];
        const int& jtype = nbh.chemmap_[jj];
        if (rij > cutoffs(itype, jtype)) continue;
        const double rijinv = 1.0 / rij;
        rb_calcderder(itype, jtype, rij);

        dist_powers[0] = 1;
        for (int k = 1; k < max_alpha_index_basic + 4; ++k) {
            dist_powers[k] = dist_powers[k - 1] * rijinv;
        }
        for (int k = 1; k < max_alpha_index_basic; ++k) {
            for (int a = 0; a < 3; a++) {
                coords_powers(k + 1, a) = coords_powers(k, a) * nbh.pos_[jj][a];
            }
        }

        for (int mu = 0; mu < max_mu; ++mu) {
            val_mu[mu] = 0;
            der_mu[mu] = 0;
            derder_mu[mu] = 0;
            int idxoffset =
                    (nbh.centertype * species_count + nbh.chemmap_[jj]) * radial_func_count * rb_size + mu * rb_size;
            for (int xi = 0; xi < rb_size; xi++) {
                val_mu[mu] += radial_coeffs[idxoffset + xi] * rb_vals[xi];
                der_mu[mu] += radial_coeffs[idxoffset + xi] * rb_ders[xi];
                derder_mu[mu] += radial_coeffs[idxoffset + xi] * rb_derders[xi];
            }
        }

        for (int i = 0; i < alpha_index_basic_count; ++i) {
            int mu = alpha_index_basic(i, 0);
            double val = val_mu[mu];
            double der = der_mu[mu];
            double derder = derder_mu[mu]; // d2f/dr2
            const auto &r = nbh.pos_[jj];
            int k = nu_arr_list[i];
            double &powk = dist_powers[k]; // pw
            double pow0 = coords_powers(alpha_index_basic(i, 1) + 1, 0);
            double pow1 = coords_powers(alpha_index_basic(i, 2) + 1, 1);
            double pow2 = coords_powers(alpha_index_basic(i, 3) + 1, 2);
            double mult0 = pow0 * pow1 * pow2; // element of tensor r ^ ("x" nv), mt
            double multa1[3] = {coords_powers(alpha_index_basic(i, 1), 0) * pow1 * pow2, 
                                pow0 * coords_powers(alpha_index_basic(i, 2), 1) * pow2,
                                pow0 * pow1 * coords_powers(alpha_index_basic(i, 3), 2)};
            double multa2[3][3];

            if (alpha_index_basic(i, 1) < 2) multa2[0][0] = 0;
            else multa2[0][0] = coords_powers(alpha_index_basic(i, 1) - 1, 0) * pow1 * pow2;
            if (alpha_index_basic(i, 2) < 2) multa2[1][1] = 0;
            else multa2[1][1] = pow0 * coords_powers(alpha_index_basic(i, 2) - 1, 1) * pow2;
            if (alpha_index_basic(i, 3) < 2) multa2[2][2] = 0;
            else multa2[2][2] = pow0 * pow1 * coords_powers(alpha_index_basic(i, 3) - 1, 2);
            // ab
            if (alpha_index_basic(i, 1) < 1 || alpha_index_basic(i, 2) < 1) multa2[0][1] = 0;
            else multa2[0][1] = coords_powers(alpha_index_basic(i, 1), 0) * coords_powers(alpha_index_basic(i, 2), 1) * pow2;
            // ac
            if (alpha_index_basic(i, 1) < 1 || alpha_index_basic(i, 3) < 1) multa2[0][2] = 0;
            else multa2[0][2] = coords_powers(alpha_index_basic(i, 1), 0) * pow1 * coords_powers(alpha_index_basic(i, 3), 2);
            // bc
            if (alpha_index_basic(i, 2) < 1 || alpha_index_basic(i, 3) < 1) multa2[1][2] = 0;
            else multa2[1][2] = pow0 * coords_powers(alpha_index_basic(i, 2), 1) * coords_powers(alpha_index_basic(i, 3), 2);
            multa2[1][0] = multa2[0][1];
            multa2[2][0] = multa2[0][2];
            multa2[2][1] = multa2[1][2];
        
            double dptmp1 = -k * dist_powers[k+2];
            double dptmp2 = k * (k+2) * dist_powers[k+4];
            double dertmp = der * dist_powers[1];
            double der2tmp = (derder - dist_powers[1] * der) * dist_powers[2];
            double ftmpr = der * dist_powers[1];
            for (int a = 0; a < 3; ++a) {
                dfdra[a] = dertmp * r[a];
                dpdra[a] = dptmp1 * r[a];
                dmdra[a] = alpha_index_basic(i, a + 1) * multa1[a];
                // a == b 
                d2fdrab[a][a] = r[a] * r[a] * der2tmp + ftmpr;
                d2mdrab[a][a] = (alpha_index_basic(i, a + 1) - 1) * alpha_index_basic(i, a + 1) * multa2[a][a];
                d2pdrab[a][a] = dptmp2 * r[a] * r[a] + dptmp1;
                for (int b = a + 1; b < 3; ++b) {
                    d2fdrab[a][b] = r[a] * r[b] * der2tmp;
                    d2fdrab[b][a] = d2fdrab[a][b];
                    d2pdrab[a][b] = dptmp2 * r[a] * r[b];
                    d2pdrab[b][a] = d2pdrab[a][b];
                    d2mdrab[a][b] = alpha_index_basic(i, b + 1) * alpha_index_basic(i, a + 1) * multa2[a][b];
                    d2mdrab[b][a] = d2mdrab[a][b];
                }
            }
            // b = f * mt * pw
            // db = df * mt * pw + f * dmt * pw + f * mt * dpw
            moment_vals[i] += val * mult0 * powk;
            for (int a = 0; a < 3; ++a) {
                moment_jacobian(i, jj, a) += dfdra[a] * mult0 * powk + val * dmdra[a] * powk + val * mult0 * dpdra[a];
                for (int b = a; b < 3; ++b) {
                    moment_hessian[i](jj, a, jj, b) += 
                        d2fdrab[a][b] * mult0 * powk + 
                        val * d2mdrab[a][b] * powk + 
                        val * mult0 * d2pdrab[a][b] +
                        dfdra[a] * (dmdra[b] * powk + mult0 * dpdra[b]) +
                        dfdra[b] * (dmdra[a] * powk + mult0 * dpdra[a]) +
                        val * (dmdra[a] * dpdra[b] + dmdra[b] * dpdra[a]);
                }
            }
        }
    }
    // fill symmetryc part of moment_hessian basic diongal terms
    for (int i = 0; i < alpha_index_basic_count; ++i) {
        for (int jj = 0; jj < nbh.nbhcount_; ++jj) {
            for (int a = 0; a < 3; ++a) {
                for (int b = 0; b < a; ++b) {
                    moment_hessian[i](jj, a, jj, b) = moment_hessian[i](jj, b, jj, a);
                }
            }
        }
    }

    // non-elementary b_i
    for (int i = 0; i < alpha_index_times_count; ++i) {
        // bk = c * bi * bj
        const int &bi = alpha_index_times(i, 0);
        const int &bj = alpha_index_times(i, 1);
        const int &bk = alpha_index_times(i, 3);
        double vbi = moment_vals[bi];
        double vbj = moment_vals[bj];
        double times = alpha_index_times(i, 2);
        moment_vals[bk] += times * vbi * vbj;
        for (int j = 0; j < nbh.nbhcount_; ++j) {
            for (int a = 0; a < 3; ++a) {
                moment_jacobian(bk, j, a) +=
                    times * (moment_jacobian(bi, j, a) * vbj + vbi * moment_jacobian(bj, j, a));
            }
        }
    }
    // moment_hessians, requires moment_jacobian calculated first
    for (int i = 0; i < alpha_index_times_count; ++i) {
        const int &bi = alpha_index_times(i, 0);
        const int &bj = alpha_index_times(i, 1);
        const int &bk = alpha_index_times(i, 3);
        double vbi = moment_vals[bi];
        double vbj = moment_vals[bj];
        double times = alpha_index_times(i, 2);
        for (int a = 0; a < 3; ++a) {
            for (int b = 0; b < 3; ++b) {
                for (int ati = 0; ati < nbh.nbhcount_; ++ati) {
                    for (int atj = ati; atj < nbh.nbhcount_; ++atj) {
                        moment_hessian[bk](ati, a, atj, b) += 
                            times * (
                                moment_hessian[bi](ati, a, atj, b) * vbj + 
                                moment_jacobian(bi, ati, a) * moment_jacobian(bj, atj, b) +
                                moment_jacobian(bi, atj, b) * moment_jacobian(bj, ati, a) +
                                vbi * moment_hessian[bj](ati, a, atj, b));
                    }
                }
            }
        }
    }
    // fill symmetric part of moment_hessian for non-basic tensors
    for (int i = alpha_index_basic_count; i < alpha_moments_count; ++i) {
        for (int a = 0; a < 3; ++a) {
            for (int b = 0; b < 3; ++b) {
                for (int ati = 0; ati < nbh.nbhcount_; ++ati) {
                    for (int atj = 0; atj < ati; ++atj) {
                        moment_hessian[i](ati, a, atj, b) = moment_hessian[i](atj, b, ati, a);
                    }
                }
            }
        }
    }


    // finish
    buff_site_energy = species_coeffs[nbh.centertype];
    for (int i = 0; i < alpha_scalar_moments; ++i) {
        buff_site_energy += moment_coeffs[i] * moment_vals[alpha_moment_mapping[i]];
    }
    buff_site_energy_ders.resize(nbh.nbhcount_, 3);
    buff_site_energy_ders.setZero();
    for (int a = 0; a < 3; ++a) {
        for (int j = 0; j < nbh.nbhcount_; ++j) {
            for (int i = 0; i < alpha_scalar_moments; ++i) {
                buff_site_energy_ders(j, a) += moment_coeffs[i] * moment_jacobian(alpha_moment_mapping[i], j, a);
            }
        }
    }
    // compute site hessian matrix
    buff_site_hessian.resize(nbh.nbhcount_);
    buff_site_hessian.setzero();
    for (int a = 0; a < 3; ++a) {
        for (int b = 0; b < 3; ++b) {
            for (int i = 0; i < alpha_scalar_moments; ++i) {
                for (int ai = 0; ai < nbh.nbhcount_; ++ai) {
                    for (int aj = ai; aj < nbh.nbhcount_; ++aj) {
                        buff_site_hessian(ai, a, aj, b) += 
                            moment_coeffs[i] * moment_hessian[alpha_moment_mapping[i]](ai, a, aj, b);
                    }
                }
            }
        }
    }
    // fill symmetric part of buff_site_hessian
    for (int a = 0; a < 3; ++a) {
        for (int b = 0; b < 3; ++b) {
            for (int ai = 0; ai < nbh.nbhcount_; ++ai) {
                for (int aj = 0; aj < ai; ++aj) {
                    buff_site_hessian(ai, a, aj, b) = buff_site_hessian(aj, b, ai, a);
                }
            }
        }
    }
}

void PotMTP::CalcSiteHessOMP(Neighborhood &nbh, ompbuffers &buffer) {
    int ns = alpha_scalar_moments + species_count;
    buffer.moment_jacobian.resize(alpha_moments_count, nbh.nbhcount_, 3);
    buffer.moment_jacobian.setZero();
    // b i alpha j beta
    buffer.moment_hessian.resize(alpha_moments_count);
    for (int i = 0; i < buffer.moment_hessian.size(); ++i) {
        buffer.moment_hessian[i].resize(nbh.nbhcount_);
    }

    std::fill(buffer.site_energy_ders_wrt_moments_.begin(),
              buffer.site_energy_ders_wrt_moments_.end(), 0);
    std::ranges::fill(buffer.moment_vals, 0);
    std::ranges::fill(buffer.rb_vals, 0);
    std::ranges::fill(buffer.rb_ders, 0);
    std::fill(buffer.rb_derders.begin(), buffer.rb_derders.end(), 0);
    std::ranges::fill(buffer.val_mu, 0);
    std::ranges::fill(buffer.der_mu, 0);
    const int& itype = nbh.centertype;
    for (int jj = 0; jj < nbh.nbhcount_; ++jj) {
        const double &rij = nbh.distances_[jj];
        const int& jtype = nbh.chemmap_[jj];
        if (rij > cutoffs(itype, jtype)) continue;
        const double rijinv = 1.0 / rij;
        rb_calcderder_omp(itype, jtype, rij, buffer);

        buffer.dist_powers[0] = 1;
        for (int k = 1; k < max_alpha_index_basic + 4; ++k) {
            buffer.dist_powers[k] = buffer.dist_powers[k - 1] * rijinv;
        }
        for (int k = 1; k < max_alpha_index_basic; ++k) {
            for (int a = 0; a < 3; a++) {
                buffer.coords_powers(k + 1, a) = buffer.coords_powers(k, a) * nbh.pos_[jj][a];
            }
        }

        for (int mu = 0; mu < max_mu; ++mu) {
            buffer.val_mu[mu] = 0;
            buffer.der_mu[mu] = 0;
            buffer.derder_mu[mu] = 0;
            int idxoffset =
                    (nbh.centertype * species_count + nbh.chemmap_[jj]) * radial_func_count * rb_size + mu * rb_size;
            for (int xi = 0; xi < rb_size; xi++) {
                buffer.val_mu[mu] += radial_coeffs[idxoffset + xi] * buffer.rb_vals[xi];
                buffer.der_mu[mu] += radial_coeffs[idxoffset + xi] * buffer.rb_ders[xi];
                buffer.derder_mu[mu] += radial_coeffs[idxoffset + xi] * buffer.rb_derders[xi];
            }
        }

        for (int i = 0; i < alpha_index_basic_count; ++i) {
            int mu = alpha_index_basic(i, 0);
            double val = buffer.val_mu[mu];
            double der = buffer.der_mu[mu];
            double derder = buffer.derder_mu[mu]; // d2f/dr2
            const auto &r = nbh.pos_[jj];
            int k = nu_arr_list[i];
            double &powk = buffer.dist_powers[k]; // pw
            double pow0 = buffer.coords_powers(alpha_index_basic(i, 1) + 1, 0);
            double pow1 = buffer.coords_powers(alpha_index_basic(i, 2) + 1, 1);
            double pow2 = buffer.coords_powers(alpha_index_basic(i, 3) + 1, 2);
            double mult0 = pow0 * pow1 * pow2; // element of tensor r ^ ("x" nv), mt
            double multa1[3] = {buffer.coords_powers(alpha_index_basic(i, 1), 0) * pow1 * pow2, 
                                pow0 * buffer.coords_powers(alpha_index_basic(i, 2), 1) * pow2,
                                pow0 * pow1 * buffer.coords_powers(alpha_index_basic(i, 3), 2)};
            double multa2[3][3];

            if (alpha_index_basic(i, 1) < 2) multa2[0][0] = 0;
            else multa2[0][0] = buffer.coords_powers(alpha_index_basic(i, 1) - 1, 0) * pow1 * pow2;
            if (alpha_index_basic(i, 2) < 2) multa2[1][1] = 0;
            else multa2[1][1] = pow0 * buffer.coords_powers(alpha_index_basic(i, 2) - 1, 1) * pow2;
            if (alpha_index_basic(i, 3) < 2) multa2[2][2] = 0;
            else multa2[2][2] = pow0 * pow1 * buffer.coords_powers(alpha_index_basic(i, 3) - 1, 2);
            // ab
            if (alpha_index_basic(i, 1) < 1 || alpha_index_basic(i, 2) < 1) multa2[0][1] = 0;
            else multa2[0][1] = buffer.coords_powers(alpha_index_basic(i, 1), 0) * buffer.coords_powers(alpha_index_basic(i, 2), 1) * pow2;
            // ac
            if (alpha_index_basic(i, 1) < 1 || alpha_index_basic(i, 3) < 1) multa2[0][2] = 0;
            else multa2[0][2] = buffer.coords_powers(alpha_index_basic(i, 1), 0) * pow1 * buffer.coords_powers(alpha_index_basic(i, 3), 2);
            // bc
            if (alpha_index_basic(i, 2) < 1 || alpha_index_basic(i, 3) < 1) multa2[1][2] = 0;
            else multa2[1][2] = pow0 * buffer.coords_powers(alpha_index_basic(i, 2), 1) * buffer.coords_powers(alpha_index_basic(i, 3), 2);
            multa2[1][0] = multa2[0][1];
            multa2[2][0] = multa2[0][2];
            multa2[2][1] = multa2[1][2];
        
            double dptmp1 = -k * buffer.dist_powers[k+2];
            double dptmp2 = k * (k+2) * buffer.dist_powers[k+4];
            double dertmp = der * buffer.dist_powers[1];
            double der2tmp = (derder - buffer.dist_powers[1] * der) * buffer.dist_powers[2];
            double ftmpr = der * buffer.dist_powers[1];
            for (int a = 0; a < 3; ++a) {
                buffer.dfdra[a] = dertmp * r[a];
                buffer.dpdra[a] = dptmp1 * r[a];
                buffer.dmdra[a] = alpha_index_basic(i, a + 1) * multa1[a];
                // a == b 
                buffer.d2fdrab[a][a] = r[a] * r[a] * der2tmp + ftmpr;
                buffer.d2mdrab[a][a] = (alpha_index_basic(i, a + 1) - 1) * alpha_index_basic(i, a + 1) * multa2[a][a];
                buffer.d2pdrab[a][a] = dptmp2 * r[a] * r[a] + dptmp1;
                for (int b = a + 1; b < 3; ++b) {
                    buffer.d2fdrab[a][b] = r[a] * r[b] * der2tmp;
                    buffer.d2fdrab[b][a] = buffer.d2fdrab[a][b];
                    buffer.d2pdrab[a][b] = dptmp2 * r[a] * r[b];
                    buffer.d2pdrab[b][a] = buffer.d2pdrab[a][b];
                    buffer.d2mdrab[a][b] = alpha_index_basic(i, b + 1) * alpha_index_basic(i, a + 1) * multa2[a][b];
                    buffer.d2mdrab[b][a] = buffer.d2mdrab[a][b];
                }
            }
            // b = f * mt * pw
            // db = df * mt * pw + f * dmt * pw + f * mt * dpw
            buffer.moment_vals[i] += val * mult0 * powk;
            for (int a = 0; a < 3; ++a) {
                buffer.moment_jacobian(i, jj, a) += buffer.dfdra[a] * mult0 * powk + val * buffer.dmdra[a] * powk + val * mult0 * buffer.dpdra[a];
                for (int b = a; b < 3; ++b) {
                    buffer.moment_hessian[i](jj, a, jj, b) += 
                        buffer.d2fdrab[a][b] * mult0 * powk + 
                        val * buffer.d2mdrab[a][b] * powk + 
                        val * mult0 * buffer.d2pdrab[a][b] +
                        buffer.dfdra[a] * (buffer.dmdra[b] * powk + mult0 * buffer.dpdra[b]) +
                        buffer.dfdra[b] * (buffer.dmdra[a] * powk + mult0 * buffer.dpdra[a]) +
                        val * (buffer.dmdra[a] * buffer.dpdra[b] + buffer.dmdra[b] * buffer.dpdra[a]);
                }
            }
        }
    }
    // fill symmetryc part of buffer.moment_hessian basic diongal terms
    for (int i = 0; i < alpha_index_basic_count; ++i) {
        for (int jj = 0; jj < nbh.nbhcount_; ++jj) {
            for (int a = 0; a < 3; ++a) {
                for (int b = 0; b < a; ++b) {
                    buffer.moment_hessian[i](jj, a, jj, b) = buffer.moment_hessian[i](jj, b, jj, a);
                }
            }
        }
    }

    // non-elementary b_i
    for (int i = 0; i < alpha_index_times_count; ++i) {
        // bk = c * bi * bj
        const int &bi = alpha_index_times(i, 0);
        const int &bj = alpha_index_times(i, 1);
        const int &bk = alpha_index_times(i, 3);
        double vbi = buffer.moment_vals[bi];
        double vbj = buffer.moment_vals[bj];
        double times = alpha_index_times(i, 2);
        buffer.moment_vals[bk] += times * vbi * vbj;
        for (int j = 0; j < nbh.nbhcount_; ++j) {
            for (int a = 0; a < 3; ++a) {
                buffer.moment_jacobian(bk, j, a) +=
                    times * (buffer.moment_jacobian(bi, j, a) * vbj + vbi * buffer.moment_jacobian(bj, j, a));
            }
        }
    }
    // buffer.moment_hessians, requires buffer.moment_jacobian calculated first
    for (int i = 0; i < alpha_index_times_count; ++i) {
        const int &bi = alpha_index_times(i, 0);
        const int &bj = alpha_index_times(i, 1);
        const int &bk = alpha_index_times(i, 3);
        double vbi = buffer.moment_vals[bi];
        double vbj = buffer.moment_vals[bj];
        double times = alpha_index_times(i, 2);
        for (int a = 0; a < 3; ++a) {
            for (int b = 0; b < 3; ++b) {
                for (int ati = 0; ati < nbh.nbhcount_; ++ati) {
                    for (int atj = ati; atj < nbh.nbhcount_; ++atj) {
                        buffer.moment_hessian[bk](ati, a, atj, b) += 
                            times * (
                                buffer.moment_hessian[bi](ati, a, atj, b) * vbj + 
                                buffer.moment_jacobian(bi, ati, a) * buffer.moment_jacobian(bj, atj, b) +
                                buffer.moment_jacobian(bi, atj, b) * buffer.moment_jacobian(bj, ati, a) +
                                vbi * buffer.moment_hessian[bj](ati, a, atj, b));
                    }
                }
            }
        }
    }
    // fill symmetric part of buffer.moment_hessian for non-basic tensors
    for (int i = alpha_index_basic_count; i < alpha_moments_count; ++i) {
        for (int a = 0; a < 3; ++a) {
            for (int b = 0; b < 3; ++b) {
                for (int ati = 0; ati < nbh.nbhcount_; ++ati) {
                    for (int atj = 0; atj < ati; ++atj) {
                        buffer.moment_hessian[i](ati, a, atj, b) = buffer.moment_hessian[i](atj, b, ati, a);
                    }
                }
            }
        }
    }

    buffer.buff_site_energy = species_coeffs[nbh.centertype];
    for (int i = 0; i < alpha_scalar_moments; ++i) {
        buffer.buff_site_energy += moment_coeffs[i] * buffer.moment_vals[alpha_moment_mapping[i]];
    }
    buffer.buff_site_energy_ders.resize(nbh.nbhcount_, 3);
    buffer.buff_site_energy_ders.setZero();
    for (int a = 0; a < 3; ++a) {
        for (int j = 0; j < nbh.nbhcount_; ++j) {
            for (int i = 0; i < alpha_scalar_moments; ++i) {
                buffer.buff_site_energy_ders(j, a) += moment_coeffs[i] * buffer.moment_jacobian(alpha_moment_mapping[i], j, a);
            }
        }
    }
    // compute site hessian matrix
    buffer.buff_site_hessian.resize(nbh.nbhcount_);
    buffer.buff_site_hessian.setzero();
    for (int a = 0; a < 3; ++a) {
        for (int b = 0; b < 3; ++b) {
            for (int i = 0; i < alpha_scalar_moments; ++i) {
                for (int ai = 0; ai < nbh.nbhcount_; ++ai) {
                    for (int aj = ai; aj < nbh.nbhcount_; ++aj) {
                        buffer.buff_site_hessian(ai, a, aj, b) += 
                            moment_coeffs[i] * buffer.moment_hessian[alpha_moment_mapping[i]](ai, a, aj, b);
                    }
                }
            }
        }
    }
    // fill symmetric part of buff_site_hessian
    for (int a = 0; a < 3; ++a) {
        for (int b = 0; b < 3; ++b) {
            for (int ai = 0; ai < nbh.nbhcount_; ++ai) {
                for (int aj = 0; aj < ai; ++aj) {
                    buffer.buff_site_hessian(ai, a, aj, b) = buffer.buff_site_hessian(aj, b, ai, a);
                }
            }
        }
    } 

}

void PotMTP::CalcHessianVectorCmpt(Atoms &atom, AnyNeighborhoods &nbhs,
    const Eigen::VectorXd &direction, Eigen::MatrixXd &components) {
    if (direction.size() != 3 * atom.size() || !direction.allFinite() ||
        nbhs.size() != atom.size()) GPERROR("Invalid basis HVP direction or neighborhoods");
    omp_init();
    auto &buffer = omp_buffers_list[0];
    components.setZero(direction.size(), lin_coeff_size);
    for (int center = 0; center < nbhs.size(); ++center) {
        auto &nbh = nbhs[center];
        auto &v = buffer.buff_site_hvp_direction;
        v.resize(nbh.nbhcount_, 3);
        for (int j = 0; j < nbh.nbhcount_; ++j)
            v.row(j) = (direction.segment<3>(3 * nbh.indices_[j]) -
                       direction.segment<3>(3 * center)).transpose();
        // Populate values, Jacobians and diagonal basic-moment curvature
        // through the matrix-free site path; its model contraction is unused.
        CalcSiteHessianVectorProductOMP(nbh, buffer, 0, v, buffer.buff_site_hvp);
        const int ndof = 3 * nbh.nbhcount_;
        using Rows = Eigen::Matrix<double, -1, -1, Eigen::RowMajor>;
        Eigen::Map<const Rows> jacobian(buffer.moment_jacobian.data(), alpha_moments_count, ndof);
        Eigen::Map<const Eigen::VectorXd> flat_v(v.data(), ndof);
        const Eigen::VectorXd directional = jacobian * flat_v;
        Rows hv = Rows::Zero(alpha_moments_count, ndof);
        for (int i = 0; i < alpha_index_basic_count; ++i)
            for (int j = 0; j < nbh.nbhcount_; ++j)
                hv.row(i).segment<3>(3 * j) =
                    (buffer.basic_moment_hessian_diag[size_t(i) * nbh.nbhcount_ + j]
                     * v.row(j).transpose()).transpose();
        for (int rule = 0; rule < alpha_index_times_count; ++rule) {
            const int i = alpha_index_times(rule, 0), j = alpha_index_times(rule, 1);
            const int k = alpha_index_times(rule, 3);
            const double times = alpha_index_times(rule, 2);
            hv.row(k) += times * (buffer.moment_vals[j] * hv.row(i) +
                buffer.moment_vals[i] * hv.row(j) + directional[i] * jacobian.row(j) +
                directional[j] * jacobian.row(i));
        }
        for (int scalar = 0; scalar < alpha_scalar_moments; ++scalar)
            for (int j = 0; j < nbh.nbhcount_; ++j) {
                const Eigen::Vector3d value = hv.row(alpha_moment_mapping[scalar]).segment<3>(3 * j);
                components.col(species_count + scalar).segment<3>(3 * nbh.indices_[j]) += value;
                components.col(species_count + scalar).segment<3>(3 * center) -= value;
            }
    }
}


void PotMTP::getEFSHessian(Atoms &atom, HessianMatrix &hessian) {
    Neighborhoods nbhs(atom, cutoff);
    getEFSHessian(atom, nbhs, hessian);
}

void PotMTP::applyHessianVectorProduct(
    Atoms &atom, AnyNeighborhoods &nbhs,
    const Eigen::VectorXd &direction, Eigen::VectorXd &product) {
    if (direction.size() != 3 * atom.size())
        GPERROR("Hessian-vector direction has the wrong size");
    if (nbhs.size() != atom.size())
        GPERROR("Hessian-vector neighborhoods have the wrong size");
    omp_init();
    Eigen::VectorXd result = Eigen::VectorXd::Zero(direction.size());

#ifdef USE_OMP
    const int active_threads = std::max(
        1, std::min(static_cast<int>(omp_buffers_list.size()), static_cast<int>(nbhs.size())));
#pragma omp parallel num_threads(active_threads)
#endif
    {
#ifdef USE_OMP
        const int ithread = omp_get_thread_num();
#else
        const int ithread = 0;
#endif
        ompbuffers &buffer = omp_buffers_list[ithread];
#ifdef USE_OMP
// Round-robin sites let later threads compute while the next site is
// accumulated in order. Contiguous chunks stall at their first ordered
// region and serialize the expensive site calculations. Keep the original
// accumulation order without allocating a dense Hessian per thread.
#pragma omp for ordered schedule(static, 1)
#endif
        for (int center = 0; center < nbhs.size(); ++center) {
            Neighborhood &nbh = nbhs[center];
            auto &local_direction = buffer.buff_site_hvp_direction;
            auto &local_product = buffer.buff_site_hvp;
            local_direction.resize(nbh.nbhcount_, 3);
            for (int j = 0; j < nbh.nbhcount_; ++j)
                local_direction.row(j) =
                    (direction.segment<3>(3 * nbh.indices_[j]) -
                     direction.segment<3>(3 * center)).transpose();
            CalcSiteHessianVectorProductOMP(
                nbh, buffer, ithread, local_direction, local_product);
#ifdef USE_OMP
#pragma omp ordered
#endif
            {
                for (int j = 0; j < nbh.nbhcount_; ++j) {
                    const Eigen::Vector3d contribution = local_product.row(j);
                    result.segment<3>(3 * nbh.indices_[j]) += contribution;
                    result.segment<3>(3 * center) -= contribution;
                }
            }
        }
    }
    product = std::move(result);
}

void PotMTP::getEFSHessian(Atoms &atom, AnyNeighborhoods &nbhs, HessianMatrix &hessian) {
    if (!add_efs) atom.resetEFS(); else atom.setEFSflag();
    hessian.resize(atom.size());

    for (int ii = 0; ii < nbhs.size(); ii++) {
        std::cout << "Currently on atom " << ii << " of " << nbhs.size() << " in total" << std::endl;
        Neighborhood &nbh = nbhs[ii];
        CalcSiteHessLowMemory(nbh);
        // compute energy
        atom.site_energies_[ii] += buff_site_energy;
        atom.energy_ += buff_site_energy;
        // compute force and stress 
        for (int i = 0; i < nbh.nbhcount_; ++i) {
            atom.forces_[ii](0) += buff_site_energy_ders(i, 0);
            atom.forces_[ii](1) += buff_site_energy_ders(i, 1);
            atom.forces_[ii](2) += buff_site_energy_ders(i, 2);
            atom.forces_[nbh.indices_[i]](0) -= buff_site_energy_ders(i, 0);
            atom.forces_[nbh.indices_[i]](1) -= buff_site_energy_ders(i, 1);
            atom.forces_[nbh.indices_[i]](2) -= buff_site_energy_ders(i, 2);

            for (int a = 0; a < 3; a++) {
                for (int b = 0; b < 3; b++) {
                    atom.stress_(a, b) -= buff_site_energy_ders(i, a) * nbh.pos_[i](b);
                }
            }
        }
        // compute hessian matrix
        /*
            1. i != j
            d2E/dxidxj = \sum_{m!=i,m!=j} d2Vm/d(xi-xm)d(xj-xm) - 
                         \sum_{n} d2Vi/d(xn-xi)d(xj-xi) -
                         \sum_{n} d2Vj/d(xn-xj)d(xi-xj)
            2. i == j
            d2E/dxidxi = \sum_{m!=i} d2Vm/d(xi-xm)2 +
                         \sum_{n} d2Vi/d(xn-xi)2
        */
        // first term, \sum_{m!=i,m!=j} d2Vm/d(xi-xm)d(xj-xm)
        for (int ai = 0; ai < nbh.nbhcount_; ++ai) {
            int ati = nbh.indices_[ai];
            for (int aj = 0; aj < nbh.nbhcount_; ++aj) {
                int atj = nbh.indices_[aj];
                for (int a = 0; a < 3; ++a) {
                    for (int b = 0; b < 3; ++b) {
                        hessian(ati, a, atj, b) += buff_site_hessian(ai, a, aj, b);
                    }
                }
            }
        }
        // second term, i != j, -\sum_{n} d2Vi/d(xn-xi)d(xj-xi) -\sum_{n} d2Vj/d(xi-xj)d(xn-xj)
        for (int jj = 0; jj < nbh.nbhcount_; ++jj) {
            int atj = nbh.indices_[jj];
            for (int a = 0; a < 3; ++a) {
                for (int b = 0; b < 3; ++b) {
                    for (int kk = 0; kk < nbh.nbhcount_; ++kk) {
                        hessian(ii, a, atj, b) -= buff_site_hessian(kk, a, jj, b);
                        hessian(atj, a, ii, b) -= buff_site_hessian(jj, a, kk, b);
                    }
                }
            }
        }
        // second term, i == j, \sum_{m,n} d2Vi/d(xn-xi)d(xm-xi)
        for (int jj = 0; jj < nbh.nbhcount_; ++jj) {
            for (int kk = 0; kk < nbh.nbhcount_; ++kk) {
                for (int a = 0; a < 3; ++a) {
                    for (int b = 0; b < 3; ++b) {
                        hessian(ii, a , ii, b) += buff_site_hessian(jj, a, kk, b);
                    }
                }
            }
        }
    }
}

void Potentials::PotMTP::getEFSHessianCell(Atoms &atom, HessianMatrix &hessian, Eigen::MatrixXd &cellhess, Eigen::MatrixXd &lrhess) {
    Neighborhoods nbhs(atom, cutoff);
    getEFSHessianCell(atom, nbhs, hessian, cellhess, lrhess);
}

void Potentials::PotMTP::getEFSHessianCell(Atoms &atom, AnyNeighborhoods &nbhs, HessianMatrix &hessian, Eigen::MatrixXd &cellhess, Eigen::MatrixXd &lrhess)
{
    if (!add_efs) atom.resetEFS(); else atom.setEFSflag();
    auto rcell = atom.cell_.inverse();
    hessian.resize(atom.size());
    cellhess.resize(9, 9);
    cellhess.setZero();
    lrhess.resize(9, 3*atom.size());
    lrhess.setZero();

    for (int ii = 0; ii < nbhs.size(); ii++) {
        std::cout << "Currently on atom " << ii << " of " << nbhs.size() << " in total" << std::endl;
        Neighborhood &nbh = nbhs[ii];
        buff_site_energy = 0;
        CalcSiteHess(nbh);
        // compute energy
        atom.site_energies_[ii] += buff_site_energy;
        atom.energy_ += buff_site_energy;
        // compute force and stress 
        for (int i = 0; i < nbh.nbhcount_; ++i) {
            /**
            dE/dxj = \sum_{i!=j}dVi/d(xj-xi) - \sum_{i}dVj/d(xi-xj)
            F = -dE/dxj
            **/
            // later term
            atom.forces_[ii](0) += buff_site_energy_ders(i, 0);
            atom.forces_[ii](1) += buff_site_energy_ders(i, 1);
            atom.forces_[ii](2) += buff_site_energy_ders(i, 2);
            // former term
            atom.forces_[nbh.indices_[i]](0) -= buff_site_energy_ders(i, 0);
            atom.forces_[nbh.indices_[i]](1) -= buff_site_energy_ders(i, 1);
            atom.forces_[nbh.indices_[i]](2) -= buff_site_energy_ders(i, 2);

            for (int a = 0; a < 3; a++) {
                for (int b = 0; b < 3; b++) {
                    atom.stress_(a, b) -= buff_site_energy_ders(i, a) * nbh.pos_[i](b);
                }
            }
            // 1st-order correction term applied on cellhess
            // d2r/dLdL
            // when do_realx = False, with this correction, equal to scale_atoms = True
            if (hc_correction) {
                Eigen::RowVector3d ri = nbh.pos_[i] * rcell;
                for (int a = 0; a < 3; ++a) {
                    for (int b = 0; b < 3; ++b) {
                        for (int l = 0; l < 3; ++l) {
                            for (int p = 0; p < 3; ++p) {
                                cellhess(3*a+b, 3*l+p) += 
                                    buff_site_energy_ders(i, b) * ri(l) * rcell(p, a) +
                                    buff_site_energy_ders(i, p) * ri(a) * rcell(b, l);
                            }
                        }
                    }
                }
            }
            // assume dr/dR != 0, but what does this mean???
            int ati = ii;
            int atj = nbh.indices_[i];
            if (rR_correction) {
                for (int a = 0; a < 3; ++a) {
                    for (int b = 0; b < 3; ++b) {
                        lrhess(3*a+b, 3*ati+a) -= 
                            rcell(a, a) * buff_site_energy_ders(i, b);
                        lrhess(3*a+b, 3*atj+a) += 
                            rcell(a, a) * buff_site_energy_ders(i, b);
                    }
                }
            }
        }
        // compute hessian matrix
        /*
            1. i != j
            d2E/dxidxj = \sum_{m!=i,m!=j} d2Vm/d(xi-xm)d(xj-xm) - 
                         \sum_{n} d2Vi/d(xn-xi)d(xj-xi) -
                         \sum_{n} d2Vj/d(xn-xj)d(xi-xj)
            2. i == j
            d2E/dxidxi = \sum_{m!=i} d2Vm/d(xi-xm)2 +
                         \sum_{n} d2Vi/d(xn-xi)2
        */
        // first term, \sum_{m!=i,m!=j} d2Vm/d(xi-xm)d(xj-xm)
        Eigen::RowVector3d ri, rj;
        for (int ai = 0; ai < nbh.nbhcount_; ++ai) {
            int ati = nbh.indices_[ai];
            ri = nbh.pos_[ai] * rcell;
            for (int aj = 0; aj < nbh.nbhcount_; ++aj) {
                int atj = nbh.indices_[aj];
                rj = nbh.pos_[aj] * rcell;
                for (int a = 0; a < 3; ++a) {
                    for (int b = 0; b < 3; ++b) {
                        hessian(ati, a, atj, b) += buff_site_hessian(ai, a, aj, b);
                        for (int l = 0; l < 3; ++l) {
                            for (int p = 0; p < 3; ++p) {
                                cellhess(3*a+b, 3*l+p) += buff_site_hessian(ai, b, aj, p) * ri(a) * rj(l);
                            }
                        }
                        for (int g = 0; g < 3; ++g) {
                            lrhess(3*a+b, 3*ii+g) -= ri(a) * buff_site_hessian(ai, b, aj, g);
                            lrhess(3*a+b, 3*atj+g) += ri(a) * buff_site_hessian(ai, b, aj, g);
                        }
                    }
                }
            }
        }
        // second term, i != j, -\sum_{n} d2Vi/d(xn-xi)d(xj-xi) -\sum_{n} d2Vj/d(xi-xj)d(xn-xj)
        for (int jj = 0; jj < nbh.nbhcount_; ++jj) {
            int atj = nbh.indices_[jj];
            for (int a = 0; a < 3; ++a) {
                for (int b = 0; b < 3; ++b) {
                    for (int kk = 0; kk < nbh.nbhcount_; ++kk) {
                        hessian(ii, a, atj, b) -= buff_site_hessian(kk, a, jj, b);
                        hessian(atj, a, ii, b) -= buff_site_hessian(jj, a, kk, b);
                    }
                }
            }
        }
        // second term, i == j, \sum_{m,n} d2Vi/d(xn-xi)d(xm-xi)
        for (int jj = 0; jj < nbh.nbhcount_; ++jj) {
            for (int kk = 0; kk < nbh.nbhcount_; ++kk) {
                for (int a = 0; a < 3; ++a) {
                    for (int b = 0; b < 3; ++b) {
                        hessian(ii, a , ii, b) += buff_site_hessian(jj, a, kk, b);
                    }
                }
            }
        }
    }
}

void Potentials::PotMTP::getEFSHessianCellSymm(Atoms& atom, HessianMatrix& hessian, Eigen::MatrixXd& cellhess, Eigen::MatrixXd& lrhess, double symprec)
{
    if (!add_efs) atom.resetEFS(); else atom.setEFSflag();
    auto rcell = atom.cell_.inverse();
    Neighborhoods nbhs(atom, cutoff);
    hessian.resize(atom.size());
    cellhess.resize(9, 9);
    cellhess.setZero();
    lrhess.resize(9, 3*atom.size());
    lrhess.setZero();
    // 1. find permitive cell and mapping relations, only translation symmetry
    auto dataset = Symmetry3D::get_sym_dataset(atom, symprec);
    auto &mapping_to_primitive = dataset.mapping_to_primitive;
    std::set<int> nonequat;
    std::vector<int> nonequatid;
    std::vector<int> nonequind;
    for (int i = 0; i < mapping_to_primitive.size(); ++i) {
        int primid = mapping_to_primitive[i];
        if (nonequat.count(primid) == 0) {
            nonequind.push_back(i);
            nonequat.insert(primid);
            nonequatid.push_back(primid);
        }
    }
    // 2. build neighbor list
    std::vector<std::pair<int, Eigen::RowVector3d>> sortind;
    for (int i = 0; i < nbhs.size(); ++i) {
        auto &nbh = nbhs[i];
        // sort
        sortind.resize(nbh.nbhcount_);
        for (int j = 0; j < nbh.nbhcount_; ++j) {
            sortind[j].first = j;
            sortind[j].second = nbh.pos_[j];
        }
        std::ranges::sort(sortind, [&symprec](const std::pair<int, Eigen::RowVector3d> &a, const std::pair<int, Eigen::RowVector3d> &b) {
            if (a.second(0) < b.second(0) - symprec) return true;
            if (a.second(0) > b.second(0) + symprec) return false;
            
            if (a.second(1) < b.second(1) - symprec) return true;
            if (a.second(1) > b.second(1) + symprec) return false;
            
            if (a.second(2) < b.second(2) - symprec) return true;
            if (a.second(2) > b.second(2) + symprec) return false;

            return false;
        });
        // reorder
        auto nbhbck = nbh;
        for (int j = 0; j < nbh.nbhcount_; ++j) {
            nbh.chemmap_[j] = nbhbck.chemmap_[sortind[j].first];
            nbh.distances_[j] = nbhbck.distances_[sortind[j].first];
            nbh.indices_[j] = nbhbck.indices_[sortind[j].first];
            nbh.pos_[j] = nbhbck.pos_[sortind[j].first];
        }
    }
    // 3. compute
    for (int in = 0; in < nonequind.size(); ++in) {
        auto &nbhp = nbhs[nonequind[in]];
        int aid = nonequind[in];
        Message(fmt::format("Currently on atom {} of {} in total", in, nonequind.size()));
        CalcSiteHess(nbhp);

        double siteen = buff_site_energy;
        for (int ii = 0; ii < nbhs.size(); ++ii) {
            if (mapping_to_primitive[ii] != nonequatid[in]) continue;
            auto &nbh = nbhs[ii];
            // energy
            atom.site_energies_[ii] += siteen;
            atom.energy_ += siteen;
            // force
            for (int i = 0; i < nbh.nbhcount_; ++i) {
                atom.forces_[ii](0) += buff_site_energy_ders(i, 0);
                atom.forces_[ii](1) += buff_site_energy_ders(i, 1);
                atom.forces_[ii](2) += buff_site_energy_ders(i, 2);
                atom.forces_[nbh.indices_[i]](0) -= buff_site_energy_ders(i, 0);
                atom.forces_[nbh.indices_[i]](1) -= buff_site_energy_ders(i, 1);
                atom.forces_[nbh.indices_[i]](2) -= buff_site_energy_ders(i, 2);
                // stress
                for (int a = 0; a < 3; a++) {
                    for (int b = 0; b < 3; b++) {
                        atom.stress_(a, b) -= buff_site_energy_ders(i, a) * nbh.pos_[i](b);
                    }
                }
                // 1st-order correction term applied on cellhess
                if (hc_correction) {
                    Eigen::RowVector3d ri = nbh.pos_[i] * rcell;
                    for (int a = 0; a < 3; ++a) {
                        for (int b = 0; b < 3; ++b) {
                            for (int l = 0; l < 3; ++l) {
                                for (int p = 0; p < 3; ++p) {
                                    cellhess(3*a+b, 3*l+p) += 
                                        buff_site_energy_ders(i, b) * ri(l) * rcell(p, a) +
                                        buff_site_energy_ders(i, p) * ri(a) * rcell(b, l);
                                }
                            }
                        }
                    }
                }
                int ati = ii;
                int atj = nbh.indices_[i];
                if (rR_correction) {
                    for (int a = 0; a < 3; ++a) {
                        for (int b = 0; b < 3; ++b) {
                            lrhess(3*a+b, 3*ati+a) -= 
                                rcell(a, a) * buff_site_energy_ders(i, b);
                            lrhess(3*a+b, 3*atj+a) += 
                                rcell(a, a) * buff_site_energy_ders(i, b);
                        }
                    }
                }
            }
            // hessian
            Eigen::RowVector3d ri, rj;
            for (int ai = 0; ai < nbh.nbhcount_; ++ai) {
                int ati = nbh.indices_[ai];
                ri = nbh.pos_[ai] * rcell;
                for (int aj = 0; aj < nbh.nbhcount_; ++aj) {
                    int atj = nbh.indices_[aj];
                    rj = nbh.pos_[aj] * rcell;
                    for (int a = 0; a < 3; ++a) {
                        for (int b = 0; b < 3; ++b) {
                            hessian(ati, a, atj, b) += buff_site_hessian(ai, a, aj, b);
                            for (int l = 0; l < 3; ++l) {
                                for (int p = 0; p < 3; ++p) {
                                    cellhess(3*a+b, 3*l+p) += buff_site_hessian(ai, b, aj, p) * ri(a) * rj(l);
                                }
                            }
                            for (int g = 0; g < 3; ++g) {
                                lrhess(3*a+b, 3*ii+g) -= ri(a) * buff_site_hessian(ai, b, aj, g);
                                lrhess(3*a+b, 3*atj+g) += ri(a) * buff_site_hessian(ai, b, aj, g);
                            }
                        }
                    }
                }
            }
            // second term, i != j, -\sum_{n} d2Vi/d(xn-xi)d(xj-xi) -\sum_{n} d2Vj/d(xi-xj)d(xn-xj)
            for (int jj = 0; jj < nbh.nbhcount_; ++jj) {
                int atj = nbh.indices_[jj];
                for (int a = 0; a < 3; ++a) {
                    for (int b = 0; b < 3; ++b) {
                        for (int kk = 0; kk < nbh.nbhcount_; ++kk) {
                            hessian(ii, a, atj, b) -= buff_site_hessian(kk, a, jj, b);
                            hessian(atj, a, ii, b) -= buff_site_hessian(jj, a, kk, b);
                        }
                    }
                }
            }
            // second term, i == j, \sum_{m,n} d2Vi/d(xn-xi)d(xm-xi)
            for (int jj = 0; jj < nbh.nbhcount_; ++jj) {
                for (int kk = 0; kk < nbh.nbhcount_; ++kk) {
                    for (int a = 0; a < 3; ++a) {
                        for (int b = 0; b < 3; ++b) {
                            hessian(ii, a , ii, b) += buff_site_hessian(jj, a, kk, b);
                        }
                    }
                }
            }
        }
    }
}

void PotMTP::getEFSHessianCellSymmOMP(Atoms &atom, HessianMatrix &hessian, Eigen::MatrixXd &cellhess, Eigen::MatrixXd &lrhess, double symprec) {
    omp_init();
    int actual_threads = 1;

    if (!add_efs) atom.resetEFS(); else atom.setEFSflag();
    auto rcell = atom.cell_.inverse();
    Neighborhoods nbhs(atom, cutoff);

    hessian.resize(atom.size());
    cellhess.resize(9,9);
    cellhess.setZero();
    lrhess.resize(9, 3*atom.size());
    lrhess.setZero();
    // 1. find permitive cell and mapping relations, only translation symmetry
    auto dataset = Symmetry3D::get_sym_dataset(atom, symprec);
    auto &mapping_to_primitive = dataset.mapping_to_primitive;
    std::set<int> nonequat;
    std::vector<int> nonequatid;
    std::vector<int> nonequind;
    for (int i = 0; i < mapping_to_primitive.size(); ++i) {
        int primid = mapping_to_primitive[i];
        if (nonequat.count(primid) == 0) {
            nonequind.push_back(i);
            nonequat.insert(primid);
            nonequatid.push_back(primid);
        }
    }
    // 2. build neighbor list
    std::vector<std::pair<int, Eigen::RowVector3d>> sortind;
    for (int i = 0; i < nbhs.size(); ++i) {
        auto &nbh = nbhs[i];
        // sort
        sortind.resize(nbh.nbhcount_);
        for (int j = 0; j < nbh.nbhcount_; ++j) {
            sortind[j].first = j;
            sortind[j].second = nbh.pos_[j];
        }
        std::ranges::sort(sortind, [&symprec](const std::pair<int, Eigen::RowVector3d> &a, const std::pair<int, Eigen::RowVector3d> &b) {
            if (a.second(0) < b.second(0) - symprec) return true;
            if (a.second(0) > b.second(0) + symprec) return false;
            
            if (a.second(1) < b.second(1) - symprec) return true;
            if (a.second(1) > b.second(1) + symprec) return false;
            
            if (a.second(2) < b.second(2) - symprec) return true;
            if (a.second(2) > b.second(2) + symprec) return false;

            return false;
        });
        // reorder
        auto nbhbck = nbh;
        for (int j = 0; j < nbh.nbhcount_; ++j) {
            nbh.chemmap_[j] = nbhbck.chemmap_[sortind[j].first];
            nbh.distances_[j] = nbhbck.distances_[sortind[j].first];
            nbh.indices_[j] = nbhbck.indices_[sortind[j].first];
            nbh.pos_[j] = nbhbck.pos_[sortind[j].first];
        }
    }

    // 3. compute
#ifdef USE_OMP
#pragma omp parallel num_threads(omp_buffers_list.size())
#endif 
{
#ifdef USE_OMP
    int ithread = omp_get_thread_num();
    if (ithread == 0) actual_threads = omp_get_num_threads();
#else 
    int ithread = 0;
#endif
    ompbuffers &buffer = omp_buffers_list[ithread];
    buffer.energy = 0;
    buffer.force.assign(atom.size(), Eigen::RowVector3d::Zero());
    buffer.siteenergy.assign(atom.size(), 0.0);
    buffer.stress = Eigen::Matrix3d::Zero();
    buffer.hessian.resize(atom.size());
    buffer.hessian.setzero();
    buffer.cellhess.resize(9,9);
    buffer.cellhess.setZero();
    buffer.lrhess.resize(9, 3*atom.size());
    buffer.lrhess.setZero();
#ifdef USE_OMP
#pragma omp for schedule(static)
#endif
    for (int in = 0; in < nonequind.size(); ++in) {
        auto &nbhp = nbhs[nonequind[in]];
        int aid = nonequind[in];
        CalcSiteHessOMP(nbhp, buffer);

        double siteen = buffer.buff_site_energy;
        for (int ii = 0; ii < nbhs.size(); ++ii) {
            if (mapping_to_primitive[ii] != nonequatid[in]) continue;
            auto &nbh = nbhs[ii];
            // energy
            buffer.siteenergy[ii] = siteen;
            buffer.energy += siteen;
            // force
            for (int i = 0; i < nbh.nbhcount_; ++i) {
                buffer.force[ii](0) += buffer.buff_site_energy_ders(i, 0);
                buffer.force[ii](1) += buffer.buff_site_energy_ders(i, 1);
                buffer.force[ii](2) += buffer.buff_site_energy_ders(i, 2);
                buffer.force[nbh.indices_[i]](0) -= buffer.buff_site_energy_ders(i, 0);
                buffer.force[nbh.indices_[i]](1) -= buffer.buff_site_energy_ders(i, 1);
                buffer.force[nbh.indices_[i]](2) -= buffer.buff_site_energy_ders(i, 2);
                // stress
                for (int a = 0; a < 3; a++) {
                    for (int b = 0; b < 3; b++) {
                        buffer.stress(a, b) -= buffer.buff_site_energy_ders(i, a) * nbh.pos_[i](b);
                    }
                }
                // 1st-order correction term applied on cellhess
                if (hc_correction) {
                    Eigen::RowVector3d ri = nbh.pos_[i] * rcell;
                    for (int a = 0; a < 3; ++a) {
                        for (int b = 0; b < 3; ++b) {
                            for (int l = 0; l < 3; ++l) {
                                for (int p = 0; p < 3; ++p) {
                                    buffer.cellhess(3*a+b, 3*l+p) += 
                                        buffer.buff_site_energy_ders(i, b) * ri(l) * rcell(p, a) +
                                        buffer.buff_site_energy_ders(i, p) * ri(a) * rcell(b, l);
                                }
                            }
                        }
                    }
                }
                int ati = ii;
                int atj = nbh.indices_[i];
                if (rR_correction) {
                    for (int a = 0; a < 3; ++a) {
                        for (int b = 0; b < 3; ++b) {
                            buffer.lrhess(3*a+b, 3*ati+a) -= 
                                rcell(a, a) * buffer.buff_site_energy_ders(i, b);
                            buffer.lrhess(3*a+b, 3*atj+a) += 
                                rcell(a, a) * buffer.buff_site_energy_ders(i, b);
                        }
                    }
                }
            }
            // hessian
            Eigen::RowVector3d ri, rj;
            for (int ai = 0; ai < nbh.nbhcount_; ++ai) {
                int ati = nbh.indices_[ai];
                ri = nbh.pos_[ai] * rcell;
                for (int aj = 0; aj < nbh.nbhcount_; ++aj) {
                    int atj = nbh.indices_[aj];
                    rj = nbh.pos_[aj] * rcell;
                    for (int a = 0; a < 3; ++a) {
                        for (int b = 0; b < 3; ++b) {
                            buffer.hessian(ati, a, atj, b) += buffer.buff_site_hessian(ai, a, aj, b);
                            for (int l = 0; l < 3; ++l) {
                                for (int p = 0; p < 3; ++p) {
                                    buffer.cellhess(3*a+b, 3*l+p) += buffer.buff_site_hessian(ai, b, aj, p) * ri(a) * rj(l);
                                }
                            }
                            for (int g = 0; g < 3; ++g) {
                                buffer.lrhess(3*a+b, 3*ii+g) -= ri(a) * buffer.buff_site_hessian(ai, b, aj, g);
                                buffer.lrhess(3*a+b, 3*atj+g) += ri(a) * buffer.buff_site_hessian(ai, b, aj, g);
                            }
                        }
                    }
                }
            }
            // second term, i != j, -\sum_{n} d2Vi/d(xn-xi)d(xj-xi) -\sum_{n} d2Vj/d(xi-xj)d(xn-xj)
            for (int jj = 0; jj < nbh.nbhcount_; ++jj) {
                int atj = nbh.indices_[jj];
                for (int a = 0; a < 3; ++a) {
                    for (int b = 0; b < 3; ++b) {
                        for (int kk = 0; kk < nbh.nbhcount_; ++kk) {
                            buffer.hessian(ii, a, atj, b) -= buffer.buff_site_hessian(kk, a, jj, b);
                            buffer.hessian(atj, a, ii, b) -= buffer.buff_site_hessian(jj, a, kk, b);
                        }
                    }
                }
            }
            // second term, i == j, \sum_{m,n} d2Vi/d(xn-xi)d(xm-xi)
            for (int jj = 0; jj < nbh.nbhcount_; ++jj) {
                for (int kk = 0; kk < nbh.nbhcount_; ++kk) {
                    for (int a = 0; a < 3; ++a) {
                        for (int b = 0; b < 3; ++b) {
                            buffer.hessian(ii, a , ii, b) += buffer.buff_site_hessian(jj, a, kk, b);
                        }
                    }
                }
            }
        }
    }
}
    // merge
    for (int i = 0; i < actual_threads; ++i) {
        auto &buffer = omp_buffers_list[i];
        atom.energy_ += buffer.energy;
        atom.stress_ += buffer.stress;
        for (int iat = 0; iat < atom.size(); ++iat) {
            atom.forces_[iat] += buffer.force[iat];
            for (int jat = 0; jat < atom.size(); ++jat) {
                hessian(iat, jat) += buffer.hessian(iat, jat);
            }
        }
        cellhess += buffer.cellhess;
        lrhess += buffer.lrhess;
    }
}

void PotMTP::getEFSHessianCellOMP(Atoms &atom, AnyNeighborhoods &nbhs, HessianMatrix &hessian, Eigen::MatrixXd &cellhess, Eigen::MatrixXd &lrhess) {
    omp_init();
    int actual_threads = 1;

    if (!add_efs) atom.resetEFS(); else atom.setEFSflag();
    auto rcell = atom.cell_.inverse();

    hessian.resize(atom.size());
    cellhess.resize(9,9);
    cellhess.setZero();
    lrhess.resize(9, 3*atom.size());
    lrhess.setZero();
#ifdef USE_OMP
#pragma omp parallel num_threads(omp_buffers_list.size())
#endif 
{
#ifdef USE_OMP
    int ithread = omp_get_thread_num();
    if (ithread == 0) actual_threads = omp_get_num_threads();
#else 
    int ithread = 0;
#endif
    ompbuffers &buffer = omp_buffers_list[ithread];
    buffer.energy = 0;
    buffer.force.assign(atom.size(), Eigen::RowVector3d::Zero());
    buffer.siteenergy.assign(atom.size(), 0.0);
    buffer.stress = Eigen::Matrix3d::Zero();
    buffer.hessian.resize(atom.size());
    buffer.hessian.setzero();
    buffer.cellhess.resize(9, 9);
    buffer.cellhess.setZero();
    buffer.lrhess.resize(9, 3*atom.size());
    buffer.lrhess.setZero();
#ifdef USE_OMP
#pragma omp for schedule(static)
#endif
    for (int ii = 0; ii < nbhs.size(); ++ii) {
        auto &nbh = nbhs[ii];
        Message(fmt::format("Currently on atom {} at proc {}, {} in total", ii, ithread, atom.size()));
        CalcSiteHessOMP(nbh, buffer);

        double siteen = buffer.buff_site_energy;
        // energy
        buffer.siteenergy[ii] = siteen;
        buffer.energy += siteen;
        // force
        for (int i = 0; i < nbh.nbhcount_; ++i) {
            buffer.force[ii](0) += buffer.buff_site_energy_ders(i, 0);
            buffer.force[ii](1) += buffer.buff_site_energy_ders(i, 1);
            buffer.force[ii](2) += buffer.buff_site_energy_ders(i, 2);
            buffer.force[nbh.indices_[i]](0) -= buffer.buff_site_energy_ders(i, 0);
            buffer.force[nbh.indices_[i]](1) -= buffer.buff_site_energy_ders(i, 1);
            buffer.force[nbh.indices_[i]](2) -= buffer.buff_site_energy_ders(i, 2);
            // stress
            for (int a = 0; a < 3; a++) {
                for (int b = 0; b < 3; b++) {
                    buffer.stress(a, b) -= buffer.buff_site_energy_ders(i, a) * nbh.pos_[i](b);
                }
            }
            // 1st-order correction term applied on cellhess
            /**
             * @brief 
             * This term is small
             * If do not change atom position according to strain, this term should be ignored.
             */
            if (hc_correction) {
            Eigen::RowVector3d ri = nbh.pos_[i] * rcell;
                for (int a = 0; a < 3; ++a) {
                    for (int b = 0; b < 3; ++b) {
                        for (int l = 0; l < 3; ++l) {
                            for (int p = 0; p < 3; ++p) {
                                buffer.cellhess(3*a+b, 3*l+p) += 
                                    buffer.buff_site_energy_ders(i, b) * ri(l) * rcell(p, a) +
                                    buffer.buff_site_energy_ders(i, p) * ri(a) * rcell(b, l);
                            }
                        }
                    }
                }
            }
            int ati = ii;
            int atj = nbh.indices_[i];
            if (rR_correction) {
                for (int a = 0; a < 3; ++a) {
                    for (int b = 0; b < 3; ++b) {
                        buffer.lrhess(3*a+b, 3*ati+a) -= 
                            rcell(a, a) * buffer.buff_site_energy_ders(i, b);
                        buffer.lrhess(3*a+b, 3*atj+a) += 
                            rcell(a, a) * buffer.buff_site_energy_ders(i, b);
                    }
                }
            }
        }
        // hessian
        Eigen::RowVector3d ri, rj;
        for (int ai = 0; ai < nbh.nbhcount_; ++ai) {
            int ati = nbh.indices_[ai];
            ri = nbh.pos_[ai] * rcell;
            for (int aj = 0; aj < nbh.nbhcount_; ++aj) {
                int atj = nbh.indices_[aj];
                rj = nbh.pos_[aj] * rcell;
                for (int a = 0; a < 3; ++a) {
                    for (int b = 0; b < 3; ++b) {
                        buffer.hessian(ati, a, atj, b) += buffer.buff_site_hessian(ai, a, aj, b);
                        for (int l = 0; l < 3; ++l) {
                            for (int p = 0; p < 3; ++p) {
                                buffer.cellhess(3*a+b, 3*l+p) += buffer.buff_site_hessian(ai, b, aj, p) * ri(a) * rj(l);
                            }
                        }
                        for (int g = 0; g < 3; ++g) {
                            buffer.lrhess(3*a+b, 3*ii+g) -= ri(a) * buffer.buff_site_hessian(ai, b, aj, g);
                            buffer.lrhess(3*a+b, 3*atj+g) += ri(a) * buffer.buff_site_hessian(ai, b, aj, g);
                        }
                    }
                }
            }
        }
        // second term, i != j, -\sum_{n} d2Vi/d(xn-xi)d(xj-xi) -\sum_{n} d2Vj/d(xi-xj)d(xn-xj)
        for (int jj = 0; jj < nbh.nbhcount_; ++jj) {
            int atj = nbh.indices_[jj];
            for (int a = 0; a < 3; ++a) {
                for (int b = 0; b < 3; ++b) {
                    for (int kk = 0; kk < nbh.nbhcount_; ++kk) {
                        buffer.hessian(ii, a, atj, b) -= buffer.buff_site_hessian(kk, a, jj, b);
                        buffer.hessian(atj, a, ii, b) -= buffer.buff_site_hessian(jj, a, kk, b);
                    }
                }
            }
        }
        // second term, i == j, \sum_{m,n} d2Vi/d(xn-xi)d(xm-xi)
        for (int jj = 0; jj < nbh.nbhcount_; ++jj) {
            for (int kk = 0; kk < nbh.nbhcount_; ++kk) {
                for (int a = 0; a < 3; ++a) {
                    for (int b = 0; b < 3; ++b) {
                        buffer.hessian(ii, a , ii, b) += buffer.buff_site_hessian(jj, a, kk, b);
                    }
                }
            }
        }
    }
}
    // merge
    for (int i = 0; i < actual_threads; ++i) {
        auto &buffer = omp_buffers_list[i];
        atom.energy_ += buffer.energy;
        atom.stress_ += buffer.stress;
        for (int iat = 0; iat < atom.size(); ++iat) {
            atom.forces_[iat] += buffer.force[iat];
            for (int jat = 0; jat < atom.size(); ++jat) {
                hessian(iat, jat) += buffer.hessian(iat, jat);
            }
        }
        cellhess += buffer.cellhess;
        lrhess += buffer.lrhess;
    }
}

// TODO: rotational symmetry
void PotMTP::getEFSHessianSymm(Atoms &atom, HessianMatrix &hessian, double symprec) {
    if (!add_efs) atom.resetEFS(); else atom.setEFSflag();
    Neighborhoods nbhs(atom, cutoff);
    hessian.resize(atom.size());
    // 1. find permitive cell and mapping relations, only translation symmetry
    auto dataset = Symmetry3D::get_sym_dataset(atom, symprec);
    auto &mapping_to_primitive = dataset.mapping_to_primitive;
    std::set<int> nonequat;
    std::vector<int> nonequatid;
    std::vector<int> nonequind;
    for (int i = 0; i < mapping_to_primitive.size(); ++i) {
        int primid = mapping_to_primitive[i];
        if (nonequat.count(primid) == 0) {
            nonequind.push_back(i);
            nonequat.insert(primid);
            nonequatid.push_back(primid);
        }
    }
    // 2. build neighbor list
    std::vector<std::pair<int, Eigen::RowVector3d>> sortind;
    for (int i = 0; i < nbhs.size(); ++i) {
        auto &nbh = nbhs[i];
        // sort
        sortind.resize(nbh.nbhcount_);
        for (int j = 0; j < nbh.nbhcount_; ++j) {
            sortind[j].first = j;
            sortind[j].second = nbh.pos_[j];
        }
        std::ranges::sort(sortind, [&symprec](const std::pair<int, Eigen::RowVector3d> &a, const std::pair<int, Eigen::RowVector3d> &b) {
            if (a.second(0) < b.second(0) - symprec) return true;
            if (a.second(0) > b.second(0) + symprec) return false;
            
            if (a.second(1) < b.second(1) - symprec) return true;
            if (a.second(1) > b.second(1) + symprec) return false;
            
            if (a.second(2) < b.second(2) - symprec) return true;
            if (a.second(2) > b.second(2) + symprec) return false;

            return false;
        });
        // reorder
        auto nbhbck = nbh;
        for (int j = 0; j < nbh.nbhcount_; ++j) {
            nbh.chemmap_[j] = nbhbck.chemmap_[sortind[j].first];
            nbh.distances_[j] = nbhbck.distances_[sortind[j].first];
            nbh.indices_[j] = nbhbck.indices_[sortind[j].first];
            nbh.pos_[j] = nbhbck.pos_[sortind[j].first];
        }
    }
    // 3. compute
    for (int in = 0; in < nonequind.size(); ++in) {
        auto &nbhp = nbhs[nonequind[in]];
        int aid = nonequind[in];
        Message(fmt::format("Currently on atom {} of {} in total", in, nonequind.size()));
        CalcSiteHessLowMemory(nbhp);

        double siteen = buff_site_energy;
        for (int ii = 0; ii < nbhs.size(); ++ii) {
            if (mapping_to_primitive[ii] != nonequatid[in]) continue;
            auto &nbh = nbhs[ii];
            // energy
            atom.site_energies_[ii] += siteen;
            atom.energy_ += siteen;
            // force
            for (int i = 0; i < nbh.nbhcount_; ++i) {
                atom.forces_[ii](0) += buff_site_energy_ders(i, 0);
                atom.forces_[ii](1) += buff_site_energy_ders(i, 1);
                atom.forces_[ii](2) += buff_site_energy_ders(i, 2);
                atom.forces_[nbh.indices_[i]](0) -= buff_site_energy_ders(i, 0);
                atom.forces_[nbh.indices_[i]](1) -= buff_site_energy_ders(i, 1);
                atom.forces_[nbh.indices_[i]](2) -= buff_site_energy_ders(i, 2);
                // stress
                for (int a = 0; a < 3; a++) {
                    for (int b = 0; b < 3; b++) {
                        atom.stress_(a, b) -= buff_site_energy_ders(i, a) * nbh.pos_[i](b);
                    }
                }
            }
            // hessian
            for (int ai = 0; ai < nbh.nbhcount_; ++ai) {
                int ati = nbh.indices_[ai];
                for (int aj = 0; aj < nbh.nbhcount_; ++aj) {
                    int atj = nbh.indices_[aj];
                    for (int a = 0; a < 3; ++a) {
                        for (int b = 0; b < 3; ++b) {
                            hessian(ati, a, atj, b) += buff_site_hessian(ai, a, aj, b);
                        }
                    }
                }
            }
            // second term, i != j, -\sum_{n} d2Vi/d(xn-xi)d(xj-xi) -\sum_{n} d2Vj/d(xi-xj)d(xn-xj)
            for (int jj = 0; jj < nbh.nbhcount_; ++jj) {
                int atj = nbh.indices_[jj];
                for (int a = 0; a < 3; ++a) {
                    for (int b = 0; b < 3; ++b) {
                        for (int kk = 0; kk < nbh.nbhcount_; ++kk) {
                            hessian(ii, a, atj, b) -= buff_site_hessian(kk, a, jj, b);
                            hessian(atj, a, ii, b) -= buff_site_hessian(jj, a, kk, b);
                        }
                    }
                }
            }
            // second term, i == j, \sum_{m,n} d2Vi/d(xn-xi)d(xm-xi)
            for (int jj = 0; jj < nbh.nbhcount_; ++jj) {
                for (int kk = 0; kk < nbh.nbhcount_; ++kk) {
                    for (int a = 0; a < 3; ++a) {
                        for (int b = 0; b < 3; ++b) {
                            hessian(ii, a , ii, b) += buff_site_hessian(jj, a, kk, b);
                        }
                    }
                }
            }
        }
    }
}

void PotMTP::getEFSHessianSymmOMP(Atoms &atom, HessianMatrix &hessian, double symprec) {
    omp_init();

    if (!add_efs) atom.resetEFS(); else atom.setEFSflag();
    Neighborhoods nbhs(atom, cutoff);

    hessian.resize(atom.size());
    // 1. find permitive cell and mapping relations, only translation symmetry
    auto dataset = Symmetry3D::get_sym_dataset(atom, symprec);
    auto &mapping_to_primitive = dataset.mapping_to_primitive;
    std::set<int> nonequat;
    std::vector<int> nonequatid;
    std::vector<int> nonequind;
    for (int i = 0; i < mapping_to_primitive.size(); ++i) {
        int primid = mapping_to_primitive[i];
        if (nonequat.count(primid) == 0) {
            nonequind.push_back(i);
            nonequat.insert(primid);
            nonequatid.push_back(primid);
        }
    }
    // 2. build neighbor list
    std::vector<std::pair<int, Eigen::RowVector3d>> sortind;
    for (int i = 0; i < nbhs.size(); ++i) {
        auto &nbh = nbhs[i];
        // sort
        sortind.resize(nbh.nbhcount_);
        for (int j = 0; j < nbh.nbhcount_; ++j) {
            sortind[j].first = j;
            sortind[j].second = nbh.pos_[j];
        }
        std::ranges::sort(sortind, [&symprec](const std::pair<int, Eigen::RowVector3d> &a, const std::pair<int, Eigen::RowVector3d> &b) {
            if (a.second(0) < b.second(0) - symprec) return true;
            if (a.second(0) > b.second(0) + symprec) return false;
            
            if (a.second(1) < b.second(1) - symprec) return true;
            if (a.second(1) > b.second(1) + symprec) return false;
            
            if (a.second(2) < b.second(2) - symprec) return true;
            if (a.second(2) > b.second(2) + symprec) return false;

            return false;
        });
        // reorder
        auto nbhbck = nbh;
        for (int j = 0; j < nbh.nbhcount_; ++j) {
            nbh.chemmap_[j] = nbhbck.chemmap_[sortind[j].first];
            nbh.distances_[j] = nbhbck.distances_[sortind[j].first];
            nbh.indices_[j] = nbhbck.indices_[sortind[j].first];
            nbh.pos_[j] = nbhbck.pos_[sortind[j].first];
        }
    }

    // 3. compute
#ifdef USE_OMP
    const int active_threads = std::max(
        1, std::min(static_cast<int>(omp_buffers_list.size()), static_cast<int>(nonequind.size())));
#pragma omp parallel num_threads(active_threads)
#endif 
{
#ifdef USE_OMP
    int ithread = omp_get_thread_num();
#else 
    int ithread = 0;
#endif
    ompbuffers &buffer = omp_buffers_list[ithread];
#ifdef USE_OMP
// Round-robin sites let later threads compute while the next site is
// accumulated in order. Contiguous chunks stall at their first ordered
// region and serialize the expensive site calculations. Keep the original
// accumulation order without allocating a dense Hessian per thread.
#pragma omp for ordered schedule(static, 1)
#endif
    for (int in = 0; in < nonequind.size(); ++in) {
        auto &nbhp = nbhs[nonequind[in]];
        int aid = nonequind[in];
        CalcSiteHessLowMemoryOMP(nbhp, buffer, ithread);

        double siteen = buffer.buff_site_energy;
#ifdef USE_OMP
#pragma omp ordered
#endif
        {
        for (int ii = 0; ii < nbhs.size(); ++ii) {
            if (mapping_to_primitive[ii] != nonequatid[in]) continue;
            auto &nbh = nbhs[ii];
            // energy
            atom.site_energies_[ii] += siteen;
            atom.energy_ += siteen;
            // force
            for (int i = 0; i < nbh.nbhcount_; ++i) {
                atom.forces_[ii](0) += buffer.buff_site_energy_ders(i, 0);
                atom.forces_[ii](1) += buffer.buff_site_energy_ders(i, 1);
                atom.forces_[ii](2) += buffer.buff_site_energy_ders(i, 2);
                atom.forces_[nbh.indices_[i]](0) -= buffer.buff_site_energy_ders(i, 0);
                atom.forces_[nbh.indices_[i]](1) -= buffer.buff_site_energy_ders(i, 1);
                atom.forces_[nbh.indices_[i]](2) -= buffer.buff_site_energy_ders(i, 2);
                // stress
                for (int a = 0; a < 3; a++) {
                    for (int b = 0; b < 3; b++) {
                        atom.stress_(a, b) -= buffer.buff_site_energy_ders(i, a) * nbh.pos_[i](b);
                    }
                }
            }
            // hessian
            for (int ai = 0; ai < nbh.nbhcount_; ++ai) {
                int ati = nbh.indices_[ai];
                for (int aj = 0; aj < nbh.nbhcount_; ++aj) {
                    int atj = nbh.indices_[aj];
                    for (int a = 0; a < 3; ++a) {
                        for (int b = 0; b < 3; ++b) {
                            hessian(ati, a, atj, b) += buffer.buff_site_hessian(ai, a, aj, b);
                        }
                    }
                }
            }
            // second term, i != j, -\sum_{n} d2Vi/d(xn-xi)d(xj-xi) -\sum_{n} d2Vj/d(xi-xj)d(xn-xj)
            for (int jj = 0; jj < nbh.nbhcount_; ++jj) {
                int atj = nbh.indices_[jj];
                for (int a = 0; a < 3; ++a) {
                    for (int b = 0; b < 3; ++b) {
                        for (int kk = 0; kk < nbh.nbhcount_; ++kk) {
                            hessian(ii, a, atj, b) -= buffer.buff_site_hessian(kk, a, jj, b);
                            hessian(atj, a, ii, b) -= buffer.buff_site_hessian(jj, a, kk, b);
                        }
                    }
                }
            }
            // second term, i == j, \sum_{m,n} d2Vi/d(xn-xi)d(xm-xi)
            for (int jj = 0; jj < nbh.nbhcount_; ++jj) {
                for (int kk = 0; kk < nbh.nbhcount_; ++kk) {
                    for (int a = 0; a < 3; ++a) {
                        for (int b = 0; b < 3; ++b) {
                            hessian(ii, a , ii, b) += buffer.buff_site_hessian(jj, a, kk, b);
                        }
                    }
                }
            }
        }
        }
    }
}
}

// assume std::pair<int, int>(i, j) : i <= j and no dups
void PotMTP::getHessian(Atoms &atom, AnyNeighborhoods &nbhs, std::vector<std::pair<int, int>> inds, HessianSparse &hessian) {
    // check if H_{ij} is zero
    // first, find inds common neighbor ids
    std::set<int> inds_candidate;
    std::vector<std::pair<int, int>> tobecalc;
    for (auto &it : inds) {
        int i = it.first;
        int j = it.second;
        auto &inbinds = nbhs[i].indices_;
        auto &jnbinds = nbhs[j].indices_;
        std::vector<int> intersection;
        std::set_intersection(inbinds.begin(), inbinds.end(), 
                              jnbinds.begin(), jnbinds.end(),
                              std::insert_iterator<std::vector<int>>(intersection, intersection.begin()));
        if (intersection.size() == 0) {
            hessian(i, j) = Eigen::Matrix3d::Zero();
        } else {
            hessian(i, j) = Eigen::Matrix3d::Zero();
            tobecalc.push_back(it);
            // case 1: i j is neighbor of each other
            if ((atom.pos_[i] - atom.pos_[j]).squaredNorm() < cutoffsq) {
                inds_candidate.insert(i);
                inds_candidate.insert(j);
                for (auto &k : intersection) {
                    inds_candidate.insert(k);
                }
            } else { // case 2: i j is not neighbor
                for (auto &k : intersection) {
                    inds_candidate.insert(k);
                }
            }
        }
    }

    for (auto &ii : inds_candidate) {
        std::cout << "Currently on atom " << ii << " of " << nbhs.size() << " in total" << std::endl;
        Neighborhood &nbh = nbhs[ii];
        buff_site_energy = 0;
        // todo: calc specific site hessian to reduce calcualtion
        CalcSiteHessLowMemory(nbh);
        for (int ai = 0; ai < nbh.nbhcount_; ++ai) {
            int ati = nbh.indices_[ai];
            for (int aj = 0; aj < nbh.nbhcount_; ++aj) {
                int atj = nbh.indices_[aj];
                if (ati > atj) continue;
                for (int a = 0; a < 3; ++a) {
                    for (int b = 0; b < 3; ++b) {
                        hessian(ati, a, atj, b) += buff_site_hessian(ai, a, aj, b);
                    }
                }
            }
        }
        // second term, i != j, -\sum_{n} d2Vi/d(xn-xi)d(xj-xi) -\sum_{n} d2Vj/d(xi-xj)d(xn-xj)
        for (int jj = 0; jj < nbh.nbhcount_; ++jj) {
            int atj = nbh.indices_[jj];
            for (int a = 0; a < 3; ++a) {
                for (int b = 0; b < 3; ++b) {
                    for (int kk = 0; kk < nbh.nbhcount_; ++kk) {
                        if (ii <= atj)
                            hessian(ii, a, atj, b) -= buff_site_hessian(kk, a, jj, b);
                        if (atj <= ii) 
                            hessian(atj, a, ii, b) -= buff_site_hessian(jj, a, kk, b);
                    }
                }
            }
        }
        // second term, i == j, \sum_{m,n} d2Vi/d(xn-xi)d(xm-xi)
        for (int jj = 0; jj < nbh.nbhcount_; ++jj) {
            for (int kk = 0; kk < nbh.nbhcount_; ++kk) {
                for (int a = 0; a < 3; ++a) {
                    for (int b = 0; b < 3; ++b) {
                        hessian(ii, a , ii, b) += buff_site_hessian(jj, a, kk, b);
                    }
                }
            }
        }
    }
}

void PotMTP::getEFSHessianOMP(Atoms &atom, AnyNeighborhoods &nbhs, HessianMatrix &hessian) {
    omp_init();

    if (!add_efs) atom.resetEFS(); else atom.setEFSflag();

    hessian.resize(atom.size());
#ifdef USE_OMP
    const int active_threads = std::max(
        1, std::min(static_cast<int>(omp_buffers_list.size()), static_cast<int>(nbhs.size())));
#pragma omp parallel num_threads(active_threads)
#endif 
{
#ifdef USE_OMP
    int ithread = omp_get_thread_num();
#else 
    int ithread = 0;
#endif
    ompbuffers &buffer = omp_buffers_list[ithread];
#ifdef USE_OMP
// Round-robin sites let later threads compute while the next site is
// accumulated in order. Contiguous chunks stall at their first ordered
// region and serialize the expensive site calculations. Keep the original
// accumulation order without allocating a dense Hessian per thread.
#pragma omp for ordered schedule(static, 1)
#endif
    for (int ii = 0; ii < nbhs.size(); ++ii) {
        auto &nbh = nbhs[ii];
        Message(fmt::format("Currently on atom {} at proc {}, {} in total", ii, ithread, atom.size()));
        CalcSiteHessLowMemoryOMP(nbh, buffer, ithread);

#ifdef USE_OMP
#pragma omp ordered
#endif
        {
        double siteen = buffer.buff_site_energy;
        // energy
        atom.site_energies_[ii] += siteen;
        atom.energy_ += siteen;
        // force
        for (int i = 0; i < nbh.nbhcount_; ++i) {
            atom.forces_[ii](0) += buffer.buff_site_energy_ders(i, 0);
            atom.forces_[ii](1) += buffer.buff_site_energy_ders(i, 1);
            atom.forces_[ii](2) += buffer.buff_site_energy_ders(i, 2);
            atom.forces_[nbh.indices_[i]](0) -= buffer.buff_site_energy_ders(i, 0);
            atom.forces_[nbh.indices_[i]](1) -= buffer.buff_site_energy_ders(i, 1);
            atom.forces_[nbh.indices_[i]](2) -= buffer.buff_site_energy_ders(i, 2);
            // stress
            for (int a = 0; a < 3; a++) {
                for (int b = 0; b < 3; b++) {
                    atom.stress_(a, b) -= buffer.buff_site_energy_ders(i, a) * nbh.pos_[i](b);
                }
            }
        }
        // hessian
        for (int ai = 0; ai < nbh.nbhcount_; ++ai) {
            int ati = nbh.indices_[ai];
            for (int aj = 0; aj < nbh.nbhcount_; ++aj) {
                int atj = nbh.indices_[aj];
                for (int a = 0; a < 3; ++a) {
                    for (int b = 0; b < 3; ++b) {
                        hessian(ati, a, atj, b) += buffer.buff_site_hessian(ai, a, aj, b);
                    }
                }
            }
        }
        // second term, i != j, -\sum_{n} d2Vi/d(xn-xi)d(xj-xi) -\sum_{n} d2Vj/d(xi-xj)d(xn-xj)
        for (int jj = 0; jj < nbh.nbhcount_; ++jj) {
            int atj = nbh.indices_[jj];
            for (int a = 0; a < 3; ++a) {
                for (int b = 0; b < 3; ++b) {
                    for (int kk = 0; kk < nbh.nbhcount_; ++kk) {
                        hessian(ii, a, atj, b) -= buffer.buff_site_hessian(kk, a, jj, b);
                        hessian(atj, a, ii, b) -= buffer.buff_site_hessian(jj, a, kk, b);
                    }
                }
            }
        }
        // second term, i == j, \sum_{m,n} d2Vi/d(xn-xi)d(xm-xi)
        for (int jj = 0; jj < nbh.nbhcount_; ++jj) {
            for (int kk = 0; kk < nbh.nbhcount_; ++kk) {
                for (int a = 0; a < 3; ++a) {
                    for (int b = 0; b < 3; ++b) {
                        hessian(ii, a , ii, b) += buffer.buff_site_hessian(jj, a, kk, b);
                    }
                }
            }
        }
        }
    }
}
}

/**
 * @brief Compute gradient of penalty terms when training, the penalty term is add to ensure orthgonality of basis.
 * 
 * @param weight weight of penalty
 * @param loss 
 * @param loss_gard 
 * 
 * For radial_mode = 1, weight_rad_w0/w1/w2 penalize sampled radial
 * integrals of |f|, |f'|, and |f''| with an r^2 weight.
 */
void PotMTP::AddPenaltyGard(const double weight, double &loss, double *loss_gard, bool fix_radial, bool fix_linear, bool fix_species) {
    //this normalizes the sum of squares of coefficients in C*C*R dimensions to 1 for each K
    if (radial_mode == 0) {
    if (!fix_radial)
    for (int k = 0; k < radial_func_count; k++) {
        double norm = 0;

        for (int i = 0; i < species_count; i++)
            for (int j = 0; j < species_count; j++)
                for (int l = 0; l < rb_size; l++) {
                    int idx = (i * species_count + j) * rb_size * radial_func_count + k * rb_size + l;
                    norm += radial_coeffs[idx] * radial_coeffs[idx];
                }

        loss += weight * (norm - 1) * (norm - 1);

        for (int i = 0; i < species_count; i++)
            for (int j = 0; j < species_count; j++)
                for (int l = 0; l < rb_size; l++) {
                    int idx = (i * species_count + j) * rb_size * radial_func_count + k * rb_size + l;
                    loss_gard[idx] += weight * 4 * (norm - 1) * radial_coeffs[idx];
                }
    }

    //this is targeted on making the radial functions for different K orthogonal
    if (!fix_radial)
    for (int k1 = 0; k1 < radial_func_count; k1++)
        for (int k2 = k1 + 1; k2 < radial_func_count; k2++) {
            double scal = 0;

            for (int i = 0; i < species_count; i++)
                for (int j = 0; j < species_count; j++)
                    for (int l = 0; l < rb_size; l++) {
                        int idx1 = (i * species_count + j) * rb_size * radial_func_count + k1 * rb_size + l;
                        int idx2 = (i * species_count + j) * rb_size * radial_func_count + k2 * rb_size + l;
                        scal += radial_coeffs[idx1] * radial_coeffs[idx2];
                    }

            loss += weight * scal * scal;

            for (int i = 0; i < species_count; i++)
                for (int j = 0; j < species_count; j++)
                    for (int l = 0; l < rb_size; l++) {
                        int idx1 = (i * species_count + j) * rb_size * radial_func_count + k1 * rb_size + l;
                        int idx2 = (i * species_count + j) * rb_size * radial_func_count + k2 * rb_size + l;
                        loss_gard[idx1] += weight * 2 * scal * radial_coeffs[idx2];
                        loss_gard[idx2] += weight * 2 * scal * radial_coeffs[idx1];
                    }
        }
    } else {
        // if radial mode is 1, change weight
        double interval = cutoff;
        double dr = interval / quad_point;
        double pref = dr / interval / interval;
        double r0 = 0;
        bool have_w0 = false;
        bool have_w1 = false;
        bool have_w2 = false;
        if (weight_rad_w2 > 0) have_w2 = true;
        if (weight_rad_w1 > 0) have_w1 = true;
        if (weight_rad_w0 > 0) have_w0 = true;
        if (have_w1 || have_w2 || have_w0)
        for (int n = 0; n < quad_point; ++n) {
            double r = r0 + n * dr;
            for (int ifunc = 0; ifunc < radial_func_count; ++ifunc) {
                for (int typei = 0; typei < species_count; ++typei) {
                    for (int typej = 0; typej < species_count; ++typej) {
                        if (have_w2) rb_calcderder_safe(typei, typej, r);
                        else if (have_w1) rb_calcder_safe(typei, typej, r);
                        else rb_calcval_safe(typei, typej, r);
                        double Qsum = 0, dQsum = 0, ddQsum = 0;
                        int idxoffset = (typei * species_count + typej) * rb_size * radial_func_count + ifunc * rb_size;
                        for (int k = 0; k < rb_size; ++k) {
                            if (have_w0)
                                Qsum += radial_coeffs[k+idxoffset] * rb_vals[k];
                            if (have_w1)
                                dQsum += radial_coeffs[k+idxoffset] * rb_ders[k];
                            if (have_w2) 
                                ddQsum += radial_coeffs[k+idxoffset] * rb_derders[k];
                        }
                        if (weight_rad_w0 > 0) {
                            loss += pref * weight_rad_w0 * r * r * fabs(Qsum);
                            for (int k = 0; k < rb_size; ++k) {
                                double sgn = 1;
                                if (Qsum < 0) sgn = -1;
                                loss_gard[k+idxoffset] += pref * weight_rad_w0 * r * r * rb_vals[k] * sgn; 
                            }
                        }
                        if (weight_rad_w1 > 0) {
                            loss += pref * weight_rad_w1 * r * r * fabs(dQsum);
                            for (int k = 0; k < rb_size; ++k) {
                                double sgn = 1;
                                if (dQsum < 0) sgn = -1;
                                loss_gard[k+idxoffset] += pref * weight_rad_w1 * r * r * rb_ders[k] * sgn; 
                            }
                        }
                        if (weight_rad_w2 > 0) {
                            loss += pref * weight_rad_w2 * r * r * fabs(ddQsum);
                            for (int k = 0; k < rb_size; ++k) {
                                double sgn = 1;
                                if (ddQsum < 0) sgn = -1;
                                loss_gard[k+idxoffset] += pref * weight_rad_w2 * r * r * rb_derders[k] * sgn; 
                            }
                        }
                    }
                }
            }
        }
    }

    //contribution from regularization of SLAE matrix
    if (!fix_linear)
    for (int i = species_count; i < lin_coeff_size; i++) {
        // L2 part
        loss += linear_coeffs[i] * linear_coeffs[i] * reg_L2_vector[i];
        // L1 part
        loss += fabs(linear_coeffs[i]) * reg_L1_vector[i];
    }

    //contribution from regularization of SLAE matrix
    int offset = radial_coeff_size;
    if (fix_radial) offset = 0;
    if (!fix_linear)
    for (int i = species_count; i < lin_coeff_size; i++) {
        // L2 part
        loss_gard[offset + i] += 2 * linear_coeffs[i] * reg_L2_vector[i];
        // L1 part
        loss_gard[offset + i] += linear_coeffs[i] > 0 ? reg_L1_vector[i] : -reg_L1_vector[i];
    }
}

void Potentials::PotMTP::AddPenaltyGardModel(double weight, double &loss, double *loss_gard)
{
    //this normalizes the sum of squares of coefficients in C*C*R dimensions to 1 for each K
    if (radial_mode == 0) {
    for (int k = 0; k < radial_func_count; k++) {
        double norm = 0;

        for (int i = 0; i < species_count; i++)
            for (int j = 0; j < species_count; j++)
                for (int l = 0; l < rb_size; l++) {
                    int idx = (i * species_count + j) * rb_size * radial_func_count + k * rb_size + l;
                    norm += radial_coeffs[idx] * radial_coeffs[idx];
                }

        loss += weight * (norm - 1) * (norm - 1);

        for (int i = 0; i < species_count; i++)
            for (int j = 0; j < species_count; j++)
                for (int l = 0; l < rb_size; l++) {
                    int idx = (i * species_count + j) * rb_size * radial_func_count + k * rb_size + l;
                    loss_gard[idx] += weight * 4 * (norm - 1) * radial_coeffs[idx];
                }
    }

    //this is targeted on making the radial functions for different K orthogonal
    for (int k1 = 0; k1 < radial_func_count; k1++)
        for (int k2 = k1 + 1; k2 < radial_func_count; k2++) {
            double scal = 0;

            for (int i = 0; i < species_count; i++)
                for (int j = 0; j < species_count; j++)
                    for (int l = 0; l < rb_size; l++) {
                        int idx1 = (i * species_count + j) * rb_size * radial_func_count + k1 * rb_size + l;
                        int idx2 = (i * species_count + j) * rb_size * radial_func_count + k2 * rb_size + l;
                        scal += radial_coeffs[idx1] * radial_coeffs[idx2];
                    }

            loss += weight * scal * scal;

            for (int i = 0; i < species_count; i++)
                for (int j = 0; j < species_count; j++)
                    for (int l = 0; l < rb_size; l++) {
                        int idx1 = (i * species_count + j) * rb_size * radial_func_count + k1 * rb_size + l;
                        int idx2 = (i * species_count + j) * rb_size * radial_func_count + k2 * rb_size + l;
                        loss_gard[idx1] += weight * 2 * scal * radial_coeffs[idx2];
                        loss_gard[idx2] += weight * 2 * scal * radial_coeffs[idx1];
                    }
        }
    } else {
        // if radial mode is 1, change weight
        double interval = cutoff;
        double dr = interval / quad_point;
        double pref = dr / interval / interval;
        double r0 = 0;
        bool have_w0 = false;
        bool have_w1 = false;
        bool have_w2 = false;
        if (weight_rad_w2 > 0) have_w2 = true;
        if (weight_rad_w1 > 0) have_w1 = true;
        if (weight_rad_w0 > 0) have_w0 = true;
        if (have_w1 || have_w2 || have_w0) {
            for (int n = 0; n < quad_point; ++n) {
                double r = r0 + n * dr;
                for (int ifunc = 0; ifunc < radial_func_count; ++ifunc) {
                    for (int typei = 0; typei < species_count; ++typei) {
                        for (int typej = 0; typej < species_count; ++typej) {
                            if (have_w2) rb_calcderder_safe(typei, typej, r);
                            else if (have_w1) rb_calcder_safe(typei, typej, r);
                            else rb_calcval_safe(typei, typej, r);
                            double Qsum = 0, dQsum = 0, ddQsum = 0;
                            int idxoffset = (typei * species_count + typej) * rb_size * radial_func_count + ifunc * rb_size;
                            for (int k = 0; k < rb_size; ++k) {
                                if (have_w0)
                                    Qsum += radial_coeffs[k+idxoffset] * rb_vals[k];
                                if (have_w1)
                                    dQsum += radial_coeffs[k+idxoffset] * rb_ders[k];
                                if (have_w2) 
                                    ddQsum += radial_coeffs[k+idxoffset] * rb_derders[k];
                            }
                            if (weight_rad_w0 > 0) {
                                loss += pref * weight_rad_w0 * r * r * fabs(Qsum);
                                for (int k = 0; k < rb_size; ++k) {
                                    double sgn = 1;
                                    if (Qsum < 0) sgn = -1;
                                    loss_gard[k+idxoffset] += pref * weight_rad_w0 * r * r * rb_vals[k] * sgn; 
                                }
                            }
                            if (weight_rad_w1 > 0) {
                                loss += pref * weight_rad_w1 * r * r * fabs(dQsum);
                                for (int k = 0; k < rb_size; ++k) {
                                    double sgn = 1;
                                    if (dQsum < 0) sgn = -1;
                                    loss_gard[k+idxoffset] += pref * weight_rad_w1 * r * r * rb_ders[k] * sgn; 
                                }
                            }
                            if (weight_rad_w2 > 0) {
                                loss += pref * weight_rad_w2 * r * r * fabs(ddQsum);
                                for (int k = 0; k < rb_size; ++k) {
                                    double sgn = 1;
                                    if (ddQsum < 0) sgn = -1;
                                    loss_gard[k+idxoffset] += pref * weight_rad_w2 * r * r * rb_derders[k] * sgn; 
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

void PotMTP::AccumulateEFSCombinationGard(Atoms &atom, AnyNeighborhoods &nbhs,
                                           double dLdE, Eigen::Matrix<double, -1, 3> &dLdF,
                                           Eigen::Matrix3d &dLdS, double *out_gard_accumulator)
{
    for (int i = 0; i < nbhs.size(); ++i) {
        Neighborhood &nbh = nbhs[i];
        tmp_se_ders_wgts.resize(nbh.nbhcount_, 3);
        tmp_se_ders_wgts.setZero();

        for (int j = 0; j < nbh.nbhcount_; ++j) {
            for (int a = 0; a < 3; ++a) {
                tmp_se_ders_wgts(j, a) += dLdF(i, a);
                tmp_se_ders_wgts(j, a) -= dLdF(nbh.indices_[j], a);
            }
        }

        for (int j = 0; j < nbh.nbhcount_; ++j) {
            for (int a = 0; a < 3; ++a)
                for (int b = 0; b < 3; ++b)
                    tmp_se_ders_wgts(j, a) -= dLdS(a, b) * nbh.pos_[j][b];
        }
        AccumulateFPGard(nbh, out_gard_accumulator, dLdE, &tmp_se_ders_wgts);
    }
}

void Potentials::PotMTP::AccumulateEFSCombinationGard(Atoms &atom, AnyNeighborhoods &nbhs, 
                        double dLdE, Eigen::Matrix<double, -1, 3> &dLdF, Eigen::Matrix3d &dLdS, 
                        double *out_gard_accumulator, int offset_r, int offset_l)
{
    for (int i = 0; i < nbhs.size(); ++i) {
        Neighborhood &nbh = nbhs[i];
        tmp_se_ders_wgts.resize(nbh.nbhcount_, 3);
        tmp_se_ders_wgts.setZero();

        for (int j = 0; j < nbh.nbhcount_; ++j) {
            for (int a = 0; a < 3; ++a) {
                tmp_se_ders_wgts(j, a) += dLdF(i, a);
                tmp_se_ders_wgts(j, a) -= dLdF(nbh.indices_[j], a);
            }
        }

        for (int j = 0; j < nbh.nbhcount_; ++j) {
            for (int a = 0; a < 3; ++a)
                for (int b = 0; b < 3; ++b)
                    tmp_se_ders_wgts(j, a) -= dLdS(a, b) * nbh.pos_[j][b];
        }
        AccumulateFPGard(nbh, out_gard_accumulator, dLdE, &tmp_se_ders_wgts, offset_r, offset_l);
    }
}

void PotMTP::AccumulateEFSCombinationGardLonly(Atoms &atom, AnyNeighborhoods &nbhs,
                                           double dLdE, double *out_gard_accumulator) {
    for (int i = 0; i < nbhs.size(); ++i) {
        Neighborhood &nbh = nbhs[i];
        AccumulateFPGardLonly(nbh, out_gard_accumulator, dLdE, nullptr);
    }
}

void PotMTP::AccumulateEFSCombinationGardLonly(Atoms &atom, AnyNeighborhoods &nbhs,
                                           double dLdE, Eigen::Matrix<double, -1, 3> &dLdF,
                                           Eigen::Matrix3d &dLdS, double *out_gard_accumulator) {
    for (int i = 0; i < nbhs.size(); ++i) {
        Neighborhood &nbh = nbhs[i];
        tmp_se_ders_wgts.resize(nbh.nbhcount_, 3);
        tmp_se_ders_wgts.setZero();

        for (int j = 0; j < nbh.nbhcount_; ++j) {
            for (int a = 0; a < 3; ++a) {
                tmp_se_ders_wgts(j, a) += dLdF(i, a);
                tmp_se_ders_wgts(j, a) -= dLdF(nbh.indices_[j], a);
            }
        }

        for (int j = 0; j < nbh.nbhcount_; ++j) {
            for (int a = 0; a < 3; ++a)
                for (int b = 0; b < 3; ++b)
                    tmp_se_ders_wgts(j, a) -= dLdS(a, b) * nbh.pos_[j][b];
        }
        AccumulateFPGardLonly(nbh, out_gard_accumulator, dLdE, &tmp_se_ders_wgts);
    }
}

void Potentials::PotMTP::AccumulateEFSCombinationGardLonly(Atoms &atom, AnyNeighborhoods &nbhs, double dLdE, Eigen::Matrix<double, -1, 3> &dLdF, Eigen::Matrix3d &dLdS, double *out_gard_accumulator, int coffset)
{
    for (int i = 0; i < nbhs.size(); ++i) {
        Neighborhood &nbh = nbhs[i];
        tmp_se_ders_wgts.resize(nbh.nbhcount_, 3);
        tmp_se_ders_wgts.setZero();

        for (int j = 0; j < nbh.nbhcount_; ++j) {
            for (int a = 0; a < 3; ++a) {
                tmp_se_ders_wgts(j, a) += dLdF(i, a);
                tmp_se_ders_wgts(j, a) -= dLdF(nbh.indices_[j], a);
            }
        }

        for (int j = 0; j < nbh.nbhcount_; ++j) {
            for (int a = 0; a < 3; ++a)
                for (int b = 0; b < 3; ++b)
                    tmp_se_ders_wgts(j, a) -= dLdS(a, b) * nbh.pos_[j][b];
        }
        AccumulateFPGardLonly(nbh, out_gard_accumulator, dLdE, &tmp_se_ders_wgts, coffset);
    }
}

void PotMTP::AccumulateEFSCombinationGardRonly(Atoms &atom, AnyNeighborhoods &nbhs,
                                           double dLdE, Eigen::Matrix<double, -1, 3> &dLdF,
                                           Eigen::Matrix3d &dLdS, double *out_gard_accumulator) {
    for (int i = 0; i < nbhs.size(); ++i) {
        Neighborhood &nbh = nbhs[i];
        tmp_se_ders_wgts.resize(nbh.nbhcount_, 3);
        tmp_se_ders_wgts.setZero();

        for (int j = 0; j < nbh.nbhcount_; ++j) {
            for (int a = 0; a < 3; ++a) {
                tmp_se_ders_wgts(j, a) += dLdF(i, a);
                tmp_se_ders_wgts(j, a) -= dLdF(nbh.indices_[j], a);
            }
        }

        for (int j = 0; j < nbh.nbhcount_; ++j) {
            for (int a = 0; a < 3; ++a)
                for (int b = 0; b < 3; ++b)
                    tmp_se_ders_wgts(j, a) -= dLdS(a, b) * nbh.pos_[j][b];
        }
        AccumulateFPGardRonly(nbh, out_gard_accumulator, dLdE, &tmp_se_ders_wgts);
    }
}

void Potentials::PotMTP::AccumulateEFSCombinationGardRonly(Atoms &atom, AnyNeighborhoods &nbhs, double dLdE, Eigen::Matrix<double, -1, 3> &dLdF, Eigen::Matrix3d &dLdS, double *out_gard_accumulator, int coffset)
{
    for (int i = 0; i < nbhs.size(); ++i) {
        Neighborhood &nbh = nbhs[i];
        tmp_se_ders_wgts.resize(nbh.nbhcount_, 3);
        tmp_se_ders_wgts.setZero();

        for (int j = 0; j < nbh.nbhcount_; ++j) {
            for (int a = 0; a < 3; ++a) {
                tmp_se_ders_wgts(j, a) += dLdF(i, a);
                tmp_se_ders_wgts(j, a) -= dLdF(nbh.indices_[j], a);
            }
        }

        for (int j = 0; j < nbh.nbhcount_; ++j) {
            for (int a = 0; a < 3; ++a)
                for (int b = 0; b < 3; ++b)
                    tmp_se_ders_wgts(j, a) -= dLdS(a, b) * nbh.pos_[j][b];
        }
        AccumulateFPGardRonly(nbh, out_gard_accumulator, dLdE, &tmp_se_ders_wgts, coffset);
    }
}

void PotMTP::compute_sensitive_matrix_with_center(Neighborhood &nbh, Eigen::MatrixXd &senmat, Eigen::VectorXd &desvec) {
    CalcSiteBasisDers(nbh);
    // der wrt self
    matrixx3dbuffer.resize(alpha_scalar_moments, 3);
    desvec.resize(alpha_scalar_moments);
    for (int bi = 0; bi < alpha_scalar_moments; ++bi)
        for (int i = 0; i < nbh.nbhcount_; ++i)
            for (int a = 0; a < 3; ++a)
                matrixx3dbuffer(bi, a) -= basis_ders(bi + 1, i, a);
    int leg = 3 * nbh.nbhcount_ + 3;
    senmat.resize(leg, leg);
    senmat.setZero();
    for (int bi = 0; bi < alpha_scalar_moments; ++bi)
    {
        for (int i = 0; i < nbh.nbhcount_; ++i)
        {
            for (int j = i; j < nbh.nbhcount_; ++j)
            {
                for (int a = 0; a < 3; ++a)
                {
                    for (int b = 0; b < 3; ++b)
                    {
                        senmat(3*i+a, 3*j+b) += basis_ders(bi + 1, i, a) * basis_ders(bi + 1, j, b);
                    }
                }
            }
        }
        for (int i = 0; i < nbh.nbhcount_; ++i)
        {
            for (int a = 0; a < 3; ++a)
            {
                for (int b = 0; b < 3; ++b)
                {
                    senmat(3 * i + a, 3 * nbh.nbhcount_ + b) += basis_ders(bi + 1, i, a) * matrixx3dbuffer(bi, b);
                }
            }
        }
        for (int a = 0; a < 3; ++a)
        {
            for (int b = a; b < 3; ++b)
            {
                senmat(3*nbh.nbhcount_ + a, 3* nbh.nbhcount_ + b) += matrixx3dbuffer(bi, a) * matrixx3dbuffer(bi, b);
            }
        }
    }
    // symm
    for (int i = 0; i < leg; ++i)
    {
        for (int j = i + 1; j < leg; ++j)
        {
            senmat(j, i) = senmat(i, j);
        }
    }
    for (int i = 0; i < alpha_scalar_moments; ++i)
    {
        desvec[i] = basis_vals[i + 1];
    }
}

void PotMTP::compute_sensitive_matrix(Neighborhood &nbh, Eigen::MatrixXd &senmat, Eigen::VectorXd &desvec) {
    CalcSiteBasisDers(nbh);
    // der wrt self
    desvec.resize(alpha_scalar_moments);
    int leg = 3 * nbh.nbhcount_;
    senmat.resize(leg, leg);
    senmat.setZero();
    for (int bi = 0; bi < alpha_scalar_moments; ++bi) {
        for (int i = 0; i < nbh.nbhcount_; ++i) {
            for (int j = i; j < nbh.nbhcount_; ++j) {
                for (int a = 0; a < 3; ++a) {
                    for (int b = 0; b < 3; ++b) {
                        senmat(3*i+a, 3*j+b) += basis_ders(bi + 1, i, a) * basis_ders(bi + 1, j, b);
                    }
                }
            }
        }
    }
    // symm
    for (int i = 0; i < leg; ++i)
    {
        for (int j = i + 1; j < leg; ++j)
        {
            senmat(j, i) = senmat(i, j);
        }
    }
    for (int i = 0; i < alpha_scalar_moments; ++i)
    {
        desvec[i] = basis_vals[i + 1];
    }
}

void PotMTP::corr_matrix(std::vector<Atoms> &atoms, 
                        Eigen::MatrixXd &corrcoeffs, 
                        Eigen::VectorXd &m,
                        Eigen::VectorXd &s,
                        int max_sample_size) {
    int dsize = alpha_scalar_moments;
    corrcoeffs.resize(dsize, dsize);
    corrcoeffs.setIdentity();
    int n = atoms.size();
    int ndesc = 0;
    int udesc = 0;
    for (int i = 0; i < n; ++i) {
        ndesc += atoms[i].size();
    }
    std::vector<int> select;
    if (ndesc > max_sample_size) {
        udesc = max_sample_size;
        select.assign(ndesc, 0);
        std::vector<int> g;
        g.resize(ndesc);
        std::iota(g.begin(), g.end(), 0);
        std::random_device rd;
        std::mt19937 mtd(rd());
        std::shuffle(g.begin(), g.end(), mtd);
        g.resize(max_sample_size);
        for (auto &it : g) {
            select[it] = 1;
        }
    } else {
        udesc = ndesc;
        select.assign(ndesc, 1);
    }
    Eigen::Matrix<double, -1, -1, Eigen::RowMajor> desc;
    desc.resize(udesc, dsize);
    std::vector<double> dbuffer;
    // desc
    int ct = 0;
    int rct = 0;
    for (int i = 0; i < n; ++i) {
_DEBUG_OUTPUT_(i);
        auto &atom = atoms[i];
        auto nbhs = Neighborhoods(atom, cutoff);
        for (int iat = 0; iat < atom.size(); ++iat) {
            if (select[ct]) {
                compute_descriptors(nbhs[iat], dbuffer);
                memcpy(desc.row(rct).data(), dbuffer.data() + 1, dsize * sizeof(double));
                ++rct;
            }
            ++ct;
        }
    }
_DEBUG_OUTPUT_(udesc);
_DEBUG_OUTPUT_(ndesc);
// compute mu and sigma for each col
_DEBUG_OUTPUT_("Compute M S");
    m.resize(dsize);
    s.resize(dsize);
    #pragma omp parallel for
    for (int i = 0; i < dsize; ++i) {
        m(i) = desc.col(i).mean();
        for (int j = 0; j < udesc; ++j) {
            desc(j, i) -= m(i);
        }
        s(i) = std::sqrt(desc.col(i).squaredNorm() / std::max(1, udesc - 1));
    }
_DEBUG_OUTPUT_("Compute COV");
    // now compute cov matrix
    for (int i = 0; i < dsize; ++i) {
_DEBUG_OUTPUT_(i);
	#pragma omp parallel for
        for (int j = i+1; j < dsize; ++j) {
            corrcoeffs(i, j) = desc.col(i).dot(desc.col(j)) / std::max(1, udesc - 1) / s(i) / s(j);
            corrcoeffs(j, i) = corrcoeffs(i, j);
        }
    }
}

void PotMTP::remove_scalar_by_moment(std::vector<int> &scalars) {
    std::set<int> ids;
    std::sort(scalars.begin(), scalars.end());
    int pt = 0;
    for (int i = 0; i < alpha_scalar_moments; ++i) {
        if (alpha_moment_mapping[i] == scalars[pt]) {
            ids.insert(i);
            ++pt;
        }
    }
    remove_scalar_by_id(ids);
}

void PotMTP::remove_scalar_by_id(std::set<int> &ids) {
    if (ids.empty()) return;
    if (*ids.begin() < 0 || *ids.rbegin() >= alpha_scalar_moments)
        GPERROR("Scalar id out of range");
    if (coeffs_handled_external)
        GPERROR("Cannot remove scalars from an externally owned coefficient buffer");
    int scalar_moments_new = alpha_scalar_moments - ids.size();
    int *alpha_moment_mapping_new = new int[scalar_moments_new];
    std::vector<double> moment_coeffs_new;
    int ct = 0;
    // adjust regression_coeffs, linear_coeffs and moment_coeffs
    for (int i = 0; i < alpha_scalar_moments; ++i) {
        if (ids.count(i)) {
            continue;
        }
        alpha_moment_mapping_new[ct] = alpha_moment_mapping[i];
        moment_coeffs_new.push_back(moment_coeffs[i]);
        ++ct;
    }
    alpha_scalar_moments = scalar_moments_new;
    delete[] alpha_moment_mapping;
    alpha_moment_mapping = alpha_moment_mapping_new;
    moment_coeff_size = scalar_moments_new;
    lin_coeff_size = scalar_moments_new + species_count;
    reg_coeff_size = radial_coeff_size + lin_coeff_size;
    // Save old pointers before allocating new coefficient buffer
    double *old_regression_coeffs = regression_coeffs;
    double *coeffs_new = new double[reg_coeff_size]{0};
    if (radial_coeff_size != 0)
        memcpy(coeffs_new, radial_coeffs, radial_coeff_size * sizeof(double));
    radial_coeffs = coeffs_new;
    species_coeffs = coeffs_new + radial_coeff_size;
    moment_coeffs = species_coeffs + species_count;
    for (int i = 0; i < species_count; ++i) {
        species_coeffs[i] = old_regression_coeffs[radial_coeff_size + i];
    }
    delete[] old_regression_coeffs;
    regression_coeffs = coeffs_new;
    if (!moment_coeffs_new.empty())
        memcpy(moment_coeffs, moment_coeffs_new.data(), moment_coeffs_new.size() * sizeof(double));
    update_layout();
}

BasisSubsetAnalysis PotMTP::analyze_basis(const std::vector<int> &scalar_ids) const {
    BasisSubsetAnalysis out;
    auto &stats = out.stats;
    out.keep.assign(alpha_moments_count, false);
    int previous = -1;
    for (int id : scalar_ids) {
        if (id <= previous || id < 0 || id >= alpha_scalar_moments)
            GPERROR("Basis analysis requires sorted unique scalar IDs in range");
        previous = id;
        out.keep[alpha_moment_mapping[id]] = true;
    }
    stats.scalars = scalar_ids.size();
    for (int i = alpha_index_times_count - 1; i >= 0; --i) {
        if (out.keep[alpha_index_times(i, 3)]) {
            out.keep[alpha_index_times(i, 0)] = true;
            out.keep[alpha_index_times(i, 1)] = true;
        }
    }
    // Existing readers require an unused basic/channel in species-only models.
    if (scalar_ids.empty() && alpha_index_basic_count > 0) out.keep[0] = true;
    out.moment_map.assign(alpha_moments_count, -1);
    for (int i = 0; i < alpha_moments_count; ++i)
        if (out.keep[i]) out.moment_map[i] = stats.moments++;
    out.channel_map.assign(radial_func_count, -1);
    for (int i = 0; i < alpha_index_basic_count; ++i) {
        if (!out.keep[i]) continue;
        ++stats.basics;
        out.channel_map[alpha_index_basic(i, 0)] = 0;
        stats.max_degree = std::max(stats.max_degree, int(alpha_index_basic.row(i).tail<3>().sum()));
    }
    for (int i = 0; i < alpha_index_times_count; ++i)
        if (out.keep[alpha_index_times(i, 3)]) ++stats.products;
    for (int mu = 0; mu < radial_func_count; ++mu) {
        if (out.channel_map[mu] < 0) continue;
        out.channel_map[mu] = stats.channels++;
        out.channels.push_back(mu);
    }
    stats.radial_bytes = std::size_t(species_count) * species_count * stats.channels * rb_size * sizeof(double);
    return out;
}

// Compact the computation DAG and radial channels without changing the function.
void PotMTP::remove_orphans() {
    if (coeffs_handled_external)
        GPERROR("Cannot compact an externally owned coefficient buffer");
    std::vector<int> scalar_ids(alpha_scalar_moments);
    std::iota(scalar_ids.begin(), scalar_ids.end(), 0);
    const auto analysis = analyze_basis(scalar_ids);
    const auto &keep = analysis.keep;
    const auto &moment_map = analysis.moment_map;
    const auto &channels = analysis.channels;
    const auto &channel_map = analysis.channel_map;
    const int moments = analysis.stats.moments, basics = analysis.stats.basics;
    const int products = analysis.stats.products, channel = analysis.stats.channels;
    const int radial_size = species_count * species_count * channel * rb_size;
    std::unique_ptr<double[]> coeffs(new double[radial_size + lin_coeff_size]);
    for (int pair = 0; pair < species_count * species_count; ++pair)
        for (int old : channels)
            std::copy_n(radial_coeffs + (pair * radial_func_count + old) * rb_size,
                        rb_size, coeffs.get() + (pair * channel + channel_map[old]) * rb_size);
    std::copy_n(linear_coeffs, lin_coeff_size, coeffs.get() + radial_size);
    Eigen::Matrix<int, -1, -1, Eigen::RowMajor> new_basic(basics, 4), new_times(products, 4);
    int row = 0;
    for (int i = 0; i < alpha_index_basic_count; ++i) {
        if (!keep[i]) continue;
        new_basic.row(row) = alpha_index_basic.row(i);
        new_basic(row++, 0) = channel_map[alpha_index_basic(i, 0)];
    }
    row = 0;
    for (int i = 0; i < alpha_index_times_count; ++i) {
        if (!keep[alpha_index_times(i, 3)]) continue;
        new_times.row(row) = alpha_index_times.row(i);
        for (int col : {0, 1, 3}) new_times(row, col) = moment_map[alpha_index_times(i, col)];
        ++row;
    }
    for (int i = 0; i < alpha_scalar_moments; ++i)
        alpha_moment_mapping[i] = moment_map[alpha_moment_mapping[i]];
    delete[] regression_coeffs;
    regression_coeffs = coeffs.release();
    radial_func_count = channel;
    radial_coeff_size = radial_size;
    reg_coeff_size = radial_size + lin_coeff_size;
    moment_coeff_size = alpha_scalar_moments;
    update_layout();
    alpha_moments_count = moments;
    alpha_index_basic_count = basics;
    alpha_index_times_count = products;
    alpha_index_basic = std::move(new_basic);
    alpha_index_times = std::move(new_times);
    delete[] nu_arr_list;
    nu_arr_list = new int[basics];
    for (int i = 0; i < basics; ++i)
        nu_arr_list[i] = alpha_index_basic.row(i).tail<3>().sum();
    max_alpha_index_basic = 0;
    max_mu = 0;
    init_basic();
    omp_inited = false;
    omp_buffers_list.clear();
    train_init();
    // Level boundaries refer to the old ordering and must not be serialized.
    max_level = 0;
    lev_scalar_nodes.clear(); lev_scalar_count.clear();
    lev_times_nodes.clear(); lev_times_count.clear();
}

void PotMTP::add_back_path_scalars_to_remove_id(std::set<int> &ids) {
    std::set<int> preserved_scalars;
    std::set<int> removed_scalars;
    std::set<int> scalars;
    for (int i = 0; i < alpha_scalar_moments; ++i) {
        if (ids.count(i)) {
            removed_scalars.insert(alpha_moment_mapping[i]);
        } else {
            preserved_scalars.insert(alpha_moment_mapping[i]);
        }
        scalars.insert(alpha_moment_mapping[i]);
    }
    // check times
    std::set<int> involved;
    std::set<int> processed;
    std::set<int> chains;
    bool state = true;
    while (state) {
        state = false;
        for (int i = 0; i < alpha_index_times_count; ++i) {
            int id1 = alpha_index_times(i, 0);
            int id2 = alpha_index_times(i, 1);
            int idx = alpha_index_times(i, 3);
            if (preserved_scalars.count(idx) || chains.count(idx)) {
                chains.insert(idx);
                if (chains.count(id1) == 0) {
                    chains.insert(id1);
                    state = true;
                }
                if (chains.count(id2) == 0) {
                    chains.insert(id2);
                    state = true;
                }
            }
        }
    }
    // add back
    for (auto &t : chains) {
        if (scalars.count(t)) {
            preserved_scalars.insert(t);
        }
    }
    ids.clear();
    for (int i = 0; i < alpha_scalar_moments; ++i) {
        int sca = alpha_moment_mapping[i];
        if (preserved_scalars.count(sca) == 0) {
            ids.insert(i);
        }
    }
}
