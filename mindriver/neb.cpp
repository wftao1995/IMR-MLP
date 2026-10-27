// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#include "neb.h"
#include "mindriver/strain_gradient.h"
#include "neb_evaluate.h"
#include "neb_symmetry.h"
#include "fire.h"
#include "sgd.h"
#include "cg.h"
#include "deplibs/LBFGSpp/include/LBFGS.h"
#include "active_learn/al_potential.h"
#include "tools/chem.h"
#include <fstream>
#include <limits>
#include <exception>

namespace {
void validate_periodic_basis(const Atoms &image) {
    if(!image.cell_.allFinite())throw NEBInvalidGeometry("NEB: non-finite cell");
    int rank=0;for(bool periodic:image.pbc_)rank+=periodic;
    if(rank==0)return;
    Eigen::MatrixXd rows(rank,3);int row=0;
    for(int i=0;i<3;++i)if(image.pbc_[i])rows.row(row++)=image.cell_.row(i);
    const Eigen::JacobiSVD<Eigen::MatrixXd> spectrum(rows);
    if(spectrum.info()!=Eigen::Success || !spectrum.singularValues().allFinite() ||
       !(spectrum.singularValues().minCoeff()>1e-12*spectrum.singularValues().maxCoeff()))
        throw NEBInvalidGeometry("NEB: periodic cell rows must be numerically independent");
}
bool singular_cartesian_fixed_neb(const NEBDriver &driver)
{
    return !driver.vcneb && !driver.images.empty()
        && !driver.images.front().is_fully_periodic()
        && std::abs(driver.refcell.determinant()) < 1e-12;
}
Eigen::RowVector3d nearest_cartesian_delta(const Eigen::RowVector3d &delta,
                                           const Eigen::Matrix3d &cell,
                                           const std::array<bool, 3> &pbc)
{
    Atoms pair;
    pair.cell_ = cell;
    pair.pbc_ = pbc;
    pair.cartesian = true;
    pair.pos_ = {Eigen::RowVector3d::Zero(), delta};
    pair.chemmap_ = {0, 0};
    // Existing exact periodic-image search subtracts only periodic lattice
    // vectors. The open-direction Cartesian residual stays in the result.
    return pair.get_rab(0, 1);
}
// Voigt ordering: [xx, xy, xz, yy, yz, zz] → full 3×3 symmetric matrix
void matrix_to_symm_voigt(const Eigen::Matrix3d &stm, Eigen::Ref<Eigen::RowVectorXd> row, int offset = 0)
{
    row[offset + 0] = stm(0, 0);
    row[offset + 1] = stm(0, 1);
    row[offset + 2] = stm(0, 2);
    row[offset + 3] = stm(1, 1);
    row[offset + 4] = stm(1, 2);
    row[offset + 5] = stm(2, 2);
}

// Interpolate two Voigt-encoded rows (no heap allocation for the temp)
Eigen::Matrix3d interpolate_symm_voigt(const Eigen::MatrixXd &x, int ia, int ib)
{
    Eigen::Matrix3d stm;
    stm << 0.5 * (x(ia, 0) + x(ib, 0)), 0.5 * (x(ia, 1) + x(ib, 1)), 0.5 * (x(ia, 2) + x(ib, 2)),
           0.5 * (x(ia, 1) + x(ib, 1)), 0.5 * (x(ia, 3) + x(ib, 3)), 0.5 * (x(ia, 4) + x(ib, 4)),
           0.5 * (x(ia, 2) + x(ib, 2)), 0.5 * (x(ia, 4) + x(ib, 4)), 0.5 * (x(ia, 5) + x(ib, 5));
    return stm;
}

Eigen::RowVector3d nearest_fractional_delta(const Eigen::RowVector3d &delta,
                                             const Eigen::Matrix3d &cell,
                                             const std::array<bool, 3> &pbc)
{
    if (!pbc[0] && !pbc[1] && !pbc[2]) return delta;
    Atoms pair;
    pair.cell_ = cell;
    pair.pbc_ = pbc;
    pair.cartesian = false;
    pair.pos_ = {Eigen::RowVector3d::Zero(), delta};
    pair.chemmap_ = {0, 0};
    // The caller selects the Cartesian metric; retain fractional output for
    // the path/VC coordinate codec after finding the exact periodic image.
    return (pair.get_rab(0, 1) * cell.inverse()).eval();
}
}

/**
 * @param atoms 
 * @bug restart incorrectly
 */

std::ofstream neb_debug_stream;

NEBDriver::NEBDriver(GPSettings settings) {
        InitSettings();
        ApplySettings(settings);
// PrintSettings();
        if (debug_level > 0) {
            neb_debug_stream = std::ofstream("debug_neb_stream.log");
        }
        pot_mindist.add_efs = true;
}

void NEBDriver::init(std::vector<Atoms>& atoms) {
    symmetry_.reset();
    images = atoms;
    subspace_active=vcneb && periodic_subspace && !images.empty() && !images.front().is_fully_periodic();
    if(subspace_active) {
        if(pressure!=0 || !external_stress.isZero() || !external_stress_str.empty())
            throw std::invalid_argument("Periodic-subspace NEB forbids three-dimensional pressure/external stress");
        if(apply_scal) throw std::invalid_argument("Periodic-subspace NEB uses scalar cellweight, not 3D cii scaling");
        int reference=iref>=0?iref:(iref==-1?int(images.size())-1:0);
        if(reference>=images.size() || images[reference].size()==0) throw std::invalid_argument("Invalid periodic-subspace reference image");
        Atoms reference_atom=images[reference];
        if(iref==-3) reference_atom.cell_=refcell;
        else if(iref==-2) reference_atom.cell_=.5*(images.front().cell_+images.back().cell_);
        subspace.init(reference_atom,recast_atomic_force);refcell=reference_atom.cell_;ivrefcell.setZero();
    }
    if(vcneb && !subspace_active && iref==-3) {
        Atoms reference;reference.cell_=refcell;validate_periodic_basis(reference);
        if(!std::isfinite(refcell.determinant()) || std::abs(refcell.determinant())<1e-12)
            throw NEBInvalidGeometry("NEB: invalid restart reference cell volume");
        ivrefcell=refcell.inverse();
        if(!ivrefcell.allFinite())throw NEBInvalidGeometry("NEB: restart reference cell inversion failed");
    }
    validate_images();
    for (auto &image : images)
        if (image.size() != 0 && !image.cartesian) image.to_cartesian();
    pot_mindist.add_efs = true;
    // select reference for vcneb
    if (vcneb && !subspace_active && symm_strain && iref > -3 && !preserve_complete_path) {
        refcell = images[0].cell_;
        ivrefcell = refcell.inverse();
        for (size_t i = 1; i < images.size(); ++i) {
            if (images[i].size() == 0) continue;
            Eigen::Matrix3d newcell = align_cell(images[i].cell_);
            images[i].set_cell(newcell);
        }
    }
    if (vcneb && !subspace_active) {
        if (iref == -3) {
            // do nothing
        } else if (iref == -2) {
            if (symm_strain) {
                Eigen::Matrix3d matC = get_align_C(images.back().cell_);
                Eigen::Matrix3d matI = Eigen::Matrix3d::Identity();
                Eigen::Matrix3d newC = (matC + matI) / 2.0;
                refcell = align_cell(newC);
            } else {
                refcell = 0.5 * (images[0].cell_ + images.back().cell_);
            }
        } else if (iref == -1) {
            refcell = images.back().cell_;
        } else if (iref >= 0 && iref < images.size()) {
            if (images[iref].size() == 0) {
                throw std::runtime_error("Image " + std::to_string(iref) + " is empty, cannot be used as reference cell.");
            }
            refcell = images[iref].cell_;
        } else {
            throw std::runtime_error("Invalid iref value: " + std::to_string(iref));
        }
        ivrefcell = refcell.inverse();
        if (symm_strain && !preserve_complete_path) {
            for (size_t i = 1; i < images.size(); ++i) {
                if (images[i].size() == 0) continue;
                Eigen::Matrix3d newcell = align_cell(images[i].cell_);
                images[i].set_cell(newcell);
            }
        }
    } else if(!subspace_active) {
        refcell = images[0].cell_;
        // Fixed-cell Cartesian paths never consume a lattice inverse.
        if (singular_cartesian_fixed_neb(*this)) ivrefcell.setZero();
        else ivrefcell = refcell.inverse();
    }
    iter = 0;
    force_evaluations = 0;
    image_force_evaluations = 0;
    accepted_steps = rejected_steps = 0;
    last_atomic_move = last_cell_change = 0;
    last_fallback = "none";
    optimizer_iterations = 0;
    converged = false;
    stop_reason = StopReason::not_run;
    max_perp_force = 0;
    max_spring_force = 0;
    max_neb_force = 0;
    saddle_g_proj = 0;
    climbing_image = -1;
    climbing_active = false;
    iclimb_ = -1;
    needs_image_adjust = false;
    al_abort_ = false;
    found_state_site = false;
    iclimbs.clear();
    ivallys.clear();
    if(subspace_active) vcoffset=subspace.cell_dofs();
    else if (vcneb && !symm_strain) vcoffset = 9;
    else if (vcneb && symm_strain) vcoffset = 6;
    else vcoffset = 0;
    if (sel_dyn) {
        if (images[0].sel_dynpos_.size() == 0) {
            std::cerr << "Warning: No positional selective dynamics info given" << std::endl;
            images[0].sel_dynpos_.resize(images[0].size(), Eigen::RowVector3d::Ones());
        }
        int ct = 0;
        for (char &c : cmask) {
            int i = ct / 3;
            int j = ct % 3;
            if (c == '0') {
                images[0].sel_dyncell_(i, j) = 0;
            } else {
                images[0].sel_dyncell_(i, j) = 1;
            }
            ct += 1;
        }
    }
    spconsts.assign(images.size() - 1, spring);
    fill_path();
    if (init_method == "idpp") {
        Message("Running IDPP interpolation");
        idpp_interpolate();
    }
    initialize_symmetry();
    auto a = images[0];
    auto b = images.back();
    std::vector<Atoms> endpoints{a,b};
    if (efs_evaluator) efs_evaluator(endpoints,0,2);
    else evaluate_neb_images(*pot,endpoints,0,2,omp_mode);
    a=std::move(endpoints[0]); b=std::move(endpoints[1]);
    image_force_evaluations += 2;
    if (mindist) {
        pot_mindist.getEFS(a);
        pot_mindist.getEFS(b);
    }
    auto store_endpoint_efs = [](Atoms &image, const Atoms &evaluated) {
        image.energy_ = evaluated.energy_;
        image.forces_ = evaluated.forces_;
        image.stress_ = evaluated.stress_;
        image.site_energies_ = evaluated.site_energies_;
        image.setEFSflag();
    };
    store_endpoint_efs(images.front(), a);
    store_endpoint_efs(images.back(), b);
    if (vcneb) {
        images[0].energy_ += pressure / eVA3_to_GPa * images[0].get_volume();
        images.back().energy_ += pressure / eVA3_to_GPa * images.back().get_volume();
    }
}

void NEBDriver::validate_images(bool require_potential) const {
    if (omp_mode!="images" && omp_mode!="atoms") throw std::invalid_argument("NEB: omp_mode must be images or atoms");
    if (require_potential && !pot) throw std::invalid_argument("NEB: potential is null");
    if (images.size() < 3)
        throw std::invalid_argument("NEB: at least 3 images (including endpoints) are required");
    if (images.front().size() == 0 || images.back().size() == 0)
        throw std::invalid_argument("NEB: endpoint images must be non-empty");
    const int n = images.front().size();
    const auto &types = images.front().chemmap_;
    const bool cell_free = !vcneb && !images.front().is_fully_periodic()
        && std::abs(images.front().cell_.determinant()) < 1e-12;
    for (const auto &image : images) {
        if (image.size() && !image.cell_.allFinite())
            throw NEBInvalidGeometry("NEB: initial cell must be finite");
        if(image.size())validate_periodic_basis(image);
        for (const auto &position : image.pos_)
            if (!position.allFinite()) throw NEBInvalidGeometry("NEB: initial positions must be finite");
        if (image.size() != 0 && image.size() != n)
            throw std::invalid_argument("NEB: all supplied images must have the same atom count");
        if (image.size() != 0 && image.chemmap_ != types)
            throw std::invalid_argument("NEB: all supplied images must have identical atom ordering/species");
        if (image.size() != 0 && image.pbc_ != images.front().pbc_)
            throw std::invalid_argument("NEB: all supplied images must have identical periodic boundary flags");
        if(image.size()!=0 && subspace_active) {
            if(!image.cartesian && !image.cell_.fullPivLu().isInvertible())
                throw std::invalid_argument("NEB: singular periodic-subspace paths require Cartesian positions");
            validate_cell(image.cell_);
        } else if (image.size() != 0 && cell_free) {
            if (!image.cartesian)
                throw std::invalid_argument("NEB: singular fixed-cell paths require Cartesian positions");
            if (!image.cell_.allFinite() ||
                (image.cell_.array() != images.front().cell_.array()).any())
                throw std::invalid_argument("NEB: singular paths require one fixed cell for every image");
        } else if (image.size() != 0 && (!std::isfinite(image.cell_.determinant()) || std::abs(image.cell_.determinant()) < 1e-12)) {
            throw NEBInvalidGeometry("NEB: image cell has a non-finite volume or is singular");
        }
        if (vcneb && !subspace_active && image.size()) {
            Atoms full_basis;full_basis.cell_=image.cell_;validate_periodic_basis(full_basis);
            Eigen::Matrix3d deformation = images.front().cell_.inverse() * image.cell_;
            if(!deformation.allFinite())throw NEBInvalidGeometry("NEB: initial cell inversion failed");
            Eigen::JacobiSVD<Eigen::Matrix3d> svd(deformation);
            if (!(deformation.determinant() > 0) ||
                !(svd.singularValues().minCoeff() > 1e-12 * svd.singularValues().maxCoeff()))
                throw NEBInvalidGeometry("NEB: initial cells must have consistent handedness and independent rows");
        }
    }
    if (cell_free) {
        // Validate the active periodic basis using the same exact-image API.
        nearest_cartesian_delta(Eigen::RowVector3d::Zero(), images.front().cell_, images.front().pbc_);
    }
    if (sel_dyn && cmask.size() != 9)
        throw std::invalid_argument("NEB: cmask must contain exactly 9 entries");
    if (nblock <= 0) throw std::invalid_argument("NEB: nblocks must be positive");
    if (image_adjust_interval <= 0)
        throw std::invalid_argument("NEB: img_adjust_interval must be positive");
    if (spring < 0 || spring_min < 0 || spring_max < spring_min)
        throw std::invalid_argument("NEB: invalid spring constants");
    if (!(neb_ftol > 0) || !(cell_weight > 0) || !std::isfinite(neb_ftol) || !std::isfinite(cell_weight))
        throw std::invalid_argument("NEB: neb_ftol and cellweight must be positive");
    if (solver_policy != "stable-v1" && solver_policy != "legacy")
        throw std::invalid_argument("NEB: solver_policy must be stable-v1 or legacy");
    if (!(max_cell_strain > 0) || !std::isfinite(max_cell_strain) || max_force_evaluations < 0)
        throw std::invalid_argument("NEB: invalid cell step or evaluation budget");
    if (min_image_count < 3 || max_image_count < min_image_count)
        throw std::invalid_argument("NEB: image-count limits must satisfy 3 <= min_images <= max_images");
    if (!(image_insert_thresh > 0) || !(image_remove_thresh >= 0))
        throw std::invalid_argument("NEB: image spacing thresholds must be non-negative");
    if (!(idpp_fmax > 0) || idpp_max_steps <= 0)
        throw std::invalid_argument("NEB: IDPP tolerance and step budget must be positive");
    if (init_method != "linear" && init_method != "idpp")
        throw std::invalid_argument("NEB: init_method must be 'linear' or 'idpp'");
    if (tangent_type < 1 || tangent_type > 3 || sproj_type < 1 || sproj_type > 3)
        throw std::invalid_argument("NEB: ttype and stype must be in [1, 3]");
    if (climb_mode != "highest" && climb_mode != "all_local")
        throw std::invalid_argument("NEB: climb_mode must be 'highest' or 'all_local'");
}

void NEBDriver::validate_cell(const Eigen::Matrix3d &cell) const {
    if (!cell.allFinite()) throw NEBInvalidGeometry("NEB: non-finite cell");
    if (!vcneb) return;
    if (subspace_active) {
        Eigen::MatrixXd rows(subspace.rank(),3);
        for(int i=0;i<subspace.rank();++i)rows.row(i)=cell.row(subspace.axes()[i]);
        if(subspace.rank()) {
            Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> spectrum(rows*rows.transpose());
            if(spectrum.info()!=Eigen::Success ||
               !(spectrum.eigenvalues().minCoeff()>1e-12*spectrum.eigenvalues().maxCoeff()))
                throw NEBInvalidGeometry("NEB: dependent or ill-conditioned periodic cell rows");
        }
        try { (void)subspace.encode_cell(cell); }
        catch(const std::invalid_argument &e) {throw NEBInvalidGeometry(e.what());}
        return;
    }
    const Eigen::Matrix3d deformation = ivrefcell * cell;
    if(!deformation.allFinite())throw NEBInvalidGeometry("NEB: non-finite relative cell deformation");
    const Eigen::JacobiSVD<Eigen::Matrix3d> svd(deformation);
    if (svd.info()!=Eigen::Success || !svd.singularValues().allFinite() || !(deformation.determinant() > 0) ||
        !(svd.singularValues().minCoeff() > 1e-12 * svd.singularValues().maxCoeff()))
        throw NEBInvalidGeometry("NEB: singular, ill-conditioned or inverted cell trial");
}

std::vector<double> NEBDriver::fire_masses() const {
    std::vector<double> result((images.size()-2)*(vcoffset+npx),1.);
    const auto explicit_mass=images.front().local_attribs_d_.find("masses");
    if(explicit_mass!=images.front().local_attribs_d_.end() && explicit_mass->second.size()!=images.front().size())
        throw std::invalid_argument("NEB: explicit masses must have one value per atom");
    std::map<int,double> species_mass;
    if(iobj) {
        species_mass=iobj->atomradii;
        if(species_mass.empty())for(const auto &[type,name]:iobj->getmap()) {
            const auto number=Chemicals::atomic_numbers.find(name);
            if(number!=Chemicals::atomic_numbers.end())species_mass[type]=Chemicals::atomic_masses[number->second];
        }
    }
    int k=0;
    for(int i=1;i<images.size()-1;++i) {
        for(int j=0;j<vcoffset;++j)result[k++]=cell_weight;
        for(int a=0;a<images[i].size();++a) {
            double mass=0;
            if(explicit_mass!=images.front().local_attribs_d_.end() && explicit_mass->second.size()==images.front().size())
                mass=explicit_mass->second[a];
            else if(species_mass.count(images.front().chemmap_[a]))mass=species_mass.at(images.front().chemmap_[a]);
            if(!(mass>0) || !std::isfinite(mass))
                throw std::invalid_argument("NEB: assign_mass requires positive explicit atomic masses or a chemistry map");
            for(int j=0;j<3;++j)result[k++]=mass;
        }
    }
    return result;
}

Eigen::RowVector3d NEBDriver::encode_position(const Atoms &image, int atom_index) const {
    if(subspace_active) return subspace.encode_position(image.pos_[atom_index],image.cell_);
    if (!vcneb) return image.pos_[atom_index];
    Eigen::RowVector3d frac = image.pos_[atom_index] * image.cell_.inverse();
    return recast_atomic_force ? frac * refcell : frac;
}

Eigen::RowVector3d NEBDriver::decode_position(const Eigen::RowVector3d &q,
                                               const Eigen::Matrix3d &cell) const {
    if(subspace_active) return subspace.decode_position(q,cell);
    if (!vcneb) return q;
    if (recast_atomic_force) return q * ivrefcell * cell;
    return q * cell;
}

double NEBDriver::distance(Atoms &a, Atoms &b, double offset) {
    double d = 0;
    if(subspace_active) {
        d=cell_weight*(subspace.encode_cell(b.cell_)-subspace.encode_cell(a.cell_)).squaredNorm();
        for(int i=0;i<a.size();++i) d+=subspace.displacement(encode_position(a,i),encode_position(b,i),refcell,check_mic).squaredNorm();
        return std::sqrt(d)+offset;
    }
    if (singular_cartesian_fixed_neb(*this)) {
        for (int i = 0; i < a.size(); ++i) {
            Eigen::RowVector3d delta = b.pos_[i] - a.pos_[i];
            if (check_mic) delta = nearest_cartesian_delta(delta, refcell, a.pbc_);
            d += delta.squaredNorm();
        }
        return std::sqrt(d) + offset;
    }
    Eigen::Matrix3d stma, stmb;
    if (vcneb) {
        stma = get_strain(a.cell_);
        stmb = get_strain(b.cell_);
        const Eigen::Matrix3d delta = stma - stmb;
        if (symm_strain && apply_scal) {
            // ciival uses xx yy zz yz xz xy; a symmetric tensor has
            // two equal off-diagonal entries in its Frobenius metric.
            d += ciival[0]*delta(0,0)*delta(0,0)
               + ciival[1]*delta(1,1)*delta(1,1)
               + ciival[2]*delta(2,2)*delta(2,2)
               + 2*ciival[3]*delta(1,2)*delta(1,2)
               + 2*ciival[4]*delta(0,2)*delta(0,2)
               + 2*ciival[5]*delta(0,1)*delta(0,1);
        } else {
            d += cell_weight * delta.squaredNorm();
        }
    }

    Atoms ra = a;
    Atoms rb = b;
    ra.to_fractional();
    rb.to_fractional();
    ra.cell_ = refcell;
    rb.cell_ = refcell;
    if (check_mic)
        for (int i = 0; i < ra.size(); ++i)
            rb.pos_[i] = ra.pos_[i] + nearest_fractional_delta(rb.pos_[i] - ra.pos_[i], refcell, a.pbc_);
    if (!vcneb || recast_atomic_force) {
        ra.to_cartesian();
        rb.to_cartesian();
    }
    for (int i = 0; i < ra.size(); ++i) {
        d += (ra.pos_[i] - rb.pos_[i]).squaredNorm();
    }
    d = sqrt(d);
    return d + offset;
}
// in case add a image on the fly, or give incomplete set initially
void NEBDriver::check_mic_images() {
    if (!check_mic) return;
    if(subspace_active) {
        int previous=0;
        for(int next=1;next<images.size();++next) {
            if(images[next].size()==0) continue;
            for(int i=0;i<images[next].size();++i) {
                const auto a=encode_position(images[previous],i),b=encode_position(images[next],i);
                images[next].pos_[i]=decode_position(a+subspace.displacement(a,b,refcell),images[next].cell_);
            }
            previous=next;
        }
        return;
    }
    if (singular_cartesian_fixed_neb(*this)) {
        if (images.front().is_nonperiodic()) return;
        int previous = 0;
        for (int next = 1; next < images.size(); ++next) {
            if (images[next].size() == 0) continue;
            for (int atom = 0; atom < images[next].size(); ++atom) {
                const Eigen::RowVector3d delta = images[next].pos_[atom] - images[previous].pos_[atom];
                images[next].pos_[atom] = images[previous].pos_[atom]
                    + nearest_cartesian_delta(delta, refcell, images[next].pbc_);
            }
            previous = next;
        }
        return;
    }
    int pst = 0;
    for (int i = 1; i < (int)images.size(); ++i) {
        if (images[i].size() == 0) continue;
        int ped = i;
        std::vector<Eigen::RowVector3d> ra, rb;
        images[pst].get_dir_pos(ra);
        images[ped].get_dir_pos(rb);
        bool reg = false;
        for (int ia = 0; ia < (int)ra.size(); ++ia) {
            const Eigen::RowVector3d old = rb[ia];
            rb[ia] = ra[ia] + nearest_fractional_delta(rb[ia] - ra[ia], refcell, images[pst].pbc_);
            reg = reg || !rb[ia].isApprox(old, 1e-14);
        }
        if (reg) {
            images[ped].set_dir_pos(rb);
        }
        pst = i;
    }
}

void NEBDriver::initialize_scaling() {
    scalval = Eigen::VectorXd::Ones(vcoffset + npx);
    if (vcneb) {
        scalval.head(vcoffset).setConstant(std::sqrt(cell_weight));
    }
    if (vcneb && !subspace_active && symm_strain && apply_scal) {
        if (!ciistr.empty()) {
            std::vector<std::string> vstr;
            auto values=ciistr;std::replace(values.begin(),values.end(),',',' ');
            str_split(values, vstr);
            if (vstr.size() < 6) {
                throw std::runtime_error("ciival requires 6 comma-separated values (xx yy zz yz xz xy), got " + std::to_string(vstr.size()));
            }
            // xx yy zz yz xz xy
            for (int i = 0; i < 6; ++i) {
                ciival[i] = std::stod(vstr[i]);
                if (!(ciival[i] > 0))
                    throw std::invalid_argument("NEB: ciival entries must be positive");
            }
        }
        scalval[0] = std::sqrt(ciival[0]);
        scalval[3] = std::sqrt(ciival[1]);
        scalval[5] = std::sqrt(ciival[2]);
        scalval[1] = std::sqrt(ciival[5]); // xy
        scalval[2] = std::sqrt(ciival[4]); // xz
        scalval[4] = std::sqrt(ciival[3]); // yz
    }
    if (vcneb && !subspace_active && symm_strain) {
        // q = sqrt(w) * C; symmetric shear appears twice in ||C||_F.
        // Use the same coordinate metric for tangents, effective forces,
        // optimizer steps and the published reaction-coordinate distance.
        for (int j : {1, 2, 4}) scalval[j] *= std::sqrt(2.0);
    }

}

void NEBDriver::fill_path() {
    if(subspace_active) vcoffset=subspace.cell_dofs();
    else if (vcneb && !symm_strain) vcoffset = 9;
    else if (vcneb && symm_strain) vcoffset = 6;
    else vcoffset = 0;
    // find empty region
    int nimg = images.size();
    natom = images[0].size();
    npx = natom * 3;
    x.resize(images.size(), vcoffset + npx);
    g.resize(images.size(), vcoffset + npx);
    if(subspace_active) g.setZero();
    assert(!(images[0].size() == 0));
    assert(!(images.back().size() == 0));
    // init
    // check MIC
    check_mic_images();
    // img to x
    for (int i = 0; i < images.size(); ++i) {
        if (images[i].size() == 0) continue;
        if(subspace_active) {
            x.row(i).head(vcoffset)=subspace.encode_cell(images[i].cell_).transpose();
            for(int ii=0;ii<images[i].size();++ii) x.row(i).segment<3>(vcoffset+3*ii)=encode_position(images[i],ii);
            continue;
        }
        if (vcneb) {
            auto stm = get_strain(images[i].cell_);
            if (symm_strain) {
                matrix_to_symm_voigt(stm, x.row(i));
            } else {
                for (int ia = 0; ia < 3; ++ia) {
                    for (int ib = 0; ib < 3; ++ib) {
                        x(i, 3*ia + ib) = stm(ia, ib);
                    }
                }
            }
            images[i].to_fractional();
        }
        Eigen::Matrix3d cv = Eigen::Matrix3d::Identity();
        if (vcneb && recast_atomic_force) {
            cv = refcell;
        }
        for (int ii = 0; ii < images[i].size(); ++ii) {
            Eigen::RowVector3d rp = images[i].pos_[ii] * cv;
            for (int jj = 0; jj < 3; ++jj) 
                x(i, vcoffset + 3*ii + jj) = rp(jj);
        }
        if (vcneb)
            images[i].to_cartesian();
    }

    int pst = 0;
    int ped = -1;
    // insert missing images
    int pst0 = 1;
    while (true) {
        // find first one 
        pst = -1;
        ped = -1;
        for (int i = pst0; i < nimg - 1; ++i) {
            if (images[i].size() == 0) {
                pst = i;
                break;
            }
        }
        if (pst == -1) break;
        for (int i = pst + 1; i < nimg - 1; ++i) {
            if (images[i].size() == 0) {
                ped = i;
                while(images[ped].size() == 0) {
                    ++ped;
                }
                --ped;
                break;
            }
        }
        // only one empty
        if (ped == -1) {
            ped = pst + 1;
            --pst;
        } else {
            --pst;
            ++ped;
        }
        // insert images, check mic between first and last
        int ninsert = ped - pst - 1;
        Eigen::VectorXd xa = x.row(pst);
        Eigen::VectorXd xb = x.row(ped);
        double f = 1.0 / (ninsert + 1);
        Eigen::VectorXd xx;
        for (int i = 1; i <= ninsert; ++i) {
            xx = (1.0 - i * f) * xa + i * f * xb;
            x.row(pst + i) = xx;
            images[pst + i].resize(images[0].size());
            images[pst + i].chemmap_ = images[0].chemmap_;
            images[pst + i].pbc_ = images[0].pbc_;
            images[pst + i].cell_ = images[0].cell_;
        }
        pst0 = ped;
    }
    X2LatPos(x);
    if (sel_dyn) {
        for (int i = 1; i < images.size(); ++i) {
            images[i].sel_dyncell_ = images[0].sel_dyncell_;
            images[i].sel_dynpos_ = images[0].sel_dynpos_;
        }
    }
    initialize_scaling();
    Leq = ((x.row(images.size() - 1) - x.row(0)).array() * scalval.transpose().array()).matrix().norm()
        / (images.size() - 1.);
    Message("DONE FILL PATH");
}

void NEBDriver::idpp_interpolate() {
    int nimgs = images.size();
    int natoms = images[0].size();
    if (nimgs < 3 || natoms == 0) return;
    int ninterior = nimgs - 2;

    // 1. Compute distance matrices for endpoint images
    auto compute_distmat = [&](const Atoms &a) {
        Eigen::MatrixXd dmat(natoms, natoms);
        dmat.setZero();
        for (int i = 0; i < natoms; ++i) {
            for (int j = i + 1; j < natoms; ++j) {
                Eigen::RowVector3d rij = a.pos_[i] - a.pos_[j];
                if (check_mic) {
                    if (subspace_active || singular_cartesian_fixed_neb(*this)) {
                        rij = nearest_cartesian_delta(rij, a.cell_, a.pbc_);
                    } else {
                        Eigen::RowVector3d rij_frac = nearest_fractional_delta(
                            rij * a.cell_.inverse(), a.cell_, a.pbc_);
                        rij = rij_frac * a.cell_;
                    }
                }
                double dist = rij.norm();
                dmat(i, j) = dist;
                dmat(j, i) = dist;
            }
        }
        return dmat;
    };

    Eigen::MatrixXd d1 = compute_distmat(images[0]);
    Eigen::MatrixXd d2 = compute_distmat(images.back());

    // 2. Compute target distance matrices for each interior image
    std::vector<Eigen::MatrixXd> targets(nimgs);
    targets[0] = d1;
    targets[nimgs - 1] = d2;
    double inv_n = 1.0 / (nimgs - 1);
    for (int img = 1; img < nimgs - 1; ++img) {
        double frac = img * inv_n;
        targets[img] = d1 + frac * (d2 - d1);
    }

    // 3. Build optimization variable x: concatenated positions of interior images
    int ndim = 3 * natoms;
    int nvars = ninterior * ndim;
    Eigen::VectorXd x_idpp(nvars);
    // Normalize each atomic coordinate to a Cartesian length scale. Without
    // this, a fractional-coordinate FIRE step grows with the cell size and
    // can escape to the spurious zero-energy limit of IDPP at infinite distance.
    std::vector<Eigen::RowVector3d> coordinate_scale(nimgs, Eigen::RowVector3d::Ones());
    if (vcneb) {
        for (int img = 1; img <= ninterior; ++img) {
            Eigen::Matrix3d jacobian = images[img].cell_;
            if(subspace_active) jacobian=subspace.atomic_jacobian(images[img].cell_);
            else if (recast_atomic_force) jacobian = ivrefcell * jacobian;
            for (int a = 0; a < 3; ++a) coordinate_scale[img](a) = jacobian.row(a).norm();
        }
    }
    for (int img = 1; img <= ninterior; ++img) {
        int offset = (img - 1) * ndim;
        for (int ia = 0; ia < natoms; ++ia) {
            for (int a = 0; a < 3; ++a) {
                x_idpp[offset + 3 * ia + a] = x(img, vcoffset + 3 * ia + a) * coordinate_scale[img](a);
            }
        }
    }

    // 4. Define IDPP functor (energy + gradient)
    struct IDPPFunctor {
        int natoms, ndim, ninterior, nimgs;
        const std::vector<Atoms> &images;
        const std::vector<Eigen::MatrixXd> &targets;
        bool check_mic;
        const NEBDriver &driver;
        const std::vector<Eigen::RowVector3d> &coordinate_scale;

        double operator()(const Eigen::VectorXd &x, Eigen::VectorXd &g) {
            g.setZero();
            double total_e = 0;
            for (int img = 1; img <= ninterior; ++img) {
                int offset = (img - 1) * ndim;
                const auto &target = targets[img];
                const auto &atom = images[img];

                // Build distance matrix and displacement tensor
                Eigen::MatrixXd d(natoms, natoms);
                d.setZero();
                // D[i][j] = r_j - r_i, shape (natoms, natoms, 3)
                std::vector<Eigen::RowVector3d> pos(natoms);
                for (int ia = 0; ia < natoms; ++ia)
                    for (int a = 0; a < 3; ++a)
                        pos[ia](a) = x[offset + 3 * ia + a] / coordinate_scale[img](a);
                for (auto &p : pos) p = driver.decode_position(p, atom.cell_);

                for (int i = 0; i < natoms; ++i) {
                    d(i, i) = 1.0; // protect against division by zero on diagonal
                    for (int j = i + 1; j < natoms; ++j) {
                        Eigen::RowVector3d rij = pos[i] - pos[j];
                        if (check_mic) {
                            if (driver.subspace_active || singular_cartesian_fixed_neb(driver)) {
                                rij = nearest_cartesian_delta(rij, atom.cell_, atom.pbc_);
                            } else {
                                Eigen::RowVector3d rij_frac = nearest_fractional_delta(
                                    rij * atom.cell_.inverse(), atom.cell_, atom.pbc_);
                                rij = rij_frac * atom.cell_;
                            }
                        }
                        double dist = rij.norm();
                        if (dist < 1e-12)
                            throw std::runtime_error("IDPP: coincident atoms; supply noncolliding interior images");
                        d(i, j) = dist;
                        d(j, i) = dist;
                    }
                }

                // Energy: E = 0.5 * Σ (d_ij - t_ij)^2 / d_ij^4
                double e_img = 0;
                for (int i = 0; i < natoms; ++i) {
                    for (int j = 0; j < natoms; ++j) {
                        if (i == j) continue;
                        double dd = d(i, j) - target(i, j);
                        double d4 = d(i, j) * d(i, j);
                        d4 *= d4; // d^4
                        e_img += dd * dd / d4;
                    }
                }
                e_img *= 0.5;
                total_e += e_img;

                // Gradient: dE/dr_k = 2 * Σ_{i≠k} dd_ik * (1 - 2*dd_ik/d_ik) / d_ik^5 * (r_k - r_i)
                for (int k = 0; k < natoms; ++k) {
                    Eigen::RowVector3d gradk = Eigen::RowVector3d::Zero();
                    for (int i = 0; i < natoms; ++i) {
                        if (i == k) continue;
                        double dik = d(k, i);
                        double dd = dik - target(k, i);
                        Eigen::RowVector3d Dki = pos[k] - pos[i];
                        if (check_mic) {
                            if (driver.subspace_active || singular_cartesian_fixed_neb(driver)) {
                                Dki = nearest_cartesian_delta(Dki, atom.cell_, atom.pbc_);
                            } else {
                                Eigen::RowVector3d Dki_frac = nearest_fractional_delta(
                                    Dki * atom.cell_.inverse(), atom.cell_, atom.pbc_);
                                Dki = Dki_frac * atom.cell_;
                            }
                        }
                        double d5 = dik * dik;
                        d5 = d5 * d5 * dik; // d^5
                        double coeff = dd * (1.0 - 2.0 * dd / dik) / d5;
                        gradk += 2.0 * coeff * Dki;
                    }
                    if (driver.vcneb) {
                        Eigen::Matrix3d jacobian = atom.cell_;
                        if(driver.subspace_active) jacobian=driver.subspace.atomic_jacobian(atom.cell_);
                        else if (driver.recast_atomic_force) jacobian = driver.ivrefcell * jacobian;
                        gradk = gradk * jacobian.transpose();
                    }
                    for (int a = 0; a < 3; ++a)
                        g[offset + 3 * k + a] = gradk(a) *
                            (driver.sel_dyn ? images[0].sel_dynpos_[k](a) : 1.0) / coordinate_scale[img](a);
                }
            }
            return total_e;
        }
    };

    IDPPFunctor idpp_functor{natoms, ndim, ninterior, nimgs, images, targets, check_mic, *this, coordinate_scale};

    // 5. Optimize with FIRE
    GPSettings empty_settings;
    SimpleFireF fire(empty_settings);
    fire.gtol = idpp_fmax;
    fire.max_iter = idpp_max_steps;
    fire.ts_init = 0.1;
    fire.max_step = 0.5;
    double f_idpp = 0;
    int nsteps = fire.minimize(idpp_functor, x_idpp, f_idpp);
    {   // Compute final max force with a SEPARATE gradient buffer.
        // idpp_functor calls g.setZero() first, which would zero x if aliased.
        Eigen::VectorXd g_dummy = x_idpp;
        double final_e = idpp_functor(x_idpp, g_dummy);
        std::cout << fmt::format("IDPP: {} after {} steps, final energy = {:.6f}, max gradient = {:.6f}\n",
                                 g_dummy.cwiseAbs().maxCoeff() < idpp_fmax ? "converged" : "max iterations",
                                 nsteps, final_e, g_dummy.cwiseAbs().maxCoeff());
    }

    // 6. Write back optimized positions to images
    for (int img = 1; img <= ninterior; ++img) {
        int offset = (img - 1) * ndim;
        for (int ia = 0; ia < natoms; ++ia) {
            Eigen::RowVector3d q = x_idpp.segment<3>(offset + 3 * ia);
            q = q.cwiseQuotient(coordinate_scale[img]);
            images[img].pos_[ia] = decode_position(q, images[img].cell_);
        }
    }

    // 7. Re-encode positions into x matrix (since positions changed)
    for (int i = 1; i < nimgs - 1; ++i) {
        for (int ii = 0; ii < natoms; ++ii) {
            Eigen::RowVector3d rp = encode_position(images[i], ii);
            for (int jj = 0; jj < 3; ++jj)
                x(i, vcoffset + 3 * ii + jj) = rp(jj);
        }
    }
}

void NEBDriver::eval() {
    for (const auto &image : images) {validate_periodic_basis(image);validate_cell(image.cell_);}
    ++force_evaluations;
    if (stable_evaluation_) ++iter;
    image_force_evaluations += images.size() - 2;
    g.setZero();
    if (efs_evaluator) efs_evaluator(images,1,images.size()-1);
    else evaluate_neb_images(*pot,images,1,images.size()-1,omp_mode);
    std::exception_ptr calculator_error;
    #pragma omp parallel 
{
    #pragma omp for
    for (int i = 1; i < images.size() - 1; ++i) {
        // int threadid = omp_get_thread_num();
        auto xold = images[i].pos_;
        try {
        images[i].wrap();
        if (mindist) {
            pot_mindist.getEFS(images[i]);
        }
        images[i].pos_ = xold;
        // fill g
        if(subspace_active) g.row(i).head(vcoffset)=-subspace.cell_gradient(images[i].cell_,images[i].stress_).transpose();
        else if (vcneb) {
            double pscal = 
            pressure / eVA3_to_GPa * images[i].get_volume();
            Eigen::Matrix3d ext = external_stress / eVA3_to_GPa * images[i].get_volume();
            // Convert energy to U + PV + Se (extstress * eps, element wise product)
            images[i].energy_ += pscal;
            Eigen::Matrix3d curr_stress = images[i].stress_ - pscal * Eigen::Matrix3d::Identity() - ext;
            // Eigen::Matrix3d gstress = refcell.transpose() * images[i].cell_.inverse().transpose() * curr_stress;
            Eigen::Matrix3d gstress = images[i].cell_.inverse().transpose() * curr_stress;
            if (symm_strain) {
                g.row(i).head<6>() = SaddleCoordinates::symmetric_pullback(
                    ivrefcell * images[i].cell_, refcell.transpose() * gstress,
                    restore_F_cholesky).transpose();
            } else {
                auto c = refcell.transpose() * gstress;
                gstress = c;
                for (int ia = 0; ia < 3; ++ia) {
                    for (int ib = 0; ib < 3; ++ib) {
                        g.row(i)(3*ia + ib) = gstress(ia, ib);
                    }
                }
            }
        }
        for (int ia = 0; ia < images[i].size(); ++ia) {
            Eigen::RowVector3d gfracforce;
            if(subspace_active) gfracforce=(subspace.atomic_jacobian(images[i].cell_)*images[i].forces_[ia]).transpose();
            else if (vcneb) {
                if (recast_atomic_force) {
                    Eigen::Matrix3d matD = ivrefcell * images[i].cell_;
                    gfracforce = matD * images[i].forces_[ia];
                } else {
                    gfracforce = images[i].cell_ * images[i].forces_[ia];
                }
            } else {
                gfracforce = images[i].forces_[ia];
            }
            for (int a = 0; a < 3; ++a) {
                g.row(i)(vcoffset + 3*ia + a) = gfracforce(a);
            }
        }
        } catch (...) {
            images[i].pos_ = xold;
            #pragma omp critical(neb_calculator_error)
            { if (!calculator_error) calculator_error = std::current_exception(); }
        }
    }
}
    if (calculator_error) std::rethrow_exception(calculator_error);
    if (keep_symm && symmetry_) for(int i=1;i<images.size()-1;++i) {
        const Eigen::RowVectorXd force=g.row(i).cwiseQuotient(scalval.transpose());
        g.row(i)=project_symmetry(force).cwiseProduct(scalval.transpose());
    }
    // ── On-the-fly AL — delegate to shared ALPotential::check() ─────────
    if (al_pot_ && !al_abort_) {
        for (int i = 1; i < static_cast<int>(images.size()) - 1; ++i) {
            images[i].features_["AL_image"] = std::to_string(i);
            if (al_pot_->check(images[i])) {
                al_abort_ = true;
                Message(fmt::format("NEB: AL threshold_break exceeded at image {} "
                                    "— stopping optimisation", i));
                break;
            }
        }
    }
}
// todo: rename to X2G
void NEBDriver::LatPos2X(const Eigen::VectorXd &xin, Eigen::VectorXd &gin)
{
    // The force is an output, including for direct public codec calls.
    gin.resize(xin.size());
    // An invalid trial must remain distinguishable from a zero residual.
    // In particular, std::max(finite, NaN) can silently retain the finite
    // operand. Do not set a persistent stop reason here: a line search or
    // the outer safeguard may restore a valid point and continue.
    auto invalid_trial = [&]() {
        f = std::numeric_limits<double>::infinity();
        gin.setConstant(xin.size(), std::numeric_limits<double>::quiet_NaN());
        max_perp_force = max_spring_force = max_neb_force = saddle_g_proj = f;
    };
    if (!xin.allFinite()) { invalid_trial(); return; }
    for (const auto &image : images) {
        const double determinant = image.cell_.determinant();
        if (!image.cell_.allFinite() || !std::isfinite(determinant) ||
            (determinant == 0 && !subspace_active && !singular_cartesian_fixed_neb(*this))) {
            invalid_trial(); return;
        }
        for (const auto &position : image.pos_)
            if (!position.allFinite()) { invalid_trial(); return; }
    }
    if (!reuse_efs_) eval();
    for (const auto &image : images) {
        if (!std::isfinite(image.energy_) || !image.stress_.allFinite() ||
            image.forces_.size() != image.pos_.size()) {
            invalid_trial(); return;
        }
        for (const auto &force : image.forces_)
            if (!force.allFinite()) { invalid_trial(); return; }
    }
    if (!g.allFinite()) { invalid_trial(); return; }
    max_perp_force = 0;
    max_spring_force = 0;
    max_neb_force = 0;
    saddle_g_proj = 0;
    // update variable spring constants
    update_spring_constants();
    // select which image is the climbing image
    int highest_interior = 1;
    double engh = images[1].energy_;
    for (int i = 1; i < images.size() - 1; ++i) {
        if (images[i].energy_ > engh) {
            engh = images[i].energy_;
            highest_interior = i;
        }
    }
    // Equal-energy interior plateaus still have a barrier above both ends.
    const bool has_internal_barrier =
        engh > std::max(images.front().energy_, images.back().energy_);
    if (!freeze_phase_ && !pin) {
        iclimbs.clear();
        ivallys.clear();
    }
    if (!freeze_phase_ && (!pin || !found_state_site)) {
        if (climb && has_internal_barrier && climb_mode == "highest")
            iclimbs.insert(highest_interior);
        for (int first = 1; first < images.size() - 1;) {
            int last = first;
            while (last + 1 < images.size() - 1 &&
                   images[last + 1].energy_ == images[first].energy_) ++last;
            const double energy = images[first].energy_;
            const int representative = (first + last) / 2;
            if (climb && has_internal_barrier && climb_mode == "all_local" &&
                energy > images[first - 1].energy_ && energy > images[last + 1].energy_)
                iclimbs.insert(representative);
            if (vally && energy < images[first - 1].energy_ && energy < images[last + 1].energy_)
                ivallys.insert(representative);
            first = last + 1;
        }
        if (pin && (!iclimbs.empty() || !ivallys.empty())) found_state_site = true;
    }
    climbing_image = iclimbs.empty() ? -1
        : (iclimbs.count(highest_interior) ? highest_interior : *iclimbs.begin());
    const int phase_step = stable_evaluation_ ? accepted_steps : iter;
    climbing_active = climb && climbing_image >= 0 && phase_step >= climb_delay;
    // project forces into // and |_ direction
    Eigen::RowVectorXd tg, spf, tgf, prf, trf, tgp, tgm;
    double mtgp, mtgm;
    for (int i = 1; i < images.size() - 1; ++i) {
        tagent(i-1, i, i+1, tg);
        if (tangent_type == 3 || sproj_type == 3) {
            tgp = (x.row(i) - x.row(i-1)).cwiseProduct(scalval.transpose());
            tgm = (x.row(i+1) - x.row(i)).cwiseProduct(scalval.transpose());
            mtgp = tgp.norm();
            mtgm = tgm.norm();
            if (mtgp < 1e-14 || mtgm < 1e-14)
                throw std::runtime_error("NEB: duplicate adjacent images make the EB spring undefined");
        }
        if (sproj_type == 1) { // improved
            const double dp = ((x.row(i+1)-x.row(i)).array() * scalval.transpose().array()).matrix().norm();
            const double dm = ((x.row(i)-x.row(i-1)).array() * scalval.transpose().array()).matrix().norm();
            spf = (dp * spconsts[i] - dm * spconsts[i-1]) * tg;
        } else if (sproj_type == 2) { // original
            Eigen::RowVectorXd spring_delta =
                ((x.row(i+1)-x.row(i)) * spconsts[i] -
                 (x.row(i)-x.row(i-1)) * spconsts[i-1]).cwiseProduct(scalval.transpose());
            spf = spring_delta.dot(tg) * tg;
        } else if (sproj_type == 3) { // EB, haven't been fully impled
            spf = (mtgm - Leq) * spconsts[i] * tgm / mtgm - (mtgp - Leq) * spconsts[i-1] * tgp / mtgp;
        }
        // tagent force
        Eigen::RowVectorXd metric_force = g.row(i).cwiseQuotient(scalval.transpose());
        tgf = metric_force.dot(tg) * tg;
        // track tangent force at climbing image(s) for convergence check
        if (iclimbs.count(i)) {
            saddle_g_proj = std::max(saddle_g_proj, std::abs(metric_force.dot(tg)));
        }
        // perpendicular force
        prf = metric_force - tgf;
        // true force to be minimized
        trf = spf + prf;
        if (climbing_active && iclimbs.count(i)) {
            trf = -tgf + prf;
        }
        if (vally && ivallys.count(i) && phase_step >= climb_delay) {
            trf = tgf + prf;
        }
        trf=project_symmetry(trf);
        prf=project_symmetry(prf);
        spf=project_symmetry(spf);
        // put them into g
        for (int j = 0; j < trf.size(); ++j) {
            gin[(vcoffset+npx) * (i-1) + j] = -trf[j] * scalval[j];
        }
        // process selective dynamics info
        if (sel_dyn) {
            if(subspace_active) {
                int ct=0;const auto &axes=subspace.axes();
                for(int a=0;a<axes.size();++a) for(int b=0;b<=a;++b,++ct)
                    gin[(vcoffset+npx)*(i-1)+ct]*=images[0].sel_dyncell_(axes[a],axes[b]);
            } else if (vcneb) {
                if (symm_strain) {
                    int ct = 0;
                    for (int a = 0; a < 3; ++a) {
                        for (int b = a; b < 3; ++b) {
                            gin[(vcoffset + npx) * (i-1) + ct] *= images[0].sel_dyncell_(a, b);
                            ++ct;
                        }
                    }
                } else {
                    for (int a = 0; a < 3; ++a) {
                        for (int b = 0; b < 3; ++b) {
                            gin[(vcoffset+npx)*(i-1) + 3*a+b] *= images[0].sel_dyncell_(a, b);
                        }
                    }
                }
            }
            for (int iat = 0; iat < images[0].size(); ++iat) {
                for (int a = 0; a < 3; ++a) {
                    gin[(vcoffset+npx)*(i-1) + vcoffset + iat * 3 + a] *= images[0].sel_dynpos_[iat](a);
                }
            }
        }

        // Convergence diagnostics must use the same active degrees of freedom
        // and metric as the force returned to the optimizer.
        Eigen::RowVectorXd masked_perp = prf;
        Eigen::RowVectorXd masked_spring = spf;
        if (sel_dyn) {
            if(subspace_active) {
                int ct=0;const auto &axes=subspace.axes();
                for(int a=0;a<axes.size();++a) for(int b=0;b<=a;++b,++ct) {
                    masked_perp[ct]*=images[0].sel_dyncell_(axes[a],axes[b]);
                    masked_spring[ct]*=images[0].sel_dyncell_(axes[a],axes[b]);
                }
            } else if (vcneb) {
                if (symm_strain) {
                    int ct = 0;
                    for (int a = 0; a < 3; ++a)
                        for (int b = a; b < 3; ++b, ++ct) {
                            masked_perp[ct] *= images[0].sel_dyncell_(a, b);
                            masked_spring[ct] *= images[0].sel_dyncell_(a, b);
                        }
                } else {
                    for (int a = 0; a < 3; ++a)
                        for (int b = 0; b < 3; ++b) {
                            masked_perp[3*a+b] *= images[0].sel_dyncell_(a, b);
                            masked_spring[3*a+b] *= images[0].sel_dyncell_(a, b);
                        }
                }
            }
            for (int ia = 0; ia < images[0].size(); ++ia)
                for (int a = 0; a < 3; ++a) {
                    masked_perp[vcoffset + 3*ia+a] *= images[0].sel_dynpos_[ia](a);
                    masked_spring[vcoffset + 3*ia+a] *= images[0].sel_dynpos_[ia](a);
                }
        }
        auto max_generalized_force = [&](const Eigen::RowVectorXd &force) {
            if (!force.allFinite()) return std::numeric_limits<double>::infinity();
            double value = 0;
            for (int j = 0; j < vcoffset; ++j) value = std::max(value, std::abs(force[j]));
            for (int ia = 0; ia < images[0].size(); ++ia)
                value = std::max(value, force.segment(vcoffset + 3*ia, 3).norm());
            return value;
        };
        Eigen::RowVectorXd masked_neb(vcoffset + npx);
        for (int j = 0; j < masked_neb.size(); ++j)
            masked_neb[j] = -gin[(vcoffset+npx) * (i-1) + j] / scalval[j];
        max_perp_force = std::max(max_perp_force, max_generalized_force(masked_perp));
        max_spring_force = std::max(max_spring_force, max_generalized_force(masked_spring));
        max_neb_force = std::max(max_neb_force, max_generalized_force(masked_neb));
    }

    f = images[0].energy_;
    for (int i = 1; i < images.size(); ++i) {
        f += images[i].energy_;
        f += 0.5 * spconsts[i-1] *
             ((x.row(i) - x.row(i-1)).array() * scalval.transpose().array()).matrix().squaredNorm();
    }
    if (!gin.allFinite() || !std::isfinite(f) || !std::isfinite(max_perp_force) ||
        !std::isfinite(max_spring_force) || !std::isfinite(max_neb_force) ||
        !std::isfinite(saddle_g_proj)) {
        invalid_trial(); return;
    }
    if (debug_level > 0) {
        double raw_gmax = 0;
        for (int i = 1; i < (int)images.size() - 1; ++i) {
            for (int j = 0; j < g.cols(); ++j)
                raw_gmax = std::max(raw_gmax, fabs(g(i, j)));
        }
        neb_debug_stream << "  raw_gmax=" << raw_gmax;
        neb_debug_stream << " climbs=[";
        for (auto &c : iclimbs) neb_debug_stream << c << ",";
        neb_debug_stream << "] valleys=[";
        for (auto &v : ivallys) neb_debug_stream << v << ",";
        neb_debug_stream << "]";
        neb_debug_stream << " max_perp=" << max_perp_force;
        neb_debug_stream << " saddle_g_proj=" << saddle_g_proj << std::endl;
    }
}
// For init only
void NEBDriver::LatPos2X(Eigen::VectorXd &xin) {
    for (int i = 1; i < images.size()-1; ++i) {
        if(subspace_active) {
            x.row(i).head(vcoffset)=subspace.encode_cell(images[i].cell_).transpose();
            for(int ii=0;ii<images[i].size();++ii) x.row(i).segment<3>(vcoffset+3*ii)=encode_position(images[i],ii);
            continue;
        }
        if (vcneb) {
            Eigen::Matrix3d stm = get_strain(images[i].cell_);
            if (symm_strain) {
                matrix_to_symm_voigt(stm, x.row(i));
            } else {
                for (int ia = 0; ia < 3; ++ia) {
                    for (int ib = 0; ib < 3; ++ib) {
                        x(i, 3*ia + ib) = stm(ia, ib);
                    }
                }
            }
        }
        // Both VC representations start from fractional positions; recasting
        // additionally maps them through the reference cell.
        if (vcneb)
            images[i].to_fractional();
        for (int ii = 0; ii < images[i].size(); ++ii) {
            auto rpos = images[i].pos_[ii];
            if (vcneb && recast_atomic_force) {
                rpos *= refcell;
            }
            for (int jj = 0; jj < 3; ++jj) {
                x(i, vcoffset + 3*ii + jj) = rpos(jj);
            }
        }
        if (vcneb)
            images[i].to_cartesian();
    }
    xin.resize((vcoffset+npx) * (images.size() - 2));
    memcpy(xin.data(), x.row(1).data(), xin.size() * sizeof(double));
}

void NEBDriver::X2LatPos(const Eigen::MatrixXd &xin) {
    if (xin.rows() != images.size() || xin.cols() != vcoffset + npx || !xin.allFinite())
        throw NEBInvalidGeometry("NEB: invalid coordinate dimensions or non-finite trial");
    // Validate every cell before mutating any image, including subspace codecs.
    if (vcneb) {
        for (int i = 1; i < images.size()-1; ++i) {
            Eigen::Matrix3d cell;
            if (subspace_active) {
                try { cell = subspace.decode_cell(xin.row(i).head(vcoffset).transpose()); }
                catch (const std::invalid_argument &e) { throw NEBInvalidGeometry(e.what()); }
            } else {
                Eigen::Matrix3d stm;
                if (symm_strain)
                    stm << xin(i,0),xin(i,1),xin(i,2),xin(i,1),xin(i,3),xin(i,4),xin(i,2),xin(i,4),xin(i,5);
                else for(int a=0;a<3;++a) for(int b=0;b<3;++b) stm(a,b)=xin(i,3*a+b);
                cell = get_cell(stm);
            }
            validate_cell(cell);
        }
    }
    for (int i = 1; i < images.size() - 1; ++i) {
        if(subspace_active) {
            images[i].cell_=subspace.decode_cell(xin.row(i).head(vcoffset).transpose());images[i].cartesian=true;
            for(int ii=0;ii<images[i].size();++ii) images[i].pos_[ii]=decode_position(xin.row(i).segment<3>(vcoffset+3*ii),images[i].cell_);
            continue;
        }
        if (vcneb) {
            Eigen::Matrix3d stm;
            if (symm_strain) {
                stm << xin(i, 0), xin(i, 1), xin(i, 2),
                       xin(i, 1), xin(i, 3), xin(i, 4),
                       xin(i, 2), xin(i, 4), xin(i, 5);
            } else {
                for (int ia = 0; ia < 3; ++ia) {
                    for (int ib = 0; ib < 3; ++ib) {
                        stm(ia, ib) = xin(i, 3*ia + ib);
                    }
                }
            }
            images[i].cell_ = get_cell(stm);
            if (!recast_atomic_force)
                images[i].cartesian = false;
        }
        for (int ii = 0; ii < images[i].size(); ++ii) {
            for (int jj = 0; jj < 3; ++jj) {
                images[i].pos_[ii](jj) = xin(i, vcoffset + 3 * ii + jj);
            }
        }
        if (vcneb && !recast_atomic_force)
            images[i].to_cartesian();
        else if (vcneb && recast_atomic_force) {
            for (int ii = 0; ii < images[i].size(); ++ii) {
                images[i].pos_[ii] = images[i].pos_[ii] * ivrefcell * images[i].cell_;
            }
        }
    }
}

void NEBDriver::X2LatPos_interf(const Eigen::VectorXd &xin) {
    if (xin.size() != (images.size()-2)*(vcoffset+npx) || !xin.allFinite())
        throw NEBInvalidGeometry("NEB: invalid flat coordinates");
    memcpy(x.data() + npx + vcoffset, xin.data(), xin.size() * sizeof(double));
    if(keep_symm && symmetry_) for(int i=1;i<images.size()-1;++i) {
        const Eigen::RowVectorXd z=x.row(i).cwiseProduct(scalval.transpose());
        x.row(i)=symmetry_->project_point(z).cwiseQuotient(scalval.transpose());
    }
    X2LatPos(x);
}

double NEBDriver::operator()(const Eigen::VectorXd &xin, Eigen::VectorXd &gin) {
    // At every time convert xin into structure and compute its energy, force and stress, then give derivate info back into gin
    // xin won't be modified, thus mic is correct here. 
    try {
        X2LatPos_interf(xin);
        LatPos2X(xin, gin);
    } catch (const NEBInvalidGeometry &) {
        f = std::numeric_limits<double>::infinity();
        gin.setConstant(xin.size(), std::numeric_limits<double>::quiet_NaN());
        max_perp_force = max_spring_force = max_neb_force = saddle_g_proj = f;
    }
    if (debug_level > 0) {
        neb_debug_stream << "Iteration " << iter + 1 << '\n';
        neb_debug_stream << "xin: " << xin.transpose() << '\n';
        neb_debug_stream << "gin: " << gin.transpose() << std::endl;
    }
    if (legacy_checkpoint_ && gin.allFinite() && std::isfinite(f)) legacy_checkpoint_();
    if (al_abort_) { gin.setZero(); return f; }
    const double gmax = *std::max_element(gin.begin(), gin.end(),
        [](double a, double b) { return fabs(a) < fabs(b); });
    ++iter;
    if (variable_image && iter % image_adjust_interval == 0) {
        check_image_spacing();
        if (needs_image_adjust) gin.setZero();
    }
    if (check_convergence()) gin.setZero();
    std::cout << fmt::format("LBFGS-F neb iteration {} f = {:<10.4f} gmax = {}", 
                iter, 
                f,
                gmax) << std::endl;
    
    if (iter == 1 && iobj && write_traj) {
        if (trjofs.is_open()) trjofs.close();
        trjofs.clear();
        trjofs.open(trajfile);
        if (!trjofs) GPERROR("Cannot open NEB trajectory: " + trajfile);
        if (!iobj->seq_write_init(trjofs, trjformat, trajfile))
            GPERROR("Cannot initialize NEB trajectory: " + trajfile);
    }
    if (iobj && write_traj && (iter == 1 || iter % nblock == 0)) {
        for (auto &it : images) {
            if (!iobj->seq_write(trjofs, it))
                GPERROR("Failed to write NEB trajectory: " + trajfile);
        }
        trjofs.flush();
        if (!trjofs) GPERROR("Failed to write NEB trajectory: " + trajfile);
    }
    return f;
}

bool NEBDriver::check_convergence() const {
    if (climb && climbing_image >= 0 && !climbing_active) return false;
    return std::isfinite(max_perp_force) && std::isfinite(max_spring_force) &&
           std::isfinite(saddle_g_proj) && std::isfinite(max_neb_force) &&
           max_neb_force < neb_ftol;
}

std::string NEBDriver::stop_reason_string() const {
    switch (stop_reason) {
        case StopReason::not_run: return "not_run";
        case StopReason::converged: return "converged";
        case StopReason::max_iterations: return "max_iterations";
        case StopReason::active_learning_abort: return "active_learning_abort";
        case StopReason::no_internal_barrier: return "no_internal_barrier";
        case StopReason::numerical_failure: return "numerical_failure";
        case StopReason::max_evaluations: return "max_evaluations";
        case StopReason::stalled: return "stalled";
    }
    return "numerical_failure";
}

int NEBDriver::minimize_guarded(const std::function<int(int)> &minimize,
                              int max_iterations, double max_move,
                              Eigen::VectorXd &xin, double &objective) {
    if (tangent_type != 2) {
        const int used = minimize(max_iterations);
        if (!al_abort_ && !needs_image_adjust) {
            Eigen::VectorXd evaluated_x;
            get_x(evaluated_x);
            // FIRE can exhaust its budget just after moving xin. Other
            // optimizers may already have evaluated their returned point.
            if ((xin.array() != evaluated_x.array()).any()) {
                Eigen::VectorXd gradient(xin.size());
                objective = (*this)(xin, gradient);
            }
        }
        return used;
    }

    // Central tangents couple images through a nonsymmetric Jacobian. Test
    // progress with the complete force residual in z = S*x, not the energy.
    // The progress threshold chooses when to try Newton; smaller valid
    // improvements are retained as a fallback, never rejected outright.
    constexpr double useful_reduction = 1e-2;
    auto interrupted = [&]() { return al_abort_ || needs_image_adjust; };
    Eigen::VectorXd scale(xin.size());
    for (int i = 0; i < scale.size(); ++i) scale(i) = scalval(i % scalval.size());
    auto residual = [&](const Eigen::VectorXd &gradient) {
        return gradient.cwiseQuotient(scale).squaredNorm();
    };
    Eigen::VectorXd gradient(xin.size());
    objective = (*this)(xin, gradient);
    int used = 0;
    int batch_size = 8;
    double damping = 0.1;
    bool reported_restart = false;
    bool reported_linear_solve = false;
    while (used < max_iterations && !interrupted() && !check_convergence()) {
        const Eigen::VectorXd previous_x = xin;
        // operator() increments iter after evaluating the force. Crossing the
        // climb/valley delay changes that force field, so residuals and Jv
        // probes from opposite sides must never share a model.
        const bool delayed_active = (climb || vally) && iter > climb_delay;
        auto phase_changed = [&]() {
            return delayed_active != ((climb || vally) && iter > climb_delay);
        };
        auto restart_phase = [&]() {
            xin = previous_x;
            objective = (*this)(xin, gradient);
        };
        const double base_norm2 = residual(gradient);
        if (!gradient.allFinite() || !std::isfinite(base_norm2)) {
            stop_reason = StopReason::numerical_failure;
            break;
        }
        if (base_norm2 == 0) {
            // A zero force before climb_delay expires is not convergence.
            ++used;
            objective = (*this)(xin, gradient);
            continue;
        }
        const double roundoff = 64 * std::numeric_limits<double>::epsilon() * base_norm2;
        double best_norm2 = base_norm2;
        Eigen::VectorXd best_x = previous_x;
        auto remember_trial = [&]() {
            const double trial_norm2 = residual(gradient);
            if (gradient.allFinite() && std::isfinite(trial_norm2) &&
                best_norm2 - trial_norm2 > roundoff) {
                best_norm2 = trial_norm2;
                best_x = xin;
            }
        };
        const int budget = std::min(batch_size, max_iterations - used);
        used += std::clamp(minimize(budget), 1, budget);
        if (interrupted()) break;
        // FIRE may have moved x just before exhausting its budget.
        objective = (*this)(xin, gradient);
        if (interrupted()) break;
        if (phase_changed()) { restart_phase(); continue; }
        if (check_convergence()) break;
        remember_trial();
        if (best_norm2 < base_norm2 * (1 - useful_reduction)) {
            batch_size = std::min(8, 2 * batch_size);
            continue;
        }
        batch_size = std::max(1, batch_size / 2);
        if (used == max_iterations) {
            xin = best_x;
            objective = (*this)(xin, gradient);
            break;
        }
        if (!reported_restart) {
            Message("NEB: restarting accelerated iterations with a damped force step (slow force-residual progress)");
            reported_restart = true;
        }
        xin = previous_x;
        objective = (*this)(xin, gradient);
        if (interrupted()) break;
        if (phase_changed()) { restart_phase(); continue; }
        if (check_convergence()) break;
        const Eigen::VectorXd base_gradient = gradient;

        // Steepest force flow in the requested metric. Preserve even a small
        // decrease, but do not let it indefinitely prevent a Newton correction.
        const Eigen::VectorXd direction = -gradient.cwiseQuotient(scale.cwiseProduct(scale));
        double step = std::min(damping, max_move / direction.stableNorm());
        for (int trial = 0; trial < 8; ++trial) {
            xin = previous_x + step * direction;
            objective = (*this)(xin, gradient);
            if (interrupted() || phase_changed() || check_convergence()) break;
            const double trial_norm2 = residual(gradient);
            remember_trial();
            if (gradient.allFinite() && base_norm2 - trial_norm2 > roundoff) {
                damping = std::min(0.1, step * 1.5);
                break;
            }
            step *= 0.5;
        }
        if (interrupted()) { ++used; break; }
        if (phase_changed()) { ++used; restart_phase(); continue; }
        if (check_convergence()) { ++used; break; }
        if (best_norm2 >= base_norm2 * (1 - useful_reduction)) {
            // Matrix-free GMRES solves J*correction = -r in z coordinates.
            // Grow beyond 20 directions when necessary: a small fixed space
            // can stagnate in a fractional-coordinate, strongly coupled band.
            if (!reported_linear_solve) {
                Message("NEB: force flow stalled; applying a local residual correction");
                reported_linear_solve = true;
            }
            const Eigen::VectorXd rhs = -base_gradient.cwiseQuotient(scale);
            const double beta = rhs.stableNorm();
            const int nk = std::min<int>(160, xin.size());
            Eigen::MatrixXd basis = Eigen::MatrixXd::Zero(xin.size(), nk + 1);
            Eigen::MatrixXd hessenberg = Eigen::MatrixXd::Zero(nk + 1, nk);
            Eigen::VectorXd target = Eigen::VectorXd::Zero(nk + 1);
            target(0) = beta;
            basis.col(0) = rhs / beta;
            Eigen::VectorXd correction = Eigen::VectorXd::Zero(xin.size());
            double best_linear_residual = beta;
            double predicted_linear = 0, predicted_quadratic = 0;
            for (int k = 0; k < nk; ++k) {
                constexpr double delta = 1e-5;
                xin = previous_x + delta * basis.col(k).cwiseQuotient(scale);
                objective = (*this)(xin, gradient);
                if (interrupted() || phase_changed() || !gradient.allFinite()) break;
                Eigen::VectorXd jv = ((gradient - base_gradient) / delta).cwiseQuotient(scale);
                if (!jv.allFinite()) break;
                // Twice-modified Gram-Schmidt protects nearly dependent bases.
                for (int pass = 0; pass < 2; ++pass)
                    for (int j = 0; j <= k; ++j) {
                        const double dot = basis.col(j).dot(jv);
                        hessenberg(j,k) += dot;
                        jv -= dot * basis.col(j);
                    }
                const double norm = jv.stableNorm();
                if (!std::isfinite(norm) || !hessenberg.col(k).allFinite()) break;
                hessenberg(k+1,k) = norm;
                const Eigen::MatrixXd hk = hessenberg.topLeftCorner(k+2, k+1);
                const Eigen::VectorXd coeff = hk.colPivHouseholderQr().solve(target.head(k+2));
                if (!coeff.allFinite()) break;
                const Eigen::VectorXd model_change = hk * coeff;
                const double linear_residual = (model_change-target.head(k+2)).stableNorm();
                const Eigen::VectorXd candidate = basis.leftCols(k+1) * coeff;
                const double linear = 2 * beta * model_change[0];
                const double quadratic = model_change.squaredNorm();
                if (std::isfinite(linear_residual) && candidate.allFinite() &&
                    std::isfinite(candidate.stableNorm()) && std::isfinite(linear) &&
                    std::isfinite(quadratic) && linear_residual < best_linear_residual) {
                    correction = candidate;
                    predicted_linear = linear;
                    predicted_quadratic = quadratic;
                    best_linear_residual = linear_residual;
                }
                if (best_linear_residual < 0.1*beta || norm < 1e-12) break;
                basis.col(k+1) = jv / norm;
            }
            if (interrupted()) { ++used; break; }
            if (phase_changed()) { ++used; restart_phase(); continue; }
            // Difference probes are not optimizer moves and may exceed
            // max_move. Restore the base before testing a bounded step; a
            // converged probe must not bypass the displacement limit.
            xin = previous_x;
            objective = (*this)(xin, gradient);
            if (interrupted()) { ++used; break; }
            if (phase_changed()) { ++used; restart_phase(); continue; }
            if (check_convergence()) { ++used; break; }
            const Eigen::VectorXd displacement = correction.cwiseQuotient(scale);
            const double displacement_norm = displacement.stableNorm();
            if (displacement.allFinite() && std::isfinite(displacement_norm) && displacement_norm > 0) {
                step = std::min(1.0, max_move / displacement_norm);
                for (int trial = 0; trial < 16; ++trial) {
                    xin = previous_x + step * displacement;
                    objective = (*this)(xin, gradient);
                    if (interrupted() || phase_changed() || check_convergence()) break;
                    // Use the model's predicted decrease at this capped step.
                    // A fixed relative decrease can reject even an exact
                    // Newton step when max_move is small relative to x.
                    const double predicted_drop = step * (predicted_linear - step * predicted_quadratic);
                    const double actual_drop = base_norm2 - residual(gradient);
                    if (gradient.allFinite() && std::isfinite(predicted_drop) && std::isfinite(actual_drop) &&
                        predicted_drop > roundoff && actual_drop > roundoff && actual_drop >= 0.1*predicted_drop) {
                        remember_trial();
                        break;
                    }
                    step *= 0.5;
                }
            }
        }
        ++used;
        if (interrupted()) break;
        if (phase_changed()) { restart_phase(); continue; }
        if (check_convergence()) break;
        xin = best_x;
        objective = (*this)(xin, gradient);
        if (!interrupted() && phase_changed()) { restart_phase(); continue; }
        if (best_norm2 == base_norm2 && !interrupted() && !check_convergence()) {
            stop_reason = StopReason::numerical_failure;
            break;
        }
    }
    return used;
}

int NEBDriver::run_legacy(const RunOptions &options) {
    const auto &driver = options.driver;
    const int max_iterations = options.max_iterations;
    const double max_move = options.max_move;
    if (max_iterations <= 0) throw std::invalid_argument("NEB: max_iterations must be positive");
    if (!(max_move > 0)) throw std::invalid_argument("NEB: max_move must be positive");
    Eigen::VectorXd xin;
    LatPos2X(xin);
    double objective = 0;
    converged = false;
    stop_reason = StopReason::not_run;
    optimizer_iterations = 0;

    auto saved_images=images;
    auto saved_x=x,saved_g=g;
    auto saved_climbs=iclimbs,saved_valleys=ivallys;
    std::array<double,5> saved_values{f,max_perp_force,max_spring_force,max_neb_force,saddle_g_proj};
    int saved_climbing=climbing_image;
    bool saved_active=climbing_active,saved_found=found_state_site;
    legacy_checkpoint_=[&](){
        saved_images=images;saved_x=x;saved_g=g;
        saved_climbs=iclimbs;saved_valleys=ivallys;
        saved_values={f,max_perp_force,max_spring_force,max_neb_force,saddle_g_proj};
        saved_climbing=climbing_image;saved_active=climbing_active;saved_found=found_state_site;
    };
    struct ClearCheckpoint {std::function<void()> &callback;~ClearCheckpoint(){callback={};}} clear_checkpoint{legacy_checkpoint_};
    auto restore=[&](){
        images=saved_images;x=saved_x;g=saved_g;iclimbs=saved_climbs;ivallys=saved_valleys;
        f=saved_values[0];max_perp_force=saved_values[1];max_spring_force=saved_values[2];
        max_neb_force=saved_values[3];saddle_g_proj=saved_values[4];
        climbing_image=saved_climbing;climbing_active=saved_active;found_state_site=saved_found;
    };

    try {
    while (optimizer_iterations < max_iterations) {
        needs_image_adjust = false;
        const int remaining = max_iterations - optimizer_iterations;
        auto should_stop = [this]() {
            return al_abort_ || needs_image_adjust || check_convergence();
        };
        int used = 0;
        if (driver == "fire") {
            SimpleFireF opt(options.optimizer_settings);
            if(opt.assignmass)opt.mass=fire_masses();
            opt.max_iter = remaining;
            opt.max_step = max_move;
            opt.gtol = 0;
            opt.converged_callback = should_stop;
            used = minimize_guarded([&](int budget) {
                opt.max_iter = budget;
                return opt.minimize(*this, xin, objective);
            }, remaining, max_move, xin, objective);
        } else if (driver == "sgd" || driver == "gd") {
            SimpleSGD opt(options.optimizer_settings);
            opt.max_iter = remaining;
            opt.max_step = max_move;
            opt.gtol = 0;
            opt.converged_callback = should_stop;
            used = minimize_guarded([&](int budget) {
                opt.max_iter = budget;
                return opt.minimize(*this, xin, objective);
            }, remaining, max_move, xin, objective);
        } else if (driver == "cg") {
            SimpleCGF opt(options.optimizer_settings);
            opt.max_iter = remaining;
            opt.max_step = max_move;
            opt.gtol = 0;
            opt.converged_callback = should_stop;
            used = minimize_guarded([&](int budget) {
                opt.max_iter = budget;
                return opt.minimize(*this, xin, objective);
            }, remaining, max_move, xin, objective);
        } else if (driver == "lbfgs") {
            LBFGSpp::LBFGSParam<double> param;
            const auto &settings=options.optimizer_settings;
            if(settings.count("lbfgs_m"))param.m=std::stoi(settings["lbfgs_m"]);
            if(settings.count("first_step"))param.first_initial_step=std::stod(settings["first_step"]);
            if(settings.count("initial_step"))param.initial_step=std::stod(settings["initial_step"]);
            if(settings.count("max_step"))param.max_step=std::stod(settings["max_step"]);
            if(settings.count("s_curv_coeff"))param.s_curv_coeff=std::stod(settings["s_curv_coeff"]);
            if(settings.count("damping"))param.damping=std::stod(settings["damping"]);
            param.max_iterations = remaining;
            param.max_step_move = max_move;
            // NEB convergence is defined by check_convergence(), which includes
            // the perpendicular, spring and complete climbing-image forces.
            // Letting LBFGS apply its component-wise tolerance here can stop a
            // run before those stricter criteria are satisfied.
            param.epsgmax = 0;
            param.epsilon = 0;
            param.epsilon_rel = 0;
            param.past = 0;
            param.linesearch = 4;
            used = minimize_guarded([&](int budget) {
                param.max_iterations = budget;
                LBFGSpp::LBFGSSolver<double, LBFGSpp::LineSearchForceOnlyA> opt(param);
                return opt.minimize(*this, xin, objective);
            }, remaining, max_move, xin, objective);
        } else {
            throw std::invalid_argument("NEB: unknown optimizer '" + driver + "'");
        }
        optimizer_iterations += std::max(1, used);

        if (al_abort_) {
            stop_reason = StopReason::active_learning_abort;
            break;
        }
        if (needs_image_adjust) {
            rebuild_path();
            get_x(xin);
            continue;
        }
        if (stop_reason == StopReason::numerical_failure) break;
        if (!std::isfinite(max_neb_force)) {
            stop_reason = StopReason::numerical_failure;
            break;
        }
        if (check_convergence()) {
            if (climb && climbing_image < 0) {
                converged = false;
                stop_reason = StopReason::no_internal_barrier;
            } else {
                converged = true;
                stop_reason = StopReason::converged;
            }
            break;
        }
    }
    } catch(...) {
        const bool numeric_failure=!std::isfinite(max_neb_force);
        restore();
        if(!numeric_failure)throw;
        converged=false;stop_reason=StopReason::numerical_failure;
    }
    if (stop_reason == StopReason::numerical_failure) restore();
    if (stop_reason == StopReason::not_run) stop_reason = StopReason::max_iterations;
    if (write_traj && trjofs.is_open()) {
        trjofs.close();
        if (!trjofs) GPERROR("Failed to write NEB trajectory: " + trajfile);
    }
    return converged ? 0 : 2;
}

void NEBDriver::update_spring_constants() {
    if (!variable_spring) return;
    double e_max = images[0].energy_;
    for (int i = 1; i < (int)images.size(); ++i)
        e_max = std::max(e_max, images[i].energy_);
    // Variable-spring NEB uses the higher endpoint as the reference: segments
    // below it receive k_min, while only the barrier region is stiffened.
    double e_ref = std::max(images[0].energy_, images.back().energy_);
    double denom = e_max - e_ref;
    if (denom < 1e-10) {
        std::fill(spconsts.begin(), spconsts.end(), spring_min);
        return;
    }
    double dk = spring_max - spring_min;
    for (int i = 0; i < (int)images.size() - 1; ++i) {
        double e_bar = std::max(images[i].energy_, images[i+1].energy_);
        spconsts[i] = e_bar <= e_ref
            ? spring_min
            : spring_max - dk * (e_max - e_bar) / denom;
    }
}

void NEBDriver::tagent(int a, int b, int c, Eigen::RowVectorXd &tg) {
    double ea = images[a].energy_;
    double eb = images[b].energy_;
    double ec = images[c].energy_;
    if (tangent_type == 1) { // improved
        if (ec > eb && eb > ea) {
        tg = (x.row(c) - x.row(b)).cwiseProduct(scalval.transpose());
        } else if (ec < eb && eb < ea) {
        tg = (x.row(b) - x.row(a)).cwiseProduct(scalval.transpose());
        } else {
            double vmax = std::max(fabs(ec - eb), fabs(ea - eb));
            double vmin = std::min(fabs(ec - eb), fabs(ea - eb));
            if (ec > ea) {
            tg = ((x.row(c) - x.row(b)) * vmax + (x.row(b) - x.row(a)) * vmin).cwiseProduct(scalval.transpose());
            } else {
            tg = ((x.row(c) - x.row(b)) * vmin + (x.row(b) - x.row(a)) * vmax).cwiseProduct(scalval.transpose());
            } 
        }
    } else if (tangent_type == 2) { // original
        const Eigen::RowVectorXd backward = (x.row(b) - x.row(a)).cwiseProduct(scalval.transpose());
        const Eigen::RowVectorXd forward = (x.row(c) - x.row(b)).cwiseProduct(scalval.transpose());
        const double backward_norm = backward.norm();
        const double forward_norm = forward.norm();
        if (backward_norm < 1e-14 || forward_norm < 1e-14)
            throw std::runtime_error("NEB: duplicate adjacent images make the tangent undefined");
        tg = backward / backward_norm + forward / forward_norm;
    } else if (tangent_type == 3) { // EB method: simple central difference
        tg = (x.row(c) - x.row(a)).cwiseProduct(scalval.transpose());
    }
    tg=project_symmetry(tg);
    if (tg.squaredNorm() < 1e-30) {
        tg = project_symmetry((x.row(c) - x.row(a)).cwiseProduct(scalval.transpose()));
        if (tg.squaredNorm() < 1e-30)
            throw std::runtime_error("NEB: degenerate image triplet makes the tangent undefined");
    }
    tg.normalize();
}

double NEBDriver::tagent_g(int a, int b) {
    Eigen::VectorXd tg;
    tg = (x.row(b) - x.row(a)).cwiseProduct(scalval.transpose());
    if (tg.squaredNorm() < 1e-30) return 0;
    tg.normalize();
    Eigen::VectorXd dg;
    dg = g.row(a).cwiseQuotient(scalval.transpose());
    // memcpy(g.data(), a.forces_[0].data(), npx * sizeof(double));
    return dg.dot(tg);
}

double NEBDriver::tagent_g(int a, int b, int c) {
    Eigen::RowVectorXd tg, dg;
    tagent(a, b, c, tg);
    // g.resize(npx+9);
    // memcpy(g.data(), b.forces_[0].data(), npx * sizeof(double));
    dg = g.row(b).cwiseQuotient(scalval.transpose());
    return -dg.dot(tg);
}

void NEBDriver::check_image_spacing() {
    if (!variable_image) return;
    int nimgs = (int)images.size();
    if (nimgs < 3) return;

    // Find current climbing image (highest energy interior image)
    int iclimb = 1;
    double emax = images[1].energy_;
    for (int i = 2; i < nimgs - 1; ++i) {
        if (images[i].energy_ > emax) {
            emax = images[i].energy_;
            iclimb = i;
        }
    }

    // 1. Compute segment lengths
    std::vector<double> seglen(nimgs - 1);
    double total_len = 0;
    for (int i = 0; i < nimgs - 1; ++i) {
        seglen[i] = distance(images[i], images[i + 1], 0.0);
        total_len += seglen[i];
    }
    double d_avg = total_len / (nimgs - 1);
    if (d_avg < 1e-10) return;

    // 2. Find longest segment (for insertion)
    int insert_at = -1;
    double d_max = 0;
    for (int i = 0; i < nimgs - 1; ++i) {
        if (seglen[i] > d_max) {
            d_max = seglen[i];
            insert_at = i;
        }
    }

    // 3. Find shortest segment (for removal), protect climbing image neighbors
    int remove_at = -1;
    double d_min = 1e100;
    for (int i = 0; i < nimgs - 2; ++i) {
        // Don't remove segments adjacent to climbing image
        if (i == iclimb - 1 || i == iclimb) continue;
        if (seglen[i] < d_min) {
            d_min = seglen[i];
            remove_at = i;
        }
    }

    // 4. Decide action (insert takes priority)
    if (d_max > image_insert_thresh * d_avg && nimgs < max_image_count) {
        std::cout << fmt::format("NEB: inserting image between {} and {} (seglen {:.3f} > {:.3f} * avg {:.3f})\n",
                                 insert_at, insert_at + 1, d_max, image_insert_thresh, d_avg);
        needs_image_adjust = true;
        // Mark: insert after image[insert_at]
        iclimb_ = insert_at;   // reuse iclimb_ to store insertion index...
        return;
    }
    // We store action type in iclimb_ sign: positive=insert, negative=remove
    // Actually, let me use a separate approach. We'll check needs_image_adjust and
    // use insert_at / remove_at logic in rebuild_path().

    if (remove_at >= 0 && d_min < image_remove_thresh * d_avg && nimgs > min_image_count) {
        std::cout << fmt::format("NEB: removing image {} (seglen {:.3f} < {:.3f} * avg {:.3f})\n",
                                 remove_at + 1, d_min, image_remove_thresh, d_avg);
        needs_image_adjust = true;
        iclimb_ = -(remove_at + 1); // negative value signals removal, stores image index
        return;
    }
}

void NEBDriver::rebuild_path() {
    int nimgs = (int)images.size();
    int natoms = images[0].size();

    if (iclimb_ >= 0) {
        // ── Insert a new image ──
        int pos = iclimb_; // insert after image[pos]
        Atoms newimg = images[0];
        newimg.pos_ = images[0].pos_;
        newimg.chemmap_ = images[0].chemmap_;

        // Interpolate cell from strain components in x matrix (VCNEB)
        if(subspace_active) newimg.cell_=subspace.decode_cell((.5*(x.row(pos).head(vcoffset)+x.row(pos+1).head(vcoffset))).transpose());
        else if (vcneb) {
            Eigen::Matrix3d stm_mid = Eigen::Matrix3d::Zero();
            if (symm_strain) {
                stm_mid = interpolate_symm_voigt(x, pos, pos + 1);
            } else {
                for (int ia = 0; ia < 3; ++ia)
                    for (int ib = 0; ib < 3; ++ib)
                        stm_mid(ia, ib) = 0.5 * (x(pos, 3*ia+ib) + x(pos + 1, 3*ia+ib));
            }
            Eigen::Matrix3d newcell = get_cell(stm_mid);
            newimg.set_cell(newcell);
        }

        // Linear interpolation of positions from x matrix
        // x stores Cartesian positions for fixed-cell NEB, fractional positions
        // for VC-NEB, or fractional * refcell for recast VC-NEB.
        Eigen::RowVector3d qmid;
        for (int ia = 0; ia < natoms; ++ia) {
            for (int a = 0; a < 3; ++a) {
                double v_mid = 0.5 * (x(pos, vcoffset + 3 * ia + a) + x(pos + 1, vcoffset + 3 * ia + a));
                qmid(a) = v_mid;
            }
            newimg.pos_[ia] = decode_position(qmid, newimg.cell_);
        }

        // Insert into images vector
        images.insert(images.begin() + pos + 1, newimg);

        std::cout << fmt::format("NEB: inserted image at position {}, total images now {}\n",
                                 pos + 1, images.size());
    } else {
        // ── Remove an image ──
        int idx = -iclimb_; // the image index to remove
        if (idx > 0 && idx < nimgs - 1) {
            images.erase(images.begin() + idx);
            std::cout << fmt::format("NEB: removed image at position {}, total images now {}\n",
                                     idx, images.size());
        }
    }

    // Rebuild all dependent structures
    nimgs = (int)images.size();
    npx = natoms * 3;
    spconsts.assign(nimgs - 1, spring);
    x.resize(nimgs, vcoffset + npx);
    g.resize(nimgs, vcoffset + npx);
    x.setZero();
    g.setZero();

    // Re-encode all images into x
    for (int i = 0; i < nimgs; ++i) {
        if (images[i].size() == 0) continue;
        if(subspace_active) {
            x.row(i).head(vcoffset)=subspace.encode_cell(images[i].cell_).transpose();
            for(int ii=0;ii<images[i].size();++ii) x.row(i).segment<3>(vcoffset+3*ii)=encode_position(images[i],ii);
            continue;
        }
        if (vcneb) {
            auto stm = get_strain(images[i].cell_);
            if (symm_strain) {
                matrix_to_symm_voigt(stm, x.row(i));
            } else {
                for (int ia = 0; ia < 3; ++ia)
                    for (int ib = 0; ib < 3; ++ib)
                        x(i, 3 * ia + ib) = stm(ia, ib);
            }
        }
        if (vcneb) {
            for (int ii = 0; ii < natoms; ++ii) {
                auto rp = encode_position(images[i], ii);
                for (int jj = 0; jj < 3; ++jj)
                    x(i, vcoffset + 3 * ii + jj) = rp(jj);
            }
        } else {
            for (int ii = 0; ii < natoms; ++ii)
                for (int jj = 0; jj < 3; ++jj)
                    x(i, vcoffset + 3 * ii + jj) = images[i].pos_[ii](jj);
        }
    }

    initialize_scaling();

    // Update Leq for EB method
    Leq = ((x.row(nimgs - 1) - x.row(0)).array() * scalval.transpose().array()).matrix().norm()
        / (nimgs - 1.0);

    // Clear the adjust flag
    needs_image_adjust = false;
    iclimbs.clear();
    ivallys.clear();
    found_state_site = false;
    climbing_image = -1;
    climbing_active = false;
    iclimb_ = -1;
}

void NEBDriver::get_x(Eigen::VectorXd &x_out) {
    int ninterior = (int)images.size() - 2;
    x_out.resize(ninterior * (vcoffset + npx));
    std::memcpy(x_out.data(), x.row(1).data(), x_out.size() * sizeof(double));
}

void NEBDriver::InitSettings()  {
    Connect(solver_policy, "solver_policy", "NEB numerical policy: stable-v1 or legacy (default legacy)");
    Connect(max_cell_strain, "max_cell_strain", "Maximum relative cell deformation per accepted step");
    Connect(max_force_evaluations, "max_force_evaluations", "Band EFS budget; zero derives 10*max_iterations+1");
    Connect(diagnostics_file, "diagnostics", "Accepted-state diagnostic CSV filename");
    Connect(spring, "spring", "spring constant");
    Connect(climb, "climb", "enable CI-NEB");
    Connect(climb_mode, "climb_mode", "climbing selection: highest or all_local");
    Connect(climb_delay, "climb_delay", "Climbing delay: accepted steps for stable-v1, force evaluations for legacy (default 0)");
    Connect(vally, "vally");
    Connect(vally, "valley", "enable valley-image refinement (alias for vally)");
    Connect(pin, "pin", "pin climbing/vally images");
    Connect(ciistr, "ciival", "", 1, 2);
    Connect(apply_scal, "apply_scal");
    Connect(iref, "iref", "reference cell index, -2: average of init and final, -1: final cell", 1, 2);

    Connect(omp_mode, "omp_mode", "OpenMP level: images (default) or atoms; MPI distributes images in either mode");
    Connect(check_mic, "check_mic", "Unwrap adjacent input images by minimum-image displacement (default true); restart skips this");
    Connect(sel_dyn, "selective_dynamics", "enable selective dynamics");
    Connect(cmask, "cmask", "mask for cell parameters, 9 compoments");
    Connect(restore_F_cholesky, "restore_F_cholesky", "Restore symmetric strain using Cholesky (true) or symmetric square root (false)", 1, 2);
    Connect(symm_strain, "symm_strain", "use symmetric strain tensor to avoid rigid rotation");
    Connect(recast_atomic_force, "recast_atomic_force", "recast atomic force when vcneb with symm_strain", 1, 2);
    Connect(keep_symm, "keep_symm", "Constrain the common labelled path subgroup (default false); full PBC, symmetric VC strain; incompatible with masks/perturbation");
    Connect(symprec, "symprec", "Symmetry position/lattice tolerance in Angstrom (default 1e-5)");
    Connect(symmetry_operations, "symmetry_operations", "Initial-image operation indices as subgroup generators; requires keep_symm; empty uses the common path group");
    Connect(symmetry_diagnostics, "symmetry_diagnostics", "Optional symmetry report filename: initial/perturbed/final groups, mappings and residuals");
    Connect(perturb_amplitude, "perturb_amplitude", "Maximum smooth atomic path perturbation in Angstrom (default 0); endpoints fixed");
    Connect(perturb_seed, "perturb_seed", "Nonnegative reproducible perturbation seed (default 0)");
    Connect(symmetry_release, "symmetry_release", "Post-run symmetry release output directory; empty disables (default); main result preserved");
    Connect(symmetry_release_trials, "symmetry_release_trials", "Seeded release trials in addition to one unperturbed trial (default 3)");
    Connect(symmetry_release_seed, "symmetry_release_seed", "First nonnegative release seed (default 0)");
    Connect(symmetry_release_steps, "symmetry_release_steps", "Independent iteration budget per release trial (default 200)");
    Connect(symmetry_release_evaluations, "symmetry_release_evaluations", "Independent band EFS budget per release trial (default 2001)");
    Connect(symmetry_release_amplitude, "symmetry_release_amplitude", "Release atomic displacement bound in Angstrom (default 0.01)");
    Connect(symmetry_release_strain, "symmetry_release_strain", "Release relative cell deformation Frobenius bound (default 0.001)");
    Connect(symmetry_release_energy_tol, "symmetry_release_energy_tol", "Total barrier reduction threshold in eV (default 1e-4)");

    Connect(write_traj, "write_traj", "writting optimization trajectories to file");
    Connect(nblock, "nblocks", "Trajectory interval: accepted steps in stable-v1, force evaluations in legacy (default 5)");
    Connect(trajfile, "traj");
    Connect(trajfile, "trjfile", "trajectory filename (alias for traj)");
    Connect(trjformat, "traj-format");
    Connect(trjformat, "trjformat", "trajectory format (alias for traj-format)");

    Connect(pressure, "pressure", "External pressure for VC-NEB in GPa (default 0)");
    Connect(cell_weight, "cellweight", "VC-NEB cell metric weight");
    Connect(vcneb, "vcneb", "enable VC-NEB");
    Connect(periodic_subspace,"periodic_subspace","Use periodic Gram cell and explicit periodic/open mixed coordinates");

    Connect(variable_spring, "variablespring", "variable spring constants");
    Connect(spring_max, "springmax", "maxium spring constant");
    Connect(spring_min, "springmin", "minium spring constant");
    Connect(variable_image, "variableimage", "auto insert/remove images during optimization");
    Connect(image_insert_thresh, "img_insert_thresh", "insert when segment length > thresh * avg");
    Connect(image_remove_thresh, "img_remove_thresh", "remove when segment length < thresh * avg");
    Connect(image_adjust_interval, "img_adjust_interval", "check image spacing every N steps");
    Connect(max_image_count, "max_images", "maximum number of images");
    Connect(min_image_count, "min_images", "minimum number of images");
    // Connect(force_proj_type, "ftype");
    Connect(tangent_type, "ttype", "Tangent: 1=improved, 2=original with force-residual safeguards, 3=EB", 1, 2);
    Connect(sproj_type, "stype", "usually should be the same as ttype", 1, 2);

    Connect(pot_mindist.cutoff, "mdist_r", "", 1, 2);
    Connect(pot_mindist.factor, "mdist_factor", "", 1, 2);
    Connect(pot_mindist.power, "mdist_power", "", 1, 2);
    Connect(mindist, "apply_mindist", "", 1, 2);

    Connect(init_method, "init_method", "path initialization method: linear or idpp");
    Connect(idpp_fmax, "idpp_fmax", "IDPP force convergence threshold");
    Connect(idpp_max_steps, "idpp_max_steps", "IDPP max optimization steps");
    Connect(neb_ftol, "neb_ftol", "Tolerance on complete active NEB residual (default 0.01); VC uses the selected generalized metric", 1, 2);
    Connect(neb_ftol, "gtol", "NEB convergence tolerance (alias for neb_ftol)");
    Connect(check_saddle_g_proj, "check_saddle", "Require saddle tangent force convergence (for CI-NEB)", 1, 5);
    Connect(debug_level, "debug", "only used for debug", 1, 5);
}
