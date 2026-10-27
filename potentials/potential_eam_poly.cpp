// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#include "potential_eam_poly.h"
#include "tools/chem.h"
#include "utils/compressed_stream.h"
#include <iomanip>
#include <cctype>
#include <charconv>
#include <limits>
#include <locale>
#include <map>
#include <string_view>

Potentials::PotentialEAMPoly::PotentialEAMPoly(int _npoly_rho, int _npoly_f, int _npoly_pair, int _nelems, Eigen::MatrixXd _rmax_pair, Eigen::MatrixXd _rmax_rho, int _basis_poly, double _b)
    : npoly_rho(_npoly_rho)
    , npoly_f(_npoly_f)
    , npoly_pair(_npoly_pair)
    , nelems(_nelems)
    , rmax_pair(_rmax_pair)
    , rmax_rho(_rmax_rho)
    , basis_poly(_basis_poly)
    , b(_b)
    , PotEAMAny()
{
    setup();
}

Potentials::PotentialEAMPoly::PotentialEAMPoly(const std::string &filename,
                         const std::map<std::string, int> &mappingtype2id)
    : PotEAMAny()
{
    Load(filename);
    // Apply element name mapping if provided
    if (!mappingtype2id.empty()) {
        type_name.resize(species_count);
        for (const auto &[name, id] : mappingtype2id) {
            if (id >= 0 && id < species_count)
                type_name[id] = name;
        }
    }
}

Potentials::PotentialEAMPoly::PotentialEAMPoly(GPSettings settings,
                         const std::map<std::string, int> &mappingtype2id)
    : PotEAMAny()
{
    // Determine nelems: prefer explicit setting, fall back to mapping size
    if (settings.count("nelems"))
        nelems = std::stoi(settings["nelems"]);
    else if (!mappingtype2id.empty())
        nelems = mappingtype2id.size();
    else
        nelems = 2;

    // Apply element name mapping before reading cutoffs (may infer nelems from it)
    if (!mappingtype2id.empty()) {
        type_name.resize(nelems);
        for (const auto &[name, id] : mappingtype2id) {
            if (id >= 0 && id < nelems)
                type_name[id] = name;
        }
    }

    // Read scalar params from settings with defaults
    npoly_rho   = settings.count("npoly_rho")   ? std::stoi(settings["npoly_rho"])   : 8;
    npoly_f     = settings.count("npoly_f")     ? std::stoi(settings["npoly_f"])     : 8;
    npoly_pair  = settings.count("npoly_pair")  ? std::stoi(settings["npoly_pair"])  : 8;
    basis_poly  = settings.count("basis_poly")  ? std::stoi(settings["basis_poly"])  : 3;
    b           = settings.count("b")           ? std::stod(settings["b"])           : -4./3.;

    // Read cutoffs matrices: rmax_pair_i_j, rmax_rho_i_j
    rmax_pair.resize(nelems, nelems);
    rmax_rho.resize(nelems, nelems);
    rmax_pair.setConstant(6.0);
    rmax_rho.setConstant(5.0);
    for (int i = 0; i < nelems; ++i) {
        for (int j = 0; j < nelems; ++j) {
            std::string kp = "rmax_pair_" + std::to_string(i) + "_" + std::to_string(j);
            if (settings.count(kp)) rmax_pair(i, j) = std::stod(settings[kp]);
            std::string kr = "rmax_rho_" + std::to_string(i) + "_" + std::to_string(j);
            if (settings.count(kr)) rmax_rho(i, j) = std::stod(settings[kr]);
        }
    }

    setup();
    species_count = nelems;

    // Optional: load coefficients if provided
    if (settings.count("coeffs")) {
        std::string coeff_str = settings["coeffs"];
        std::vector<std::string> tokens;
        str_split(coeff_str, tokens);
        for (size_t i = 0; i < tokens.size() && i < (size_t)n_coeffs; ++i)
            regression_coeffs[i] = std::stod(tokens[i]);
    }
}

Potentials::PotentialEAMPoly::~PotentialEAMPoly()
{
    for (auto *b : basis_rho)
        if (b) delete b;
    for (auto *b : basis_pair)
        if (b) delete b;
    basis_rho.clear();
    basis_pair.clear();
}

void Potentials::PotentialEAMPoly::setup()
{
    // Free old basis objects (in case setup() is called again, e.g. from Load())
    for (auto *b : basis_rho) if (b) delete b;
    for (auto *b : basis_pair) if (b) delete b;
    basis_rho.clear();
    basis_pair.clear();

    // 系数布局
    n_density_coeffs = nelems * nelems * npoly_rho;
    n_embeeding_coeffs = nelems * npoly_f;
    n_pair_coeffs = (nelems + 1) * nelems / 2 * npoly_pair;
    n_coeffs = n_density_coeffs + n_embeeding_coeffs + n_pair_coeffs + nelems;
    regression_coeffs.resize(n_coeffs, 0.0);
    density_coeffs = regression_coeffs.data();
    embedding_coeffs = regression_coeffs.data() + n_density_coeffs;
    pair_coeffs = regression_coeffs.data() + n_density_coeffs + n_embeeding_coeffs;
    species_coeffs = regression_coeffs.data() + n_density_coeffs + n_embeeding_coeffs + n_pair_coeffs;

    // 基类必需的成员
    species_count = nelems;
    cutoffs.resize(nelems, nelems);
    cutoffsqs.resize(nelems, nelems);
    r_mins.resize(nelems, nelems);

    // 计算全局 cutoff 并初始化 cutoffs 矩阵
    cutoff = 0;
    for (int i = 0; i < nelems; ++i) {
        for (int j = 0; j < nelems; ++j) {
            double rc = std::max(rmax_pair(i, j), rmax_rho(i, j));
            cutoffs(i, j) = rc;
            cutoffsqs(i, j) = rc * rc;
            r_mins(i, j) = 0.0;
            if (rc > cutoff) cutoff = rc;
        }
    }
    cutoffsq = cutoff * cutoff;

    // 构造 pair 基函数 (上三角: i >= j)
    basis_rho.resize(nelems * nelems);
    basis_pair.resize((nelems + 1) * nelems / 2);
    auto make_basis = [&](double rmax, int nbasis) -> AnyBasis* {
        switch (basis_poly) {
        case 1: return new RBasis_Chebyshev(0.0, rmax, nbasis, 1.0);
        case 2:
            return new RBasis_ChebyshevConfigurable(
                0.0, rmax, nbasis,
                Cutoff::make_cutoff(
                    Cutoff::CutoffType::NormalizedQuadratic, 0.0, rmax),
                XTransForm::make_shift(
                    XTransForm::ShiftType::Linear, 0.0, rmax));
        case 3: {
            Cutoff::Parameters parameters;
            parameters.b = b;
            return new RBasis_ChebyshevConfigurable(
                0.0, rmax, nbasis,
                Cutoff::make_cutoff(
                    Cutoff::CutoffType::Poly3, 0.0, rmax, parameters),
                XTransForm::make_shift(
                    XTransForm::ShiftType::Linear, 0.0, rmax));
        }
        case 4: return new RBasis_ChebyshevPoly4(0.0, rmax, nbasis, b);
        case 5: return new RBasis_ChebyshevPoly5(0.0, rmax, nbasis, b);
        default: throw std::runtime_error("basis_poly must be 2, 3, 4 or 5");
        }
    };
    for (int i = 0; i < nelems; ++i) {
        for (int j = 0; j < nelems; ++j) {
            basis_rho[get_rhoid(i, j)] = make_basis(rmax_rho(i, j), npoly_rho);
        }
    }
    for (int i = 0; i < nelems; ++i) {
        for (int j = 0; j <= i; ++j) {
            basis_pair[get_pairid(i, j)] = make_basis(rmax_pair(i, j), npoly_pair);
        }
    }
}

void Potentials::PotentialEAMPoly::getEFS(Atoms& atom, AnyNeighborhoods& nbhs)
{
    if (!add_efs) atom.resetEFS();
    for (int ii = 0; ii < nbhs.size(); ii++) {
        Neighborhood &nbh = nbhs[ii];
        double buff_site_energy = 0;
        CalcSiteEFS(nbh, buff_site_energy, buff_site_energy_ders, buff_drhodx);
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
        }
    }
}

void Potentials::PotentialEAMPoly::getEFSHessian(Atoms &atom, AnyNeighborhoods &nbhs,
    HessianMatrix &hessian) {
    assemble_hessian(atom,nbhs,hessian,nullptr,nullptr);
}

void Potentials::PotentialEAMPoly::assemble_hessian(Atoms &atom,AnyNeighborhoods &nbhs,
    HessianMatrix &hessian,Eigen::MatrixXd *cellhess,Eigen::MatrixXd *lrhess) {
    if (nbhs.size() != atom.size())
        GPERROR("Hessian neighborhoods have the wrong size");
    getEFS(atom, nbhs);
    hessian.resize(atom.size());
    Eigen::Matrix3d inverse = Eigen::Matrix3d::Zero();
    if (cellhess) {
        inverse = atom.cell_.inverse();
        *cellhess = Eigen::MatrixXd::Zero(9,9);
        *lrhess = Eigen::MatrixXd::Zero(9,3*atom.size());
    }
    std::vector<double> vals, ders, derders;
    struct Evaluation { double value = 0., first = 0., second = 0.; };
    auto evaluate = [&](AnyBasis *basis, const double *coefficients, int count, double r) {
        vals.resize(count); ders.resize(count); derders.resize(count);
        basis->calc_derder(r, vals, ders, derders);
        Evaluation out;
        for (int k = 0; k < count; ++k) {
            out.value += coefficients[k] * vals[k];
            out.first += coefficients[k] * ders[k];
            out.second += coefficients[k] * derders[k];
        }
        return out;
    };
    struct Radial { double pair_first, pair_second, density_first, density_second; };
    std::vector<Radial> radial;
    for (int i = 0; i < nbhs.size(); ++i) {
        const auto &n = nbhs[i];
        const int a = n.centertype;
        radial.resize(n.nbhcount_);
        double rho = 0.;
        for (int j = 0; j < n.nbhcount_; ++j) {
            const int b = n.chemmap_[j];
            const double r = n.distances_[j];
            Evaluation pair, density;
            if (r < rmax_pair(a, b)) {
                const int id = get_pairid(a, b);
                pair = evaluate(basis_pair[id], pair_coeffs + id * npoly_pair, npoly_pair, r);
            }
            if (r < rmax_rho(a, b)) {
                const int id = get_rhoid(a, b);
                density = evaluate(basis_rho[id], density_coeffs + id * npoly_rho, npoly_rho, r);
            }
            radial[j] = {pair.first, pair.second, density.first, density.second};
            // Self images contribute to the embedding background even though
            // their atomic displacement differences vanish below.
            rho += density.value;
        }
        double first = 0., second = 0., power = 1.;
        for (int k = 1; k <= npoly_f; ++k) {
            first += k * embedding_coeffs[a * npoly_f + k - 1] * power;
            power *= rho;
        }
        power = 1.;
        for (int k = 2; k <= npoly_f; ++k) {
            second += k * (k - 1) * embedding_coeffs[a * npoly_f + k - 1] * power;
            power *= rho;
        }
        Eigen::Matrix<double,9,1> cell_gradient = Eigen::Matrix<double,9,1>::Zero();
        std::map<int, Eigen::Vector3d> density_gradient;
        density_gradient.emplace(i, Eigen::Vector3d::Zero());
        for (int j = 0; j < n.nbhcount_; ++j) {
            const int source = n.indices_[j];
            const double r = n.distances_[j];
            const auto &v = radial[j];
            const Eigen::Vector3d u = n.pos_[j].transpose() / r;
            const double tangential = (.5 * v.pair_first + first * v.density_first) / r;
            const double longitudinal = .5 * v.pair_second + first * v.density_second;
            const Eigen::Matrix3d curvature = tangential * Eigen::Matrix3d::Identity()
                + (longitudinal - tangential) * u * u.transpose();
            if (cellhess) {
                const Eigen::RowVector3d fractional = n.pos_[j]*inverse;
                for (int a=0;a<3;++a)
                    for (int b=0;b<3;++b) {
                        cell_gradient[3*a+b] += fractional(a)*v.density_first*u(b);
                        for (int l=0;l<3;++l)
                            for (int q=0;q<3;++q)
                                (*cellhess)(3*a+b,3*l+q) += fractional(a)*curvature(b,q)*fractional(l);
                        for (int q=0;q<3;++q) {
                            const double mixed = fractional(a)*curvature(b,q);
                            (*lrhess)(3*a+b,3*source+q) += mixed;
                            (*lrhess)(3*a+b,3*i+q) -= mixed;
                        }
                    }
            }
            if (source == i) continue;
            hessian(i, i) += curvature;
            hessian(source, source) += curvature;
            hessian(i, source) -= curvature;
            hessian(source, i) -= curvature;
            const Eigen::Vector3d gradient = v.density_first * u;
            density_gradient.try_emplace(source, Eigen::Vector3d::Zero()).first->second += gradient;
            density_gradient.at(i) -= gradient;
        }
        // Aggregate repeated images by source before forming the cross term.
        for (const auto &[a_source, qa] : density_gradient)
            for (const auto &[b_source, qb] : density_gradient)
                hessian(a_source, b_source) += second * qa * qb.transpose();
        if (cellhess) {
            *cellhess += second*cell_gradient*cell_gradient.transpose();
            for (const auto &[source,q]:density_gradient)
                lrhess->middleCols(3*source,3) += second*cell_gradient*q.transpose();
        }
    }
}

void Potentials::PotentialEAMPoly::getEFSHessianCellOMP(Atoms &atom,AnyNeighborhoods &nbhs,
    HessianMatrix &hessian,Eigen::MatrixXd &cellhess,Eigen::MatrixXd &lrhess) {
    if (!atom.cell_.allFinite() || !atom.cell_.fullPivLu().isInvertible())
        GPERROR("Analytical EAMPoly cell Hessian requires a finite invertible cell");
    assemble_hessian(atom,nbhs,hessian,&cellhess,&lrhess);
}
void Potentials::PotentialEAMPoly::getEFSHessianCellSymmOMP(Atoms &atom,HessianMatrix &hessian,
    Eigen::MatrixXd &cellhess,Eigen::MatrixXd &lrhess,double) {
    Neighborhoods nbhs(atom,cutoff);
    getEFSHessianCellOMP(atom,nbhs,hessian,cellhess,lrhess);
}
void Potentials::PotentialEAMPoly::getEFSHessianCellSymm(Atoms &atom,HessianMatrix &hessian,
    Eigen::MatrixXd &cellhess,Eigen::MatrixXd &lrhess,double symprec) {
    getEFSHessianCellSymmOMP(atom,hessian,cellhess,lrhess,symprec);
}

void Potentials::PotentialEAMPoly::getEFSHessian(Atoms &atom, HessianMatrix &hessian)
{
    Neighborhoods nbhs(atom, cutoff);
    getEFSHessian(atom, nbhs, hessian);
}

void Potentials::PotentialEAMPoly::getEFSHessianSymm(Atoms &atom, HessianMatrix &hessian, double)
{
    getEFSHessian(atom, hessian);
}

void Potentials::PotentialEAMPoly::applyHessianVectorProduct(Atoms &atom,AnyNeighborhoods &nbhs,
    const Eigen::VectorXd &direction,Eigen::VectorXd &product)
{
    if (direction.size()!=3*atom.size())
        GPERROR("Hessian-vector direction has the wrong size");
    if (nbhs.size()!=atom.size())
        GPERROR("Hessian-vector neighborhoods have the wrong size");
    // External basis outputs leave the model's EFS buffers and observed rho
    // range untouched. Capacity is reused across neighbors and sites.
    std::vector<double> vals,ders,derders;
    struct Evaluation { double value=0.,first=0.,second=0.; };
    auto evaluate=[&](AnyBasis *basis,const double *coefficients,int count,double r) {
        vals.resize(count); ders.resize(count); derders.resize(count);
        basis->calc_derder(r,vals,ders,derders);
        Evaluation out;
        for (int k=0;k<count;++k) {
            out.value+=coefficients[k]*vals[k];
            out.first+=coefficients[k]*ders[k];
            out.second+=coefficients[k]*derders[k];
        }
        return out;
    };
    struct Radial { double pair_first,pair_second,density_first,density_second,alpha; };
    std::vector<Radial> radial;
    Eigen::VectorXd result=Eigen::VectorXd::Zero(direction.size());
    for (int i=0;i<nbhs.size();++i) {
        const auto &n=nbhs[i]; const int a=n.centertype;
        radial.resize(n.nbhcount_);
        double rho=0.,delta_rho=0.;
        for (int j=0;j<n.nbhcount_;++j) {
            const int b=n.chemmap_[j]; const double r=n.distances_[j];
            Evaluation pair,density;
            if (r<rmax_pair(a,b)) {
                const int id=get_pairid(a,b);
                pair=evaluate(basis_pair[id],pair_coeffs+id*npoly_pair,npoly_pair,r);
            }
            if (r<rmax_rho(a,b)) {
                const int id=get_rhoid(a,b);
                density=evaluate(basis_rho[id],density_coeffs+id*npoly_rho,npoly_rho,r);
            }
            const Eigen::Vector3d w=direction.segment<3>(3*n.indices_[j])-direction.segment<3>(3*i);
            const double alpha=n.pos_[j].dot(w)/r;
            radial[j]={pair.first,pair.second,density.first,density.second,alpha};
            rho+=density.value; delta_rho+=density.first*alpha;
        }
        // Integer powers also cover zero and negative raw densities without
        // dividing by rho or changing the model's polynomial embedding.
        double first=0.,second=0.,power=1.;
        for (int k=1;k<=npoly_f;++k) {
            first+=k*embedding_coeffs[a*npoly_f+k-1]*power;
            power*=rho;
        }
        power=1.;
        for (int k=2;k<=npoly_f;++k) {
            second+=k*(k-1)*embedding_coeffs[a*npoly_f+k-1]*power;
            power*=rho;
        }
        for (int j=0;j<n.nbhcount_;++j) {
            const double r=n.distances_[j]; const auto &v=radial[j];
            const Eigen::Vector3d u=n.pos_[j].transpose()/r;
            const Eigen::Vector3d w=direction.segment<3>(3*n.indices_[j])-direction.segment<3>(3*i);
            const double radial_first=.5*v.pair_first+first*v.density_first;
            const double radial_second=.5*v.pair_second+first*v.density_second;
            const Eigen::Vector3d contribution=radial_second*v.alpha*u
                +(radial_first/r)*(w-v.alpha*u)+second*delta_rho*v.density_first*u;
            result.segment<3>(3*n.indices_[j])+=contribution;
            result.segment<3>(3*i)-=contribution;
        }
    }
    product=std::move(result);
}

void Potentials::PotentialEAMPoly::CalcSiteEFS(
    Neighborhood& nbh, double& site_en,
    Eigen::Matrix<double, -1, 3, Eigen::RowMajor>& site_energy_ders)
{
    CalcSiteEFS(nbh, site_en, site_energy_ders, buff_drhodx);
}

void Potentials::PotentialEAMPoly::CalcSiteE(Neighborhood& nbh,
                                             double& site_en)
{
    const int itype = nbh.centertype;
    site_en = species_coeffs[itype];
    double rho = 0.0;
    for (int j = 0; j < nbh.nbhcount_; ++j) {
        const int jtype = nbh.chemmap_[j];
        const double r = nbh.distances_[j];
        if (r < rmax_pair(itype, jtype)) {
            const int pairid = get_pairid(itype, jtype);
            basis_pair[pairid]->calc_der(r);
            for (int i = 0; i < npoly_pair; ++i) {
                site_en += 0.5 * pair_coeffs[pairid * npoly_pair + i]
                         * basis_pair[pairid]->vals[i];
            }
        }
        if (r < rmax_rho(itype, jtype)) {
            const int rhoid = get_rhoid(itype, jtype);
            basis_rho[rhoid]->calc_der(r);
            for (int i = 0; i < npoly_rho; ++i) {
                rho += density_coeffs[rhoid * npoly_rho + i]
                     * basis_rho[rhoid]->vals[i];
            }
        }
    }
    double rho_power = rho;
    for (int i = 0; i < npoly_f; ++i) {
        site_en += embedding_coeffs[itype * npoly_f + i] * rho_power;
        rho_power *= rho;
    }
}

// U_i = E0_a + 1/2 \sum_{j!=i} V_{ab}(r_ij) + F_a(\sum_j \phi_{ab}(r_ij))
void Potentials::PotentialEAMPoly::CalcSiteEFS(Neighborhood& nbh, double& site_en, Eigen::Matrix<double, -1, 3, Eigen::RowMajor>& site_energy_ders, Eigen::Matrix<double, -1, 3, Eigen::RowMajor> &drhodx)
{
    site_energy_ders.resize(nbh.nbhcount_, 3);
    site_energy_ders.setZero();
    drhodx.resize(nbh.nbhcount_, 3);
    drhodx.setZero();
    int itype = nbh.centertype;
    site_en = species_coeffs[itype];
    double rho = 0;
    // pair interaction
    for (int j = 0; j < nbh.nbhcount_; ++j) {
        int jtype = nbh.chemmap_[j];
        double rc = rmax_pair(itype, jtype);
        double rd = rmax_rho(itype, jtype);
        if (nbh.distances_[j] < rc) {
            int pairid = get_pairid(itype, jtype);
            basis_pair[pairid]->calc_der(nbh.distances_[j]);
            for (int i = 0; i < npoly_pair; ++i) {
                site_en += 0.5 * pair_coeffs[pairid * npoly_pair + i] * basis_pair[pairid]->vals[i];
                site_energy_ders.row(j) += 0.5 * pair_coeffs[pairid * npoly_pair + i] * basis_pair[pairid]->ders[i] * nbh.pos_[j] / nbh.distances_[j];
            }
        }
        if (nbh.distances_[j] < rd) {
            int rhoid = get_rhoid(itype, jtype);
            basis_rho[rhoid]->calc_der(nbh.distances_[j]);
            for (int i = 0; i < npoly_rho; ++i) {
                rho += density_coeffs[rhoid * npoly_rho + i] * basis_rho[rhoid]->vals[i];
                drhodx.row(j) += density_coeffs[rhoid * npoly_rho + i] * basis_rho[rhoid]->ders[i] * nbh.pos_[j] / nbh.distances_[j];
            }
        }
    }
    if (rho_min > rho) [[unlikely]] {
        rho_min = rho;
    }
    if (rho_max < rho) [[unlikely]] {
        rho_max = rho;
    }
    // embedding term
    double dFdrho = 0;
    double prho = 1.0;
    for (int i = 1; i <= npoly_f; ++i) {
        site_en += embedding_coeffs[itype * npoly_f + i - 1] * prho * rho;
        dFdrho += i * embedding_coeffs[itype * npoly_f + i - 1] * prho;
        prho *= rho;
    }
    for (int j = 0; j < nbh.nbhcount_; ++j) {
        int jtype = nbh.chemmap_[j];
        double rd = rmax_rho(itype, jtype);
        if (nbh.distances_[j] < rd) {
            site_energy_ders.row(j) += dFdrho * drhodx.row(j);
        }
    }
}

/**
 * Tag-value file format. Lines starting with # are comments, tags may appear in any order.
 * If the "coeffs" tag is absent, coefficients are zero-initialised (fit-ready template).
 *
 *   nelems       2
 *   npoly_rho    4
 *   npoly_f      3
 *   npoly_pair   4
 *   basis_poly   3
 *   b            -1.3333333
 *   rmax_pair    6.0  5.5
 *                5.5  5.0
 *   rmax_rho     5.0  4.5
 *                4.5  4.0
 *   rho_min      0.0
 *   rho_max      10.0
 *   coeffs       0.5 -0.3  0.1  ...   (optional)
 */
void Potentials::PotentialEAMPoly::Load(const std::string &filename)
{
    auto ifs = open_ifile(filename);
    if (!ifs) throw std::runtime_error("Failed to open file: " + filename);
    load_stream(*ifs, filename);
}

void Potentials::PotentialEAMPoly::Load(std::istream &is)
{
    load_stream(is, "EAMPoly stream");
}

void Potentials::PotentialEAMPoly::load_stream(std::istream &is, const std::string &source)
{
    if (!is) throw std::runtime_error("Failed to read " + source);
    struct ExceptionMaskGuard {
        std::istream &stream;
        std::ios_base::iostate mask = stream.exceptions();
        ~ExceptionMaskGuard() {
            // exceptions() restores the mask before reporting an existing
            // error state. Do not replace the parser's error during unwinding.
            try { stream.exceptions(mask); }
            catch (const std::ios_base::failure &) {}
        }
    } exception_mask{is};
    is.exceptions(std::ios_base::goodbit);
    bool has_coeffs = false, has_pair_cutoffs = false, has_density_cutoffs = false;
    std::vector<double> pair_cutoffs, density_cutoffs, coeffs;
    std::vector<double> *values = nullptr;
    int next_nelems = 0, next_npoly_rho = 0, next_npoly_f = 0, next_npoly_pair = 0;
    int next_basis_poly = 3;
    double next_b = -4./3., next_rho_min = 0.0, next_rho_max = 0.0;

    // Read first, then initialize: arrays may precede dimensions or span lines.
    // Repeated cutoff tags also accept the rows emitted by older Save versions.
    std::string tag;
    auto parse_number = [&](const std::string &token) {
        // from_chars is independent of both the stream and global C locales.
        // Strip the sign and 0x prefix explicitly to retain the existing
        // positive-sign, hexadecimal and negative-zero input syntax.
        std::string_view text(token);
        const bool negative = !text.empty() && text.front() == '-';
        if (!text.empty() && (text.front() == '+' || text.front() == '-'))
            text.remove_prefix(1);
        const bool hexadecimal = text.starts_with("0x") || text.starts_with("0X");
        if (hexadecimal) text.remove_prefix(2);
        double value = 0.0;
        const auto result = std::from_chars(text.data(), text.data() + text.size(), value,
            hexadecimal ? std::chars_format::hex : std::chars_format::general);
        if (text.empty() || text.front() == '-' || text.front() == '+'
            || result.ec != std::errc{} || result.ptr != text.data() + text.size())
            throw std::runtime_error("Invalid numeric value '" + token + "' in " + source);
        if (!std::isfinite(value)) throw std::runtime_error("Non-finite value in " + source);
        return negative ? -value : value;
    };
    auto read_scalar = [&]<class T>(T &value) {
        std::string token;
        if (!(is >> token)) throw std::runtime_error("Invalid " + tag + " in " + source);
        if constexpr (std::is_floating_point_v<T>) value = parse_number(token);
        else {
            std::string_view text(token);
            if (!text.empty() && text.front() == '+') text.remove_prefix(1);
            const auto result = std::from_chars(text.data(), text.data() + text.size(), value);
            if (text.empty() || (token.front() == '+' && text.front() == '-')
                || result.ec != std::errc{} || result.ptr != text.data() + text.size())
                throw std::runtime_error("Invalid " + tag + " in " + source + ": " + token);
        }
        if (!std::isfinite(static_cast<double>(value)))
            throw std::runtime_error("Non-finite " + tag + " in " + source);
    };
    while (is >> tag) {
        if (tag.front() == '#') { std::getline(is, tag); continue; }
        if (tag == "rmax_pair") { values = &pair_cutoffs; has_pair_cutoffs = true; continue; }
        if (tag == "rmax_rho") { values = &density_cutoffs; has_density_cutoffs = true; continue; }
        if (tag == "coeffs") { values = &coeffs; has_coeffs = true; continue; }
        std::string lower = tag;
        std::transform(lower.begin(), lower.end(), lower.begin(),
                       [](unsigned char ch) { return std::tolower(ch); });
        if (values && (std::isdigit(static_cast<unsigned char>(tag.front()))
                       || tag.front() == '-' || tag.front() == '+' || tag.front() == '.'
                       || lower.starts_with("nan") || lower.starts_with("inf"))) {
            values->push_back(parse_number(tag));
            continue;
        }
        values = nullptr;
        if      (tag == "nelems")     read_scalar(next_nelems);
        else if (tag == "npoly_rho")  read_scalar(next_npoly_rho);
        else if (tag == "npoly_f")    read_scalar(next_npoly_f);
        else if (tag == "npoly_pair") read_scalar(next_npoly_pair);
        else if (tag == "basis_poly") read_scalar(next_basis_poly);
        else if (tag == "b")          read_scalar(next_b);
        else if (tag == "rho_min")    read_scalar(next_rho_min);
        else if (tag == "rho_max")    read_scalar(next_rho_max);
        else { std::getline(is, tag); continue; } // skip unknown tag
    }
    if (is.bad() || (is.fail() && !is.eof()))
        throw std::runtime_error("Failed to read " + source);
    // The terminating token extraction reaches EOF normally, not a parse error.
    if (is.eof()) is.clear(is.rdstate() & ~std::ios_base::failbit);
    if (next_nelems <= 0 || next_npoly_rho <= 0 || next_npoly_f <= 0 || next_npoly_pair <= 0)
        throw std::runtime_error("Missing required parameters in " + source);
    if (next_basis_poly < 1 || next_basis_poly > 5)
        throw std::runtime_error("basis_poly must be 1-5 in " + source);

    // setup() stores sizes in int; validate before its multiplications or allocations.
    const size_t species = next_nelems;
    size_t expected_coeffs = 0;
    auto add_coeffs = [&](size_t blocks, size_t count) {
        if (blocks > (std::numeric_limits<int>::max() - expected_coeffs) / count)
            throw std::runtime_error("Too many EAMPoly parameters in " + source);
        expected_coeffs += blocks * count;
    };
    add_coeffs(species * species, next_npoly_rho);
    add_coeffs(species, next_npoly_f);
    add_coeffs((species + 1) * species / 2, next_npoly_pair);
    add_coeffs(species, 1);
    if (coeffs.size() > expected_coeffs)
        throw std::runtime_error("Too many coefficients in " + source);

    auto make_cutoffs = [&](const std::vector<double> &data, bool supplied,
                            double default_value, const std::string &tag) {
        if (supplied && data.size() != species * species)
            throw std::runtime_error("Wrong number of " + tag + " values in " + source);
        Eigen::MatrixXd matrix(next_nelems, next_nelems);
        for (int i = 0; i < next_nelems; ++i) {
            for (int j = 0; j < next_nelems; ++j) {
                const double value = supplied ? data[i * next_nelems + j] : default_value;
                if (value <= 0.0) throw std::runtime_error("Non-positive " + tag + " in " + source);
                matrix(i, j) = value;
            }
        }
        return matrix;
    };
    auto next_pair = make_cutoffs(pair_cutoffs, has_pair_cutoffs, 6.0, "rmax_pair");
    auto next_density = make_cutoffs(density_cutoffs, has_density_cutoffs, 5.0, "rmax_rho");
    PotentialEAMPoly next(next_npoly_rho, next_npoly_f, next_npoly_pair, next_nelems,
                         std::move(next_pair), std::move(next_density), next_basis_poly, next_b);
    std::copy(coeffs.begin(), coeffs.end(), next.regression_coeffs.begin());
    next.rho_min = next_rho_min;
    next.rho_max = next_rho_max;
    if (has_coeffs && coeffs.size() < expected_coeffs) {
        std::cerr << "Warning: " << source << ": read " << coeffs.size() << " of "
                  << expected_coeffs << " coefficients; missing coefficients initialized to zero for training\n";
    }

    // Commit the fully initialized model without copying owning basis pointers.
    // The temporary then destroys the previous model's basis objects.
    using std::swap;
    swap(nelems, next.nelems);
    swap(npoly_rho, next.npoly_rho);
    swap(npoly_f, next.npoly_f);
    swap(npoly_pair, next.npoly_pair);
    swap(basis_poly, next.basis_poly);
    swap(b, next.b);
    swap(n_density_coeffs, next.n_density_coeffs);
    swap(n_embeeding_coeffs, next.n_embeeding_coeffs);
    swap(n_pair_coeffs, next.n_pair_coeffs);
    swap(n_coeffs, next.n_coeffs);
    regression_coeffs.swap(next.regression_coeffs);
    swap(density_coeffs, next.density_coeffs);
    swap(embedding_coeffs, next.embedding_coeffs);
    swap(pair_coeffs, next.pair_coeffs);
    swap(species_coeffs, next.species_coeffs);
    basis_rho.swap(next.basis_rho);
    basis_pair.swap(next.basis_pair);
    rmax_pair.swap(next.rmax_pair);
    rmax_rho.swap(next.rmax_rho);
    swap(species_count, next.species_count);
    swap(cutoff, next.cutoff);
    swap(cutoffsq, next.cutoffsq);
    cutoffs.swap(next.cutoffs);
    cutoffsqs.swap(next.cutoffsqs);
    r_mins.swap(next.r_mins);
    swap(rho_min, next.rho_min);
    swap(rho_max, next.rho_max);
}

void Potentials::PotentialEAMPoly::Save(const std::string &potfile)
{
    if (save_gpm_if_requested(*this, potfile)) return;
    auto ofs = open_ofile(potfile);
    if (!ofs) throw std::runtime_error("Failed to open file: " + potfile);
    Save(*ofs);
    if (!close_ofile(ofs)) throw std::runtime_error("Failed to write potential file: " + potfile);
}

void Potentials::PotentialEAMPoly::Save(std::ostream &os)
{
    struct FormatGuard {
        std::ostream &stream;
        std::ios_base::fmtflags flags = stream.flags();
        std::streamsize precision = stream.precision();
        std::locale locale = stream.getloc();
        ~FormatGuard() {
            stream.flags(flags);
            stream.precision(precision);
            stream.imbue(locale);
        }
    } format{os};
    os.imbue(std::locale::classic());
    os << std::dec << std::defaultfloat << std::setprecision(std::numeric_limits<double>::max_digits10);
    os << "# EAM potential (polynomial basis)\n";
    os << "nelems       " << nelems << "\n";
    os << "npoly_rho    " << npoly_rho << "\n";
    os << "npoly_f      " << npoly_f << "\n";
    os << "npoly_pair   " << npoly_pair << "\n";
    os << "basis_poly   " << basis_poly << "\n";
    os << "b            " << b << "\n\n";

    os << "# Pair cutoffs (nelems x nelems, row-major)\n";
    os << "rmax_pair    ";
    for (int i = 0; i < nelems; ++i) {
        for (int j = 0; j < nelems; ++j)
            os << rmax_pair(i, j) << " ";
        os << "\n";
    }
    os << "\n# Density cutoffs (nelems x nelems, row-major)\n";
    os << "rmax_rho     ";
    for (int i = 0; i < nelems; ++i) {
        for (int j = 0; j < nelems; ++j)
            os << rmax_rho(i, j) << " ";
        os << "\n";
    }
    os << "\n# Rho range for embedding function\n";
    os << "rho_min      " << rho_min << "\n";
    os << "rho_max      " << rho_max << "\n\n";

    os << "# Regression coefficients ("
        << n_density_coeffs << " density + "
        << n_embeeding_coeffs << " embed + "
        << n_pair_coeffs << " pair + "
        << nelems << " species = " << n_coeffs << ")\n";
    os << "coeffs       ";
    for (int i = 0; i < n_coeffs; ++i) {
        os << regression_coeffs[i] << " ";
    }
    os << "\n";
    if (!os) throw std::runtime_error("Failed to write EAMPoly stream");
}

void Potentials::PotentialEAMPoly::random_init()
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dis(-1.0, 1.0);
    for (int i = 0; i < n_coeffs; ++i) {
        regression_coeffs[i] = dis(gen);
    }
}

void Potentials::PotentialEAMPoly::nl_get_coeffs(double* x) const
{
    for (int i = 0; i < n_coeffs; ++i) {
        x[i] = regression_coeffs[i];
    }
}

void Potentials::PotentialEAMPoly::nl_set_coeffs(const double* x)
{
    for (int i = 0; i < n_coeffs; ++i) {
        regression_coeffs[i] = x[i];
    }
}

void Potentials::PotentialEAMPoly::AccumulateFPGard(Neighborhood& nbh,
            double* out_gard_accumulator,
            double se_weight,
            Eigen::Matrix<double, -1, 3, Eigen::RowMajor>* f_weight)
{
    int itype = nbh.centertype;
    buff_drhodx.resize(nbh.nbhcount_, 3);
    buff_drhodx.setZero();
    // species coeffs
    out_gard_accumulator[n_density_coeffs + n_embeeding_coeffs + n_pair_coeffs + nbh.centertype] += se_weight;
    // density coeffs precalc
    double rho = 0;
    for (int j = 0; j < nbh.nbhcount_; ++j) {
        int jtype = nbh.chemmap_[j];
        double rd = rmax_rho(itype, jtype);
        if (nbh.distances_[j] < rd) {
            int rhoid = get_rhoid(itype, jtype);
            basis_rho[rhoid]->calc_der(nbh.distances_[j]);
            for (int i = 0; i < npoly_rho; ++i) {
                rho += density_coeffs[rhoid * npoly_rho + i] * basis_rho[rhoid]->vals[i];
                buff_drhodx.row(j) += density_coeffs[rhoid * npoly_rho + i] * basis_rho[rhoid]->ders[i] * nbh.pos_[j] / nbh.distances_[j];
            }
        }
    }
    // embeeding coeffs: F(rho) = sum_{k=1}^{npoly_f} l_k * rho^k
    double prho = 1.0;
    double dFdrho = 0;
    for (int i = 1; i <= npoly_f; ++i) {
        out_gard_accumulator[n_density_coeffs + itype * npoly_f + i - 1] += se_weight * prho * rho;
        dFdrho += i * embedding_coeffs[itype * npoly_f + i - 1] * prho;
        if (f_weight != nullptr) {
            for (int j = 0; j < nbh.nbhcount_; ++j) {
                int jtype = nbh.chemmap_[j];
                double rd = rmax_rho(itype, jtype);
                if (nbh.distances_[j] < rd) {
                    for (int a = 0; a < 3; ++a) {
                        out_gard_accumulator[n_density_coeffs + itype * npoly_f + i - 1] += (*f_weight)(j, a) * i * prho * buff_drhodx.row(j)(a);
                    }
                }
            }
        }
        prho *= rho;
    }
    // F''(rho) = sum_{k=2}^{npoly_f} k*(k-1) * l_k * rho^(k-2)
    double dF2drho = 0;
    {
        double pprho = 1.0; // rho^0
        for (int i = 2; i <= npoly_f; ++i) {
            dF2drho += i * (i - 1) * embedding_coeffs[itype * npoly_f + i - 1] * pprho;
            pprho *= rho;
        }
    }
    // density coeffs
    // F'' couples a density-parameter change at any neighbor to all forces.
    const double weighted_drho = f_weight == nullptr ? 0.0
        : (f_weight->array() * buff_drhodx.array()).sum();
    for (int j = 0; j < nbh.nbhcount_; ++j) {
        int jtype = nbh.chemmap_[j];
        double rd = rmax_rho(itype, jtype);
        if (nbh.distances_[j] < rd) {
            int rhoid = get_rhoid(itype, jtype);
            basis_rho[rhoid]->calc_der(nbh.distances_[j]);
            double r_hat_inv = 1.0 / nbh.distances_[j];
            for (int i = 0; i < npoly_rho; ++i) {
                out_gard_accumulator[rhoid * npoly_rho + i] +=
                    (se_weight * dFdrho + dF2drho * weighted_drho) * basis_rho[rhoid]->vals[i];
                if (f_weight != nullptr) {
                    for (int a = 0; a < 3; ++a) {
                        out_gard_accumulator[rhoid * npoly_rho + i] += (*f_weight)(j, a)
                            * dFdrho * basis_rho[rhoid]->ders[i] * nbh.pos_[j](a) * r_hat_inv;
                    }
                }
            }
        }
    }
    // pair_coeffs (offset: n_density_coeffs + n_embeeding_coeffs)
    for (int j = 0; j < nbh.nbhcount_; ++j) {
        int jtype = nbh.chemmap_[j];
        if (nbh.distances_[j] < rmax_pair(itype, jtype)) {
            int pairid = get_pairid(itype, jtype);
            int pair_offset = n_density_coeffs + n_embeeding_coeffs + pairid * npoly_pair;
            basis_pair[pairid]->calc_der(nbh.distances_[j]);
            for (int i = 0; i < npoly_pair; ++i) {
                out_gard_accumulator[pair_offset + i] += se_weight * 0.5 * basis_pair[pairid]->vals[i];
                if (f_weight != nullptr) {
                    for (int a = 0; a < 3; ++a) {
                        out_gard_accumulator[pair_offset + i] += (*f_weight)(j, a) * 0.5 * basis_pair[pairid]->ders[i] * nbh.pos_[j](a) / nbh.distances_[j];
                    }
                }
            }
        }
    }
}

void Potentials::PotentialEAMPoly::train_init()
{
    reg_L1_vector.assign(n_coeffs, 0.0);
    reg_L2_vector.assign(n_coeffs, 0.0);
}

void Potentials::PotentialEAMPoly::AddPenaltyGard(double weight, double &loss, double *loss_gard)
{
    for (int i = 0; i < n_coeffs; ++i) {
        double l2 = reg_L2_vector[i] * weight;
        loss += 0.5 * l2 * regression_coeffs[i] * regression_coeffs[i];
        loss_gard[i] += l2 * regression_coeffs[i];
    }
}

double Potentials::PotentialEAMPoly::eval_pair(int itype, int jtype, double r)
{
    if (r <= 0 || r >= rmax_pair(itype, jtype)) return 0.0;
    int pairid = get_pairid(itype, jtype);
    basis_pair[pairid]->calc_val(r);
    double v = 0;
    for (int k = 0; k < npoly_pair; ++k)
        v += pair_coeffs[pairid * npoly_pair + k] * basis_pair[pairid]->vals[k];
    return v;
}

double Potentials::PotentialEAMPoly::eval_density(int itype, int jtype, double r)
{
    if (r < 0 || r >= rmax_rho(itype, jtype)) return 0.0;
    int rhoid = get_rhoid(itype, jtype);
    basis_rho[rhoid]->calc_val(r);
    double v = 0;
    for (int k = 0; k < npoly_rho; ++k)
        v += density_coeffs[rhoid * npoly_rho + k] * basis_rho[rhoid]->vals[k];
    return v;
}

double Potentials::PotentialEAMPoly::eval_embed(int itype, double rho)
{
    // HE format allows negative rho
    double f = 0, prho = rho;
    for (int k = 1; k <= npoly_f; ++k) {
        f += embedding_coeffs[itype * npoly_f + k - 1] * prho;
        prho *= rho;
    }
    return f;
}

void Potentials::PotentialEAMPoly::OutputLAMMPSset(std::string outfile, int nr, int nrho, double rhomin, double rhomax)
{
    // Auto-estimate rho range if not specified (-1 sentinel)
    if (rhomin < 0 && rhomax < 0) {
        rhomin = std::min(0.0, rho_min) * 1.2;
        rhomax = std::max(rho_max * 1.2, 1.0);
        if (rhomax <= rhomin) { rhomin = -1.0; rhomax = 10.0; }
    }

    if (nr < 3 || nrho < 3 || !std::isfinite(cutoff) || cutoff <= 0
        || !std::isfinite(rhomin) || !std::isfinite(rhomax) || rhomax <= rhomin)
        throw std::runtime_error("Invalid EAM export grid for " + outfile);
    auto names = type_name;
    if (names.empty()) {
        if (species_count < 1 || species_count > 118)
            throw std::runtime_error("Invalid EAM species count for " + outfile);
        for (int i = 0; i < species_count; ++i) names.push_back(Chemicals::chemical_symbols[i + 1]);
    }
    if (names.size() != static_cast<size_t>(species_count))
        throw std::runtime_error("Incomplete EAM element names for " + outfile);
    for (const auto &name : names) {
        const auto element = Chemicals::atomic_numbers.find(name);
        if (element == Chemicals::atomic_numbers.end() || element->second <= 0 || element->second > 118)
            throw std::runtime_error("Unknown EAM element " + name + " for " + outfile);
    }
    auto ofs = open_ofile(outfile);
    if (!ofs) throw std::runtime_error("Failed to open EAM output file: " + outfile);
    *ofs << std::setprecision(17);
    *ofs << "Created by IMRMLP PotentialEAMPoly\n";
    *ofs << "For LAMMPS eam/he pair_style\n";
    *ofs << "In setfl format\n";
    *ofs << "   " << species_count;
    for (const auto &name : names) *ofs << " " << name;
    *ofs << "\n";

    double delta_r = cutoff / static_cast<double>(nr - 1);
    double drho = (rhomax - rhomin) / static_cast<double>(nrho - 1);
    const int mlrs = 5;

    // HE format: Nrho, drho, Nr, dr, cutoff, rhomax
    *ofs << "  " << nrho << "  " << drho << "  " << nr << "  "
        << delta_r << "  " << cutoff << "  " << rhomax << '\n';

    for (int i = 0; i < species_count; ++i) {
        // Element header: atomic number, mass, lattice constant, lattice type
        const int atomic_number = Chemicals::atomic_numbers.at(names[i]);
        *ofs << "  " << atomic_number << "  " << Chemicals::atomic_masses[atomic_number]
             << "  " << 5.0 << "  fcc";

        // Embedding function F(rho): from rhomin to rhomax
        for (int ii = 0; ii < nrho; ++ii) {
            double rho = rhomin + drho * static_cast<double>(ii);
            if (ii % mlrs == 0) *ofs << '\n';
            *ofs << species_coeffs[i] + eval_embed(i, rho) << " ";
        }

        // The section species i is the neighbor; rows enumerate host species j.
        for (int j = 0; j < species_count; ++j) {
            for (int ii = 0; ii < nr; ++ii) {
                double rcurr = delta_r * static_cast<double>(ii);
                if (ii % mlrs == 0) *ofs << '\n';
                *ofs << eval_density(j, i, rcurr) << " ";
            }
        }
        *ofs << '\n';
    }

    // Pair tables store r * V_ab(r), with i >= j.
    for (int i = 0; i < species_count; ++i) {
        for (int j = 0; j <= i; ++j) {
            for (int ii = 0; ii < nr; ++ii) {
                double rcurr = delta_r * static_cast<double>(ii);
                if (ii % mlrs == 0) *ofs << '\n';
                *ofs << rcurr * eval_pair(i, j, rcurr) << " ";
            }
        }
    }
    *ofs << '\n';
    if (!close_ofile(ofs)) throw std::runtime_error("Failed to write EAM output file: " + outfile);
}
