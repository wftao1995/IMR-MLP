// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#include "relax_new.h"
#include <charconv>
#include <limits>


// Modifiers
void FilterFixCM::forward(RelaxationDrvV2* pdrv)
{
    double ff[3] { 0, 0, 0 };
    auto& g = pdrv->g;
    // auto& x = pdrv->x;
    auto& atom = pdrv->atom;
    for (int i = 0; i < atom.size(); ++i) {
        for (int a = 0; a < 3; ++a) {
            ff[a] += atom.forces_[i](a);
        }
    }
    for (int a = 0; a < 3; ++a) {
        ff[a] /= atom.size();
        for (int i = 0; i < atom.size(); ++i) {
            atom.forces_[i](a) -= ff[a];
        }
    }
}

void FilterFixSmallGD::forward(RelaxationDrvV2* pdrv)
{
    auto& g = pdrv->g;
    auto& tol = pdrv->tol;
    for (int i = 0; i < g.size(); ++i) {
        if (abs(g[i]) < tol.ztol)
            g[i] = 0;
    }
}

void FilterLF::forward(RelaxationDrvV2* pdrv)
{
    auto& x = pdrv->x;
    auto& g = pdrv->g;
    x = matrixA * x + vectorb;
    // g = invmatrixA.transpose() * g;
    g = inv_qr.solve(g);
}

void FilterLF::reverse(RelaxationDrvV2* pdrv)
{
    auto& x = pdrv->x;
    // x = invmatrixA * (x - vectorb);
    x = qr.solve(x - vectorb);
}

void FilterLF::solve()
{
    qr = Eigen::ColPivHouseholderQR<Eigen::MatrixXd>(matrixA);
    inv_qr = Eigen::ColPivHouseholderQR<Eigen::MatrixXd>(matrixA.transpose());
    if (qr.isInvertible())
        invmatrixA = qr.inverse();
    else {
        throw std::runtime_error("matrixA is not invertible");
    }
}

void FilterLF::init(RelaxationDrvV2* pdrv, void* ptr)
{
    std::string lfile = *reinterpret_cast<std::string*>(ptr);
    std::ifstream ifs(lfile);
    if (!ifs) {
        throw std::runtime_error("Failed to open file: " + lfile);
    }
    // now load matrixA, vectorb
    int k, n;
    ifs >> k >> n;
    if (k != n) {
        throw std::runtime_error("k != n");
    }
    if (n != pdrv->x.size()) {
        throw std::runtime_error("n != x.size()");
    }
    matrixA.resize(k, n);
    vectorb.resize(k);
    for (int i = 0; i < k; ++i) {
        for (int j = 0; j < n; ++j) {
            ifs >> matrixA(i, j);
        }
    }
    for (int i = 0; i < k; ++i) {
        ifs >> vectorb(i);
    }
    solve();
}

void FilterLF::forward_scaling(RelaxationDrvV2* pdrv)
{
    auto scal_b = pdrv->prec_scaling;
    pdrv->prec_scaling.setZero();
    // d2F/dyi = \sum_{m,n} A^-1_{im} A^-1_{in} d2F/dxm d2F/dxn, need FORCE_CONSTANT. TODO: solve this
    // assume H_{nm,m!=n} == 0
    for (int i = 0; i < pdrv->x.size(); ++i) {
        for (int m = 0; m < pdrv->x.size(); ++m) {
            pdrv->prec_scaling[i] += invmatrixA(m, i) * invmatrixA(m, i) * scal_b[m];
        }
    }
}

void FilterLC::forward(RelaxationDrvV2* pdrv)
{
    auto& x = pdrv->x;
    auto& g = pdrv->g;
    if (algo == 0) { // this is more stable that T
        Eigen::VectorXd v = Tcore.solve(x - b0);
        x = v;
    } else if (algo == 1) {
        x = T * (x - b0);
    }
    g = Q2.transpose() * g;
}

void FilterLC::reverse(RelaxationDrvV2* pdrv)
{
    auto& x = pdrv->x;
    auto& g = pdrv->g;
    x = b0 + Q2 * x;
    g.resize(x.size());
    // g.setZero();
}

void FilterLC::init(RelaxationDrvV2* pdrv, void* ptr)
{
    std::string lfile = *reinterpret_cast<std::string*>(ptr);
    std::ifstream ifs(lfile);
    if (!ifs) {
        throw std::runtime_error("Failed to open file: " + lfile);
    }
    // now load matrixA, vectorb
    ifs >> k >> n;
    if (k > n) {
        throw std::runtime_error("k > n");
    }
    if (n != pdrv->x.size()) {
        throw std::runtime_error("n != x.size()");
    }
    matrixA.resize(k, n);
    vectorb.resize(k);
    for (int i = 0; i < k; ++i) {
        for (int j = 0; j < n; ++j) {
            ifs >> matrixA(i, j);
        }
    }
    for (int i = 0; i < k; ++i) {
        ifs >> vectorb(i);
    }
    solve();
}

void FilterLC::forward_scaling(RelaxationDrvV2* pdrv)
{
    auto scal_b = pdrv->prec_scaling;
    // d2F/dyi = \sum_{m,n} Q2_{im} Q2_{in} d2F/dxm d2F/dxn, need FORCE_CONSTANT. TODO: solve this
    // assume H_{nm,m!=n} == 0
    pdrv->prec_scaling.resize(n - k);
    pdrv->prec_scaling.setZero();
    for (int i = 0; i < pdrv->prec_scaling.size(); ++i) {
        for (int m = 0; m < scal_b.size(); ++m) {
            pdrv->prec_scaling[i] += Q2(m, i) * Q2(m, i) * scal_b[m];
        }
    }
}

void FilterLC::solve()
{
    /*
     * Linear constraint filter:
     *   A x = b
     * is parameterized as:
     *   x = b0 + Q2 v
     * where b0 is one feasible point (A b0 = b), and columns of Q2 span
     * null(A).  The optimizer then works in the reduced variable v with
     * dimension n-k, so every reverse() result satisfies the linear constraint.
     *
     * The previous implementation attempted to derive Q1/Q2 from a QR
     * decomposition of A directly.  For A with shape k x n, that Q is not the
     * n-dimensional variable-space basis needed here, and the block extraction
     * for Q2 had incompatible dimensions.  It also depended on member k/n
     * being set correctly; those members were previously shadowed in init().
     * Keep the old sketch below for reference, but use SVD because V gives an
     * explicit orthonormal basis in variable space and V.rightCols(n-k) is the
     * nullspace basis when A has full row rank.
     *
     * Legacy sketch:
     *
     * Eigen::HouseholderQR<Eigen::MatrixXd> qr(matrixA);
     * Eigen::MatrixXd matQ = qr.householderQ();
     * Eigen::MatrixXd matR = qr.matrixQR().triangularView<Eigen::Upper>();
     * R1 = matR.block(0, 0, k, k);
     * Q1 = matQ.block(0, 0, k, n);
     * Q2 = matQ.block(0, k, n, n);
     * b0.noalias() = Q1 * (R1.inverse()).transpose() * vectorb;
     * T.noalias() = (Q2.transpose() * Q2).inverse() * Q2.transpose();
     * Tcore.compute(Q2);
     */
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(matrixA, Eigen::ComputeFullU | Eigen::ComputeFullV);
    double tol = std::max(matrixA.rows(), matrixA.cols()) *
                 std::numeric_limits<double>::epsilon() *
                 svd.singularValues().array().abs().maxCoeff();
    int rank = 0;
    for (int i = 0; i < svd.singularValues().size(); ++i) {
        if (svd.singularValues()[i] > tol) {
            ++rank;
        }
    }
    if (rank != k) {
        throw std::runtime_error("Linear constraint matrix must have full row rank");
    }

    Eigen::MatrixXd V = svd.matrixV();
    Q1 = V.leftCols(k);
    Q2 = V.rightCols(n - k);
    b0 = matrixA.transpose() * (matrixA * matrixA.transpose()).ldlt().solve(vectorb);
    T = Q2.transpose();
    Tcore.compute(Q2);
}

std::vector<int> FilterSelDyn::read_mask(std::istream &input, Eigen::Index full_size)
{
    const std::string prefix = "Invalid optimization-coordinate mask: ";
    std::string token;
    long long count = -1;
    if (!(input >> token))
        throw std::invalid_argument(prefix + "missing dimension");
    const auto result = std::from_chars(token.data(), token.data() + token.size(), count);
    if (result.ec != std::errc{} || result.ptr != token.data() + token.size() ||
        count < 0 || full_size < 0 || count != full_size || count > std::numeric_limits<int>::max())
        throw std::invalid_argument(prefix + "dimension must equal the full optimizer size " + std::to_string(full_size));
    std::vector<int> selected;
    for (int i = 0; i < count; ++i) {
        if (!(input >> token) || (token != "0" && token != "1"))
            throw std::invalid_argument(prefix + "expected 0 or 1 at coordinate " + std::to_string(i));
        if (token == "1") selected.push_back(i);
    }
    if (input >> token)
        throw std::invalid_argument(prefix + "unexpected trailing data");
    if (!input.eof())
        throw std::invalid_argument(prefix + "failed while reading the file");
    return selected;
}

void FilterSelDyn::forward(RelaxationDrvV2* pdrv)
{
    auto& x = pdrv->x;
    auto& g = pdrv->g;
    x_save = x;
    g_new.resize(ids_pass.size());
    x.resize(ids_pass.size());
    for (int i = 0; i < ids_pass.size(); ++i) {
        x[i] = x_save[ids_pass[i]];
        g_new[i] = g[ids_pass[i]];
    }
    g = g_new;
}

void FilterSelDyn::reverse(RelaxationDrvV2* pdrv)
{
    auto& x = pdrv->x;
    auto& g = pdrv->g;
    for (int i = 0; i < ids_pass.size(); ++i) {
        x_save[ids_pass[i]] = x[i];
    }
    x = x_save;
    g.resize(x.size());
    // g.setZero();
}

void FilterSelDyn::forward_scaling(RelaxationDrvV2* pdrv)
{
    auto x = pdrv->prec_scaling;
    pdrv->prec_scaling.resize(ids_pass.size());
    for (int i = 0; i < ids_pass.size(); ++i) {
        pdrv->prec_scaling[i] = x[ids_pass[i]];
    }
}

void FilterSelDyn::init(RelaxationDrvV2* pdrv, void* ptr)
{
    std::vector<int> selected;
    if (ptr) {
        selected = *reinterpret_cast<std::vector<int>*>(ptr);
    } else {
        // lattice related
        if (pdrv->flags.fix_shape) {
            selected.push_back(0);
        } else if (pdrv->flags.relax_cell && pdrv->subspace_active) {
            int k=0;const auto &axes=pdrv->subspace.axes();
            for(int i=0;i<pdrv->subspace.rank();++i)for(int j=0;j<=i;++j,++k)
                if(pdrv->atom.sel_dyncell_(axes[i],axes[j])>0)selected.push_back(k);
        } else if (pdrv->flags.relax_cell) {
            if (pdrv->flags.relax_cell_use_strain && pdrv->flags.symm_strain) {
                for (int i = 0; i < 6; ++i) {
                    if (pdrv->symm_strain_mask[i] > 0) {
                        selected.push_back(i);
                    }
                }
            } else {
                for (int i = 0; i < 9; ++i) {
                    if (pdrv->atom.sel_dyncell_(i / 3, i % 3) > 0) {
                        selected.push_back(i);
                    }
                }
            }
        }
        // position related
        if (pdrv->flags.relax_atom) {
            if (pdrv->atom.sel_dynpos_.size() != pdrv->atom.size())
                throw std::invalid_argument("Selective dynamics atomic mask size does not match the structure");
            for (int i = pdrv->xposoffset; i < pdrv->x.size(); ++i) {
                int iat = (i - pdrv->xposoffset) / 3;
                int a = (i - pdrv->xposoffset) % 3;
                if (pdrv->atom.sel_dynpos_[iat](a) > 0) {
                    selected.push_back(i);
                }
            }
        }
    }
    std::vector<bool> used(pdrv->x.size(), false);
    for (int id : selected) {
        if (id < 0 || id >= pdrv->x.size() || used[id])
            throw std::invalid_argument("Invalid optimization-coordinate mask: duplicate or out-of-range coordinate");
        used[id] = true;
    }
    ids_pass.swap(selected);
}
// TODO: move this into modifiers_list_carg
void FilterVIB::forward(RelaxationDrvV2* pdrv)
{
    auto& g = pdrv->g;
    auto c = Eigen::Map<Eigen::VectorXd>(g.data(), g.size());
    c = projection_matrix * c;
    for (int i = 0; i < mode_factor.size(); ++i) {
        c[i] *= mode_factor[i];
    }
    c = projection_matrix.transpose() * c;
}

void FilterVIB::init(RelaxationDrvV2* pdrv, void* ptr)
{
    std::string fnm = *reinterpret_cast<std::string*>(ptr);
    std::ifstream ifs(fnm);
    if (!ifs) {
        throw std::runtime_error("Failed to open file: " + fnm);
    }
    // now load projection matrix
    int n;
    ifs >> n;
    projection_matrix.resize(n, n);
    mode_factor.resize(n);
    pdrv->mode_displacement_init.resize(n);
    for (int i = 0; i < n; ++i) {
        ifs >> mode_factor[i];
    }
    for (int i = 0; i < n; ++i) {
        ifs >> pdrv->mode_displacement_init[i];
    }
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            ifs >> projection_matrix(i, j);
        }
    }
    // apply initial displacements to system
    pdrv->x += projection_matrix.transpose() * pdrv->mode_displacement_init;
}

void FilterVIB2::forward(RelaxationDrvV2* pdrv)
{
    auto& g = pdrv->g;
    auto& x = pdrv->x;
    auto c = Eigen::Map<Eigen::VectorXd>(g.data(), g.size());
    c = projection_matrix * c;
    for (int i = 0; i < mode_factor.size(); ++i) {
        c[i] *= mode_factor[i];
    }
    x = projection_matrix * (x - x0);
}

void FilterVIB2::reverse(RelaxationDrvV2* pdrv)
{
    auto& x = pdrv->x;
    // x = x.transpose() * projection_matrix;
    x = projection_matrix.transpose() * x + x0;
}

void FilterVIB2::init(RelaxationDrvV2* pdrv, void* ptr)
{
    std::string fnm = *reinterpret_cast<std::string*>(ptr);
    std::ifstream ifs(fnm);
    if (!ifs) {
        throw std::runtime_error("Failed to open file: " + fnm);
    }
    // now load projection matrix
    int n;
    ifs >> n;
    projection_matrix.resize(n, n);
    eigenvalues.resize(n);
    mode_factor.resize(n);
    pdrv->mode_displacement_init.resize(n);
    for (int i = 0; i < n; ++i) {
        ifs >> mode_factor[i];
    }
    for (int i = 0; i < n; ++i) {
        ifs >> eigenvalues[i];
        if (fabs(eigenvalues[i]) < pdrv->tol.ztol) {
            mode_factor[i] = 0;
        }
    }
    for (int i = 0; i < n; ++i) {
        ifs >> pdrv->mode_displacement_init[i];
    }
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            ifs >> projection_matrix(i, j);
        }
    }
    x0 = pdrv->x;
    // apply initial displacements to system
    pdrv->x += projection_matrix.transpose() * pdrv->mode_displacement_init;
}
// zero eigenvalues should be masked
void FilterVIB2::forward_scaling(RelaxationDrvV2* pdrv)
{
    for (int i = 0; i < eigenvalues.size(); ++i) {
        pdrv->prec_scaling[i] = fabs(eigenvalues[i]);
        if (pdrv->prec_scaling[i] < pdrv->tol.ztol) {
            pdrv->prec_scaling[i] = pdrv->tol.ztol;
        }
    }
}

void FilterSymmForceStress::forward(RelaxationDrvV2* pdrv)
{
    // Forces and stress from pot->getEFS() are in Cartesian coordinates.
    // Rotations from spglib (rots[]) act on fractional row coordinates.
    // Under a rotation R, vectors (forces) transform as  f → R·f,
    // and rank-2 tensors (stress) transform as  σ → R·σ·Rᵀ.
    //
    // For a RowVector3d force f (1×3): f_rotated = f · Rᵀ = f · R.transpose()
    // For a Matrix3d stress σ:          σ_rotated = R · σ · Rᵀ

    if (pdrv->flags.relax_atom) {
        auto& atom = pdrv->atom;
        std::vector<Eigen::RowVector3d> forces_symm(atom.size());
        std::fill(forces_symm.begin(), forces_symm.end(), Eigen::RowVector3d::Zero());
        for (int iop = 0; iop < (int)rots.size(); ++iop) {
            const Eigen::Matrix3d R =
                (pdrv->atom.cell_.inverse() * rots[iop] * pdrv->atom.cell_).transpose();
            for (int iat = 0; iat < atom.size(); ++iat) {
                int jat = mapping(iop, iat);
                // f → R·f: convert to column, rotate, convert back to row
                Eigen::Vector3d f_col = atom.forces_[iat].transpose();
                forces_symm[jat] += (R * f_col).transpose();
            }
        }
        double inv_n = 1.0 / rots.size();
        for (int iat = 0; iat < atom.size(); ++iat)
            atom.forces_[iat] = forces_symm[iat] * inv_n;
    }

    if (pdrv->flags.relax_cell) {
        auto& stress = pdrv->atom.stress_;
        Eigen::Matrix3d stress_symm = Eigen::Matrix3d::Zero();
        for (int iop = 0; iop < (int)rots.size(); ++iop) {
            const Eigen::Matrix3d R =
                (pdrv->atom.cell_.inverse() * rots[iop] * pdrv->atom.cell_).transpose();
            // σ_rotated = R · σ · Rᵀ
            stress_symm += R * stress * R.transpose();
        }
        pdrv->atom.stress_ = stress_symm / rots.size();
    }
}

void FilterSymmForceStress::reverse(RelaxationDrvV2* pdrv)
{
    // do nothing
}

void FilterSymmForceStress::init(RelaxationDrvV2* pdrv, void* ptr)
{
    auto &atom = pdrv->atom;
    double symprec = *reinterpret_cast<double*>(ptr);
    using namespace Symmetry3D;
    symdataset dataset = get_sym_dataset(atom, symprec, true);
    rots.resize(dataset.n_operations);
    mapping = dataset.symop_mapping;;
    for (int i = 0; i < dataset.n_operations; ++i) {
        rots[i] = dataset.symm_ops[i].first;
    }
}
