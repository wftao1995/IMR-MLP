// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#pragma once

#include "config.h"
#include <array>
#include <string>
#include <map>
#include <Eigen/Dense>
#include <vector>
#include "gpmask.h"
#include <set>
#include <stdexcept>
#include <span>


/**
 * @brief Core atomic structure container holding cell, positions, species, and all per-atom/per-structure properties.
 * @ingroup core
 *
 * The Atoms class is the central data type used throughout the codebase. It stores:
 * - Unit cell matrix (row-major, each row is a lattice vector) and atomic positions
 * - Chemical species via integer map (chemmap_) with string-to-int mapping in GPIO
 * - Physical properties: energy, forces, stress, velocities, charges, magnetic moments
 * - Molecular topology: bonds, angles, molecule IDs
 * - Fitting weights: per-structure fitness, per-atom fitting factors
 * - Extra arbitrary attributes (double/int/vector/matrix per atom or global)
 * - Selective dynamics flags and masks for constrained relaxation
 *
 * Coordinates are stored in Cartesian by default; use to_fractional() / to_cartesian() to convert.
 * Cell transformations are applied via left-multiplication (see mksuperL, deformL).
 *
 * @note Thread safety: Not thread-safe. Each thread should operate on its own copy.
 * @see Neighborhoods, GPIO
 */
class Atoms
{
private:
public:
    // -----------Basic
    /// @name Geometry
    /// @{
    Eigen::Matrix3d cell_ = Eigen::Matrix3d::Zero();   ///< Unit cell matrix (row vectors are lattice vectors)
    std::vector<Eigen::RowVector3d> pos_;              ///< Atomic positions in Cartesian coordinates (default)
    std::vector<int> chemmap_;                          ///< Chemical species index per atom
    std::array<bool, 3> pbc_{{true, true, true}};      ///< Periodicity along the three lattice directions
    bool cartesian = true;                              ///< Whether pos_ is Cartesian (true) or fractional (false)
    /// @}

    /// @name Ghost atoms (MPI)
    /// @{
    bool have_ghost = false;                            ///< Whether ghost atoms have been built
    int nlocal_ = 0, nghost_ = 0;                               ///< Number of local and ghost atoms
    std::vector<int> map_ghost_;                        ///< Map ghost atom index to lattice atom index
    void build_ghost(double r);                          ///< Build ghost atoms for MPI communication
    /// @}

    /// @name Identifiers
    /// @{
    std::vector<int> ids_;                              ///< Per-atom unique identifiers
    /// @}

    /// @name Molecular topology
    /// @{
    std::vector<int> molid_;                            ///< Molecule ID per atom (same size as chemmap_)
    std::vector<int> molsize_;                          ///< Number of atoms per molecule
    int molnum_ = 0;                                        ///< Number of molecules
    std::vector<int> mol_types_;                        ///< Molecule type labels (e.g. SOL, SO4)
    std::vector<std::array<int, 2>> bonds_;             ///< Bond list (pairs of atom indices)
    std::vector<std::array<int, 3>> angles_;            ///< Angle list (triplets of atom indices)
    /// @}

    /// @name Physical properties
    /// @{
    std::vector<Eigen::Vector3d> forces_;               ///< Atomic forces (eV/A)
    std::vector<Eigen::Vector3d> velocities_;           ///< Atomic velocities
    double energy_ = 0;                                     ///< Total energy (eV)
    double fenergy_ = 0;                                ///< Free energy (eV)
    Eigen::Matrix3d stress_ = Eigen::Matrix3d::Zero();                            ///< Stress tensor (eV)
    Eigen::MatrixXd hessian_;                           ///< Cartesian Hessian matrix (3N x 3N, eV/A^2)
    Eigen::MatrixXd hvp_directions_;                    ///< Dimensionless Cartesian probes, columns of length 3N
    Eigen::MatrixXd hvp_targets_;                       ///< H * probe, same shape, eV/A^2
    std::vector<double> site_energies_;                 ///< Per-atom site energies
    std::vector<double> charges_;                       ///< Per-atom charges
    std::vector<double> magmom_scalar_;                 ///< Collinear magnetic moments (scalar)
    std::vector<Eigen::Vector3d> magmoms_;              ///< Non-collinear magnetic moments (vector)
    std::vector<Eigen::Matrix3d> born_eff_chgs_;        ///< Born effective charge tensors
    bool collinear_ = true;                             ///< Whether magnetism is collinear
    bool have_energy = false;                           ///< Flag: energy is present
    bool have_fenergy = false;                          ///< Flag: free energy is present
    bool have_stress = false;                           ///< Flag: stress is present
    bool have_hessian = false;                          ///< Flag: Hessian is present
    bool have_eff_chg = false;                          ///< Flag: Born effective charges present
    /// @}

    /// @name Fitting weights
    /// @{
    std::vector<double> fitting_factor;                 ///< Per-structure fitting factor (size 1 or 1+N or 1+N+1)
    double fitness = 1.0;                               ///< Structure weight for training
    std::vector<double> atomic_fitness;                 ///< Per-atom weights for training
    /// @}

    /// @name Masks and flags
    /// @{
    std::vector<int> masks;                             ///< Per-atom mask/group labels
    int group_count = 0;                                ///< Number of distinct groups in masks
    inline bool has_mask() const { return masks.size() > 0; };
    /// @}

    /// @name Selective dynamics
    /// @{
    Eigen::Matrix3d sel_dyncell_ = Eigen::Matrix3d::Zero();                       ///< Selective dynamics flags for cell
    std::vector<Eigen::RowVector3d> sel_dynpos_;        ///< Selective dynamics flags for positions
    /// @}

    /// @name Extra attributes (key-value stores)
    /// @{
    std::map<std::string, double> global_attribs_d_;                     ///< Global scalar (double) attributes
    std::map<std::string, Eigen::VectorXd> local_attribs_d_;             ///< Per-atom scalar attributes
    std::map<std::string, Eigen::VectorXd> global_attribs_v_;            ///< Global vector attributes
    std::map<std::string, Eigen::MatrixXd> local_attribs_v_;             ///< Per-atom vector attributes (col per atom)
    std::map<std::string, int> global_attribs_i_;                        ///< Global integer attributes
    std::map<std::string, Eigen::VectorXi> global_attribs_vi_;           ///< Global integer vector attributes
    std::map<std::string, Eigen::VectorXi> local_attribs_i_;             ///< Per-atom integer attributes
    std::map<std::string, Eigen::MatrixXi> local_attribs_vi_;            ///< Per-atom integer vector attributes
    inline void alloc_ld(std::string name) { local_attribs_d_[name].resize(size()); }
    inline void alloc_li(std::string name) { local_attribs_i_[name].resize(size()); }
    inline void alloc_lvd(std::string name, size_t len) { local_attribs_v_[name].resize(len, size()); }
    inline void alloc_lvi(std::string name, size_t len) { local_attribs_vi_[name].resize(len, size()); }
    /// @}

    /// @name Metadata
    /// @{
    std::map<std::string, std::string> features_;       ///< Arbitrary string-keyed metadata
    double mvgrade_ = 0;                                    ///< Grade metric from MAXVOL
    double mindist_ = 0;                                    ///< Minimum interatomic distance (A)
    double maxdist_ = 0;                                    ///< Maximum interatomic distance (A)
    int id = 0;                                         ///< Structure index / identifier
    /// @}

    /// @name Cell operations
    /// @{

    /// @brief Create supercell by left-multiplying cell_ with integer matrix; retain coordinate representation
    void mksuperL(const Eigen::Matrix3i &mat);
    /// @brief Swap lattice axes; mode encodes the permutation (e.g. 132 = xyz -> xzy)
    void swap_axis(int mode, bool keephand = true);
    /// @brief Create supercell using Hermite Normal Form; retain coordinate representation
    void mksuperL_hnf(const Eigen::Matrix3i &mat);
    /// @brief Create diagonal supercell, retaining coordinate representation
    void mksuper(const Eigen::Vector3i &vec);
    /// @brief Return Cartesian supercell geometry with source PBC, without changing the source.
    /// Does not copy physical labels. Replaces s2p with supercell-index to parent-index mappings.
    /// The integer matrix must be nonsingular; either determinant sign is supported.
    Atoms create_supercell(const Eigen::Matrix3i &mat, std::map<int, int> &s2p) const;
    /// @brief Construct supercell by real left transformation, retaining coordinate representation
    void createSuperL(const Eigen::Matrix3d &mat);
    /// @brief Right-multiply cell_, preserving direct coordinates and coordinate representation
    void deformR(const Eigen::Matrix3d &mat);
    /// @brief Left-multiply cell_, preserving direct coordinates and coordinate representation
    void deformL(const Eigen::Matrix3d &mat);
    /// @brief Wrap atoms back into the simulation cell
    void wrap();
    /// @brief Wrap a single position into the cell
    void wrap(Eigen::RowVector3d &pos);
    /// @brief High-precision wrap (less precision loss, slower)
    void hwrap();
    /// @brief Scale cell and positions by given factor
    void scale_by_factor(const double factor);
    /// @brief Scale cell to target volume (isotropic)
    void scale_to_volume(const double volume);
    /// @brief Resize atom arrays to new size
    void resize(int newsize);
    /// @brief Reserve capacity for atom arrays
    void reserve(int newsize);
    /// @brief Make cell lower triangular (standard orientation)
    void mkLower();
    /// @brief Translate by a Cartesian vector without wrapping, retaining coordinate representation
    void translate(Eigen::Vector3d vec);
    /// @brief Rotate atoms and/or cell around a Cartesian axis/center, retaining coordinate representation
    template<bool ROT_ATOMS = true, bool ROT_CELL = true>
    void rotate(Eigen::Vector3d axis, double angle, Eigen::Vector3d center = Eigen::Vector3d::Zero()) {
        const bool restore_fractional = !cartesian && (ROT_ATOMS || ROT_CELL);
        if (restore_fractional) to_cartesian();
        axis.normalize();
        Eigen::AngleAxisd rotation(angle, axis);
        if constexpr (ROT_ATOMS) {
            for (auto &p : pos_) {
                p = (center + rotation * (p.transpose() - center)).transpose();
            }
        }
        if constexpr (ROT_CELL) {
            cell_ = (rotation * cell_.transpose()).transpose();
        }
        if (restore_fractional) to_fractional();
    }
    /// @brief Rotate atoms and/or cell around a lattice direction [ijk]
    template<bool ROT_ATOMS = true, bool ROT_CELL = true>
    void rotate(int i, int j, int k, double angle) {
        Eigen::Vector3d axis = (i * cell_.row(0) + j * cell_.row(1) + k * cell_.row(2)).transpose();
        rotate<ROT_ATOMS, ROT_CELL>(axis, angle);
    }
    /// @brief Convert positions from Cartesian to fractional coordinates
    void to_fractional();
    /// @brief Convert positions from fractional to Cartesian coordinates
    void to_cartesian();
    /// @}

    /// @name Molecular construction
    /// @{
    /// @brief Build molecular bonds between atom types t1-t2 within cutoff rc
    void build_bonds_mol(const double &rc, int t1, int t2);
    /// @brief Build molecular bond angles t1-t2-t3 within cutoff rc
    void build_angles_mol(const double &rc, int t1, int t2, int t3);
    /// @}

    /// @name Property flags
    /// @{
    inline bool has_site_energies() const { return site_energies_.size() > 0; };
    inline bool has_forces() const { return forces_.size() > 0; };
    inline bool has_velo() const { return velocities_.size() > 0; };
    inline bool has_charges() const { return charges_.size() > 0; };
    inline bool has_eff_charges() const { return born_eff_chgs_.size() > 0 ; };
    inline bool has_magmom() const { return magmom_scalar_.size() > 0 || magmoms_.size() > 0; };
    inline bool has_energy() const { return have_energy; };
    inline bool has_fenergy() const { return have_fenergy; };
    inline bool has_stresses() const { return have_stress; };
    inline bool has_hessian() const { return have_hessian && hessian_.rows() == 3 * size() && hessian_.cols() == 3 * size(); };
    bool has_hvp() const { return hvp_directions_.cols() > 0; }
    void clear_hvp() { hvp_directions_.resize(0, 0); hvp_targets_.resize(0, 0); }
    void validate_hvp() const {
        if (hvp_directions_.size() == 0 && hvp_targets_.size() == 0 &&
            hvp_directions_.cols() == 0 && hvp_targets_.cols() == 0) return;
        if (size() == 0 || hvp_directions_.rows() != 3 * size() ||
            hvp_targets_.rows() != 3 * size() || hvp_directions_.cols() == 0 ||
            hvp_directions_.cols() != hvp_targets_.cols() ||
            !hvp_directions_.allFinite() || !hvp_targets_.allFinite())
            throw std::invalid_argument("HVP directions and targets must be finite paired 3N x m matrices");
        for (int k = 0; k < hvp_directions_.cols(); ++k)
            if (!(hvp_directions_.col(k).stableNorm() > 0))
                throw std::invalid_argument("HVP direction must be nonzero");
    }
    inline bool has_molecule() const { return molid_.size() > 0; };
    inline bool has_bonds() const { return bonds_.size() > 0; };
    inline bool has_angle() const { return angles_.size() > 0; };
    inline bool has_g_attrib_d() const { return global_attribs_d_.size() > 0; };
    inline bool has_g_attrib_v() const { return global_attribs_v_.size() > 0; };
    inline bool has_l_attrib_d() const { return local_attribs_d_.size() > 0; };
    inline bool has_l_attrib_v() const { return local_attribs_v_.size() > 0; };
    inline bool has_ids() const { return ids_.size() > 0; };
    inline bool is_right_hand() const { return cell_.row(0).dot(cell_.row(1).cross(cell_.row(2))) > 0; };
    /// @}

    /// @name Atom deletion
    /// @{
    enum class DuplicateAtomMode {
        Delete,
        Merge
    };
    struct DuplicateAtomStats {
        int initial_atoms = 0;
        int final_atoms = 0;
        int removed_atoms = 0;
        int duplicate_groups = 0;
    };
    /// @brief Remove or merge same-type duplicate atoms within distance tolerance
    DuplicateAtomStats remove_duplicate_atoms(double rtol, DuplicateAtomMode mode = DuplicateAtomMode::Delete);
    /// @brief Delete atoms at given indices (sorted)
    void delete_atoms(std::vector<int> &inds);
    /// @brief Delete a single atom by index
    void delete_atom(int &id);
    /// @}

    /// @name Geometry computation
    /// @{
    void find_mindist(void);                            ///< Compute minimum distance including nonzero periodic self images; no candidate returns 1e10
    void find_maxdist(void);                            ///< Compute and cache maximum interatomic distance
    double report_mindist(void);                        ///< Recompute and return minimum distance, preserving source coordinates
    double report_maxforce(void);                       ///< Return maximum force magnitude
    inline Eigen::Matrix3d get_metric() const { return cell_ * cell_.transpose(); };  ///< Metric tensor G = A^T A
    inline double get_volume(void) const { return cell_.determinant(); };             ///< Cell volume
    inline double get_volume_pow2() const { return get_metric().determinant(); };     ///< Squared volume (det(G))
    double get_distance(int a, int b);                  ///< Distance between atoms a and b (minimum image)
    Eigen::VectorXd get_voigt_stress() const;            ///< Stress in Voigt notation (xx,yy,zz,yz,xz,xy)
    Eigen::RowVector3d get_rab(int a, int b);            ///< Vector from atom a to b (minimum image)
    inline int size() const { return pos_.size(); };     ///< Number of atoms
    inline bool is_fully_periodic() const { return pbc_[0] && pbc_[1] && pbc_[2]; }
    inline bool is_nonperiodic() const { return !pbc_[0] && !pbc_[1] && !pbc_[2]; }
    inline bool has_partial_pbc() const { return !is_fully_periodic() && !is_nonperiodic(); }
    bool isorth() const;                                 ///< Whether cell is orthogonal
    Eigen::Matrix3d get_norm_cell(void) const;           ///< Normalized cell (each row normalized)
    Eigen::Matrix3d get_lower_cell(void) const;          ///< Lower-triangular form of cell
    Eigen::Matrix3d get_rec() const { return cell_.inverse().transpose(); };          ///< Reciprocal lattice (no 2pi)
    Eigen::Matrix3d get_recip() const { return 2.0 * M_PI * cell_.inverse().transpose();}; ///< Reciprocal lattice (with 2pi)
    Eigen::Matrix3d get_strain(Atoms &ref) const;        ///< Strain tensor relative to reference structure
    Eigen::RowVector3d get_cm(void) const;               ///< Center of mass
    /// @}

    /// @name Serialization
    /// @{
    std::istream &read_from_bin(std::istream &is);
    std::ostream &write_to_bin(std::ostream &os) const;
    /// @}

    /// @name Lifecycle
    /// @{
    Atoms(){};
    void clear(void);                                   ///< Reset all data to default state
    /// @}

    /// @name Coordinate utilities
    /// @{
    /// @brief Convert fractional coordinates to Cartesian (multiply by cell_)
    inline Eigen::RowVector3d dir2car(const Eigen::RowVector3d &pos) const { return pos * cell_; };
    /// @brief Convert Cartesian coordinates to fractional (multiply by inverse cell)
    inline Eigen::RowVector3d car2dir(const Eigen::RowVector3d &pos) const { return pos * cell_.inverse(); };
    /// @brief Fill pos with all atomic positions in fractional coordinates
    void get_dir_pos(std::vector<Eigen::RowVector3d> &pos) const
    {
        if (!cartesian) {
            pos = pos_;
            return;
        }
        pos.resize(size());
        auto invcell = cell_.inverse();
        for (int i = 0; i < size(); ++i)
            pos[i] = pos_.at(i) * invcell;
    }
    std::map<int, int> types_count_map(void) const;     ///< Count atoms per species
    std::vector<double> get_cell_parameters(void) const; ///< Cell params (a,b,c,alpha,beta,gamma)
    std::vector<int> get_types(void) const;              ///< List of unique chemical types
    std::set<int> get_types_set(void) const;             ///< Set of unique chemical types
    /// @brief Handedness: +1 = right-handed, -1 = left-handed, 0 = degenerate
    int get_handedness() const {
        double d = cell_.determinant();
        if (fabs(d) < 1e-14) return 0;
        return (d > 0 ? 1 : -1);
    }
    void to_lower_standard();                            ///< Standardize: lower triangular, right-handed, a aligned to x
    void to_righthand();                                 ///< Ensure right-handed cell orientation
    /// @brief Compute standard form of cell (QR decomposition), returns lower or upper triangular
    Eigen::Matrix3d standard_form(bool upper = false);
    /// Set the cell, preserving coordinate representation and the existing handedness convention.
    void set_cell(double, double, double, double, double, double);
    /// With sc_pos=false, preserve Cartesian positions instead of fractional coordinates.
    void set_cell(double, double, double, double, double, double, bool);
    void set_cell(Eigen::Matrix3d &cell);
    void set_dir_pos(std::vector<Eigen::RowVector3d> pos); ///< Set direct-coordinate input, storing Cartesian positions
    /// Reorder all supported atom fields with new[i] = old[order[i]].
    /// Requires an N-element permutation and complete nonempty atom fields.
    /// Molecular topology and ghost layouts are unsupported. Failure leaves
    /// the structure unchanged; Hessian/HVP Cartesian blocks follow the atoms.
    void reorder(std::span<const int> order);
    /// Compatibility entry point: caller must supply at least size() integers.
    /// A raw pointer cannot provide a buffer length; prefer the span overload.
    void reorder(int *map);
    void resetEFS();                                       ///< Set energy/stress/forces to zero
    void resetE();                                         ///< Set energy to zero
    void clearEFS();                                       ///< Remove energy/stress/forces data
    void setEFSflag() { have_energy = true, have_stress = true; };
    void setEflag() { have_energy = true, have_stress = false; };
    /// @}

    /// @name Operators and merging
    /// @{
    friend std::ostream &operator<<(std::ostream &os, const Atoms &atom);
    friend Atoms merge_atoms(const Atoms &a1, const Atoms &a2);
    void merge_atoms(const Atoms &opt);
    /// @brief Extract substructure for atoms matching mask
    Atoms extract_substructures_from_mask(const GPMask &) const;
    friend Atoms extract_substructures_from_mask(const Atoms &a1, const GPMask &);
    friend void extract_substructures_from_mask(const Atoms &a1, const GPMask &, Atoms &masked, Atoms &nonmasked);
    /// @}

};

#ifdef EXTERN_EXPCLIT_TEMPLATE
extern template class std::vector<Atoms>;
#endif
