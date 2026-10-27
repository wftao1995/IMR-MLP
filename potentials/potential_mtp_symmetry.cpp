#include "potential_mtp.h"
#include "external/spglib_driver.h"
#include <limits>
#include <map>
#ifdef USE_OMP
#include <omp.h>
#endif

namespace Potentials {

bool PotMTP::try_full_symmetry_efs(Atoms &atom, AnyNeighborhoods &nbhs,
                                   double symprec) {
    if (typeid(*this) != typeid(PotMTP))
        GPERROR("full_symmetry_reuse currently supports ordinary MTP only");
    last_symmetry_site_evaluations = atom.size();
    if (!atom.is_fully_periodic() || atom.size() == 0 ||
        nbhs.size() != atom.size()) return false;

    Symmetry3D::symdataset dataset;
    try { dataset = Symmetry3D::get_sym_dataset(atom, symprec, true); }
    catch (const std::exception &) { return false; }

    struct SiteMap {
        int representative;
        Eigen::Matrix3d rotation;
        std::vector<int> neighbor;
    };
    std::vector<SiteMap> sites(atom.size());
    std::map<int, std::vector<int>> orbits;
    for (int i = 0; i < atom.size(); ++i)
        orbits[dataset.equivalent_atoms[i]].push_back(i);

    // spglib's tolerance identifies candidates. Reuse requires an accurate
    // Cartesian neighbour bijection, so near-symmetric input is not projected
    // onto a different geometry. Periodic images are matched as separate rows.
    const double tolerance = std::max(1e-12, 128. * std::numeric_limits<double>::epsilon() *
                                      std::max({1., cutoff, atom.cell_.norm()}));
    const Eigen::Matrix3d inverse_cell = atom.cell_.inverse();
    for (const auto &[key, members] : orbits) {
        const int representative = members.front();
        const auto &source = nbhs[representative];
        for (int target : members) {
            auto &mapping = sites[target];
            mapping.representative = representative;
            const auto &destination = nbhs[target];
            if (destination.nbhcount_ != source.nbhcount_) return false;
            bool matched = false;
            for (int op = 0; op < dataset.n_operations && !matched; ++op) {
                if (dataset.symop_mapping(op, representative) != target) continue;
                const Eigen::Matrix3d rotation = inverse_cell * dataset.symm_ops[op].first * atom.cell_;
                if ((rotation.transpose() * rotation - Eigen::Matrix3d::Identity()).norm() > 1e-10)
                    continue;
                std::vector<int> rows(source.nbhcount_, -1);
                std::vector<bool> used(destination.nbhcount_, false);
                bool valid = true;
                for (int j = 0; j < source.nbhcount_ && valid; ++j) {
                    const Eigen::RowVector3d rotated = source.pos_[j] * rotation;
                    const int index = dataset.symop_mapping(op, source.indices_[j]);
                    for (int k = 0; k < destination.nbhcount_; ++k) {
                        if (!used[k] && destination.indices_[k] == index &&
                            destination.chemmap_[k] == source.chemmap_[j] &&
                            (destination.pos_[k] - rotated).norm() <= tolerance) {
                            rows[j] = k;
                            used[k] = true;
                            break;
                        }
                    }
                    valid = rows[j] >= 0;
                }
                if (valid) {
                    mapping.rotation = rotation;
                    mapping.neighbor = std::move(rows);
                    matched = true;
                }
            }
            if (!matched) return false;
        }
    }

    omp_init();
    if (!add_efs) atom.resetEFS(); else atom.setEFSflag();
    std::vector<std::vector<int>> groups;
    for (const auto &[key, members] : orbits) groups.push_back(members);
    int actual_threads = 1;
#ifdef USE_OMP
#pragma omp parallel num_threads(omp_buffers_list.size())
#endif
    {
#ifdef USE_OMP
        const int thread = omp_get_thread_num();
        if (thread == 0) actual_threads = omp_get_num_threads();
#else
        const int thread = 0;
#endif
        auto &buffer = omp_buffers_list[thread];
        buffer.energy = 0.;
        buffer.force.assign(atom.size(), Eigen::RowVector3d::Zero());
        buffer.siteenergy.assign(atom.size(), 0.);
        buffer.stress.setZero();
#ifdef USE_OMP
#pragma omp for schedule(static)
#endif
        for (int group = 0; group < groups.size(); ++group) {
            const int representative = groups[group].front();
            buffer.buff_site_energy = 0.;
            CalcSiteEFSOMP(nbhs[representative], buffer);
            for (int target : groups[group]) {
                const auto &mapping = sites[target];
                const auto &destination = nbhs[target];
                buffer.energy += buffer.buff_site_energy;
                buffer.siteenergy[target] += buffer.buff_site_energy;
                for (int j = 0; j < mapping.neighbor.size(); ++j) {
                    const int k = mapping.neighbor[j];
                    const Eigen::RowVector3d gradient = buffer.buff_site_energy_ders.row(j) * mapping.rotation;
                    buffer.force[target] += gradient;
                    buffer.force[destination.indices_[k]] -= gradient;
                    buffer.stress.noalias() -= gradient.transpose() * destination.pos_[k];
                }
            }
        }
    }
    for (int thread = 0; thread < actual_threads; ++thread) {
        const auto &buffer = omp_buffers_list[thread];
        atom.energy_ += buffer.energy;
        atom.stress_ += buffer.stress;
        for (int i = 0; i < atom.size(); ++i) {
            atom.forces_[i] += buffer.force[i];
            atom.site_energies_[i] += buffer.siteenergy[i];
        }
    }
    last_symmetry_site_evaluations = groups.size();
    return true;
}

} // namespace Potentials
