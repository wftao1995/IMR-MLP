// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#include "potential_mtpnn1.h"
#include "NNBase/coeff_io.h"
#include "utils/compressed_stream.h"
#include <filesystem>
#include <fmt/format.h>
#include <fmt/ranges.h>

#ifdef USE_OMP
#include <omp.h>
#endif 

/**
 * ;
 */
namespace fs = std::filesystem;

Potentials::PotMTPNN1::PotMTPNN1(GPSettings settings) {
    InitSettings();
    ApplySettings(settings);
    if (nnlay != 1 && nnlay != 2) GPERROR("MTPNN1 nnlay must be 1 or 2");
    if (fs::exists(nnfile)) nnlay = IMRMLPNN::CoeffIO::layers(nnfile, nnlay);
    std::unique_ptr<PotMTP> mtp;
    std::unique_ptr<IMRMLPNN::NN1lay> nn;
    std::unique_ptr<IMRMLPNN::NN2lay> nn2;
    bool can_load_nn = false;
    bool can_load_mtp = false;
    bool can_load_sc = false;
    if (fs::exists(mtpfile)) {
        mtp = std::make_unique<PotMTP>(mtpfile);
        p_mtp = mtp.get();
        can_load_mtp = p_mtp->inited();
    }
    else 
        GPERROR("POTENTIAL FILE NOT FOUND -- .MTP FILE");
    species_count = p_mtp->species_count;
    cutoff = p_mtp->cutoff;
    cutoffsq = p_mtp->cutoffsq;
    cutoffs = p_mtp->cutoffs;
    cutoffsqs = p_mtp->cutoffsqs;
    if (nnlay > 1) {
        nn2 = std::make_unique<IMRMLPNN::NN2lay>();
        p_nn2 = nn2.get();
    } else {
        nn = std::make_unique<IMRMLPNN::NN1lay>();
        p_nn = nn.get();
    }
    if (fs::exists(nnfile)) {
        if (nnlay > 1)
            p_nn2->load_coeffs(nnfile);
        else {
            Message("Loading nn coeffs");
            p_nn->load_coeffs(nnfile);
        }
        const int input_size = nnlay == 2 ? p_nn2->n_input : p_nn->n_input;
        if (input_size != p_mtp->alpha_scalar_moments + 1)
            GPERROR("MTPNN1 input dimension does not match MTP descriptors: " + nnfile);
        can_load_nn = true;
    } else {
        Warning("NN1lay coeffs file not found, use an empty init");
        if (nnlay > 1)
            p_nn2->setup(p_mtp->alpha_scalar_moments + 1, nn_nhidden, nn_nhidden2);
        else 
            p_nn->setup(p_mtp->alpha_scalar_moments + 1, nn_nhidden);
    }
    species_coeffs.resize(p_mtp->species_count, 0);
    if (fs::exists(spcfile)) {
        auto ifs = open_ifile(spcfile);
        if (!ifs) {
            GPERROR("MTPNN1 failed to read species coeffs from: " + spcfile);
        }
        for (int i = 0; i < species_coeffs.size(); ++i) {
            if (!(*ifs >> species_coeffs[i])) {
                GPERROR("MTPNN1 failed to read species coeffs from: " + spcfile);
            }
        }
        can_load_sc = true;
    } else {
        Warning("Species coeffs file not found, use an empty init");
    }
    if (can_load_mtp && can_load_nn && can_load_sc) {
        nl_inited = true;
    }
    if(nnlay > 1) {
        nn_coeff_size = p_nn2->get_coeff_size();
    } else {
        nn_coeff_size = p_nn->get_coeff_size();
    }
    mtp.release();
    nn.release();
    nn2.release();
}

void Potentials::PotMTPNN1::CalcSiteEFS(
    Neighborhood &nbh, double &energy, SiteCoordinateMatrix &gradient)
{
    if (!supports_site_efs()) GPERROR("NNMTP site EFS requires an initialized model");
    energy = 0.0;
    if (nnlay > 1)
        p_mtp->CalcSiteEFS<IMRMLPNN::NN2lay, 0>(nbh, energy, gradient, *p_nn2);
    else
        p_mtp->CalcSiteEFS<IMRMLPNN::NN1lay, 0>(nbh, energy, gradient, *p_nn);
    energy += species_coeffs[nbh.centertype];
}

void Potentials::PotMTPNN1::getEFS(Atoms &atom, AnyNeighborhoods &nbhs)
{
    if (!add_efs) atom.resetEFS(); else atom.setEFSflag();

    for (int ii = 0; ii < nbhs.size(); ii++) {
        Neighborhood &nbh = nbhs[ii];
        double buff_site_energy = 0;
        if (nnlay > 1) {
            p_mtp->CalcSiteEFS<IMRMLPNN::NN2lay, 0>(nbh, buff_site_energy, p_mtp->buff_site_energy_ders, *p_nn2);
        }
        else {
            p_mtp->CalcSiteEFS<IMRMLPNN::NN1lay, 0>(nbh, buff_site_energy, p_mtp->buff_site_energy_ders, *p_nn);
        }
        buff_site_energy += species_coeffs[nbh.centertype];
        // extract energy, force, virual, etc
        atom.energy_ += buff_site_energy;
        atom.site_energies_[ii] += buff_site_energy;

        for (int i = 0; i < nbh.nbhcount_; ++i) {
            // dE/dxj = \sum_{i!=j}dVi/d(xj-xi) - \sum_{i}dVj/d(xi-xj)
            atom.forces_[ii](0) += p_mtp->buff_site_energy_ders(i, 0);
            atom.forces_[ii](1) += p_mtp->buff_site_energy_ders(i, 1);
            atom.forces_[ii](2) += p_mtp->buff_site_energy_ders(i, 2);
            // for each local configuration n_j, r is acutally r_ii - r_j, thus we have negative terms.
            atom.forces_[nbh.indices_[i]](0) -= p_mtp->buff_site_energy_ders(i, 0);
            atom.forces_[nbh.indices_[i]](1) -= p_mtp->buff_site_energy_ders(i, 1);
            atom.forces_[nbh.indices_[i]](2) -= p_mtp->buff_site_energy_ders(i, 2);

            for (int a = 0; a < 3; a++) {
                for (int b = 0; b < 3; b++) {
                    atom.stress_(a, b) -= p_mtp->buff_site_energy_ders(i, a) * nbh.pos_[i](b);
                }
            }
        }
    }
}

void Potentials::PotMTPNN1::getEFSOMP(Atoms &atom, AnyNeighborhoods &nbhs)
{
    omp_init();
    int actual_threads = 1;
    if (!add_efs) atom.resetEFS(); else atom.setEFSflag();
    #ifdef USE_OMP
#pragma omp parallel num_threads(p_mtp->omp_buffers_list.size())
#endif 
{
#ifdef USE_OMP
    int ithread = omp_get_thread_num();
    if (ithread == 0) actual_threads = omp_get_num_threads();
#else 
    int ithread = 0;
#endif
    auto &buffer = p_mtp->omp_buffers_list[ithread];
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
        if (nnlay > 1)
            p_mtp->CalcSiteEFSOMP<IMRMLPNN::NN2lay, 0>(nbh, buffer, *p_nn2, ithread);
        else 
            p_mtp->CalcSiteEFSOMP<IMRMLPNN::NN1lay, 0>(nbh, buffer, *p_nn, ithread);
        // extract energy, force, virual, etc
        buffer.buff_site_energy += species_coeffs[nbh.centertype];
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
        auto &buffer = p_mtp->omp_buffers_list[i];
        atom.energy_ += buffer.energy;
        atom.stress_ += buffer.stress;
        for (int iat = 0; iat < atom.size(); ++iat) {
            atom.forces_[iat] += buffer.force[iat];
        }
    }
}

void Potentials::PotMTPNN1::getE(Atoms &atom, AnyNeighborhoods &nbhs)
{
    if (!add_efs) atom.resetE(); else atom.setEflag();

    for (int ii = 0; ii < nbhs.size(); ii++) {
        Neighborhood &nbh = nbhs[ii];
        double buff_site_energy = 0;
        if (nnlay > 1)
            p_mtp->CalcSiteEFS<IMRMLPNN::NN2lay, 0, 0>(nbh, buff_site_energy, p_mtp->buff_site_energy_ders, *p_nn2);
        else 
            p_mtp->CalcSiteEFS<IMRMLPNN::NN1lay, 0, 0>(nbh, buff_site_energy, p_mtp->buff_site_energy_ders, *p_nn);
        buff_site_energy += species_coeffs[nbh.centertype];
        // extract energy, force, virual, etc
        atom.energy_ += buff_site_energy;
        atom.site_energies_[ii] += buff_site_energy;
    }
}

void Potentials::PotMTPNN1::getEOMP(Atoms &atom, AnyNeighborhoods &nbhs)
{
    omp_init();
    int actual_threads = 1;
    if (!add_efs) atom.resetE(); else atom.setEflag();
    #ifdef USE_OMP
#pragma omp parallel num_threads(p_mtp->omp_buffers_list.size())
#endif 
{
#ifdef USE_OMP
    int ithread = omp_get_thread_num();
    if (ithread == 0) actual_threads = omp_get_num_threads();
#else 
    int ithread = 0;
#endif
    auto &buffer = p_mtp->omp_buffers_list[ithread];
    buffer.energy = 0;
    buffer.siteenergy.assign(atom.size(), 0.0);
#ifdef USE_OMP
#pragma omp for schedule(static)
#endif
    for (int ii = 0; ii < nbhs.size(); ii++) {
        Neighborhood &nbh = nbhs[ii];
        buffer.buff_site_energy = 0;
        if (nnlay > 1)
            p_mtp->CalcSiteEFSOMP<IMRMLPNN::NN2lay, 0, 0>(nbh, buffer, *p_nn2, ithread);
        else 
            p_mtp->CalcSiteEFSOMP<IMRMLPNN::NN1lay, 0, 0>(nbh, buffer, *p_nn, ithread);
        // extract energy, force, virual, etc
        buffer.buff_site_energy += species_coeffs[nbh.centertype];
        buffer.energy += buffer.buff_site_energy;
        atom.site_energies_[ii] += buffer.buff_site_energy;
    }
}
    // merge
    for (int i = 0; i < actual_threads; ++i) {
        auto &buffer = p_mtp->omp_buffers_list[i];
        atom.energy_ += buffer.energy;
    }
}

void Potentials::PotMTPNN1::omp_init()
{
    p_mtp->omp_init();
    if (nnlay > 1) {
        p_nn2->resize_omp_buffers(static_cast<int>(p_mtp->omp_buffers_list.size()));
    }
    else  {
        p_nn->resize_omp_buffers(static_cast<int>(p_mtp->omp_buffers_list.size()));
    }
}

void Potentials::PotMTPNN1::random_init()
{
    if (!train_nn_only && random_init_mtp)
        p_mtp->random_init();
    if (nnlay > 1)
        p_nn2->random_init();
    else
        p_nn->random_init();
}

void Potentials::PotMTPNN1::train_init()
{
    p_mtp->train_init();
    // L1, L2 reg vector only applies on NN model
    reg_L1_vector.resize(nn_coeff_size, 0);
    reg_L2_vector.resize(nn_coeff_size, 0);
}

void Potentials::PotMTPNN1::set_scale(double s)
{
    p_mtp->set_scale(s);
}

int Potentials::PotMTPNN1::get_NL_coeffs_count() const
{
    if (train_nn_only) {
        if (nnlay > 1)
            return p_nn2->n_coeffs + species_coeffs.size();
        else
            return p_nn->n_coeffs + species_coeffs.size();
    } else {
        if (nnlay > 1)
            return p_nn2->n_coeffs + p_mtp->radial_coeff_size + species_coeffs.size() * 2;
        else
            return p_nn->n_coeffs + p_mtp->radial_coeff_size + species_coeffs.size() * 2; // species_coeffs * 2 because of B0 of MTP
    }
}

std::vector<Potentials::ConditionalLinearBlock> Potentials::PotMTPNN1::conditional_linear_blocks(double) const
{
    if (typeid(*this) != typeid(PotMTPNN1)) return {};
    ConditionalLinearBlock block; block.name="output";
    const int nn_count=nnlay>1 ? p_nn2->n_coeffs : p_nn->n_coeffs;
    const int output_count=nnlay>1 ? p_nn2->n_neuron2 : p_nn->n_neuron;
    const int offset=train_nn_only ? 0 : p_mtp->radial_coeff_size+p_mtp->species_count;
    block.composition_count=species_coeffs.size();
    for (int i=0;i<int(species_coeffs.size());++i) block.append(offset+nn_count+i);
    const double l1=reg_L1_vector.empty() ? 0. : reg_L1_vector[0];
    const double l2=reg_L2_vector.empty() ? 0. : reg_L2_vector[0];
    for (int i=nn_count-output_count;i<nn_count;++i) block.append(offset+i,l1,l2);
    return {std::move(block)};
}

// [mtp_radial, mtp_species, nn_weights, species_coeffs]
void Potentials::PotMTPNN1::nl_get_coeffs(double *x) const
{
    int offset = 0;
    if (!train_nn_only) {
        p_mtp->model_get_coeffs(x);
        offset += p_mtp->radial_coeff_size + p_mtp->species_count;
    }
    if (nnlay > 1) {
        p_nn2->get_coeffs(x + offset);
        offset += p_nn2->get_coeff_size();
    }
    else {
        p_nn->get_coeffs(x + offset);
        offset += p_nn->get_coeff_size();
    }
    for (int i = 0; i < species_coeffs.size(); ++i) {
        x[offset + i] = species_coeffs[i];
    }    
}

void Potentials::PotMTPNN1::nl_set_coeffs(const double *x)
{
    int offset = 0;
    if (!train_nn_only) {
        p_mtp->model_set_coeffs(x);
        offset += p_mtp->radial_coeff_size + p_mtp->species_count;
    }
    if (nnlay > 1)
        p_nn2->set_coeffs(x + offset);
    else
        p_nn->set_coeffs(x + offset);
    offset += nn_coeff_size;
    for (int i = 0; i < species_coeffs.size(); ++i) {
        species_coeffs[i] = x[offset + i];
    }  
}

void Potentials::PotMTPNN1::AccumulateFPGard(Neighborhood &nbh, 
            double *out_gard_accumulator, double se_weight, Eigen::Matrix<double, -1, 3, Eigen::RowMajor> *f_weight)
{
    // This will accumulate all coeffs except species_coeffs
    if (train_nn_only) {
        if (nnlay > 1) {
            if (f_weight != nullptr) {
                p_mtp->AccumulateFPGard<IMRMLPNN::NN2lay, 0, 0, 0, 1, 1>(nbh, out_gard_accumulator, se_weight, f_weight, *p_nn2);
            } else {
                p_mtp->AccumulateFPGard<IMRMLPNN::NN2lay, 0, 0, 0, 1, 0>(nbh, out_gard_accumulator, se_weight, f_weight, *p_nn2);
            }
        }
        else {
            if (f_weight != nullptr) {
                p_mtp->AccumulateFPGard<IMRMLPNN::NN1lay, 0, 0, 0, 1, 1>(nbh, out_gard_accumulator, se_weight, f_weight, *p_nn);
            } else {
                p_mtp->AccumulateFPGard<IMRMLPNN::NN1lay, 0, 0, 0, 1, 0>(nbh, out_gard_accumulator, se_weight, f_weight, *p_nn);
            }
        }
    } else  {
        if (nnlay > 1) {
            if (f_weight != nullptr) {
                p_mtp->AccumulateFPGard<IMRMLPNN::NN2lay, 0, 1, 1, 1, 1>(nbh, out_gard_accumulator, se_weight, f_weight, *p_nn2);
            } else {
                p_mtp->AccumulateFPGard<IMRMLPNN::NN2lay, 0, 1, 1, 1, 0>(nbh, out_gard_accumulator, se_weight, f_weight, *p_nn2);
            }
        }
        else {
            if (f_weight != nullptr) {
                p_mtp->AccumulateFPGard<IMRMLPNN::NN1lay, 0, 1, 1, 1, 1>(nbh, out_gard_accumulator, se_weight, f_weight, *p_nn);
            } else {
                p_mtp->AccumulateFPGard<IMRMLPNN::NN1lay, 0, 1, 1, 1, 0>(nbh, out_gard_accumulator, se_weight, f_weight, *p_nn);
            }
        }
    }
    // species_coeffs
    int offset = 0;
    if (!train_nn_only)
       offset += p_mtp->radial_coeff_size + p_mtp->species_count;
    offset += nn_coeff_size;
    out_gard_accumulator[offset + nbh.centertype] += se_weight;
}

void Potentials::PotMTPNN1::Save(const std::string &potfile)
{
    if (save_gpm_if_requested(*this, potfile)) return;
    p_mtp->Save(potfile + ".mtp");
    if (nnlay > 1)
        p_nn2->dump_coeffs(potfile + ".nn");
    else 
        p_nn->dump_coeffs(potfile + ".nn");
    auto ofs = open_ofile(potfile + ".sc");
    if (!ofs) throw std::runtime_error("Failed to open species coefficient file: " + potfile + ".sc");
    *ofs << fmt::format("{}\n", fmt::join(species_coeffs, " "));
    if (!close_ofile(ofs))
        throw std::runtime_error("Failed to write species coefficient file: " + potfile + ".sc");
}

void Potentials::PotMTPNN1::Load(const std::string &potfile)
{
    if (nnlay != 1 && nnlay != 2) GPERROR("MTPNN1 nnlay must be 1 or 2");
    auto mtp = std::make_unique<PotMTP>(potfile + ".mtp");
    const int layers = IMRMLPNN::CoeffIO::layers(potfile + ".nn", nnlay);
    std::unique_ptr<IMRMLPNN::NN1lay> nn;
    std::unique_ptr<IMRMLPNN::NN2lay> nn2;
    if (layers == 2) nn2 = std::make_unique<IMRMLPNN::NN2lay>(potfile + ".nn");
    else nn = std::make_unique<IMRMLPNN::NN1lay>(potfile + ".nn");
    const int input_size = layers == 2 ? nn2->n_input : nn->n_input;
    if (input_size != mtp->alpha_scalar_moments + 1)
        GPERROR("MTPNN1 input dimension does not match MTP descriptors: " + potfile + ".nn");
    std::vector<double> species(mtp->species_count);
    auto ifs = open_ifile(potfile + ".sc");
    if (!ifs) GPERROR("MTPNN1 species coeffs file not found: " + potfile + ".sc");
    for (auto &value : species) {
        if (!(*ifs >> value) || !std::isfinite(value))
            GPERROR("MTPNN1 failed to read species coeffs from: " + potfile + ".sc");
    }

    // Replace the model only after every component has loaded successfully.
    delete p_mtp;
    delete p_nn;
    delete p_nn2;
    p_mtp = mtp.release();
    p_nn = nn.release();
    p_nn2 = nn2.release();
    nnlay = layers;
    species_coeffs = std::move(species);
    species_count = p_mtp->species_count;
    cutoff = p_mtp->cutoff;
    cutoffsq = p_mtp->cutoffsq;
    cutoffs = p_mtp->cutoffs;
    cutoffsqs = p_mtp->cutoffsqs;
    nn_coeff_size = layers == 2 ? p_nn2->get_coeff_size() : p_nn->get_coeff_size();
    omp_inited = false;
    nl_inited = true;
}

void Potentials::PotMTPNN1::AddPenaltyGard(double weight, double &loss, double *loss_gard)
{
    // Add MTP's pnal weight
    if (!train_nn_only)
        p_mtp->AddPenaltyGardModel(weight, loss, loss_gard);
    // Add L1/L2
    double &reg_L1 = reg_L1_vector[0];
    double &reg_L2 = reg_L2_vector[0];
    int offset = 0;
    if (!train_nn_only)
        offset += p_mtp->radial_coeff_size + p_mtp->species_count;
    if (nnlay > 1)
        p_nn2->add_pnal(reg_L1, reg_L2, loss, loss_gard + offset);
    else 
        p_nn->add_pnal(reg_L1, reg_L2, loss, loss_gard + offset);
}

namespace {
// CalcSiteHessImpl contracts the moment Jacobians, weighted moment Hessians
// and the external model's input Hessian. Preserve the legacy SFLAG=0 input
// layout: the first neural input is the MTP species constant.
template<class Network>
struct LegacyNeuralFC2 {
    Network &network;
    void propagate(const std::vector<double> &basis, double &energy,
        Eigen::VectorXd &gradient, Eigen::MatrixXd &hessian, int &tag, int thread) {
        const Eigen::Map<const Eigen::VectorXd> input(basis.data(),basis.size());
        network.propagate(input,energy,gradient,hessian,tag,thread);
    }
};
}

bool Potentials::PotMTPNN1::supports_site_hessian() const {
    if (typeid(*this) != typeid(PotMTPNN1) || !p_mtp ||
        typeid(*p_mtp) != typeid(PotMTP) || p_mtp->hc_correction || p_mtp->rR_correction)
        return false;
    if (nnlay == 1 && p_nn)
        return p_nn->acfunc == IMRMLPNN::TANH || p_nn->acfunc == IMRMLPNN::SIGMOID;
    if (nnlay == 2 && p_nn2)
        return p_nn2->acfunc == IMRMLPNN::TANH || p_nn2->acfunc == IMRMLPNN::SIGMOID;
    return false;
}
bool Potentials::PotMTPNN1::supports_site_hvp() const { return supports_site_hessian(); }
bool Potentials::PotMTPNN1::supports_analytical_hessian() const { return supports_site_hessian(); }
bool Potentials::PotMTPNN1::supports_analytical_hessian_vector_product() const { return supports_site_hvp(); }
bool Potentials::PotMTPNN1::supports_analytical_cell_hessian() const { return supports_site_hessian(); }

void Potentials::PotMTPNN1::evaluate_site_kernel(Neighborhood &nbh, double &energy,
    SiteCoordinateMatrix &gradient, HessianMatrix &hessian,
    const SiteCoordinateMatrix *direction, SiteCoordinateMatrix *product) {
    if (!supports_site_hessian()) GPERROR("Legacy neural MTP curvature requires a classic uncorrected MTP and smooth NN1/NN2");
    PotMTP::ompbuffers unused;
    if (nnlay == 1) {
        LegacyNeuralFC2<IMRMLPNN::NN1lay> model{*p_nn};
        p_mtp->CalcSiteHessImpl<false,decltype(model),0>(
            nbh,unused,energy,gradient,hessian,model,0,direction,product);
    } else {
        LegacyNeuralFC2<IMRMLPNN::NN2lay> model{*p_nn2};
        p_mtp->CalcSiteHessImpl<false,decltype(model),0>(
            nbh,unused,energy,gradient,hessian,model,0,direction,product);
    }
    energy += species_coeffs[nbh.centertype];
}

void Potentials::PotMTPNN1::evaluate_site_hessian(Neighborhood &nbh, double &energy,
    SiteCoordinateMatrix &gradient, HessianMatrix &hessian) {
    evaluate_site_kernel(nbh,energy,gradient,hessian);
}
void Potentials::PotMTPNN1::apply_site_hvp(Neighborhood &nbh, const SiteCoordinateMatrix &direction,
    double &energy, SiteCoordinateMatrix &gradient, SiteCoordinateMatrix &product) {
    if (direction.rows() != nbh.nbhcount_ || !direction.allFinite())
        GPERROR("Invalid legacy neural MTP site HVP direction");
    const SiteCoordinateMatrix input = direction;
    HessianMatrix unused;
    evaluate_site_kernel(nbh,energy,gradient,unused,&input,&product);
}

void Potentials::PotMTPNN1::applyHessianVectorProduct(Atoms &atom, AnyNeighborhoods &nbhs,
    const Eigen::VectorXd &direction, Eigen::VectorXd &product) {
    if (!supports_analytical_hessian_vector_product()) GPERROR("Unsupported legacy neural MTP HVP");
    if (direction.size() != 3*atom.size() || !direction.allFinite())
        GPERROR("Hessian-vector direction must be finite with size 3*N");
    if (nbhs.size() != atom.size()) GPERROR("Hessian-vector neighborhoods have the wrong size");
    Eigen::VectorXd result = Eigen::VectorXd::Zero(direction.size());
    for (int i = 0; i < nbhs.size(); ++i) {
        auto &n = nbhs[i];
        SiteCoordinateMatrix relative(n.nbhcount_,3), gradient, local;
        for (int j = 0; j < n.nbhcount_; ++j)
            relative.row(j) = (direction.segment<3>(3*n.indices_[j])-direction.segment<3>(3*i)).transpose();
        double energy;
        apply_site_hvp(n,relative,energy,gradient,local);
        for (int j = 0; j < n.nbhcount_; ++j) {
            result.segment<3>(3*n.indices_[j]) += local.row(j).transpose();
            result.segment<3>(3*i) -= local.row(j).transpose();
        }
    }
    product = std::move(result);
}

void Potentials::PotMTPNN1::assemble_hessian(Atoms &atom, AnyNeighborhoods &nbhs,
    HessianMatrix &hessian, Eigen::MatrixXd *cellhess, Eigen::MatrixXd *lrhess) {
    if (!supports_analytical_hessian()) GPERROR("Unsupported legacy neural MTP Hessian");
    if (nbhs.size() != atom.size()) GPERROR("Hessian neighborhoods have the wrong size");
    if (cellhess && (!atom.cell_.allFinite() || !atom.cell_.fullPivLu().isInvertible()))
        GPERROR("Analytical legacy neural MTP cell Hessian requires a finite invertible cell");
    if (!add_efs) atom.resetEFS(); else atom.setEFSflag();
    hessian.resize(atom.size());
    Eigen::Matrix3d inverse = Eigen::Matrix3d::Zero();
    if (cellhess) {
        inverse = atom.cell_.inverse();
        *cellhess = Eigen::MatrixXd::Zero(9,9);
        *lrhess = Eigen::MatrixXd::Zero(9,3*atom.size());
    }
    for (int i = 0; i < nbhs.size(); ++i) {
        auto &n = nbhs[i];
        double energy;
        SiteCoordinateMatrix gradient;
        HessianMatrix local;
        evaluate_site_hessian(n,energy,gradient,local);
        atom.energy_ += energy;
        atom.site_energies_[i] += energy;
        for (int j = 0; j < n.nbhcount_; ++j) {
            atom.forces_[i] += gradient.row(j).transpose();
            atom.forces_[n.indices_[j]] -= gradient.row(j).transpose();
            atom.stress_ -= gradient.row(j).transpose()*n.pos_[j];
        }
        for (int j = 0; j < n.nbhcount_; ++j)
            for (int k = 0; k < n.nbhcount_; ++k) {
                const auto block = local(j,k);
                const int sj = n.indices_[j], sk = n.indices_[k];
                hessian(sj,sk) += block;
                hessian(i,sk) -= block;
                hessian(sj,i) -= block;
                hessian(i,i) += block;
                if (cellhess) {
                    Eigen::Matrix<double,3,9> aj = Eigen::Matrix<double,3,9>::Zero();
                    Eigen::Matrix<double,3,9> ak = Eigen::Matrix<double,3,9>::Zero();
                    const Eigen::RowVector3d fj = n.pos_[j]*inverse, fk = n.pos_[k]*inverse;
                    for (int a = 0; a < 3; ++a)
                        for (int b = 0; b < 3; ++b) {
                            aj(b,3*a+b) = fj(a);
                            ak(b,3*a+b) = fk(a);
                        }
                    *cellhess += aj.transpose()*block*ak;
                    const Eigen::Matrix<double,9,3> mixed = aj.transpose()*block;
                    lrhess->middleCols(3*sk,3) += mixed;
                    lrhess->middleCols(3*i,3) -= mixed;
                }
            }
    }
}

void Potentials::PotMTPNN1::getEFSHessian(Atoms &atom, AnyNeighborhoods &nbhs, HessianMatrix &hessian) {
    assemble_hessian(atom,nbhs,hessian,nullptr,nullptr);
}
void Potentials::PotMTPNN1::getEFSHessian(Atoms &atom, HessianMatrix &hessian) {
    Neighborhoods nbhs(atom,cutoff);
    getEFSHessian(atom,nbhs,hessian);
}
void Potentials::PotMTPNN1::getEFSHessianOMP(Atoms &atom, AnyNeighborhoods &nbhs, HessianMatrix &hessian) {
    getEFSHessian(atom,nbhs,hessian);
}
void Potentials::PotMTPNN1::getEFSHessianSymm(Atoms &atom, HessianMatrix &hessian, double) {
    getEFSHessian(atom,hessian);
}
void Potentials::PotMTPNN1::getEFSHessianCell(Atoms &atom, AnyNeighborhoods &nbhs, HessianMatrix &hessian,
    Eigen::MatrixXd &cellhess, Eigen::MatrixXd &lrhess) {
    assemble_hessian(atom,nbhs,hessian,&cellhess,&lrhess);
}
void Potentials::PotMTPNN1::getEFSHessianCell(Atoms &atom, HessianMatrix &hessian,
    Eigen::MatrixXd &cellhess, Eigen::MatrixXd &lrhess) {
    Neighborhoods nbhs(atom,cutoff);
    getEFSHessianCell(atom,nbhs,hessian,cellhess,lrhess);
}
void Potentials::PotMTPNN1::getEFSHessianCellOMP(Atoms &atom, AnyNeighborhoods &nbhs, HessianMatrix &hessian,
    Eigen::MatrixXd &cellhess, Eigen::MatrixXd &lrhess) {
    getEFSHessianCell(atom,nbhs,hessian,cellhess,lrhess);
}
void Potentials::PotMTPNN1::getEFSHessianCellSymm(Atoms &atom, HessianMatrix &hessian,
    Eigen::MatrixXd &cellhess, Eigen::MatrixXd &lrhess, double) {
    getEFSHessianCell(atom,hessian,cellhess,lrhess);
}
void Potentials::PotMTPNN1::getEFSHessianCellSymmOMP(Atoms &atom, HessianMatrix &hessian,
    Eigen::MatrixXd &cellhess, Eigen::MatrixXd &lrhess, double) {
    getEFSHessianCell(atom,hessian,cellhess,lrhess);
}
