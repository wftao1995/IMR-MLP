// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#include "potential_hybmtp.h"

void Potentials::PotMTPHyb3::CalcSiteEFS(
    Neighborhood &nbh, double &energy, SiteCoordinateMatrix &gradient) {
    if (!supports_site_efs()) GPERROR("HMTP site EFS requires initialized local submodels");
    energy = 0.0;
    gradient.setZero(nbh.nbhcount_, 3);
    std::vector<int> rows;
    SiteCoordinateMatrix child_gradient;
    for (auto *pot : {pot1, pot2, pot3}) {
        if (!pot) continue;
        Neighborhood site;
        site.centertype = nbh.centertype;
        site.indices_.clear(); site.chemmap_.clear();
        site.pos_.clear(); site.distances_.clear();
        rows.clear();
        for (int i = 0; i < nbh.nbhcount_; ++i) {
            if (nbh.distances_[i] >= pot->cutoff) continue;
            rows.push_back(i);
            site.indices_.push_back(nbh.indices_[i]);
            site.chemmap_.push_back(nbh.chemmap_[i]);
            site.pos_.push_back(nbh.pos_[i]);
            site.distances_.push_back(nbh.distances_[i]);
        }
        site.nbhcount_ = rows.size();
        double value = 0.0;
        pot->CalcSiteEFS(site, value, child_gradient);
        energy += value;
        for (int i = 0; i < site.nbhcount_; ++i)
            gradient.row(rows[i]) += child_gradient.row(i);
    }
}


/**
 * @brief 
 * 
 * @param atom 
 * @param nbhs with largest cutoff
 */
void Potentials::PotMTPHyb3::getEFS(Atoms &atom, AnyNeighborhoods &nbhs) {
    AnyNeighborhoods nbhs1, nbhs2, nbhs3;
    nbhs1 = nbhs.sub_neighborhoods(r1);
    nbhs2 = nbhs.sub_neighborhoods(r2);
    if (pot3) 
        // nbhs3 = nbhs.sub_neighborhoods(r1, r3);
        nbhs3 = nbhs.sub_neighborhoods(r3);
    getEFS(atom, nbhs1, nbhs2, nbhs3);
}

void Potentials::PotMTPHyb3::getEFS(Atoms &atom) {
    AnyNeighborhoods nbhs1, nbhs2, nbhs3;
    Neighborhoods nbhs(atom, rc);
    nbhs1 = nbhs.sub_neighborhoods(r1);
    nbhs2 = nbhs.sub_neighborhoods(r2);
    if (pot3) 
        // nbhs3 = nbhs.sub_neighborhoods(r1, r3);
        nbhs3 = nbhs.sub_neighborhoods(r3);
    getEFS(atom, nbhs1, nbhs2, nbhs3);
}

void Potentials::PotMTPHyb3::getEFSOMP(Atoms &atom) {
    AnyNeighborhoods nbhs1, nbhs2, nbhs3;
    Neighborhoods nbhs(atom, rc);
    nbhs1 = nbhs.sub_neighborhoods(r1);
    nbhs2 = nbhs.sub_neighborhoods(r2);
    if (pot3) 
        // nbhs3 = nbhs.sub_neighborhoods(r1, r3);
        nbhs3 = nbhs.sub_neighborhoods(r3);
    getEFSOMP(atom, nbhs1, nbhs2, nbhs3);
}

void Potentials::PotMTPHyb3::getEFSOMP(Atoms &atom, AnyNeighborhoods &nbhs1, AnyNeighborhoods &nbhs2, AnyNeighborhoods &nbhs3) {
    atom.energy_ = 0;
    atom.stress_ = Eigen::Matrix3d::Zero();
    atom.forces_.assign(atom.size(), Eigen::RowVector3d::Zero());
    atom.site_energies_.assign(atom.size(), 0.0);
    atom.have_energy = true;
    atom.have_stress = true;

    pot1->getEFSOMPadd(atom, nbhs1);
    pot2->getEFSOMPadd(atom, nbhs2);  
    if (pot3) pot3->getEFSOMPadd(atom, nbhs3);
}

void Potentials::PotMTPHyb3::getEFSOMP(Atoms &atom, AnyNeighborhoods &nbhs) {
    AnyNeighborhoods nbhs1, nbhs2, nbhs3;
    nbhs1 = nbhs.sub_neighborhoods(r1);
    nbhs2 = nbhs.sub_neighborhoods(r2);
    if (pot3) 
        // nbhs3 = nbhs.sub_neighborhoods(r1, r3);
        nbhs3 = nbhs.sub_neighborhoods(r3);
    getEFSOMP(atom, nbhs1, nbhs2, nbhs3);
}

void Potentials::PotMTPHyb3::getEFS(Atoms &atom, 
        AnyNeighborhoods &nbhs1, 
        AnyNeighborhoods &nbhs2, 
        AnyNeighborhoods &nbhs3) {
    atom.energy_ = 0;
    atom.stress_ = Eigen::Matrix3d::Zero();
    atom.forces_.assign(atom.size(), Eigen::RowVector3d::Zero());
    atom.site_energies_.assign(atom.size(), 0.0);
    atom.have_energy = true;
    atom.have_stress = true;

    pot1->getEFSadd(atom, nbhs1);
    pot2->getEFSadd(atom, nbhs2);  
    if (pot3) pot3->getEFSadd(atom, nbhs3);
}

// void Potentials::PotMTPHyb3::CalcEFSCmpt(Atoms &atom, 
//         AnyNeighborhoods &nbhs1, 
//         AnyNeighborhoods &nbhs2, 
//         std::vector<double> &energy_cmpts, 
//         Eigen::Tensor<double, 3> &force_cmpts, 
//         Eigen::Tensor<double, 3> &stress_cmpts) {
//     energy_cmpts.assign(lin_coeff_size, 0);
//     force_cmpts.resize(atom.size(), lin_coeff_size, 3);
//     force_cmpts.setZero();
//     stress_cmpts.resize(lin_coeff_size, 3, 3);
//     stress_cmpts.setZero();
//     int offset1 = 0;
//     int offset2 = pot1->lin_coeff_size;

//     pot1->CalcEFSCmpt(atom, nbhs1, energy_cmpts, force_cmpts, stress_cmpts, offset1);
//     pot2->CalcEFSCmpt(atom, nbhs2, energy_cmpts, force_cmpts, stress_cmpts, offset2);
// }

void Potentials::PotMTPHyb3::CalcEFSCmpt(Atoms &atom, 
        AnyNeighborhoods &nbhs1, 
        AnyNeighborhoods &nbhs2, 
        AnyNeighborhoods &nbhs3,
        std::vector<double> &energy_cmpts, 
        Eigen::Tensor<double, 3> &force_cmpts, 
        Eigen::Tensor<double, 3> &stress_cmpts) {
    energy_cmpts.assign(lin_coeff_size, 0);
    force_cmpts.resize(atom.size(), lin_coeff_size, 3);
    force_cmpts.setZero();
    stress_cmpts.resize(lin_coeff_size, 3, 3);
    stress_cmpts.setZero();
    int offset1 = 0;
    int offset2 = pot1->lin_coeff_size;
    int offset3 = offset2 + pot2->lin_coeff_size;

    pot1->CalcEFSCmpt(atom, nbhs1, energy_cmpts, force_cmpts, stress_cmpts, offset1);
    pot2->CalcEFSCmpt(atom, nbhs2, energy_cmpts, force_cmpts, stress_cmpts, offset2);
    if (pot3) pot3->CalcEFSCmpt(atom, nbhs3, energy_cmpts, force_cmpts, stress_cmpts, offset3);
}

// void Potentials::PotMTPHyb3::CalcECmpt(Atoms &atom, 
//         AnyNeighborhoods &nbhs1, 
//         AnyNeighborhoods &nbhs2, 
//         std::vector<double> &energy_cmpts) {
//     energy_cmpts.assign(lin_coeff_size, 0);
//     int offset1 = 0;
//     int offset2 = pot1->lin_coeff_size;

//     pot1->CalcECmpt(atom, nbhs1, energy_cmpts, offset1);
//     pot2->CalcECmpt(atom, nbhs2, energy_cmpts, offset2);
// }

void Potentials::PotMTPHyb3::CalcECmpt(Atoms &atom, 
        AnyNeighborhoods &nbhs1, 
        AnyNeighborhoods &nbhs2, 
        AnyNeighborhoods &nbhs3,
        std::vector<double> &energy_cmpts) {
    energy_cmpts.assign(lin_coeff_size, 0);
    int offset1 = 0;
    int offset2 = pot1->lin_coeff_size;
    int offset3 = offset2 + pot2->lin_coeff_size;

    // _DEBUG_OUTPUT_(nbhs1);
    // _DEBUG_OUTPUT_(nbhs2.size());
    // _DEBUG_OUTPUT_(nbhs3.size());

    pot1->CalcECmpt(atom, nbhs1, energy_cmpts, offset1);
    pot2->CalcECmpt(atom, nbhs2, energy_cmpts, offset2);
    if (pot3) pot3->CalcECmpt(atom, nbhs3, energy_cmpts, offset3);
}

// void Potentials::PotMTPHyb3::getEFS(Atoms &atom, 
//         AnyNeighborhoods &nbhs1, 
//         AnyNeighborhoods &nbhs2) {
//     atom.energy_ = 0;
//     atom.stress_ = Eigen::Matrix3d::Zero();
//     atom.forces_.assign(atom.size(), Eigen::RowVector3d::Zero());
//     atom.site_energies_.resize(atom.size());
//     atom.have_energy = true;
//     atom.have_stress = true;

//     pot1->getEFSadd(atom, nbhs1);
//     pot2->getEFSadd(atom, nbhs2);
// }

void Potentials::PotMTPHyb3::Save(const std::string &potname) {
    if (save_gpm_if_requested(*this, potname)) return;
    if (pot1) pot1->Save(potname + "_sub1");
    if (pot2) pot2->Save(potname + "_sub2");
    if (pcount == 3)
        if (pot3) pot3->Save(potname + "_sub3");
}

void Potentials::PotMTPHyb3::Load(const std::string potname, int pcount) {
    if (pcount != 2 && pcount != 3) GPERROR("HMTP supports two or three submodels");
    needfree = true;
    std::string p1, p2, p3;
    p1 = potname + "_sub1";
    p2 = potname + "_sub2";
    p3 = potname + "_sub3";
    pot1 = new Potentials::PotMTP(p1);
    pot2 = new Potentials::PotMTP(p2);
    if (pcount == 3) pot3 = new Potentials::PotMTP(p3);
    if (pcount == 2) {
        if (pot1->inited() && pot2->inited())
            empty = false;
    }
    if (pcount == 3) {
        if (pot1->inited() && pot2->inited() && pot3->inited())
            empty = false;
    }
}

void Potentials::PotMTPHyb3::AccumulateEFSCombinationGard(
    Atoms &atom, AnyNeighborhoods &nbhs, double dLdE,
    Eigen::Matrix<double, -1, 3> &dLdF, Eigen::Matrix3d &dLdS,
    Eigen::Matrix<double, -1, 3, Eigen::RowMajor> &buffer, double *gradient)
{
    auto nbhs1 = nbhs.sub_neighborhoods(r1);
    auto nbhs2 = nbhs.sub_neighborhoods(r2);
    static_cast<AnyPotential &>(*pot1).AccumulateEFSCombinationGard(atom, nbhs1, dLdE, dLdF, dLdS, buffer, gradient);
    static_cast<AnyPotential &>(*pot2).AccumulateEFSCombinationGard(atom, nbhs2, dLdE, dLdF, dLdS, buffer, gradient + c_offset2);
    if (pot3) {
        auto nbhs3 = nbhs.sub_neighborhoods(r3);
        static_cast<AnyPotential &>(*pot3).AccumulateEFSCombinationGard(atom, nbhs3, dLdE, dLdF, dLdS, buffer, gradient + c_offset3);
    }
}

void Potentials::PotMTPHyb3::AccumulateEFSCombinationGard(
    Atoms &atom, AnyNeighborhoods &nbhs, double dLdE, double *gradient)
{
    auto nbhs1 = nbhs.sub_neighborhoods(r1);
    auto nbhs2 = nbhs.sub_neighborhoods(r2);
    static_cast<AnyPotential &>(*pot1).AccumulateEFSCombinationGard(atom, nbhs1, dLdE, gradient);
    static_cast<AnyPotential &>(*pot2).AccumulateEFSCombinationGard(atom, nbhs2, dLdE, gradient + c_offset2);
    if (pot3) {
        auto nbhs3 = nbhs.sub_neighborhoods(r3);
        static_cast<AnyPotential &>(*pot3).AccumulateEFSCombinationGard(atom, nbhs3, dLdE, gradient + c_offset3);
    }
}

void Potentials::PotMTPHyb3::AddPenaltyGard(const double weight, double &loss, double *loss_gard) {
    pot1->AddPenaltyGard(weight, loss, loss_gard + c_offset1);
    pot2->AddPenaltyGard(weight, loss, loss_gard + c_offset2);
    if (pcount == 3)
        pot3->AddPenaltyGard(weight, loss, loss_gard + c_offset3);
}

void Potentials::PotMTPHyb3::AccumulateEFSCombinationGard(Atoms &atom,
                                          AnyNeighborhoods &nbhs1,
                                          AnyNeighborhoods &nbhs2,
                                          AnyNeighborhoods &nbhs3,
                                          double dLdE,
                                          Eigen::Matrix<double, -1, 3> &dLdF,
                                          Eigen::Matrix3d &dLdS,
                                          double *out_gard_accumulator) {
    pot1->AccumulateEFSCombinationGard(atom, nbhs1, dLdE, dLdF, dLdS, out_gard_accumulator + c_offset1);
    pot2->AccumulateEFSCombinationGard(atom, nbhs2, dLdE, dLdF, dLdS, out_gard_accumulator + c_offset2);
    if (pcount == 3)
        pot3->AccumulateEFSCombinationGard(atom, nbhs3, dLdE, dLdF, dLdS, out_gard_accumulator + c_offset3);
}
