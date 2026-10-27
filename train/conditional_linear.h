#pragma once

#include "potentials/potential.h"
#include "hvp_fit.h"
#include <algorithm>
#include <set>
#include <sstream>

namespace Training {

// Legacy selectors retain their meaning; all other tokens name exact advertised
// blocks. Models must advertise jointly affine blocks, never alternative fits.
inline std::vector<std::string> conditional_linear_selection(Potentials::AnyPotential &model,
                                                            const std::string &mode) {
    if (mode=="all") return {};
    if (mode=="sr_sog") return {"sr","sog"};
    if (mode=="head") {
        for (const auto &block : model.conditional_linear_blocks(0.))
            if (block.name=="output") return {"output"};
        for (const auto &block : model.conditional_linear_blocks(0.))
            if (block.name=="linear") return {"linear"};
        GPERROR("Potential has no affine output head");
    }
    if (mode.empty() || mode=="off") GPERROR("Select a conditional linear block");
    std::vector<std::string> names;
    std::istringstream input(mode);std::string name;
    while (std::getline(input,name,',')) {
        const auto begin=name.find_first_not_of(" \t\r\n");
        const auto end=name.find_last_not_of(" \t\r\n");
        if (begin==std::string::npos) GPERROR("Empty conditional linear block");
        names.push_back(name.substr(begin,end-begin+1));
    }
    if (mode.back()==',') GPERROR("Empty conditional linear block");
    return names;
}

// Model-owned affine claims, with selected packed-parameter scatter. This
// adapter never infers linearity from a parameter-group name or inheritance.
class ConditionalLinearAdapter {
public:
    Potentials::AnyPotential &potential;
    std::vector<int> indices;
    std::vector<std::string> blocks;
    int composition_count = 0;
    Eigen::VectorXd l1, l2;

    ConditionalLinearAdapter(Potentials::AnyPotential &model,
        const std::vector<std::string> &selected, double penalty_weight)
        : potential(model) {
        const auto available=model.conditional_linear_blocks(penalty_weight);
        std::set<std::string> pending(selected.begin(),selected.end());
        if (pending.size()!=selected.size()) GPERROR("Duplicate conditional linear block");
        struct Column { int index; double l1,l2; bool composition; };
        std::vector<Column> columns;
        std::set<int> seen;
        for (const auto &block : available) {
            if (!selected.empty() && !pending.erase(block.name)) continue;
            if (block.composition_count<0 || block.composition_count>int(block.indices.size()) ||
                block.l1.size()!=block.indices.size() || block.l2.size()!=block.indices.size())
                GPERROR("Invalid model conditional linear block metadata");
            blocks.push_back(block.name);
            for (int j=0;j<int(block.indices.size());++j) {
                const int index=block.indices[j];
                if (index<0 || index>=model.get_NL_coeffs_count() || !seen.insert(index).second ||
                    !std::isfinite(block.l1[j]) || !std::isfinite(block.l2[j]))
                    GPERROR("Invalid conditional linear parameter or penalty");
                columns.push_back({index,block.l1[j],block.l2[j],j<block.composition_count});
            }
        }
        if (!pending.empty()) GPERROR("Unsupported conditional linear block: "+*pending.begin());
        if (columns.empty()) GPERROR("Potential has no selected conditional linear parameters");
        // Composition columns from multiple children stay contiguous for the
        // composition-rank elimination used by conditional least squares.
        std::stable_partition(columns.begin(),columns.end(),[](const Column &c){return c.composition;});
        l1.resize(columns.size());l2.resize(columns.size());
        for (int j=0;j<int(columns.size());++j) {
            const auto &c=columns[j];indices.push_back(c.index);l1[j]=c.l1;l2[j]=c.l2;
            composition_count+=c.composition;
        }
    }

    int size() const { return indices.size(); }
    bool selected(int index) const {
        return std::find(indices.begin(),indices.end(),index)!=indices.end();
    }
    Eigen::VectorXd full_coefficients() const {
        Eigen::VectorXd full(potential.get_NL_coeffs_count());potential.nl_get_coeffs(full.data());return full;
    }
    Eigen::VectorXd get() const {
        const auto full=full_coefficients();Eigen::VectorXd out(size());
        for (int i=0;i<size();++i) out[i]=full[indices[i]];
        return out;
    }
    void set(const Eigen::VectorXd &values) {
        if (values.size()!=size() || !values.allFinite()) GPERROR("Invalid conditional linear coefficients");
        potential.conditional_linear_set_coeffs(indices,values);
    }

    static Eigen::VectorXd efs_values(const Atoms &atom) {
        Eigen::VectorXd out(1+3*atom.size()+9);out[0]=atom.energy_;
        for (int i=0;i<atom.size();++i) for (int d=0;d<3;++d) out[1+3*i+d]=atom.forces_[i][d];
        for (int i=0;i<3;++i) for (int j=0;j<3;++j) out[1+3*atom.size()+3*i+j]=atom.stress_(i,j);
        return out;
    }
    Eigen::VectorXd prediction(const Atoms &reference) {
        Atoms atom=reference;Neighborhoods neighbors(atom,potential.cutoff);
        atom.resetEFS();potential.getEFS(atom,neighbors);return efs_values(atom);
    }
    Eigen::MatrixXd basis(const Atoms &reference) {
        Atoms atom=reference;Neighborhoods neighbors(atom,potential.cutoff);
        const int rows=1+3*atom.size()+9;
        Eigen::MatrixXd out(rows,size());
        Eigen::Matrix<double,-1,3> force=Eigen::Matrix<double,-1,3>::Zero(atom.size(),3);
        Eigen::Matrix3d stress=Eigen::Matrix3d::Zero();
        Eigen::Matrix<double,-1,3,Eigen::RowMajor> work;
        Eigen::VectorXd gradient(potential.get_NL_coeffs_count());
        for (int row=0;row<rows;++row) {
            // Several backends cache neural/long-range state in getEFS. Each
            // independent cotangent therefore starts from a fresh forward pass.
            atom.resetEFS();potential.getEFS(atom,neighbors);
            force.setZero();stress.setZero();gradient.setZero();
            if (row>0 && row<=3*atom.size()) force((row-1)/3,(row-1)%3)=1.;
            if (row>3*atom.size()) stress((row-1-3*atom.size())/3,(row-1-3*atom.size())%3)=1.;
            potential.AccumulateEFSCombinationGard(atom,neighbors,row==0 ? 1. : 0.,force,stress,work,gradient.data());
            for (int j=0;j<size();++j) out(row,j)=gradient[indices[j]];
        }
        return out;
    }
    Eigen::VectorXd fixed_contribution(const Atoms &reference, const Eigen::MatrixXd &columns) {
        return prediction(reference)-columns*get();
    }
    // Same displaced-force operator as training; columns are selected affine
    // coefficients, not derivatives of a surrogate Hessian implementation.
    Eigen::MatrixXd hvp_basis(const Atoms &reference, double step) {
        ForceObservation forces=[&](Atoms &atom,Neighborhoods &) {
            return basis(atom).middleRows(1,3*atom.size()).eval();
        };
        return finite_difference_hvp_observations(reference,step,potential.cutoff,forces);
    }
};

} // namespace Training
