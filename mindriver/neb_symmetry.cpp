#include "neb_symmetry.h"
#include "neb_perturbation.h"
#include "neb.h"
#include "relax_symmetry_selection.h"
#include <numbers>
#include <random>
#include <iomanip>

namespace {
Eigen::Matrix3d symmetric_component(int k) {
    Eigen::Matrix3d m=Eigen::Matrix3d::Zero(); int index=0;
    for(int a=0;a<3;++a) for(int b=a;b<3;++b,++index)
        if(index==k) m(a,b)=m(b,a)=1;
    return m;
}
Eigen::MatrixXd null_modes(const Eigen::MatrixXd &normal) {
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eig(normal);
    if(eig.info()!=Eigen::Success) throw std::runtime_error("NEB symmetry mode decomposition failed");
    int count=0;
    const double tolerance=1e-10*std::max(1.,eig.eigenvalues().maxCoeff());
    while(count<eig.eigenvalues().size() && eig.eigenvalues()[count]<tolerance) ++count;
    return eig.eigenvectors().leftCols(count);
}
}

NEBPathSymmetry::NEBPathSymmetry(const NEBDriver &d,const std::string &selection)
    : cell_dofs_(d.vcoffset), tolerance_(d.symprec) {
    Atoms first=d.images.front();
    initial=Symmetry3D::get_sym_dataset(first,tolerance_,true);
    const Eigen::Matrix3d coordinates=d.vcneb && !d.recast_atomic_force
        ? Eigen::Matrix3d::Identity().eval() : d.refcell;
    const Eigen::Matrix3d inverse_coordinates=coordinates.inverse();
    std::vector<Eigen::MatrixXd> fractional;
    for(const auto &image:d.images) {
        Eigen::MatrixXd positions(image.size(),3);
        for(int i=0;i<image.size();++i) positions.row(i)=image.pos_[i]*image.cell_.inverse();
        fractional.push_back(positions);
    }
    std::vector<Operation> candidates;
    for(int index=0;index<initial.n_operations;++index) {
        Operation op; op.index=index;
        op.fractional=initial.symm_ops[index].first;
        op.translation=initial.symm_ops[index].second;
        op.coordinate=inverse_coordinates*op.fractional*coordinates;
        for(int i=0;i<first.size();++i) {
            const int j=initial.symop_mapping(index,i); op.permutation.push_back(j);
            const Eigen::RowVector3d delta=fractional[0].row(i)*op.fractional+op.translation-fractional[0].row(j);
            op.lattice_images.push_back(delta.array().round().matrix());
        }
        bool common=true;
        for(int image=0;image<d.images.size() && common;++image) {
            const auto &cell=d.images[image].cell_;
            const Eigen::Matrix3d metric=cell*cell.transpose();
            if((op.fractional*metric*op.fractional.transpose()-metric).norm()
                > tolerance_*std::max(1.,cell.norm())) common=false;
            for(int i=0;i<first.size() && common;++i) {
                const Eigen::RowVector3d delta=fractional[image].row(i)*op.fractional+op.translation
                    -fractional[image].row(op.permutation[i])-op.lattice_images[i];
                if((delta*cell).norm()>tolerance_) common=false;
            }
        }
        if(common) {common_operations.push_back(index); candidates.push_back(std::move(op));}
    }
    std::string generators=selection;
    if(generators.empty()) for(int index:common_operations) {
        if(!generators.empty()) generators+=',';
        generators+=std::to_string(index);
    }
    const auto selected=RelaxSymmetry::select_initial(initial,generators,"");
    selected_operations=selected.operations;
    for(int index:selected_operations) {
        auto it=std::find_if(candidates.begin(),candidates.end(),[&](const auto &op){return op.index==index;});
        if(it==candidates.end()) throw std::invalid_argument("NEB symmetry operation "+std::to_string(index)+
            " is not shared by every labelled, unwrapped image (including subgroup closure)");
        operations_.push_back(*it);
    }
    origin_=Eigen::RowVectorXd::Zero(d.vcoffset+d.npx);
    for(const auto &op:operations_) for(int i=0;i<first.size();++i) {
        origin_.segment<3>(cell_dofs_+3*op.permutation[i]) +=
            (fractional[0].row(i)*op.fractional+op.translation-op.lattice_images[i])*coordinates/double(operations_.size());
    }
    // Orbit-local construction uses at most 3x3 eigensystems, avoiding a dense
    // 3N projector while allowing nonorthogonal fractional representations.
    std::vector<bool> visited(first.size(),false);
    for(int rep=0;rep<first.size();++rep) {
        if(visited[rep]) continue;
        Orbit orbit; std::map<int,Eigen::Matrix3d> transforms;
        for(const auto &op:operations_) transforms.try_emplace(op.permutation[rep],op.coordinate.transpose());
        Eigen::Matrix3d normal=Eigen::Matrix3d::Zero();
        for(const auto &[i,transform]:transforms) for(const auto &op:operations_) {
            const auto found=transforms.find(op.permutation[i]);
            if(found==transforms.end()) throw std::runtime_error("NEB symmetry orbit is not closed");
            const Eigen::Matrix3d defect=found->second-op.coordinate.transpose()*transform;
            normal+=defect.transpose()*defect;
        }
        const Eigen::MatrixXd basis=null_modes(normal);
        Eigen::MatrixXd gram=Eigen::MatrixXd::Zero(basis.cols(),basis.cols());
        for(const auto &[i,transform]:transforms) {
            visited[i]=true; orbit.atoms.push_back(i); orbit.maps.push_back(transform*basis);
            gram+=orbit.maps.back().transpose()*orbit.maps.back();
        }
        if(basis.cols()) {
            Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eig(gram);
            const Eigen::MatrixXd normalize=eig.eigenvectors()*eig.eigenvalues().cwiseSqrt().cwiseInverse().asDiagonal()*eig.eigenvectors().transpose();
            for(auto &map:orbit.maps) map=(map*normalize).eval();
        }
        atom_dofs_+=basis.cols(); orbits_.push_back(std::move(orbit));
    }
    cell_basis_.resize(0,0);
    if(cell_dofs_) {
        Eigen::MatrixXd normal=Eigen::MatrixXd::Zero(cell_dofs_,cell_dofs_);
        for(const auto &op:operations_) {
            Eigen::MatrixXd constraints(9,cell_dofs_);
            for(int k=0;k<cell_dofs_;++k) {
                const Eigen::Matrix3d metric=d.refcell*symmetric_component(k)*d.refcell.transpose()/d.refcell.squaredNorm();
                const Eigen::Matrix3d defect=op.fractional*metric*op.fractional.transpose()-metric;
                constraints.col(k)=Eigen::Map<const Eigen::VectorXd>(defect.data(),9);
            }
            normal+=constraints.transpose()*constraints;
        }
        const Eigen::MatrixXd native_basis=null_modes(normal);
        const Eigen::MatrixXd scaled_basis=d.scalval.head(cell_dofs_).asDiagonal()*native_basis;
        Eigen::HouseholderQR<Eigen::MatrixXd> qr(scaled_basis);
        cell_basis_=qr.householderQ()*Eigen::MatrixXd::Identity(cell_dofs_,native_basis.cols());
    }
}

Eigen::RowVectorXd NEBPathSymmetry::project(const Eigen::RowVectorXd &vector) const {
    Eigen::RowVectorXd result=Eigen::RowVectorXd::Zero(vector.size());
    if(cell_dofs_) result.head(cell_dofs_)=vector.head(cell_dofs_)*cell_basis_*cell_basis_.transpose();
    for(const auto &orbit:orbits_) {
        Eigen::VectorXd reduced=Eigen::VectorXd::Zero(orbit.maps.front().cols());
        for(int i=0;i<orbit.atoms.size();++i) reduced+=orbit.maps[i].transpose()*vector.segment<3>(cell_dofs_+3*orbit.atoms[i]).transpose();
        for(int i=0;i<orbit.atoms.size();++i) result.segment<3>(cell_dofs_+3*orbit.atoms[i])=(orbit.maps[i]*reduced).transpose();
    }
    return result;
}
Eigen::RowVectorXd NEBPathSymmetry::project_point(const Eigen::RowVectorXd &point) const {
    return origin_+project(point-origin_);
}

void NEBPathSymmetry::report(std::ostream &out,const NEBDriver &d,const std::string &stage) const {
    out<<"# common_operations_scope initialized_path\n# stage "<<stage<<"\n# symprec_A "<<tolerance_<<"\n# initial_spacegroup "<<initial.space_group_number
       <<"\n# common_operations";
    for(int index:common_operations) out<<' '<<index;
    out<<"\n# selected_operations"; for(int index:selected_operations) out<<' '<<index;
    out<<"\n# allowed_atomic_dofs "<<atom_dofs_<<"\n# allowed_cell_dofs "<<cell_dofs()
       <<"\n# operation: initial_index, fractional row-vector rotation, translation, atom permutation\n";
    for(int index=0;index<initial.n_operations;++index) {
        out<<"# operation "<<index;
        for(int a=0;a<3;++a) for(int b=0;b<3;++b) out<<' '<<initial.symm_ops[index].first(a,b);
        for(int a=0;a<3;++a) out<<' '<<initial.symm_ops[index].second[a];
        for(int atom=0;atom<initial.n_atoms;++atom) out<<' '<<initial.symop_mapping(index,atom);
        out<<'\n';
    }
    out<<"stage,image,spacegroup,operations,selected_residual\n";
    for(int i=0;i<d.images.size();++i) {
        Atoms image=d.images[i]; const auto dataset=Symmetry3D::get_sym_dataset(image,tolerance_);
        const Eigen::RowVectorXd z=d.x.row(i).cwiseProduct(d.scalval.transpose());
        out<<stage<<','<<i<<','<<dataset.space_group_number<<','<<dataset.n_operations<<','<<(project_point(z)-z).norm()<<'\n';
    }
}

void NEBDriver::initialize_symmetry() {
    symmetry_.reset();
    if(!std::isfinite(symprec) || symprec<=0 || !std::isfinite(perturb_amplitude) || perturb_amplitude<0 || perturb_seed<0)
        throw std::invalid_argument("NEB: positive symprec, nonnegative perturb_amplitude and perturb_seed required");
    if(!symmetry_operations.empty() && !keep_symm)
        throw std::invalid_argument("NEB: symmetry_operations requires keep_symm=true");
    if(keep_symm && (sel_dyn || perturb_amplitude>0))
        throw std::invalid_argument("NEB: keep_symm cannot be combined with selective_dynamics or symmetry-breaking perturbation");
    const bool analyse=keep_symm || !symmetry_diagnostics.empty() || perturb_amplitude>0;
    if(!analyse) return;
    const bool supported=images.front().is_fully_periodic() && (!vcneb || symm_strain);
    if(keep_symm && !supported)
        throw std::invalid_argument("NEB: keep_symm requires full periodicity and symmetric VC strain coordinates");
    if(supported) symmetry_=std::make_shared<NEBPathSymmetry>(*this,symmetry_operations);
    record_symmetry("initial",false);
    if(perturb_amplitude>0) {
        NEBPathSeeds::initial(*this,symmetry_.get());
        Message("NEB: smooth path perturbation, amplitude="+std::to_string(perturb_amplitude)+" Angstrom, seed="+std::to_string(perturb_seed));
        record_symmetry("perturbed",true);
    }
    if(keep_symm) {
        for(int i=1;i<images.size()-1;++i) {
            const Eigen::RowVectorXd z=x.row(i).cwiseProduct(scalval.transpose());
            x.row(i)=symmetry_->project_point(z).cwiseQuotient(scalval.transpose());
        }
        X2LatPos(x);
        Message("NEB: common symmetry constraints: "+std::to_string(symmetry_->selected_operations.size())+
            " operations, "+std::to_string(symmetry_->atomic_dofs())+" atomic modes, "+std::to_string(symmetry_->cell_dofs())+" cell modes");
        record_symmetry("constrained",true);
    }
}

void NEBDriver::record_symmetry(const std::string &stage,bool append) const {
    if(symmetry_diagnostics.empty()) return;
    std::ofstream out(symmetry_diagnostics,append?std::ios::app:std::ios::out);
    if(!out) throw std::runtime_error("Cannot open NEB symmetry diagnostics: "+symmetry_diagnostics);
    out<<std::setprecision(17)<<"# keep_symm "<<keep_symm<<"\n# perturb_amplitude_A "<<perturb_amplitude<<"\n# perturb_seed "<<perturb_seed<<'\n';
    if(symmetry_) {
        if(stage!="initial") {
            NEBPathSymmetry current(*this,"");
            out<<"# current_common_operations";
            for(int index:current.common_operations) out<<' '<<index;
            out<<'\n';
        }
        symmetry_->report(out,*this,stage);
    }
    else out<<"# stage "<<stage<<"\n# symmetry analysis unavailable: requires full periodicity and symmetric VC strain coordinates\n";
    out.close(); if(!out) throw std::runtime_error("Failed to write NEB symmetry diagnostics");
}

Eigen::RowVectorXd NEBDriver::project_symmetry(const Eigen::RowVectorXd &vector) const {
    return keep_symm && symmetry_ ? symmetry_->project(vector) : vector;
}

void NEBDriver::project_symmetry_step(Eigen::VectorXd &step) const {
    if(!keep_symm || !symmetry_) return;
    const int stride=vcoffset+npx;
    for(int offset=0;offset<step.size();offset+=stride) {
        const Eigen::RowVectorXd z=step.segment(offset,stride).cwiseProduct(scalval).transpose();
        step.segment(offset,stride)=symmetry_->project(z).transpose().cwiseQuotient(scalval);
    }
}
