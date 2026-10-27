#pragma once

// Included inside namespace Potentials after the complete PotMTP_M class.
template<class T>
LatentCurvatureCallbacks PotMTP_M<T>::curvature_callbacks(Atoms& atom,AnyNeighborhoods& nbhs,
    EvaluationWorkspace& workspace) {
    prepare_site_workspace(workspace.site);
    LatentCurvatureCallbacks callbacks;
    callbacks.finish_efs=[this,&atom,&nbhs,&workspace] {getEFS(atom,nbhs,workspace);};
    callbacks.whole_hvp=[this,&atom,&nbhs,&workspace](const Eigen::VectorXd& direction,Eigen::VectorXd& product) {
        applyHessianVectorProduct(atom,nbhs,direction,product,workspace);
    };
    callbacks.charge_hvp=[this,&workspace](Neighborhood& nbh,int channel,const SiteCoordinateMatrix& direction,
        double& charge,SiteCoordinateMatrix& gradient,SiteCoordinateMatrix& product) {
        auto& buffer=workspace.site.moments;
        WorkspaceChargeHead head{les_species+channel*species_count,
            les_moment+channel*alpha_scalar_moments,alpha_scalar_moments};
        CalcSiteHessImpl<true,WorkspaceChargeHead,0>(nbh,buffer,charge,gradient,buffer.buff_site_hessian,
            head,0,&direction,&product);
    };
    callbacks.charge_second=[this,&workspace](Neighborhood& nbh,int channel,const SiteCoordinateMatrix& v,
        const SiteCoordinateMatrix& w,double& charge,SiteCoordinateMatrix& gradient,
        SiteCoordinateMatrix& hv,SiteCoordinateMatrix& hw) {
        auto& buffer=workspace.site.moments;
        // The full moment graph is head-independent. Its ordinary linear E/F
        // outputs are unused here; only values/Jacobians/Hessians are retained.
        PotMTP::CalcSiteHessOMP(nbh,buffer);
        const auto* coefficients=les_moment+channel*alpha_scalar_moments;
        charge=les_species[channel*species_count+nbh.centertype];
        gradient.setZero(nbh.nbhcount_,3);hv=gradient;hw=gradient;
        for(int m=0;m<alpha_scalar_moments;++m) {
            const double weight=coefficients[m];if(weight==0.) continue;
            const int index=alpha_moment_mapping[m];charge+=weight*buffer.moment_vals[index];
            for(int j=0;j<nbh.nbhcount_;++j) for(int a=0;a<3;++a) {
                gradient(j,a)+=weight*buffer.moment_jacobian(index,j,a);
                for(int k=0;k<nbh.nbhcount_;++k) for(int b=0;b<3;++b) {
                    const double value=weight*buffer.moment_hessian[index](j,a,k,b);
                    hv(j,a)+=value*v(k,b);hw(j,a)+=value*w(k,b);
                }
            }
        }
    };
    return callbacks;
}

template<class T>
void PotMTP_M<T>::getEFSHessian(Atoms& atom,AnyNeighborhoods& nbhs,HessianMatrix& hessian,
    EvaluationWorkspace& workspace) {
    if(!supports_analytical_hessian_vector_product()) GPERROR("Unsupported caller-owned MTP-M Hessian");
    auto callbacks=curvature_callbacks(atom,nbhs,workspace);
    assemble_cartesian_hessian(*this,atom,nbhs,hessian,callbacks);
}

template<class T>
void PotMTP_M<T>::getEFSHessianCell(Atoms& atom,AnyNeighborhoods& nbhs,HessianMatrix& hessian,
    Eigen::MatrixXd& cell,Eigen::MatrixXd& mixed,EvaluationWorkspace& workspace) {
    if(!supports_analytical_hessian_vector_product()) GPERROR("Unsupported caller-owned MTP-M native Hessian");
    auto callbacks=curvature_callbacks(atom,nbhs,workspace);
    const LocalCartesianHVP short_range=[this,&workspace](Neighborhood& n,const SiteCoordinateMatrix& direction,
        SiteCoordinateMatrix& product) {CalcSiteHessianVectorProductM(n,direction,product,workspace.site);};
    assemble_latent_native_hessian(*this,*this,model_les,atom,nbhs,les_species,les_moment,
        les_channels(),lr_les,short_range,hessian,cell,mixed,callbacks,workspace.reciprocal);
}
