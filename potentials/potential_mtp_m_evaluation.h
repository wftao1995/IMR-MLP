#pragma once

// Included inside namespace Potentials after the complete PotMTP_M class.
// These kernels use caller-owned site and reciprocal workspaces throughout.
template<class T>
void PotMTP_M<T>::compute_les_charges_and_descriptors(Atoms& atom,AnyNeighborhoods& nbhs,
    EvaluationWorkspace& workspace) {
    prepare_site_workspace(workspace.site);
    if(!lr_les || nbhs.size()!=atom.size()) GPERROR("Invalid MTP-M latent descriptor request");
    workspace.descriptors.resize(atom.size(),alpha_scalar_moments);
    workspace.latents.resize(atom.size(),les_channels());
    workspace.atom_types.resize(atom.size());
    auto& moments=workspace.site.moments;
    for(int i=0;i<atom.size();++i) {
        auto& nbh=nbhs[i];CalcSiteBasisOMP(nbh,moments);
        workspace.atom_types[i]=nbh.centertype;
        for(int a=0;a<alpha_scalar_moments;++a)
            workspace.descriptors(i,a)=moments.moment_vals[alpha_moment_mapping[a]];
        for(int channel=0;channel<les_channels();++channel) {
            double charge=les_species[channel*species_count+nbh.centertype];
            for(int a=0;a<alpha_scalar_moments;++a)
                charge+=les_moment[channel*alpha_scalar_moments+a]*workspace.descriptors(i,a);
            workspace.latents(i,channel)=charge;
        }
    }
}

template<class T>
void PotMTP_M<T>::charge_gradient_workspace(Neighborhood& nbh,const double* charge_moments,
    SiteWorkspace& workspace,SiteCoordinateMatrix& gradient) {
    prepare_site_workspace(workspace);
    auto& buffer=workspace.moments;
    CalcSiteDescriptorsOMP(nbh,buffer);
    auto& adjoint=buffer.site_energy_ders_wrt_moments_;
    std::fill(adjoint.begin(),adjoint.end(),0.);
    for(int a=0;a<alpha_scalar_moments;++a) adjoint[alpha_moment_mapping[a]]+=charge_moments[a];
    for(int p=alpha_index_times_count-1;p>=0;--p) {
        const int left=alpha_index_times(p,0),right=alpha_index_times(p,1),out=alpha_index_times(p,3);
        const double weight=alpha_index_times(p,2)*adjoint[out];
        adjoint[left]+=weight*buffer.moment_vals[right];
        adjoint[right]+=weight*buffer.moment_vals[left];
    }
    gradient.setZero(nbh.nbhcount_,3);
    for(int m=0;m<alpha_index_basic_count;++m)
        for(int j=0;j<nbh.nbhcount_;++j) for(int a=0;a<3;++a)
            gradient(j,a)+=adjoint[m]*buffer.moment_jacobian(m,j,a);
}

template<class T> template<int REQ_F>
void PotMTP_M<T>::getEFS_impl(Atoms& atom,AnyNeighborhoods& nbhs,EvaluationWorkspace& workspace) {
    require_loaded_model();
    if(nbhs.size()!=atom.size()) GPERROR("MTP-M EFS neighborhoods have the wrong size");
    if(!lr_les) {get_short_range_efs<REQ_F>(atom,nbhs,workspace.site);return;}
    LES::validate_periodic_geometry(atom);
    prepare_site_workspace(workspace.site);
    workspace.descriptors.resize(atom.size(),alpha_scalar_moments);
    workspace.latents.resize(atom.size(),les_channels());
    workspace.atom_types.resize(atom.size());
    if(!add_efs) atom.resetEFS();else atom.setEFSflag();
    auto& buffer=workspace.site.moments;
    for(int center=0;center<nbhs.size();++center) {
        auto& nbh=nbhs[center];
        CalcSiteEFS<REQ_F>(nbh,buffer.buff_site_energy,buffer.buff_site_energy_ders,workspace.site);
        atom.energy_+=buffer.buff_site_energy;atom.site_energies_[center]+=buffer.buff_site_energy;
        workspace.atom_types[center]=nbh.centertype;
        for(int a=0;a<alpha_scalar_moments;++a)
            workspace.descriptors(center,a)=buffer.moment_vals[alpha_moment_mapping[a]];
        for(int channel=0;channel<les_channels();++channel) {
            double q=les_species[channel*species_count+nbh.centertype];
            for(int a=0;a<alpha_scalar_moments;++a)
                q+=les_moment[channel*alpha_scalar_moments+a]*workspace.descriptors(center,a);
            workspace.latents(center,channel)=q;
        }
        if constexpr(REQ_F) for(int j=0;j<nbh.nbhcount_;++j) {
            atom.forces_[center]+=buffer.buff_site_energy_ders.row(j);
            atom.forces_[nbh.indices_[j]]-=buffer.buff_site_energy_ders.row(j);
            for(int a=0;a<3;++a) for(int b=0;b<3;++b)
                atom.stress_(a,b)-=buffer.buff_site_energy_ders(j,a)*nbh.pos_[j][b];
        }
    }
    atom.charges_.clear();
    for(int channel=0;channel<les_channels();++channel) {
        workspace.q=workspace.latents.col(channel);
        workspace.evaluated=model_les.evaluate(atom,workspace.q,workspace.reciprocal,channel);
        const auto& evaluated=workspace.evaluated;
        atom.energy_+=evaluated.energy;
        for(int i=0;i<atom.size();++i) atom.site_energies_[i]+=evaluated.per_atom_energy[i];
        if constexpr(REQ_F) {
            atom.stress_+=evaluated.virial;
            for(int center=0;center<atom.size();++center) {
                atom.forces_[center]+=evaluated.forces.row(center);
                auto& nbh=nbhs[center];
                charge_gradient_workspace(nbh,les_moment+channel*alpha_scalar_moments,
                    workspace.site,workspace.local_gradient);
                for(int j=0;j<nbh.nbhcount_;++j) {
                    const Eigen::RowVector3d contribution=evaluated.dEdq[center]*workspace.local_gradient.row(j);
                    atom.forces_[center]+=contribution;atom.forces_[nbh.indices_[j]]-=contribution;
                    for(int a=0;a<3;++a) for(int b=0;b<3;++b)
                        atom.stress_(a,b)-=contribution[a]*nbh.pos_[j][b];
                }
            }
        }
    }
    workspace.q=workspace.latents.col(0);
    if(!uses_sog()) atom.charges_.assign(workspace.q.data(),workspace.q.data()+workspace.q.size());
}

template<class T>
void PotMTP_M<T>::AccumulateEFSCombinationGard(Atoms& atom,AnyNeighborhoods& nbhs,double energy_weight,
    const Eigen::Matrix<double,-1,3>& force_weight,const Eigen::Matrix3d& stress_weight,
    double* gradient,EvaluationWorkspace& workspace) {
    require_loaded_model();
    if(nbhs.size()!=atom.size() || force_weight.rows()!=atom.size() || !force_weight.allFinite() || !stress_weight.allFinite())
        GPERROR("Invalid MTP-M EFS gradient inputs");
    prepare_site_workspace(workspace.site);
    if(req_radical || req_species || req_moment || req_model) {
        auto& weights=workspace.site.moments.tmp_se_ders_wgts;
        for(int center=0;center<nbhs.size();++center) {
            auto& nbh=nbhs[center];weights.setZero(nbh.nbhcount_,3);
            for(int j=0;j<nbh.nbhcount_;++j) if(nbh.distances_[j]<cutoffs(nbh.centertype,nbh.chemmap_[j]))
                for(int a=0;a<3;++a) {
                    weights(j,a)=force_weight(center,a)-force_weight(nbh.indices_[j],a);
                    for(int b=0;b<3;++b) weights(j,a)-=stress_weight(a,b)*nbh.pos_[j][b];
                }
            AccumulateFPGard(nbh,gradient,energy_weight,&weights,workspace.site);
        }
    }
    accumulate_les_workspace(atom,nbhs,energy_weight,&force_weight,&stress_weight,gradient,workspace);
}

template<class T>
void PotMTP_M<T>::AccumulateEFSCombinationGard(Atoms& atom,AnyNeighborhoods& nbhs,double energy_weight,
    double* gradient,EvaluationWorkspace& workspace) {
    require_loaded_model();
    if(nbhs.size()!=atom.size()) GPERROR("MTP-M gradient neighborhoods have the wrong size");
    if(req_radical || req_species || req_moment || req_model)
        for(int i=0;i<nbhs.size();++i) AccumulateFPGard(nbhs[i],gradient,energy_weight,nullptr,workspace.site);
    accumulate_les_workspace(atom,nbhs,energy_weight,nullptr,nullptr,gradient,workspace);
}

template<class T>
void PotMTP_M<T>::accumulate_les_workspace(Atoms& atom,AnyNeighborhoods& nbhs,double energy_weight,
    const Eigen::Matrix<double,-1,3>* forces,const Eigen::Matrix3d* stress,double* gradient,
    EvaluationWorkspace& workspace) {
    if(!lr_les || (!req_les && !req_radical && !(uses_sog() && req_sog))) return;
    LES::validate_periodic_geometry(atom);
    compute_les_charges_and_descriptors(atom,nbhs,workspace);
    workspace.zero_force.setZero(atom.size(),3);
    const auto& force_weight=forces ? *forces : workspace.zero_force;
    const Eigen::Matrix3d zero_stress=Eigen::Matrix3d::Zero();
    const auto& stress_weight=stress ? *stress : zero_stress;
    const bool response_needed=force_weight.squaredNorm()>1e-30 || stress_weight.squaredNorm()>1e-30;
    auto& buffer=workspace.site.moments;
    for(int channel=0;channel<les_channels();++channel) {
        const auto* charge_moments=les_moment+channel*alpha_scalar_moments;
        workspace.q=workspace.latents.col(channel);
        workspace.dEdq=model_les.apply_charge_hessian(atom,workspace.q,workspace.reciprocal,channel);
        workspace.charge_vjp.setZero(atom.size());workspace.response.setZero(atom.size());
        workspace.moment_vjp.setZero(alpha_scalar_moments);
        if(response_needed) {
            model_les.compute_les_dLdq_from_forces(atom,workspace.q,force_weight,stress_weight,
                workspace.direct_vjp,workspace.reciprocal,channel);
            auto& directional=buffer.dloss_dsenders;directional.resize(alpha_moments_count);
            for(int center=0;center<atom.size();++center) {
                auto& nbh=nbhs[center];CalcSiteDescriptorsOMP(nbh,buffer);
                auto& weights=buffer.tmp_se_ders_wgts;weights.setZero(nbh.nbhcount_,3);
                for(int j=0;j<nbh.nbhcount_;++j) for(int a=0;a<3;++a) {
                    weights(j,a)=force_weight(center,a)-force_weight(nbh.indices_[j],a);
                    for(int b=0;b<3;++b) weights(j,a)-=stress_weight(a,b)*nbh.pos_[j][b];
                }
                std::fill(directional.begin(),directional.end(),0.);
                for(int m=0;m<alpha_index_basic_count;++m)
                    for(int j=0;j<nbh.nbhcount_;++j) for(int a=0;a<3;++a)
                        directional[m]+=weights(j,a)*buffer.moment_jacobian(m,j,a);
                for(int p=0;p<alpha_index_times_count;++p) {
                    const int left=alpha_index_times(p,0),right=alpha_index_times(p,1),out=alpha_index_times(p,3);
                    directional[out]+=alpha_index_times(p,2)*(directional[left]*buffer.moment_vals[right]
                        +directional[right]*buffer.moment_vals[left]);
                }
                for(int a=0;a<alpha_scalar_moments;++a) {
                    const double value=directional[alpha_moment_mapping[a]];
                    workspace.response[center]+=charge_moments[a]*value;
                    workspace.moment_vjp[a]+=workspace.dEdq[center]*value;
                }
            }
            workspace.charge_vjp=workspace.direct_vjp+
                model_les.apply_charge_hessian(atom,workspace.response,workspace.reciprocal,channel);
        }
        if(uses_sog() && req_sog) {
            const int terms=model_les.sog_weights.size(),offset=sog_fitting_offset();
            workspace.kernel_gradient.assign(2*terms,0.);
            model_les.accumulate_kernel_vjp(atom,workspace.q,energy_weight,force_weight,stress_weight,
                workspace.response,workspace.kernel_gradient.data(),workspace.reciprocal,channel);
            const double span=sog_log_bandwidth_max-sog_log_bandwidth_min;
            const double mid=.5*(sog_log_bandwidth_max+sog_log_bandwidth_min);
            for(int m=0;m<terms;++m) {
                const double cosine=std::cos(M_PI*(model_les.sog_log_bandwidths[m]-mid)/span);
                gradient[offset+m]+=workspace.kernel_gradient[m];
                gradient[offset+terms+m]+=workspace.kernel_gradient[terms+m]*span/M_PI*cosine*cosine;
            }
        }
        workspace.dLdq=energy_weight*workspace.dEdq+workspace.charge_vjp;
        if(req_les) {
            const int offset=les_fitting_offset();
            for(int i=0;i<atom.size();++i)
                gradient[offset+channel*species_count+workspace.atom_types[i]]+=workspace.dLdq[i];
            for(int a=0;a<alpha_scalar_moments;++a)
                gradient[offset+les_channels()*species_count+channel*alpha_scalar_moments+a]+=
                    workspace.dLdq.dot(workspace.descriptors.col(a))+workspace.moment_vjp[a];
        }
        if(req_radical) accumulate_charge_radial_workspace(atom,nbhs,charge_moments,workspace.dLdq,
            workspace.dEdq,force_weight,stress_weight,gradient,workspace);
    }
}

template<class T>
void PotMTP_M<T>::applyHessianVectorProduct(Atoms& atom,AnyNeighborhoods& nbhs,
    const Eigen::VectorXd& direction,Eigen::VectorXd& product,EvaluationWorkspace& workspace) {
    require_loaded_model();
    if(!supports_analytical_hessian_vector_product()) GPERROR("Unsupported MTP-M HVP");
    if(direction.size()!=3*atom.size() || !direction.allFinite() || nbhs.size()!=atom.size())
        GPERROR("Invalid MTP-M Hessian-vector direction or neighborhoods");
    if(lr_les) LES::validate_periodic_geometry(atom);
    prepare_site_workspace(workspace.site);
    workspace.product.setZero(direction.size());
    const Eigen::Map<const SiteCoordinateMatrix> velocity(direction.data(),atom.size(),3);
    auto relative_direction=[&](Neighborhood& nbh,int center) {
        workspace.relative.resize(nbh.nbhcount_,3);
        for(int j=0;j<nbh.nbhcount_;++j)
            workspace.relative.row(j)=velocity.row(nbh.indices_[j])-velocity.row(center);
    };
    auto scatter=[&](const Neighborhood& nbh,int center,const SiteCoordinateMatrix& local) {
        for(int j=0;j<nbh.nbhcount_;++j) {
            workspace.product.template segment<3>(3*nbh.indices_[j])+=local.row(j).transpose();
            workspace.product.template segment<3>(3*center)-=local.row(j).transpose();
        }
    };
    for(int center=0;center<nbhs.size();++center) {
        auto& nbh=nbhs[center];relative_direction(nbh,center);
        CalcSiteHessianVectorProductM(nbh,workspace.relative,workspace.local_product,workspace.site);
        scatter(nbh,center,workspace.local_product);
    }
    if(lr_les) {
        auto& buffer=workspace.site.moments;
        workspace.q.resize(atom.size());workspace.q_directional.resize(atom.size());
        for(int channel=0;channel<les_channels();++channel) {
            WorkspaceChargeHead head{les_species+channel*species_count,
                les_moment+channel*alpha_scalar_moments,alpha_scalar_moments};
            auto evaluate=[&](Neighborhood& nbh,int center,double& charge) {
                relative_direction(nbh,center);
                CalcSiteHessImpl<true,WorkspaceChargeHead,0>(nbh,buffer,charge,workspace.local_gradient,
                    buffer.buff_site_hessian,head,0,&workspace.relative,&workspace.local_product);
            };
            for(int center=0;center<atom.size();++center) {
                evaluate(nbhs[center],center,workspace.q[center]);
                workspace.q_directional[center]=(workspace.local_gradient.array()*workspace.relative.array()).sum();
            }
            workspace.directional=model_les.evaluate_cartesian_directional(atom,workspace.q,
                workspace.q_directional,velocity,workspace.reciprocal,channel);
            for(int center=0;center<atom.size();++center) {
                workspace.product.template segment<3>(3*center)-=workspace.directional.force_directional.row(center).transpose();
                double charge;evaluate(nbhs[center],center,charge);
                workspace.local_combination=workspace.directional.dEdq_directional[center]*workspace.local_gradient
                    +workspace.directional.dEdq[center]*workspace.local_product;
                scatter(nbhs[center],center,workspace.local_combination);
            }
        }
    }
    product=workspace.product;
}

template<class T>
void PotMTP_M<T>::accumulate_charge_radial_workspace(Atoms& atom,AnyNeighborhoods& nbhs,
    const double* charge_moments,const Eigen::VectorXd& dLdq,const Eigen::VectorXd& dEdq,
    const Eigen::Matrix<double,-1,3>& dLdF,const Eigen::Matrix3d& dLdS,double* out_gard,
    EvaluationWorkspace& workspace) {
    prepare_site_workspace(workspace.site);
    auto& buffer=workspace.site.moments;
    buffer.moment_jacobian_coeffs.resize(alpha_index_basic_count,species_count,radial_func_count*rb_size);
    buffer.moment_jacobian_coeff_coord.resize(alpha_index_basic_count,species_count,radial_func_count*rb_size);
    buffer.dloss_dmom.resize(alpha_moments_count);buffer.dloss_dsenders.resize(alpha_moments_count);
    auto& moment_vals=buffer.moment_vals;
    auto& moment_jacobian=buffer.moment_jacobian;
    auto& moment_jacobian_coeffs=buffer.moment_jacobian_coeffs;
    auto& moment_jacobian_coeff_coord=buffer.moment_jacobian_coeff_coord;
    auto& dloss_dmom=buffer.dloss_dmom;
    auto& dloss_dsenders=buffer.dloss_dsenders;
    auto& tmp_se_ders_wgts=buffer.tmp_se_ders_wgts;
    auto& rb_vals=buffer.rb_vals;
    auto& rb_ders=buffer.rb_ders;
    auto& val_mu=buffer.val_mu;
    auto& der_mu=buffer.der_mu;
    auto& dist_powers=buffer.dist_powers;
    auto& coords_powers=buffer.coords_powers;

    const int n_atoms = atom.size();
    const bool has_force = dLdF.squaredNorm() > 1e-30;
    const bool has_stress = dLdS.squaredNorm() > 1e-30;
    for (int i = 0; i < n_atoms; ++i) {
        Neighborhood &nbh = nbhs[i];
        CalcSiteDescriptorsOMP(nbh,buffer);

        const double dLdq_i = dLdq(i);

        // q_der is dq/d(moment node), propagated through the product graph.
        auto& q_der=workspace.q_der;
        q_der.assign(alpha_moments_count,0.);
        for (int a = 0; a < alpha_scalar_moments; ++a) {
            int k = alpha_moment_mapping[a];
            q_der[k] += charge_moments[a];
        }
        for (int p = alpha_index_times_count - 1; p >= 0; --p) {
            double val0 = moment_vals[alpha_index_times(p, 0)];
            double val1 = moment_vals[alpha_index_times(p, 1)];
            int    val2 = alpha_index_times(p, 2);
            int    id3  = alpha_index_times(p, 3);
            q_der[alpha_index_times(p, 0)] += q_der[id3] * val2 * val1;
            q_der[alpha_index_times(p, 1)] += q_der[id3] * val2 * val0;
        }

        std::fill(dloss_dmom.begin(), dloss_dmom.end(), 0.0);
        for (int m=0; m<alpha_index_basic_count; ++m)
            dloss_dmom[m] = dLdq_i*q_der[m];

        // The explicit indirect-force term is
        // g_i w·d/dc(dq_i/dr), with g_i=dE/dq_i.  It contains both the
        // product-graph mixed derivative and the basic radial mixed
        // derivative accumulated in moment_jacobian_coeff_coord below.
        std::fill(dloss_dsenders.begin(), dloss_dsenders.end(), 0.0);
        if (has_force || has_stress) {
            tmp_se_ders_wgts.setZero(nbh.nbhcount_, 3);
            for (int j=0; j<nbh.nbhcount_; ++j)
                for (int a=0; a<3; ++a) {
                    tmp_se_ders_wgts(j,a)=dLdF(i,a)-dLdF(nbh.indices_[j],a);
                    for (int b=0; b<3; ++b)
                        tmp_se_ders_wgts(j,a)-=dLdS(a,b)*nbh.pos_[j](b);
                }
            for (int m=0; m<alpha_index_basic_count; ++m)
                for (int j=0; j<nbh.nbhcount_; ++j)
                    for (int a=0; a<3; ++a)
                        dloss_dsenders[m] += tmp_se_ders_wgts(j,a)
                            * moment_jacobian(m,j,a);
            for (int p=0; p<alpha_index_times_count; ++p) {
                const int left=alpha_index_times(p,0), right=alpha_index_times(p,1);
                const int result=alpha_index_times(p,3);
                const double factor=alpha_index_times(p,2);
                dloss_dsenders[result] += factor
                    *(dloss_dsenders[left]*moment_vals[right]
                      +dloss_dsenders[right]*moment_vals[left]);
            }

            auto& mixed=workspace.mixed;
            mixed.assign(alpha_moments_count,0.);
            for (int p=0; p<alpha_index_times_count; ++p) {
                const int left=alpha_index_times(p,0), right=alpha_index_times(p,1);
                const int result=alpha_index_times(p,3);
                const double factor=alpha_index_times(p,2)*dEdq(i)*q_der[result];
                mixed[left] += factor*dloss_dsenders[right];
                mixed[right] += factor*dloss_dsenders[left];
            }
            for (int p=alpha_index_times_count-1; p>=0; --p) {
                const int left=alpha_index_times(p,0), right=alpha_index_times(p,1);
                const int result=alpha_index_times(p,3);
                const double factor=alpha_index_times(p,2);
                mixed[left] += mixed[result]*factor*moment_vals[right];
                mixed[right] += mixed[result]*factor*moment_vals[left];
            }
            for (int m=0; m<alpha_index_basic_count; ++m)
                dloss_dmom[m] += mixed[m];
        }

        // Zero the Jacobian buffers
        moment_jacobian_coeffs.setZero();
        moment_jacobian_coeff_coord.setZero();
        std::ranges::fill(rb_vals, 0);
        std::ranges::fill(rb_ders, 0);
        std::ranges::fill(val_mu, 0);
        std::ranges::fill(der_mu, 0);
        std::ranges::fill(moment_vals, 0);

        // Compute Jacobian for each neighbor
        const int itype = nbh.centertype;
        for (int jj = 0; jj < nbh.nbhcount_; ++jj) {
            const double rij = nbh.distances_[jj];
            const int jtype = nbh.chemmap_[jj];
            if (rij > cutoffs(itype, jtype)) continue;
            const double rijinv = 1.0 / rij;
            rb_calcder_omp(itype,jtype,rij,buffer);

            dist_powers[0] = 1;
            for (int k = 1; k < max_alpha_index_basic; ++k) {
                dist_powers[k] = dist_powers[k-1] * rijinv;
                for (int a = 0; a < 3; a++)
                    coords_powers(k+1, a) = coords_powers(k, a) * nbh.pos_[jj][a];
            }

            for (int mu = 0; mu < max_mu; ++mu) {
                val_mu[mu] = 0; der_mu[mu] = 0;
                int off = (itype * species_count + jtype) * radial_func_count * rb_size + mu * rb_size;
                for (int xi = 0; xi < rb_size; ++xi) {
                    val_mu[mu] += radial_coeffs[off + xi] * rb_vals[xi];
                    der_mu[mu] += radial_coeffs[off + xi] * rb_ders[xi];
                }
            }

            for (int ii = 0; ii < alpha_index_basic_count; ++ii) {
                int mu = alpha_index_basic(ii, 0);
                double val = val_mu[mu], der = der_mu[mu];
                int k = nu_arr_list[ii];
                double powk = dist_powers[k];
                val *= powk;
                double pow0 = coords_powers(alpha_index_basic(ii, 1)+1, 0);
                double pow1 = coords_powers(alpha_index_basic(ii, 2)+1, 1);
                double pow2 = coords_powers(alpha_index_basic(ii, 3)+1, 2);
                double mult0 = pow0 * pow1 * pow2;
                moment_vals[ii] += val * mult0;

                for (int xi = 0; xi < rb_size; xi++) {
                    moment_jacobian_coeffs(ii, jtype, mu * rb_size + xi) +=
                        rb_vals[xi] * powk * mult0;
                    if (has_force || has_stress) {
                        const double tail=(rb_ders[xi]*powk*mult0
                            -rb_vals[xi]*k*powk*mult0/rij);
                        double derx=nbh.pos_[jj][0]/rij*tail;
                        double dery=nbh.pos_[jj][1]/rij*tail;
                        double derz=nbh.pos_[jj][2]/rij*tail;
                        derx += rb_vals[xi]*powk*alpha_index_basic(ii,1)
                            *coords_powers(alpha_index_basic(ii,1),0)*pow1*pow2;
                        dery += rb_vals[xi]*powk*alpha_index_basic(ii,2)
                            *coords_powers(alpha_index_basic(ii,2),1)*pow0*pow2;
                        derz += rb_vals[xi]*powk*alpha_index_basic(ii,3)
                            *coords_powers(alpha_index_basic(ii,3),2)*pow0*pow1;
                        moment_jacobian_coeff_coord(ii,jtype,mu*rb_size+xi) +=
                            tmp_se_ders_wgts(jj,0)*derx
                            +tmp_se_ders_wgts(jj,1)*dery
                            +tmp_se_ders_wgts(jj,2)*derz;
                    }
                }
            }
        }

        // Multiply by dloss_dmom and accumulate radial gradient
        // dloss_dmom already contains the full dL/d(moment), no extra factor needed
        for (int ii = 0; ii < alpha_index_basic_count; ++ii) {
            for (int j = 0; j < species_count; ++j) {
                for (int k = 0; k < radial_func_count * rb_size; ++k) {
                    int idx = (itype * species_count + j) * radial_func_count * rb_size + k;
                    out_gard[idx] += dloss_dmom[ii] * moment_jacobian_coeffs(ii, j, k);
                    if (has_force || has_stress)
                        out_gard[idx] += dEdq(i)*q_der[ii]
                            *moment_jacobian_coeff_coord(ii,j,k);
                }
            }
        }
    }
}
