#pragma once

// Included inside namespace Potentials after the complete PotMTP_M class.
template<class T>
AnyPotential::SpeciesAppendResult PotMTP_M<T>::append_species(
    const std::vector<int>& clone_sources,const std::vector<std::string>& new_elements) {
    if(!supports_species_append()) GPERROR("Species append requires a known owned MTP-M head without corrections");
    require_loaded_model();
    const int old_count=species_count,old_radial=radial_coeff_size,old_species=species_coeff_size;
    const int old_total=reg_coeff_size,old_packed=calc_NL_coeffs_count();
    const size_t expected=size_t(old_radial)+old_species+moment_coeff_size+model_coeff_size
        +(lr_les ? size_t(les_channels())*(old_count+alpha_scalar_moments) : 0);
    if(n_props<1 || old_species!=(size_t(n_props)+1)*old_count || expected!=size_t(old_total)
        || moment_coeff_size!=size_t(n_props)*alpha_scalar_moments || (req_les && !lr_les))
        GPERROR("MTP-M species append requires a consistent loaded head layout");
    if(lr_les) model_les.validate();
    for(int source:clone_sources)
        if(source<0 || source>=old_count) GPERROR("Species clone source must be an existing species ID");
    if(!new_elements.empty() && new_elements.size()!=clone_sources.size())
        GPERROR("Appended element declarations must match clone sources");
    SpeciesAppendResult result;
    if(clone_sources.empty()) {
        result.old_to_new_parameters.resize(old_packed);
        std::iota(result.old_to_new_parameters.begin(),result.old_to_new_parameters.end(),0);
        return result;
    }
    const size_t count=size_t(old_count)+clone_sources.size();
    const size_t stride=size_t(radial_func_count)*rb_size;
    const size_t maximum=std::numeric_limits<int>::max();
    if(count>maximum || !stride || count>maximum/stride/count)
        GPERROR("Appended MTP-M parameter layout is too large");
    const size_t radial=count*count*stride;
    const size_t species=count*(size_t(n_props)+1);
    const size_t latent=lr_les ? size_t(les_channels())*(count+alpha_scalar_moments) : 0;
    if(species>maximum || radial+species+moment_coeff_size+model_coeff_size+latent>maximum)
        GPERROR("Appended MTP-M parameter layout is too large");
    const int new_count=count,new_radial=radial,new_species=species;
    const int new_total=radial+species+moment_coeff_size+model_coeff_size+latent;

    // Ordinary MTP owns the established directed-pair/basis/potential_tag append
    // operation. Use an independent staging copy only for those shared parts;
    // its temporary linear head is discarded, never installed on this model.
    PotMTP staged(*this);
    staged.reg_L1_vector.clear();staged.reg_L2_vector.clear();
    staged.append_species(clone_sources,new_elements);
    std::vector<int> sources(new_count);
    std::iota(sources.begin(),sources.begin()+old_count,0);
    std::copy(clone_sources.begin(),clone_sources.end(),sources.begin()+old_count);
    auto coefficients=std::make_unique<double[]>(new_total);
    std::vector<int> source_parameter(new_total),physical_map(old_total,-1);
    auto copy=[&](int destination,int source,bool original) {
        coefficients[destination]=regression_coeffs[source];source_parameter[destination]=source;
        if(original) physical_map[source]=destination;
    };
    for(int i=0;i<new_count;++i) for(int j=0;j<new_count;++j)
        for(size_t a=0;a<stride;++a)
            copy((i*new_count+j)*stride+a,(sources[i]*old_count+sources[j])*stride+a,
                i<old_count && j<old_count);
    for(int i=0;i<new_count;++i)
        copy(new_radial+i,old_radial+sources[i],i<old_count);
    for(int p=0;p<n_props;++p) for(int i=0;i<new_count;++i)
        copy(new_radial+new_count+p*new_count+i,old_radial+old_count+p*old_count+sources[i],i<old_count);
    for(int a=0;a<moment_coeff_size+model_coeff_size;++a)
        copy(new_radial+new_species+a,old_radial+old_species+a,true);
    const int old_latent=old_radial+old_species+moment_coeff_size+model_coeff_size;
    const int new_latent=new_radial+new_species+moment_coeff_size+model_coeff_size;
    if(lr_les) {
        for(int c=0;c<les_channels();++c) for(int i=0;i<new_count;++i)
            copy(new_latent+c*new_count+i,old_latent+c*old_count+sources[i],i<old_count);
        for(int a=0;a<les_channels()*alpha_scalar_moments;++a)
            copy(new_latent+les_channels()*new_count+a,old_latent+les_channels()*old_count+a,true);
    }
    auto active_physical=[&](int n,int r,int s) {
        std::vector<int> indices;
        auto append=[&](int first,int size) {for(int i=0;i<size;++i) indices.push_back(first+i);};
        if(req_radical) append(0,r);
        if(req_species) append(r,n);
        if(req_moment) {append(r+n,s-n);append(r+s,moment_coeff_size);}
        if(req_model) append(r+s+moment_coeff_size,model_coeff_size);
        if(req_les) append(r+s+moment_coeff_size+model_coeff_size,les_channels()*(n+alpha_scalar_moments));
        return indices;
    };
    const auto old_active=active_physical(old_count,old_radial,old_species);
    const auto new_active=active_physical(new_count,new_radial,new_species);
    const int sog_count=uses_sog() && req_sog ? 2*model_les.sog_weights.size() : 0;
    const int new_packed=new_active.size()+sog_count;
    std::vector<int> old_inverse(old_total,-1),new_inverse(new_total,-1);
    for(int i=0;i<old_active.size();++i) old_inverse[old_active[i]]=i;
    for(int i=0;i<new_active.size();++i) new_inverse[new_active[i]]=i;
    result.old_to_new_parameters.resize(old_packed);
    for(int i=0;i<old_active.size();++i) {
        if(physical_map[old_active[i]]<0) GPERROR("Incomplete MTP-M append parameter mapping");
        result.old_to_new_parameters[i]=new_inverse[physical_map[old_active[i]]];
    }
    for(int i=0;i<sog_count;++i)
        result.old_to_new_parameters[old_active.size()+i]=new_active.size()+i;
    auto regularization=[&](const std::vector<double>& previous) {
        std::vector<double> updated;
        if(previous.empty()) return updated;
        if(previous.size()!=size_t(old_total) && previous.size()!=size_t(old_packed))
            GPERROR("Reinitialize MTP-M regularization before species append");
        updated.assign(previous.size()==size_t(old_packed) ? new_packed : new_total,0.);
        for(int i=0;i<new_active.size();++i) {
            const int source=old_inverse[source_parameter[new_active[i]]];
            if(source>=0 && source<previous.size()) updated[i]=previous[source];
        }
        for(int i=0;i<sog_count && new_active.size()+i<updated.size();++i)
            if(old_active.size()+i<previous.size()) updated[new_active.size()+i]=previous[old_active.size()+i];
        return updated;
    };
    auto l1=regularization(reg_L1_vector),l2=regularization(reg_L2_vector);
    Eigen::Tensor<double,3,Eigen::RowMajor> radial_jacobian(alpha_index_basic_count,new_count,radial_func_count*rb_size);
    Eigen::Tensor<double,3,Eigen::RowMajor> radial_mixed(alpha_index_basic_count,new_count,radial_func_count*rb_size);
    // Allocate delayed NN1 maps before changing any owned model storage.
    std::unique_ptr<Eigen::Map<Matrix>> input_map;
    std::unique_ptr<Eigen::Map<Vector>> hidden_map,bias_map;
    double* new_model=coefficients.get()+new_radial+new_species+moment_coeff_size;
    if constexpr(std::is_same_v<T,m_model_nn1>) {
        input_map=std::make_unique<Eigen::Map<Matrix>>(new_model,model.n_input,model.n_neuron);
        hidden_map=std::make_unique<Eigen::Map<Vector>>(new_model+model.n_input*model.n_neuron,model.n_neuron);
        bias_map=std::make_unique<Eigen::Map<Vector>>(new_model+model.n_input*model.n_neuron+model.n_neuron,model.n_neuron);
    }

    delete[] regression_coeffs;regression_coeffs=coefficients.release();
    species_count=new_count;radial_coeff_size=new_radial;species_coeff_size=new_species;reg_coeff_size=new_total;
    lin_coeff_size=species_coeff_size+moment_coeff_size;
    radial_coeffs=regression_coeffs;linear_coeffs=species_coeffs=radial_coeffs+radial_coeff_size;
    moment_coeffs=species_coeffs+species_coeff_size;model_coeffs=moment_coeffs+moment_coeff_size;
    model.coeffs=model_coeffs;
    if(lr_les) {
        les_species=model_coeffs+model_coeff_size;les_moment=les_species+les_channels()*species_count;les_coeffs=les_species;
    }
    if constexpr(std::is_same_v<T,m_model_nn1>) {
        model.w_input_map=std::move(input_map);model.w_hidden_map=std::move(hidden_map);model.b_hidden_map=std::move(bias_map);
    } else model.postinit();
    std::swap(p_radical_basis,staged.p_radical_basis);
    p_radical_basis_pair.swap(staged.p_radical_basis_pair);
    cutoffs=std::move(staged.cutoffs);cutoffsqs=std::move(staged.cutoffsqs);r_mins=std::move(staged.r_mins);
    potential_tag=std::move(staged.potential_tag);
    moment_jacobian_coeffs=std::move(radial_jacobian);moment_jacobian_coeff_coord=std::move(radial_mixed);
    reg_L1_vector=std::move(l1);reg_L2_vector=std::move(l2);
    omp_buffers_list.clear();omp_inited=false;
    LES_all_qs.resize(0);LES_all_latents.resize(0,0);
    n_coeffs_fitting=new_packed;
    return result;
}
