#include "neb_perturbation.h"
#include "neb_symmetry.h"
#include <numbers>
#include <random>

void NEBPathSeeds::initial(NEBDriver &d,const NEBPathSymmetry *symmetry) {
    std::mt19937_64 rng(d.perturb_seed);
    Eigen::RowVectorXd direction=Eigen::RowVectorXd::Zero(d.vcoffset+d.npx);
    for(int j=d.vcoffset;j<direction.size();++j) direction[j]=2.*double(rng()>>11)/9007199254740992.-1.;
    // Remove the invariant part only when there are nontrivial common
    // operations. P1 paths still receive a reproducible smooth perturbation.
    if(symmetry && symmetry->common_operations.size()>1) direction-=symmetry->project(direction);
    Eigen::RowVector3d mean=Eigen::RowVector3d::Zero();
    for(int a=0;a<d.natom;++a) mean+=direction.segment<3>(d.vcoffset+3*a)/double(d.natom);
    if(d.natom>1) for(int a=0;a<d.natom;++a) direction.segment<3>(d.vcoffset+3*a)-=mean;
    if(d.sel_dyn) for(int a=0;a<d.natom;++a) direction.segment<3>(d.vcoffset+3*a).array()*=d.images[0].sel_dynpos_[a].array();
    auto proposal=d.x;
    for(int i=1;i<d.images.size()-1;++i) proposal.row(i)+=std::sin(std::numbers::pi*i/(d.images.size()-1))*direction;
    const auto original=d.images;
    d.X2LatPos(proposal);
    double maximum=0;
    for(int i=1;i<d.images.size()-1;++i) for(int a=0;a<d.natom;++a)
        maximum=std::max(maximum,(d.images[i].pos_[a]-original[i].pos_[a]).norm());
    d.images=original;
    if(maximum<1e-14) throw std::invalid_argument("NEB: no symmetry-breaking displacement remains after constraints");
    d.x+=(proposal-d.x)*(d.perturb_amplitude/maximum);
    d.X2LatPos(d.x);
}

std::pair<double,double> NEBPathSeeds::release(NEBDriver &trial,const NEBPathSymmetry &symmetry,int seed,double amplitude,double strain) {
    const auto source_images=trial.images;double actual_atom=0,actual_cell=0;
    std::mt19937_64 rng(seed);
    Eigen::RowVectorXd atoms=Eigen::RowVectorXd::Zero(trial.x.cols()),cell=atoms;
    for(int j=0;j<trial.x.cols();++j) {
        const double random=2.*double(rng()>>11)/9007199254740992.-1.;
        (j<trial.vcoffset?cell:atoms)[j]=random;
    }
    atoms-=symmetry.project(atoms);cell-=symmetry.project(cell);
    Eigen::RowVector3d mean=Eigen::RowVector3d::Zero();
    for(int a=0;a<trial.natom;++a) mean+=atoms.segment<3>(trial.vcoffset+3*a)/double(trial.natom);
    if(trial.natom>1) for(int a=0;a<trial.natom;++a) atoms.segment<3>(trial.vcoffset+3*a)-=mean;
    auto displace=[&](Eigen::RowVectorXd direction, double bound, bool is_cell) {
        if(bound==0 || direction.norm()<1e-12) return;
        direction=direction.cwiseQuotient(trial.scalval.transpose());
        direction*=std::min(1.,bound)/std::max(1.,direction.norm());
        const auto base=trial.x;
        const auto original=trial.images;
        double factor=1;
        // Bound the decoded geometry, not its generalized coordinates.
        for(int attempt=0;attempt<60;++attempt) {
            auto proposal=base;
            for(int i=1;i<source_images.size()-1;++i)
                proposal.row(i)+=factor*std::sin(std::numbers::pi*i/(source_images.size()-1))*direction;
            bool valid=true;double maximum=0;
            try {
                trial.X2LatPos(proposal);trial.validate_images();
                for(int i=1;i<source_images.size()-1;++i) {
                    if(is_cell) maximum=std::max(maximum,(original[i].cell_.inverse()*trial.images[i].cell_-Eigen::Matrix3d::Identity()).norm());
                    else for(int a=0;a<trial.natom;++a) maximum=std::max(maximum,(trial.images[i].pos_[a]-original[i].pos_[a]).norm());
                }
            } catch(const NEBInvalidGeometry &) {valid=false;}
            if(valid && maximum>1e-14 && maximum<=bound*(1+1e-10) && (maximum>=bound*.99 || attempt>50)) {
                trial.x=proposal;return;
            }
            trial.images=original;
            if(!valid || !std::isfinite(maximum)) factor*=.5;
            else if(maximum<1e-14) return;
            else factor*=std::min(2.,.999*bound/maximum);
        }
        throw std::runtime_error("Cannot construct a valid bounded release perturbation");
    };
    displace(cell,strain,true);
    // Cell motion carries atoms affinely; the atomic bound applies to the additional displacement.
    const auto affine=trial.images;
    displace(atoms,amplitude,false);
    for(int i=1;i<source_images.size()-1;++i) {
        actual_cell=std::max(actual_cell,(source_images[i].cell_.inverse()*trial.images[i].cell_-Eigen::Matrix3d::Identity()).norm());
        for(int a=0;a<trial.natom;++a) actual_atom=std::max(actual_atom,(trial.images[i].pos_[a]-affine[i].pos_[a]).norm());
    }
    return {actual_atom,actual_cell};
}

void NEBDriver::prepare_complete_path(const std::vector<Atoms> &path,const Eigen::Matrix3d &reference) {
    images=path;subspace_active=false;
    if(images.size()<3 || images.front().size()==0)throw std::invalid_argument("A complete path requires at least three nonempty images");
    for(auto &a:images) {
        if(a.size()!=images.front().size())throw std::invalid_argument("Every image must be supplied; interpolation is disabled");
        if(!a.cartesian)a.to_cartesian();
    }
    if(vcneb && (!images.front().is_fully_periodic() || !symm_strain || periodic_subspace))
        throw std::invalid_argument("Path seeds require fully periodic symmetric-strain VC coordinates");
    refcell=vcneb?reference:images.front().cell_;
    if(vcneb && (!refcell.allFinite() || std::abs(refcell.determinant())<1e-12))
        throw std::invalid_argument("A finite invertible reference cell is required for VC path seeds");
    ivrefcell.setZero();if(std::abs(refcell.determinant())>=1e-12)ivrefcell=refcell.inverse();
    iref=-3;check_mic=false;init_method="linear";keep_symm=false;perturb_amplitude=0;
    validate_images(false);
    if(sel_dyn) {
        if(images.front().sel_dynpos_.empty())images.front().sel_dynpos_.assign(images.front().size(),Eigen::RowVector3d::Ones());
        for(int i=0;i<9;++i)images.front().sel_dyncell_(i/3,i%3)=cmask[i]=='1';
        for(auto &a:images) {
            if(a.sel_dynpos_.empty())a.sel_dynpos_=images.front().sel_dynpos_;
            if(a.sel_dynpos_!=images.front().sel_dynpos_)throw std::invalid_argument("Path constraints must have one fixed atom mapping");
        }
    }
    natom=images.front().size();npx=3*natom;vcoffset=vcneb?6:0;
    x.resize(images.size(),vcoffset+npx);g=decltype(g)::Zero(images.size(),x.cols());initialize_scaling();
    x.setZero();
    Eigen::VectorXd flat;LatPos2X(flat);
    // LatPos2X(flat) intentionally encodes only optimizer (interior) rows.
    // A standalone geometry must also own initialized endpoint coordinates.
    for(int i:{0,int(images.size())-1}) {
        if(vcneb) {
            const auto strain=get_strain(images[i].cell_);int j=0;
            for(int a=0;a<3;++a)for(int b=a;b<3;++b)x(i,j++)=strain(a,b);
        }
        for(int a=0;a<natom;++a)x.row(i).segment<3>(vcoffset+3*a)=encode_position(images[i],a);
    }
    const auto original=images;X2LatPos(x);
    for(int i=1;i<images.size()-1;++i) {
        if(!images[i].cell_.isApprox(original[i].cell_,1e-10))throw std::invalid_argument("Path cell is outside the supplied reference coordinate chart");
        for(int a=0;a<natom;++a)if(!images[i].pos_[a].isApprox(original[i].pos_[a],1e-10))
            throw std::invalid_argument("Path coordinates are outside the supplied reference chart");
    }
    images=original;
}

NEBSeedResult NEBPathSeeds::generate(const NEBDriver &source,const NEBSeedOptions &options) {
    if(options.mode!="sine1" && options.mode!="sine2" && options.mode!="mixed3")throw std::invalid_argument("Unknown path seed mode");
    if((options.sign!=1 && options.sign!=-1) || !std::isfinite(options.amplitude) || options.amplitude<0 ||
       !std::isfinite(options.strain) || options.strain<0 || !std::isfinite(options.radius) || options.radius<0)
        throw std::invalid_argument("Invalid seed sign, amplitude, strain or radius");
    NEBDriver d;d.ApplySettings(source.CurrentSettings());d.ciival=source.ciival;
    d.prepare_complete_path(source.images,source.refcell);
    const auto original=d.images;
    NEBSeedResult result;result.images=original;
    if(options.center>=d.natom || options.center<-1 || (options.radius>0 && options.center<0))throw std::invalid_argument("A valid center atom is required for a local radius");
    std::vector<double> weights(d.natom,options.atoms.empty()?1.:0.);
    for(int a:options.atoms) {if(a<0 || a>=d.natom)throw std::invalid_argument("Local atom index out of range");weights[a]=1.;}
    if(options.center>=0 && options.radius==0)throw std::invalid_argument("A center requires a positive radius");
    if(options.radius>0) {
        auto middle=original[original.size()/2];
        for(int a=0;a<d.natom;++a) {
            const double r=a==options.center?0.:middle.get_rab(options.center,a).norm();
            weights[a]*=r>=options.radius?0.:.5*(1+std::cos(std::numbers::pi*r/options.radius));
        }
    }
    for(int a=0;a<d.natom;++a)if(weights[a]>0)result.selected_atoms.push_back(a);
    std::unique_ptr<NEBPathSymmetry> symmetry;
    if(original.front().is_fully_periodic())symmetry=std::make_unique<NEBPathSymmetry>(d,"");
    const bool complement=symmetry && (symmetry->atomic_dofs()<d.npx || symmetry->cell_dofs()<d.vcoffset);
    result.subspace=complement?"symmetry_complement":"free";
    const int dimensions=d.x.cols();
    auto feasible=[&](Eigen::RowVectorXd direction,bool cell) {
        std::vector<int> allowed;
        for(int j=cell?0:d.vcoffset;j<(cell?d.vcoffset:dimensions);++j) {
            bool active=true;
            if(cell && d.sel_dyn) {const int map[]={0,1,2,4,5,8};active=d.cmask[map[j]]=='1';}
            if(!cell) {const int atom=(j-d.vcoffset)/3,axis=(j-d.vcoffset)%3;
                active=weights[atom]>0 && (!d.sel_dyn || original[0].sel_dynpos_[atom][axis]!=0);
                direction[j]*=weights[atom];}
            if(active)allowed.push_back(j);else direction[j]=0;
        }
        // Full support admits pure translations. Strictly local support does
        // not: removing its mean would incorrectly delete single-atom motion.
        const bool translations=!cell && allowed.size()==d.npx && d.natom>1;
        if(allowed.empty())return Eigen::RowVectorXd::Zero(dimensions).eval();
        Eigen::MatrixXd normal=Eigen::MatrixXd::Zero(allowed.size(),allowed.size());
        if(complement)for(int col=0;col<allowed.size();++col) {
            auto basis=Eigen::RowVectorXd::Zero(dimensions).eval();basis[allowed[col]]=1.;const auto p=symmetry->project(basis);
            for(int row=0;row<allowed.size();++row)normal(row,col)=p[allowed[row]];
        }
        if(translations)for(int a=0;a<allowed.size();++a)for(int b=0;b<allowed.size();++b)
            if((allowed[a]-d.vcoffset)%3==(allowed[b]-d.vcoffset)%3)normal(a,b)+=1./d.natom;
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eig((normal+normal.transpose())*.5);
        if(eig.info()!=Eigen::Success)throw std::runtime_error("Seed feasible-subspace decomposition failed");
        int count=0;while(count<allowed.size() && eig.eigenvalues()[count]<1e-10)++count;
        Eigen::VectorXd local(allowed.size());for(int i=0;i<allowed.size();++i)local[i]=direction[allowed[i]];
        local=(eig.eigenvectors().leftCols(count)*(eig.eigenvectors().leftCols(count).transpose()*local)).eval();
        auto projected=Eigen::RowVectorXd::Zero(dimensions).eval();for(int i=0;i<allowed.size();++i)projected[allowed[i]]=local[i];
        return projected;
    };
    Eigen::MatrixXd atomic=Eigen::MatrixXd::Zero(d.x.rows(),dimensions),cell=atomic;
    std::mt19937_64 rng(options.seed);
    const int modes=options.mode=="mixed3"?std::min(3,int(d.images.size())-2):1;
    for(int m=1;m<=modes;++m) {
        const int k=options.mode=="sine2"?2:m;
        if(k>=d.images.size()-1)continue;
        auto a=Eigen::RowVectorXd::Zero(dimensions).eval(),c=a;
        for(int j=0;j<dimensions;++j)(j<d.vcoffset?c:a)[j]=2.*double(rng()>>11)/9007199254740992.-1.;
        a=feasible(a,false);c=feasible(c,true);
        for(int i=1;i<d.images.size()-1;++i) {
            const double factor=options.sign*std::sin(k*std::numbers::pi*i/(d.images.size()-1))/(k*k);
            atomic.row(i)+=factor*a.cwiseQuotient(d.scalval.transpose());cell.row(i)+=factor*c.cwiseQuotient(d.scalval.transpose());
        }
    }
    auto displace=[&](const Eigen::MatrixXd &delta,double bound,bool is_cell) {
        if(bound==0 || delta.norm()<1e-12)return;
        const auto base=d.x;const auto before=d.images;double factor=bound/std::max(1.,delta.norm());
        for(int attempt=0;attempt<80;++attempt) {
            bool valid=true;double maximum=0;const auto proposal=(base+factor*delta).eval();
            try {
                d.X2LatPos(proposal);d.validate_images(false);
                for(int i=1;i<d.images.size()-1;++i) {
                    if(is_cell)maximum=std::max(maximum,(before[i].cell_.inverse()*d.images[i].cell_-Eigen::Matrix3d::Identity()).norm());
                    else for(int a=0;a<d.natom;++a)maximum=std::max(maximum,(d.images[i].pos_[a]-before[i].pos_[a]).norm());
                }
            } catch(const NEBInvalidGeometry &) {valid=false;}
            if(valid && maximum<=bound*(1+1e-10) && maximum>=.99*bound) {d.x=proposal;return;}
            d.images=before;
            if(!valid || !std::isfinite(maximum))factor*=.5;
            else if(maximum<1e-14)return;
            else factor*=std::min(2.,.999*bound/maximum);
        }
        throw std::runtime_error("Cannot construct a valid bounded path seed");
    };
    displace(cell,options.strain,true);const auto affine=d.images;displace(atomic,options.amplitude,false);
    for(int i=1;i<d.images.size()-1;++i) {
        if(d.vcneb)result.cell_deformation=std::max(result.cell_deformation,(original[i].cell_.inverse()*d.images[i].cell_-Eigen::Matrix3d::Identity()).norm());
        for(int a=0;a<d.natom;++a)result.atomic_displacement=std::max(result.atomic_displacement,(d.images[i].pos_[a]-affine[i].pos_[a]).norm());
    }
    if(result.atomic_displacement<1e-14 && result.cell_deformation<1e-14) {result.status="skipped";result.reason="No nonzero direction in the requested mode/support/constraint intersection";}
    result.images=d.images;return result;
}
