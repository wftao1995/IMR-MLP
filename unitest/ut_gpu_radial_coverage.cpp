#include <catch2/catch_all.hpp>
#include "basis/basis.h"
#include "runtime/mtp/gpu-direct/radial_gpu_math.h"
#include "potentials/potential_mtp_m.h"
#include "runtime/mtp/mlip.cpp"
#include <filesystem>
#include <fstream>
#include <unistd.h>
#include <array>
#include <cstdlib>
#include <sstream>

#ifdef IMRMLP_GPU
#include "potentials/mtp_gpu_adapter.h"
#include "potentials/potential_mtp_LES.h"
#include "potentials/potential_mtpnn1.h"
#endif

namespace {
using namespace MLIPCore::GpuDirect;
using namespace Potentials;
const std::array<const char *,12> cutoff_names{{"quadratic","cosine","poly5","normalized_quadratic",
    "cosine_left","cosine_unscaled","cosine_squared","sine_squared","poly3","poly4","smoothstep5","smoothstep7"}};
const std::array<const char *,5> transform_names{{"linear","exponential","exponential_left","i2","exponential_tanh"}};
std::string radial_block(const std::string &name, int cutoff=0, int transform=0,
                         bool flipped=false, double lambda=0.7, double scale=0.8)
{
    std::ostringstream out;
    out << "radial_basis_type = " << name << "\nmin_dist = 0.5\nmax_dist = 5\nradial_basis_size = 4\n";
    if (name=="RBChebyshevPoly4" || name=="RBChebyshevPoly5") out << "rb_b = -0.7\n";
    if (name=="RBChebyshevConfigurable") {
        const std::string cut=cutoff_names[cutoff],shift=transform_names[transform];
        out << "basis_function_type = " << (flipped?"one_minus_chebyshev":"chebyshev")
            << "\ncutoff_type = " << cut << '\n';
        if (cut=="poly3"||cut=="poly4"||cut=="poly5") out << "cutoff_b = -0.7\n";
        if (cut=="smoothstep7") out << "cutoff_r1 = 1.4\ncutoff_r2 = 3.7\n";
        out << "shift_type = " << shift << '\n';
        if (shift!="linear"&&shift!="i2") out << "shift_lambda = " << lambda << '\n';
        if (shift=="exponential_tanh") out << "shift_scale_lambda = " << scale << '\n';
    }
    return out.str();
}
std::string model_text(const std::vector<std::string> &radials, bool linear_head=false)
{
    const int species=radials.size()==4?2:1;
    std::ostringstream out;
    out << "MTP\nversion = 1.1.0\npotential_name = coverage\nscaling = 0.65\nspecies_count = " << species << "\npotential_tag =\n";
    if(species==2) {
        out << "radial_basis_split = 1\n";
        for(int i=0;i<2;++i)for(int j=0;j<2;++j)out << "pair " << i << ' ' << j << '\n' << radials[2*i+j];
    } else out << radials.front();
    out << "radial_funcs_count = 1\nradial_coeffs\n";
    for(int i=0;i<species;++i)for(int j=0;j<species;++j)
        out << "  " << i << '-' << j << "\n {0.7, -0.2, 0.12, -0.05}\n";
    out << "alpha_moments_count = 5\nalpha_index_basic_count = 4\n"
        << "alpha_index_basic = {{0,0,0,0}, {0,1,0,0}, {0,0,1,0}, {0,0,0,1}}\n"
        << "alpha_index_times_count = 3\nalpha_index_times = {{1,1,1,4}, {2,2,1,4}, {3,3,1,4}}\n"
        << "alpha_scalar_moments = 2\nalpha_moment_mapping = {0,4}\n";
    if(linear_head)out << "species_coeffs = {0.13, -0.04}\nmoment_coeffs = {0.2,-0.1}\nmodel_coeffs = {}\n";
    else out << (species==2?"species_coeffs = {0.3,-0.1}\n":"species_coeffs = {0.3}\n") << "moment_coeffs = {0.12,-0.04}\n";
    return out.str();
}
}

TEST_CASE("GPU radial formulas cover all native configurable components", "[radial][gpu-radial-coverage]")
{
    for(int cut=0;cut<12;++cut)for(int shift=0;shift<5;++shift)for(bool flip:{false,true}) {
        CAPTURE(cutoff_names[cut],transform_names[shift],flip);
        std::istringstream input(radial_block("RBChebyshevConfigurable",cut,shift,flip));
        std::string key,equal,name;input>>key>>equal>>name;
        RBasis_ChebyshevConfigurable basis(input,0.65);
        for(double r:{0.5,0.500001,1.4-1e-8,1.4,1.4+1e-8,2.1,3.7-1e-8,3.7,3.7+1e-8,4.999999,5.0}) {
            CAPTURE(r);
            double x,dx,fc,dfc;
            RadialMath::coordinate(static_cast<RadialTransform>(shift),r,.5,5.,.7,.8,x,dx);
            RadialMath::envelope(static_cast<RadialCutoff>(cut),r,.5,5.,-.7,1.4,3.7,fc,dfc);
            std::vector<double> values(4),derivatives(4);basis.calc_der(r,values,derivatives);
            double previous=1,current=x,dprevious=0,dcurrent=dx;
            for(int n=0;n<4;++n) {
                const double p=flip?1-current:previous,dp=flip?-dcurrent:dprevious;
                REQUIRE(.65*p*fc==Catch::Approx(values[n]).margin(2e-11).epsilon(2e-10));
                REQUIRE(.65*(dp*fc+p*dfc)==Catch::Approx(derivatives[n]).margin(2e-11).epsilon(2e-10));
                const double next=2*x*current-previous,dnext=2*(dx*current+x*dcurrent)-dprevious;
                previous=current;current=next;dprevious=dcurrent;dcurrent=dnext;
            }
        }
    }
}

TEST_CASE("Standalone GPU reader preserves all configurable radial metadata", "[radial][gpu-radial-coverage][reader]")
{
    const auto directory=std::filesystem::temp_directory_path()/
        ("gpml-gpu-radial-reader-"+std::to_string(getpid()));
    std::filesystem::create_directories(directory);
    struct Cleanup {std::filesystem::path p;~Cleanup(){std::error_code error;std::filesystem::remove_all(p,error);}} cleanup{directory};
    const auto path=directory/"model.mtp";
    std::array<int,MLIPCore::mlip_g_s> map{};
    for(int cut=0;cut<12;++cut)for(int shift=0;shift<5;++shift)for(bool flip:{false,true}) {
        CAPTURE(cutoff_names[cut],transform_names[shift],flip);
        std::istringstream input(model_text({radial_block("RBChebyshevConfigurable",cut,shift,flip)}));
        PotMTP native;native.Load(input);native.Save(path.string());
        std::string filename=path.string();double cutoff;
        std::unique_ptr<MLIPCore::mlip,decltype(&MLIPCore::destroy_mlip)> parsed(
            MLIPCore::setup(filename.data(),map.data(),&cutoff),MLIPCore::destroy_mlip);
        REQUIRE(parsed->pair_radial_basis_count==1);
        const auto &basis=parsed->pair_radial_bases[0];
        REQUIRE(basis.one_minus_chebyshev==flip);
        REQUIRE(static_cast<int>(basis.cutoff)==cut);
        REQUIRE(static_cast<int>(basis.transform)==shift);
        std::unique_ptr<MLIPCore::mlip_buffers,decltype(&MLIPCore::destroy_mlip_buffer)> buffer(
            MLIPCore::create_mlip_buffer(parsed.get()),MLIPCore::destroy_mlip_buffer);
        for(double r:{.5,.500001,1.4,2.1,3.7,4.999999,5.}) {
            MLIPCore::RB_CalcDerPair(parsed.get(),&basis,buffer.get(),r);
            native.p_radical_basis->calc_der(r);
            for(int n=0;n<native.rb_size;++n) {
                REQUIRE(buffer->rb_vals[n]==Catch::Approx(native.p_radical_basis->vals[n]).margin(2e-11).epsilon(2e-10));
                REQUIRE(buffer->rb_ders[n]==Catch::Approx(native.p_radical_basis->ders[n]).margin(2e-11).epsilon(2e-10));
            }
        }
    }
}

#ifdef IMRMLP_GPU
namespace {
bool gpu_available()
{
    int count=0;std::string error;
    const bool available=MTPGpuDirect::available_device_count(&count,&error)&&count>0;
    if(std::getenv("IMRMLP_REQUIRE_GPU")){INFO(error);REQUIRE(available);}
    return available;
}
Atoms geometry(int species=1,double distance=-1)
{
    Atoms a;a.cell_=20*Eigen::Matrix3d::Identity();
    if(distance>=0)a.pos_={{1,1,1},{1+distance,1,1}};
    else a.pos_={{1,1,1},{2.1,1.2,1},{1.3,2.6,1.4},{4.7,1.1,1.2},{1.22,1.04,1}};
    for(int i=0;i<int(a.pos_.size());++i)a.chemmap_.push_back(i%species);
    a.resetEFS();return a;
}
void equal_efs(const Atoms &a,const Atoms &b)
{
    REQUIRE(a.energy_==Catch::Approx(b.energy_).epsilon(2e-9).margin(2e-9));
    for(int i=0;i<a.size();++i) {
        REQUIRE(a.site_energies_[i]==Catch::Approx(b.site_energies_[i]).epsilon(2e-9).margin(2e-9));
        for(int d=0;d<3;++d) REQUIRE(a.forces_[i][d]==Catch::Approx(b.forces_[i][d]).epsilon(5e-9).margin(5e-9));
    }
    for(int i=0;i<3;++i)for(int j=0;j<3;++j)
        REQUIRE(a.stress_(i,j)==Catch::Approx(b.stress_(i,j)).epsilon(5e-9).margin(5e-9));
}
template<class Potential> void verify(Potential &p,bool gradients=true)
{
    auto a=geometry(p.species_count),cpu=a,gpu=a;Neighborhoods n(a,p.cutoff);
    p.getEFS(cpu,n);MTPGpuEvaluator evaluator;MTPGpuBatch batch;std::string error;
    INFO(error);REQUIRE(evaluator.initialize(p,0,&error));REQUIRE(batch.build(a,n,&error));
    REQUIRE(batch.evaluate(evaluator,gpu,false,&error));equal_efs(gpu,cpu);
    auto energy=a;REQUIRE(batch.evaluate_energy(evaluator,energy,true,&error));
    REQUIRE(energy.energy_==Catch::Approx(cpu.energy_).epsilon(2e-9).margin(2e-9));
    // Energy-only clears the force forward workspace; refresh before requesting EFS gradients.
    REQUIRE(batch.evaluate(evaluator,gpu,true,&error));
    if(!gradients)return;
    p.train_init();
    Eigen::Matrix<double,-1,3> fw(a.size(),3);fw.setConstant(.03);
    fw(0,0)=.13;fw(1,2)=-.07;
    Eigen::Matrix3d sw;sw<<.07,.02,-.01,.04,-.03,.08,-.05,.06,.02;
    Eigen::Matrix<double,-1,3,Eigen::RowMajor> unused;
    std::vector<double> reference(p.get_NL_coeffs_count(),0),actual(reference.size(),0),f(3*a.size()),stress(9);
    if constexpr (std::is_same_v<Potential,PotMTP>)
        p.AccumulateEFSCombinationGard(a,n,.61,fw,sw,reference.data());
    else p.AccumulateEFSCombinationGard(a,n,.61,fw,sw,unused,reference.data());
    for(int i=0;i<a.size();++i)for(int d=0;d<3;++d)f[3*i+d]=fw(i,d);
    for(int i=0;i<3;++i)for(int j=0;j<3;++j)stress[3*i+j]=sw(i,j);
    REQUIRE(batch.accumulate_gradient(evaluator,{.61},f,stress,actual.data(),actual.size(),false,&error));
    for(size_t i=0;i<actual.size();++i){CAPTURE(i);REQUIRE(actual[i]==Catch::Approx(reference[i]).epsilon(3e-8).margin(1e-7));}
}
}

TEST_CASE("GPU native adapter covers 120 radial combinations and six legacy bases", "[gpu][gpu-radial-coverage]")
{
    if(!gpu_available())SKIP("No GPU device");
    for(int cut=0;cut<12;++cut)for(int shift=0;shift<5;++shift)for(bool flip:{false,true}) {
        CAPTURE(cutoff_names[cut],transform_names[shift],flip);
        std::istringstream input(model_text({radial_block("RBChebyshevConfigurable",cut,shift,flip)}));
        PotMTP p;p.Load(input);verify(p);
    }
    for(const auto *name:{"RBChebyshev","RBChebyCos","RBChebyshevPoly4","RBChebyshevPoly5","RBPolyB","RBPolyBCos"}) {
        CAPTURE(name);std::istringstream input(model_text({radial_block(name)}));
        PotMTP p;p.Load(input);verify(p);
    }
}

TEST_CASE("GPU complete radial descriptors preserve boundaries split pairs and zero transform limits", "[gpu][gpu-radial-coverage]")
{
    if(!gpu_available())SKIP("No GPU device");
    std::istringstream split(model_text({radial_block("RBPolyB"),radial_block("RBChebyshevConfigurable",11,4,true),
        radial_block("RBPolyBCos"),radial_block("RBChebyshevPoly4")}));
    PotMTP split_p;split_p.Load(split);verify(split_p);
    for(double lambda:{0.,-.7})for(double scale:{0.,.8})for(int shift:{1,2,4}) {
        std::istringstream input(model_text({radial_block("RBChebyshevConfigurable",11,shift,true,lambda,scale)}));
        PotMTP p;p.Load(input);verify(p);
        MTPGpuEvaluator evaluator;std::string error;INFO(error);REQUIRE(evaluator.initialize(p,0,&error));
        for(double r:{.5-1e-8,.5,.5+1e-8,1.4-1e-8,1.4,1.4+1e-8,3.7-1e-8,3.7,3.7+1e-8,5.-1e-8,5.,5.+1e-8}) {
            CAPTURE(lambda,scale,shift,r);
            auto a=geometry(1,r),cpu=a,gpu=a;Neighborhoods n(a,p.cutoff);p.getEFS(cpu,n);
            MTPGpuBatch batch;REQUIRE(batch.build(a,n,&error));REQUIRE(batch.evaluate(evaluator,gpu,false,&error));equal_efs(gpu,cpu);
        }
    }
}

TEST_CASE("GPU linear MTP-M preserves descriptor layout coefficient updates and frozen groups", "[gpu][mtp-m][gpu-radial-coverage]")
{
    if(!gpu_available())SKIP("No GPU device");
    std::istringstream input(model_text({radial_block("RBChebyshevConfigurable",8,4,true)},true));
    PotMTP_M<m_model_mtpli> p;p.Load(input);verify(p);
    p.req_radical=false;p.req_species=false;verify(p);
    MTPGpuEvaluator evaluator;std::string error;INFO(error);REQUIRE(evaluator.initialize(p,0,&error));
    p.moment_coeffs[0]*=1.7;p.species_coeffs[1]-=.1;REQUIRE(evaluator.update_coefficients(p,&error));
    auto a=geometry(),cpu=a,gpu=a;Neighborhoods n(a,p.cutoff);p.getEFS(cpu,n);
    MTPGpuBatch batch;REQUIRE(batch.build(a,n,&error));REQUIRE(batch.evaluate(evaluator,gpu,false,&error));equal_efs(gpu,cpu);
    REQUIRE_FALSE(evaluator.initialize(p,0,MTPGpuPrecision::Mixed,&error));
}

TEST_CASE("GPU radial regularization mode is inference only", "[gpu][gpu-radial-coverage]")
{
    if(!gpu_available())SKIP("No GPU device");
    std::istringstream input(model_text({radial_block("RBPolyBCos")}));
    PotMTP p;p.Load(input);p.radial_mode=1;verify(p,false);
    MTPGpuEvaluator evaluator;std::string error;REQUIRE(evaluator.initialize(p,0,&error));
    MTPGpuGradientRequest request;
    REQUIRE_FALSE(evaluator.accumulate_gradient(request,&error));
    REQUIRE(error.find("radial_mode=0")!=std::string::npos);
}

namespace {
template<class Potential> void verify_weighted(Potential &p)
{
    auto a=geometry();Neighborhoods n(a,p.cutoff);MTPGpuBatch batch;
    MTPGpuEvaluator evaluator;std::string error;INFO(error);
    if constexpr (std::is_same_v<Potential,PotMTPNN1>)
        REQUIRE(evaluator.initialize_legacy_nn(p,0,&error));
    else REQUIRE(evaluator.initialize(p,0,&error));
    REQUIRE(batch.build(a,n,&error));
    const auto full=batch.request();
    // Nontrivial center order and one ghost verify atom-indexed weight lookup.
    std::vector<int> centers{3,0,2},offsets{0},neighbors;
    std::vector<double> vectors;
    for(int center:centers) {
        for(int k=full.offsets[center];k<full.offsets[center+1];++k) {
            neighbors.push_back(full.neighbors[k]);
            for(int d=0;d<3;++d)vectors.push_back(full.neighbor_vectors[3*k+d]);
        }
        offsets.push_back(neighbors.size());
    }
    std::vector<double> weights{.7,0.,-.3,1.4,0.},force(15),site(4),virial(30);
    double energy=0.;
    auto req=full;req.center_count=centers.size();req.nlocal=4;
    req.centers=centers.data();req.offsets=offsets.data();req.neighbors=neighbors.data();
    req.neighbor_vectors=vectors.data();req.center_energy_weights=weights.data();
    req.total_energy=&energy;req.force=force.data();req.site_energy=site.data();req.site_virial=virial.data();
    req.topology_token={};req.input_token={};
    const auto reference_energy=[&](Atoms b) {
        Neighborhoods nb(b,p.cutoff);p.getEFS(b,nb);double sum=0.;
        for(int atom:centers)sum+=weights[atom]*b.site_energies_[atom];return sum;
    };
    for(bool ordered:{false,true})for(auto kernel:{MTPGpuForceKernel::BasicTile,MTPGpuForceKernel::CenterNeighbors}) {
        CAPTURE(ordered,static_cast<int>(kernel));
        req.compute_forces=true;req.site_virial=virial.data();req.deterministic_reduction=ordered;req.force_kernel=kernel;
        if (ordered && kernel==MTPGpuForceKernel::CenterNeighbors) {
            REQUIRE_FALSE(evaluator.context().compute_batch(req,&error));
            REQUIRE(error.find("unavailable with deterministic")!=std::string::npos);
            continue;
        }
        REQUIRE(evaluator.context().compute_batch(req,&error));
        REQUIRE(energy==Catch::Approx(reference_energy(a)).margin(2e-9).epsilon(2e-9));
        auto native=a;p.getEFS(native,n);
        for(int atom:centers)REQUIRE(site[atom]==Catch::Approx(weights[atom]*native.site_energies_[atom]).margin(2e-9).epsilon(2e-9));
        const double h=1e-6;
        for(int atom=0;atom<a.size();++atom)for(int d=0;d<3;++d) {
            auto plus=a,minus=a;plus.pos_[atom][d]+=h;minus.pos_[atom][d]-=h;
            REQUIRE(force[3*atom+d]==Catch::Approx(-(reference_energy(plus)-reference_energy(minus))/(2*h)).margin(2e-7).epsilon(2e-6));
        }
        for(int d=0;d<3;++d) {
            auto plus=a,minus=a;for(int atom=0;atom<a.size();++atom) {plus.pos_[atom][d]*=1+h;minus.pos_[atom][d]*=1-h;}
            double total_virial=0.;for(int atom=0;atom<a.size();++atom)total_virial+=virial[6*atom+d];
            REQUIRE(total_virial==Catch::Approx(-(reference_energy(plus)-reference_energy(minus))/(2*h)).margin(2e-7).epsilon(2e-6));
        }
        MTPGpuGradientRequest gradient;REQUIRE_FALSE(evaluator.context().accumulate_gradient(gradient,&error));
        REQUIRE(error.find("inference-only")!=std::string::npos);
        MTPGpuLossRequest loss;REQUIRE_FALSE(evaluator.context().prepare_loss(loss,&error));
        REQUIRE(error.find("inference-only")!=std::string::npos);
        req.compute_forces=false;req.site_virial=nullptr;
        REQUIRE(evaluator.context().compute_batch(req,&error));
        REQUIRE(energy==Catch::Approx(reference_energy(a)).margin(2e-9).epsilon(2e-9));
    }
    // The weight upload is independent of topology and coordinate cache reuse.
    weights[0]=-.8;req.reuse_topology=true;
    req.force_kernel=MTPGpuForceKernel::BasicTile;req.compute_forces=false;req.site_virial=nullptr;
    REQUIRE(evaluator.context().compute_batch(req,&error));
    REQUIRE(energy==Catch::Approx(reference_energy(a)).margin(2e-9).epsilon(2e-9));
    req.center_energy_weights=nullptr;
    REQUIRE(evaluator.context().compute_batch(req,&error));
    auto native=a;p.getEFS(native,n);double sum=0.;for(int atom:centers)sum+=native.site_energies_[atom];
    REQUIRE(energy==Catch::Approx(sum).margin(2e-9).epsilon(2e-9));
}
}

TEST_CASE("GPU center energy weights seed local model adjoints", "[gpu][gpu-radial-coverage][weighted]")
{
    if(!gpu_available())SKIP("No GPU device");
    SECTION("linear MTP") {
        std::istringstream input(model_text({radial_block("RBPolyB")}));PotMTP p;p.Load(input);verify_weighted(p);
    }
    SECTION("linear MTP-M") {
        std::istringstream input(model_text({radial_block("RBChebyshevConfigurable",8,4,true)},true));
        PotMTP_M<m_model_mtpli> p;p.Load(input);verify_weighted(p);
    }
    SECTION("neural MTP-M") {
        GPSettings settings;settings.Modify("n_input","1");settings.Modify("n_neuron","2");settings.Modify("activation_function","TANH");
        PotMTP_M<m_model_nn1> p;p.model.init(settings);
        auto text=model_text({radial_block("RBChebyshevPoly4")},true);
        text.replace(text.find("model_coeffs = {}"),std::string("model_coeffs = {}").size(),"model_coeffs = {0.3,-0.2,0.1,-0.4,0.5,0.6}");
        std::istringstream input(text);p.Load(input);verify_weighted(p);
    }
}

TEST_CASE("GPU short range primitive omits explicitly delegated LES", "[gpu][gpu-radial-coverage][short-range]")
{
    if(!gpu_available())SKIP("No GPU device");
    PotMTP_LES p("tests/case-mtp-les/les.mtp");p.train_init();
    REQUIRE(p.lr_les);REQUIRE_FALSE(p.uses_sog());
    MTPGpuEvaluator evaluator;std::string error;
    REQUIRE_FALSE(evaluator.initialize(p,0,&error));
    REQUIRE(evaluator.initialize_short_range(p,0,&error));REQUIRE(evaluator.sog_definition()==nullptr);
    auto a=geometry(2),cpu=a,gpu=a;Neighborhoods n(a,p.cutoff);
    PotMTP reference("tests/case-mtp-les/les.mtp");reference.getEFS(cpu,n);
    MTPGpuBatch batch;REQUIRE(batch.build(a,n,&error));
    REQUIRE(batch.evaluate(evaluator,gpu,false,&error));equal_efs(gpu,cpu);
    REQUIRE(p.lr_les);REQUIRE(evaluator.update_coefficients(p,&error));
    MTPGpuGradientRequest gradient;REQUIRE_FALSE(evaluator.accumulate_gradient(gradient,&error));
    REQUIRE(error.find("inference-only")!=std::string::npos);
}

TEST_CASE("GPU linear head override retains radial graph and caller ownership", "[gpu][gpu-radial-coverage][linear-head]")
{
    if(!gpu_available())SKIP("No GPU device");
    const auto text=model_text({radial_block("RBChebyshevConfigurable",11,4,true)});
    std::istringstream input(text),reference_input(text);PotMTP p,reference;
    p.Load(input);reference.Load(reference_input);
    std::vector<double> species{-.17},moments{.4,-.3};
    reference.species_coeffs[0]=species[0];std::copy(moments.begin(),moments.end(),reference.moment_coeffs);
    MTPGpuEvaluator evaluator;std::string error;
    REQUIRE(evaluator.initialize_linear_head(p,species.data(),moments.data(),0,&error));
    // Initialization copies the caller's coefficients into the device model.
    std::fill(species.begin(),species.end(),99.);std::fill(moments.begin(),moments.end(),99.);
    auto a=geometry(),cpu=a,gpu=a;Neighborhoods n(a,p.cutoff);reference.getEFS(cpu,n);
    MTPGpuBatch batch;REQUIRE(batch.build(a,n,&error));REQUIRE(batch.evaluate(evaluator,gpu,false,&error));equal_efs(gpu,cpu);
    REQUIRE(p.species_coeffs[0]==Catch::Approx(.3));REQUIRE_FALSE(evaluator.update_coefficients(p,&error));
    REQUIRE(error.find("reinitialization")!=std::string::npos);
}

TEST_CASE("GPU legacy neural heads preserve normalization species bias and weighted adjoints", "[gpu][gpu-radial-coverage][legacy-nn]")
{
    if(!gpu_available())SKIP("No GPU device");
    const auto directory=std::filesystem::temp_directory_path()/
        ("gpml-gpu-legacy-nn-"+std::to_string(getpid()));
    std::filesystem::create_directories(directory);
    struct Cleanup {std::filesystem::path p;~Cleanup(){std::error_code error;std::filesystem::remove_all(p,error);}} cleanup{directory};
    std::istringstream input(model_text({radial_block("RBChebyshevConfigurable",11,4,true)}));
    PotMTP base;base.Load(input);base.Save((directory/"model.mtp").string());
    for(int layers:{1,2})for(auto activation:{IMRMLPNN::TANH,IMRMLPNN::SIGMOID,IMRMLPNN::RELU}) {
        CAPTURE(layers,static_cast<int>(activation));
        GPSettings settings;settings.Modify("mtpcoeff",(directory/"model.mtp").string());
        settings.Modify("nncoeff",(directory/"absent.nn").string());settings.Modify("spcoeff",(directory/"absent.sc").string());
        settings.Modify("nnlay",std::to_string(layers));settings.Modify("nnhidden","4");settings.Modify("nnhidden2","3");
        PotMTPNN1 p(settings);std::vector<double> coefficients(p.nn_coeff_size);
        for(int i=0;i<coefficients.size();++i)coefficients[i]=.31*std::sin(.7*(i+1));
        const auto configure=[&](auto &network) {
            network.set_coeffs(coefficients.data());network.acfunc=activation;
            for(int i=0;i<network.n_input;++i){network.norm_mean[i]=.17*(i+1);network.norm_irstd[i]=.8+.2*i;}
        };
        if(layers==1)configure(*p.p_nn);else configure(*p.p_nn2);
        p.species_coeffs={-.31};
        auto a=geometry(),cpu=a,gpu=a;Neighborhoods n(a,p.cutoff);p.getEFS(cpu,n);
        MTPGpuEvaluator evaluator;std::string error;REQUIRE(evaluator.initialize_legacy_nn(p,0,&error));
        MTPGpuBatch batch;REQUIRE(batch.build(a,n,&error));REQUIRE(batch.evaluate(evaluator,gpu,false,&error));equal_efs(gpu,cpu);
        verify_weighted(p);
    }
}
#endif
