// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#include <catch2/catch_all.hpp>
#include "runtime/mtp/gpu-direct/gpu_backend.h"

#ifdef IMRMLP_GPU
#include "potentials/mtp_gpu_adapter.h"
#include "potentials/potential_mtp_LES.h"
#include "potentials/potential_mtp_m.h"
#include "train/train_nonlinear_potential.h"
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <sstream>

using namespace Potentials;
using Catch::Approx;
namespace {
bool available() {
    int count=0;std::string error;
    const bool ok=MLIPCore::GpuDirect::MTPGpuDirect::available_device_count(&count,&error)&&count>0;
    if(std::getenv("IMRMLP_REQUIRE_GPU")){INFO(error);REQUIRE(ok);}
    return ok;
}
Atoms geometry() {
    Atoms a;a.cell_<<10.,0,0,.8,10.5,0,-.3,.6,11.;
    a.pos_={{2.,2.,2.},{4.1,2.2,2.},{2.3,4.2,2.4},{5.,4.8,3.1}};
    a.chemmap_={0,1,0,1};a.fitness=1.;a.id=0;return a;
}
void initialize(PotMTP_LES &p,int channels) {
    p.les_kernel="sog";p.sog_terms=2;p.sog_channels=channels;p.train_init();
    for(int c=0;c<channels;++c){
        p.les_species_coeffs[2*c]=.23+.03*c;p.les_species_coeffs[2*c+1]=-.17;
        auto *b=p.les_moment_coeffs+c*p.alpha_scalar_moments;
        std::fill_n(b,p.alpha_scalar_moments,0.);b[0]=.35;b[1]=-.21;b[3]=.12/(c+1);
        p.model_les.sog_weights[2*c]=4.2/(c+1);p.model_les.sog_weights[2*c+1]=-1.3;
        p.model_les.sog_log_bandwidths[2*c]=std::log(.7);p.model_les.sog_log_bandwidths[2*c+1]=std::log(1.2);
    }
    p.train_init();p.set_inited();
}
void equal(double actual,double expected) { REQUIRE(std::abs(actual-expected)<=1e-8+2e-9*std::max(std::abs(actual),std::abs(expected))); }
void equal_efs(const Atoms &actual,const Atoms &expected) {
    equal(actual.energy_,expected.energy_);
    for(int i=0;i<actual.size();++i)for(int d=0;d<3;++d)equal(actual.forces_[i][d],expected.forces_[i][d]);
    for(int i=0;i<3;++i)for(int j=0;j<3;++j)equal(actual.stress_(i,j),expected.stress_(i,j));
}
}

TEST_CASE("SOG GPU full EFS preserves channels zero self and energy-only", "[sog][gpu]") {
    if(!available())SKIP("No GPU device");
    const int channels=GENERATE(1,3,16);
    const bool zero=GENERATE(false,true),self=GENERATE(false,true),ordered=GENERATE(false,true);
    CAPTURE(channels,zero,self,ordered);
    PotMTP_LES p("tests/case-mtp-les/les.mtp");initialize(p,channels);
    p.model_les.sog_include_zero=zero;p.remove_self_interaction=self;p.train_init();
    auto cpu=geometry(),gpu=cpu;p.getEFS(cpu);Neighborhoods nbhs(gpu,p.cutoff);
    MTPGpuEvaluator evaluator;MTPGpuBatch batch;std::string error;
    REQUIRE(evaluator.initialize(p,0,&error));INFO(error);
    batch.configure_gradient_reduction(ordered,128);REQUIRE(batch.build(gpu,nbhs,&error));
    REQUIRE(batch.evaluate(evaluator,gpu,true,&error));equal_efs(gpu,cpu);
    auto energy=geometry();REQUIRE(batch.evaluate_energy(evaluator,energy,true,&error));equal(energy.energy_,cpu.energy_);
    auto repeat=geometry();REQUIRE(batch.evaluate(evaluator,repeat,true,&error));equal_efs(repeat,cpu);
    if(ordered){REQUIRE(repeat.energy_==gpu.energy_);for(int i=0;i<gpu.size();++i)REQUIRE(repeat.forces_[i]==gpu.forces_[i]);}
    // Refresh the same topology with new reciprocal geometry and coordinates.
    auto moved=geometry();moved.cell_(2,1)+=.4;moved.pos_[0][0]+=.013;
    Neighborhoods next(moved,p.cutoff);REQUIRE(batch.refresh_compatible(moved,next,&error));
    auto ref=moved;p.getEFS(ref);REQUIRE(batch.evaluate(evaluator,moved,true,&error));equal_efs(moved,ref);
    p.model_les.sog_weights[0]*=1.3;p.les_moment_coeffs[0]+=.05;
    REQUIRE(evaluator.update_coefficients(p,&error));ref=moved;p.getEFS(ref);
    REQUIRE(batch.evaluate(evaluator,moved,true,&error));equal_efs(moved,ref);
}

TEST_CASE("SOG GPU complete parameter VJP matches CPU and finite differences", "[sog][gpu][grad]") {
    if(!available())SKIP("No GPU device");
    const int channels=GENERATE(1,3);const bool ordered=GENERATE(false,true);
    PotMTP_LES p("tests/case-mtp-les/les.mtp");initialize(p,channels);
    auto a=geometry();Neighborhoods nbhs(a,p.cutoff);std::string error;
    MTPGpuEvaluator evaluator;MTPGpuBatch batch;REQUIRE(evaluator.initialize(p,0,&error));INFO(error);
    batch.configure_gradient_reduction(ordered,128);REQUIRE(batch.build(a,nbhs,&error));REQUIRE(batch.evaluate(evaluator,a,true,&error));
    std::vector<double> x(p.get_NL_coeffs_count());p.nl_get_coeffs(x.data());
    for(int mode=0;mode<4;++mode){
        const double ew=(mode==0||mode==3)?.37:0.;
        Eigen::Matrix<double,-1,3> fw(4,3);fw<<.2,-.1,.3,-.5,.7,.1,.3,.2,-.4,.1,.2,.4;
        Eigen::Matrix3d sw;sw<<.1,.2,-.1,.3,-.2,.5,-.1,.2,.7;
        if(mode!=1&&mode!=3)fw.setZero();if(mode!=2&&mode!=3)sw.setZero();
        std::vector<double> force(12),stress(9),cpu(x.size(),0.),gpu(x.size(),0.);
        for(int i=0;i<4;++i)for(int j=0;j<3;++j)force[3*i+j]=fw(i,j);
        for(int i=0;i<3;++i)for(int j=0;j<3;++j)stress[3*i+j]=sw(i,j);
        Eigen::Matrix<double,-1,3,Eigen::RowMajor> unused;
        p.AccumulateEFSCombinationGard(a,nbhs,ew,fw,sw,unused,cpu.data());
        REQUIRE(batch.accumulate_gradient(evaluator,{ew},force,stress,gpu.data(),gpu.size(),false,&error));
        for(size_t i=0;i<x.size();++i){CAPTURE(mode,channels,ordered,i,cpu[i],gpu[i]);equal(gpu[i],cpu[i]);}
        // Exercise a new direction without rebuilding the retained forward.
        auto again=gpu;REQUIRE(batch.accumulate_gradient(evaluator,{ew},force,stress,again.data(),again.size(),false,&error));
        if(ordered)REQUIRE(std::memcmp(gpu.data(),again.data(),gpu.size()*sizeof(double))==0);
        if(mode==3){
            const auto loss=[&](){auto b=geometry();REQUIRE(evaluator.update_coefficients(p,&error));REQUIRE(batch.evaluate(evaluator,b,true,&error));double l=ew*b.energy_+(sw.array()*b.stress_.array()).sum();for(int i=0;i<4;++i)l+=fw.row(i).dot(b.forces_[i]);return l;};
            for(const auto &group:p.nl_parameter_groups()){
                const int i=group.begin;auto plus=x,minus=x;const double h=1e-5;
                plus[i]+=h;minus[i]-=h;p.nl_set_coeffs(plus.data());const double lp=loss();p.nl_set_coeffs(minus.data());const double lm=loss();
                CAPTURE(group.name,i);REQUIRE(gpu[i]==Approx((lp-lm)/(2*h)).margin(2e-6).epsilon(2e-5));
            }
            p.nl_set_coeffs(x.data());
        }
    }
}

TEST_CASE("SOG GPU device loss chunks and frozen kernel fitting", "[sog][gpu][training]") {
    if(!available())SKIP("No GPU device");
    PotMTP_LES p("tests/case-mtp-les/les.mtp");initialize(p,3);
    std::vector<Atoms> data;
    for(int i=0;i<3;++i){auto a=geometry();a.pos_[1][0]+=.15*i;a.id=i;p.getEFS(a);data.push_back(a);}
    p.model_les.sog_weights[0]+=1.;p.train_init();
    std::vector<Neighborhoods> nbhs;for(auto &a:data)nbhs.emplace_back(a,p.cutoff);
    GPSettings settings;settings.Modify("energy_weight","1");settings.Modify("force_weight","1");settings.Modify("stress_weight","0.1");
    settings.Modify("charge_weight","0");settings.Modify("penalty_weight","0");settings.Modify("parameter_scaling","none");
    settings.Modify("bfgs_grad_tol","1e-14");settings.Modify("bfgs_step_tol","0");settings.Modify("bfgs_conv_tol","0");
    TrainNonlinear cpu(settings);cpu.init(data,nbhs,&p);double lc=0.;Eigen::VectorXd gc(p.get_NL_coeffs_count());cpu.AccumulateGard(lc,gc);
    settings.Modify("device",MLIPCore::GpuDirect::MTPGpuDirect::backend_name());settings.Modify("gpu_id","0");settings.Modify("gpu_batch_structures","1");
    TrainNonlinear gpu(settings);gpu.init(data,nbhs,&p);double lg=0.;Eigen::VectorXd gg(gc.size());gpu.AccumulateGard(lg,gg);
    equal(lg,lc);for(int i=0;i<gc.size();++i){CAPTURE(i);equal(gg[i],gc[i]);}
    gpu.ConfigureTrainableGroups({"sog_weights"});std::vector<double> before(gc.size()),after(gc.size());p.nl_get_coeffs(before.data());
    gpu.TrainNonLinearBFGS(10);double fitted=0.;gpu.AccumulateGard(fitted,gg);p.nl_get_coeffs(after.data());REQUIRE(fitted<lg);
    for(const auto &group:p.nl_parameter_groups())if(group.name!="sog_weights")for(int i=group.begin;i<group.end;++i)REQUIRE(before[i]==Approx(after[i]).margin(1e-12));
}

TEMPLATE_TEST_CASE("GPU resident inputs survive updates and reject stale batch state",
                   "[gpu][cache]", PotMTP, PotMTP_LES) {
    if (!available()) SKIP("No GPU device");
    const bool ordered = GENERATE(false, true);
    TestType p("tests/case-mtp-les/les.mtp");
    if constexpr (std::is_same_v<TestType, PotMTP_LES>) initialize(p, 3);
    else { p.train_init(); p.set_inited(); }
    auto a = geometry(), reference = a;
    Neighborhoods nbhs(a, p.cutoff);
    MTPGpuEvaluator evaluator, other;
    MTPGpuBatch batch, second;
    std::string error;
    INFO(error);
    REQUIRE(evaluator.initialize(p, 0, &error));
    REQUIRE(other.initialize(p, 0, &error));
    batch.configure_gradient_reduction(ordered, 128);
    REQUIRE(batch.build(a, nbhs, &error));
    REQUIRE(batch.evaluate(evaluator, a, true, &error));
    p.getEFS(reference); equal_efs(a, reference);
    const auto first = evaluator.context().runtime_stats();
    REQUIRE(batch.evaluate(evaluator, a, true, &error));
    auto stats = evaluator.context().runtime_stats();
    REQUIRE(stats.coordinate_upload_bytes == first.coordinate_upload_bytes);
    REQUIRE(stats.type_upload_bytes == first.type_upload_bytes);
    REQUIRE(stats.input_reuses == first.input_reuses + 1);
    REQUIRE(stats.sog_grid_upload_bytes == first.sog_grid_upload_bytes);
    if constexpr (std::is_same_v<TestType, PotMTP_LES>) {
        REQUIRE(first.sog_grid_upload_bytes > 0);
        REQUIRE(stats.sog_grid_reuses == first.sog_grid_reuses + 1);
    }

    std::vector<double> target_e{reference.energy_ + .1}, ew{.7};
    std::vector<double> target_f(12, .03), fw(4, .2), target_s(9, .02), sw{.1};
    std::array<double, 7> metrics{}, baseline{};
    std::vector<double> gradient(p.get_NL_coeffs_count()), baseline_gradient(gradient.size());
    const auto loss = [&](MTPGpuBatch &b, auto &values, auto &g) {
        REQUIRE(b.evaluate_loss(evaluator, target_e, ew, target_f, fw, target_s, sw,
                                values, true, true, &error));
        REQUIRE(b.accumulate_prepared_gradient(evaluator, g.data(), g.size(), true, true, &error));
    };
    loss(batch, baseline, baseline_gradient);
    stats = evaluator.context().runtime_stats();
    loss(batch, metrics, gradient);
    REQUIRE(evaluator.context().runtime_stats().label_upload_bytes == stats.label_upload_bytes);
    for (size_t i=0; i<metrics.size(); ++i) equal(metrics[i], baseline[i]);
    for (size_t i=0; i<gradient.size(); ++i) equal(gradient[i], baseline_gradient[i]);
    if (ordered) REQUIRE(gradient == baseline_gradient);

    // Changing each label/weight family must update the loss, even in-place.
    for (auto *values : {&target_e, &ew, &target_f, &fw, &target_s, &sw}) {
        (*values)[0] += .017;
        const auto before = evaluator.context().runtime_stats().label_upload_bytes;
        loss(batch, metrics, gradient);
        REQUIRE(evaluator.context().runtime_stats().label_upload_bytes > before);
        auto cold = batch; // copy owns fresh cache identities
        loss(cold, baseline, baseline_gradient);
        for (size_t i=0; i<metrics.size(); ++i) equal(metrics[i], baseline[i]);
        for (size_t i=0; i<gradient.size(); ++i) equal(gradient[i], baseline_gradient[i]);
    }

    // Different neighbor ordering, same counts: counters coincide across evaluators.
    auto reordered = nbhs;
    for (int i=0; i<reordered.size(); ++i) {
        std::reverse(reordered[i].indices_.begin(), reordered[i].indices_.end());
        std::reverse(reordered[i].pos_.begin(), reordered[i].pos_.end());
    }
    REQUIRE(second.build(reference, reordered, &error));
    REQUIRE(second.evaluate(other, a, true, &error));
    REQUIRE(batch.evaluate(other, a, true, &error));
    equal_efs(a, reference);
    REQUIRE(other.context().runtime_stats().topology_uploads == 2);
    int devices = 0;
    REQUIRE(MLIPCore::GpuDirect::MTPGpuDirect::available_device_count(&devices, &error));
    if (devices > 1) {
        REQUIRE(other.initialize(p, 1, &error));
        REQUIRE(second.evaluate(other, a, true, &error));
        REQUIRE(batch.evaluate(evaluator, a, true, &error));
        equal_efs(a, reference);
    }

    // A->B->A, including a copied batch that later changes geometry.
    auto copy = batch;
    auto moved = geometry(); moved.pos_[1][0] += .021;
    Neighborhoods moved_nbhs(moved, p.cutoff);
    REQUIRE(copy.refresh_compatible(moved, moved_nbhs, &error));
    REQUIRE(copy.evaluate(evaluator, moved, true, &error));
    REQUIRE_FALSE(batch.accumulate_prepared_gradient(evaluator, gradient.data(), gradient.size(), true, true, &error));
    REQUIRE(batch.evaluate(evaluator, a, true, &error)); equal_efs(a, reference);
    const auto before_update = evaluator.context().runtime_stats();
    p.radial_coeffs[0] += .003;
    if constexpr (std::is_same_v<TestType, PotMTP_LES>) p.model_les.sog_weights[0] *= 1.01;
    REQUIRE(evaluator.update_coefficients(p, &error));
    REQUIRE_FALSE(batch.accumulate_prepared_gradient(evaluator, gradient.data(), gradient.size(), true, true, &error));
    REQUIRE(batch.evaluate(evaluator, a, true, &error)); p.getEFS(reference); equal_efs(a, reference);
    REQUIRE(evaluator.context().runtime_stats().coordinate_upload_bytes == before_update.coordinate_upload_bytes);
    REQUIRE(evaluator.context().runtime_stats().sog_grid_upload_bytes == before_update.sog_grid_upload_bytes);

    const auto context = evaluator.context().context_identity();
    REQUIRE(evaluator.initialize(p, 0, &error));
    REQUIRE(evaluator.context().context_identity() != context);
    REQUIRE(batch.evaluate(evaluator, a, true, &error)); equal_efs(a, reference);

    loss(batch, metrics, gradient);
    auto invalid = batch.request(); invalid.offsets = nullptr;
    REQUIRE_FALSE(evaluator.context().compute_batch(invalid, &error));
    REQUIRE_FALSE(batch.accumulate_prepared_gradient(evaluator, gradient.data(), gradient.size(), true, true, &error));
    loss(batch, metrics, gradient); // valid retry after failure
    MLIPCore::GpuDirect::MTPGpuLossRequest invalid_loss;
    REQUIRE_FALSE(evaluator.context().prepare_loss(invalid_loss, &error));
    REQUIRE_FALSE(batch.accumulate_prepared_gradient(evaluator, gradient.data(), gradient.size(), true, true, &error));
    loss(batch, metrics, gradient);
    REQUIRE_FALSE(batch.accumulate_gradient(evaluator, {}, target_f, target_s,
                                            gradient.data(), gradient.size(), false, &error));
    REQUIRE_FALSE(batch.accumulate_prepared_gradient(evaluator, gradient.data(), gradient.size(), true, true, &error));
    loss(batch, metrics, gradient);
    REQUIRE_FALSE(batch.accumulate_prepared_gradient(evaluator, nullptr, gradient.size(), true, true, &error));
    REQUIRE_FALSE(batch.accumulate_prepared_gradient(evaluator, gradient.data(), gradient.size(), true, true, &error));
    loss(batch, metrics, gradient);
    REQUIRE_FALSE(batch.build(std::vector<Atoms>{a}, std::vector<Neighborhoods>{}, &error));
    REQUIRE_FALSE(batch.accumulate_prepared_gradient(evaluator, gradient.data(), gradient.size(), true, true, &error));
    REQUIRE_FALSE(batch.evaluate(evaluator, a, true, &error));
    REQUIRE(batch.build(a, nbhs, &error));
    loss(batch, metrics, gradient);
    invalid = batch.request(); invalid.nlocal = invalid.ntotal = std::numeric_limits<int>::max();
    REQUIRE_FALSE(evaluator.context().compute_batch(invalid, &error));
    loss(batch, metrics, gradient);
    const auto warm = evaluator.context().runtime_stats();
    moved = geometry(); moved.cell_(2, 1) += .3;
    Neighborhoods cell_nbhs(moved, p.cutoff);
    REQUIRE(batch.refresh_compatible(moved, cell_nbhs, &error));
    REQUIRE(batch.evaluate(evaluator, moved, true, &error));
    auto expected = moved; p.getEFS(expected); equal_efs(moved, expected);
    REQUIRE(evaluator.context().runtime_stats().coordinate_upload_bytes > warm.coordinate_upload_bytes);
    if constexpr (std::is_same_v<TestType, PotMTP_LES>)
        REQUIRE(evaluator.context().runtime_stats().sog_grid_upload_bytes > warm.sog_grid_upload_bytes);
}

TEST_CASE("SOG GPU handles empty grids and maximum Gaussian count", "[sog][gpu][edge]") {
    if(!available())SKIP("No GPU device");
    const bool empty_grid=GENERATE(false,true);
    PotMTP_LES p("tests/case-mtp-les/les.mtp");initialize(p,1);
    p.sog_terms=256;p.model_les.sog_weights.resize(256,.001);
    p.model_les.sog_log_bandwidths.resize(256,-.3);
    if(empty_grid){p.dl=100.;p.model_les.sog_include_zero=false;}
    p.train_init();
    std::vector<Atoms> data{geometry(),geometry()};data[1].cell_(0,0)+=2.;data[1].pos_[0][1]+=.12;
    auto cpu=data;std::vector<Neighborhoods> nbhs;for(size_t i=0;i<data.size();++i){p.getEFS(cpu[i]);nbhs.emplace_back(data[i],p.cutoff);}
    MTPGpuEvaluator evaluator;MTPGpuBatch batch;std::string error;
    REQUIRE(evaluator.initialize(p,0,&error));REQUIRE(batch.build(data,nbhs,&error));
    REQUIRE(batch.evaluate(evaluator,data,true,&error));
    for(size_t i=0;i<data.size();++i)equal_efs(data[i],cpu[i]);
    std::vector<double> gc(p.get_NL_coeffs_count(),0.),gg(gc.size());
    Eigen::Matrix<double,-1,3> fw(4,3);fw.setConstant(.13);Eigen::Matrix3d sw=Eigen::Matrix3d::Identity()*.2;
    Eigen::Matrix<double,-1,3,Eigen::RowMajor> unused;
    for(size_t i=0;i<data.size();++i)p.AccumulateEFSCombinationGard(data[i],nbhs[i],.37,fw,sw,unused,gc.data());
    std::vector<double> f(24,.13),w(18,0.);for(int i=0;i<6;++i)w[3*i+i%3]=.2;
    REQUIRE(batch.accumulate_gradient(evaluator,{.37,.37},f,w,gg.data(),gg.size(),false,&error));
    for(size_t i=0;i<gc.size();++i){CAPTURE(empty_grid,i);equal(gg[i],gc[i]);}
    // Explicitly requested mixed precision cannot silently drop the long range.
    MTPGpuEvaluator mixed;
    REQUIRE_FALSE(mixed.initialize(p,0,MLIPCore::GpuDirect::MTPGpuPrecision::Mixed,&error));
    REQUIRE(error.find("FP64")!=std::string::npos);
}

TEMPLATE_TEST_CASE("SOG GPU MTP-M heads preserve frozen optimizer layouts",
                   "[sog][gpu][mtp-m]", m_model_mtpq2,m_model_mtpfs,m_model_nn1,m_model_nn2) {
    if(!available())SKIP("No GPU device");
    const int freeze=GENERATE(0,1,2,3);
    PotMTP_M<TestType> p;
    GPSettings settings;settings.Modify("n_input","3");settings.Modify("n_neuron","4");
    settings.Modify("n_neuron2","3");settings.Modify("activation_function","TANH");
    p.model.init(settings);
    std::ifstream file("tests/case-mtp-les/les.mtp");
    std::string source((std::istreambuf_iterator<char>(file)),{});
    source.resize(source.find("species_coeffs"));
    PotMTP ref("tests/case-mtp-les/les.mtp");
    source+="les_species_coeffs = {0.23, -0.17, 0.2, -0.1, 0.1, 0.1}\nles_moment_coeffs = {";
    for(int i=0;i<3*ref.alpha_scalar_moments;++i){if(i)source+=", ";source+=std::to_string(.03*std::sin(i+1.));}
    source+="}\n";
    SOGModel sog;sog.channels=3;sog.weights={4.2,-1.3,1.,-.7,.5,.2};sog.log_bandwidths={-.4,.2,-.4,.2,-.4,.2};
    std::ostringstream metadata;sog.write(metadata);source+=metadata.str();
    std::istringstream input(source);p.Load(input);p.train_init();
    std::vector<double> coeff(p.get_NL_coeffs_count());p.nl_get_coeffs(coeff.data());
    for(const auto &g:p.nl_parameter_groups())if(g.name=="sr_species"||g.name=="sr_moment"||g.name=="sr_model")
        for(int i=g.begin;i<g.end;++i)coeff[i]=.03*std::sin(i+1.);
    p.nl_set_coeffs(coeff.data());p.set_inited();
    if (!static_cast<const PotMTP &>(p).inited()) p.set_empty_flag();
    if(freeze==1)p.req_les=false;
    if(freeze==2){p.req_radical=false;p.req_species=false;p.req_model=false;p.req_sog=false;}
    if(freeze==3){p.req_radical=false;p.req_species=false;p.req_moment=false;p.req_model=false;p.req_les=false;}
    p.train_init();
    auto a=geometry(),cpu=a;Neighborhoods nbhs(a,p.cutoff);p.getEFS(cpu);
    MTPGpuEvaluator evaluator;MTPGpuBatch batch;std::string error;
    const bool initialized=evaluator.initialize(p,0,&error);INFO(error);REQUIRE(initialized);REQUIRE(batch.build(a,nbhs,&error));
    REQUIRE(batch.evaluate(evaluator,a,true,&error));equal_efs(a,cpu);
    Eigen::Matrix<double,-1,3> fw(4,3);fw<<.2,-.1,.3,-.5,.7,.1,.3,.2,-.4,.1,.2,.4;
    Eigen::Matrix3d sw;sw<<.1,.2,-.1,.3,-.2,.5,-.1,.2,.7;
    Eigen::Matrix<double,-1,3,Eigen::RowMajor> unused;
    std::vector<double> f(12),w(9),gc(p.get_NL_coeffs_count(),0.),gg(gc.size(),0.);
    for(int i=0;i<4;++i)for(int j=0;j<3;++j)f[3*i+j]=fw(i,j);
    for(int i=0;i<3;++i)for(int j=0;j<3;++j)w[3*i+j]=sw(i,j);
    p.AccumulateEFSCombinationGard(a,nbhs,.37,fw,sw,unused,gc.data());
    REQUIRE(batch.accumulate_gradient(evaluator,{.37},f,w,gg.data(),gg.size(),false,&error));
    for(size_t i=0;i<gc.size();++i){CAPTURE(freeze,i);equal(gg[i],gc[i]);}
    REQUIRE(batch.accumulate_gradient(evaluator,{.37},f,w,gg.data(),gg.size(),true,&error));
    for(size_t i=0;i<gc.size();++i)equal(gg[i],2*gc[i]);
    REQUIRE(batch.accumulate_gradient_chunks(evaluator,{.37},f,w,gg.data(),gg.size(),true,false,&error));
    REQUIRE(batch.accumulate_gradient_chunks(evaluator,{.37},f,w,gg.data(),gg.size(),false,true,&error));
    for(size_t i=0;i<gc.size();++i)equal(gg[i],2*gc[i]);
    // A compact native optimizer vector must also survive actual device loss.
    std::vector<Atoms> data{cpu,cpu};std::vector<Neighborhoods> neighbors{nbhs,nbhs};
    data[0].energy_+=.1;data[1].forces_[0][0]+=.01;
    settings.Modify("charge_weight","0");settings.Modify("penalty_weight","0");
    TrainNonlinear tc(settings);tc.init(data,neighbors,&p);double lc;Eigen::VectorXd vc(gc.size());tc.AccumulateGard(lc,vc);
    settings.Modify("device",MLIPCore::GpuDirect::MTPGpuDirect::backend_name());settings.Modify("gpu_id","0");settings.Modify("gpu_batch_structures","1");
    TrainNonlinear tg(settings);tg.init(data,neighbors,&p);double lg;Eigen::VectorXd vg(gc.size());tg.AccumulateGard(lg,vg);
    equal(lg,lc);for(int i=0;i<vc.size();++i){CAPTURE(freeze,i);equal(vg[i],vc[i]);}
}

#endif
