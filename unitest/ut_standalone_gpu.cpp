#include <catch2/catch_all.hpp>
#if __has_include("mlip.h")
#include "gpu-direct/gpu_backend.h"
#ifdef IMRMLP_GPU
#include "gpu-direct/mtp_gpu_direct.h"
#endif
#else
#include "runtime/mtp/gpu-direct/gpu_backend.h"
#ifdef IMRMLP_GPU
#include "runtime/mtp/gpu-direct/mtp_gpu_direct.h"
#endif
#endif
#ifdef IMRMLP_GPU
#if __has_include("standalone/tests/radial_fixture.h")
#include "radial_cache.h"
#include "cpu/mtp_cpu.h"
#include "standalone/tests/radial_fixture.h"
#include "opt/mlip_opt_virial.h"
#else
#include "runtime/mtp/radial_cache.h"
#include "runtime/mtp/cpu/mtp_cpu.h"
#include "LAMMPS/USER-MLIP/standalone/tests/radial_fixture.h"
#include "LAMMPS/USER-MLIP/opt/mlip_opt_virial.h"
#endif
#include <array>
#include <cmath>
#include <cstdlib>
#include <string>
#include <vector>

namespace {
using namespace MLIPCore::GpuDirect;

void require_standalone_device()
{
    int count = 0;
    std::string error;
    if (!MTPGpuDirect::available_device_count(&count, &error) || !count) {
        if (std::getenv("IMRMLP_REQUIRE_GPU")) FAIL("Required GPU unavailable: " << error);
        SKIP("No GPU: " << error);
    }
}

struct StandaloneModel {
    std::array<int, 2> mapping{{-1, 0}};
    std::array<int, 8> basic{{0,0,0,0, 1,0,0,0}};
    std::array<int, 8> times{{0,1,1,2, 0,0,2,3}};
    std::array<int, 4> scalars{{0,2,2,3}}; // repeated scalar mapping is intentional
    std::array<int, 2> nu{{0,0}};
    std::array<double, 4> radial{{0.07,-0.02,0.02,0.03}};
    std::array<double, 1> species{{0.3}};
    std::vector<double> bias, projection, head;
    MTPGpuPotentialView view;
    explicit StandaloneModel(MTPGpuModel kind) {
        view.species_count=1; view.max_type=2; view.spmap=mapping.data();
        view.alpha_moments_count=4; view.alpha_index_basic_count=2;
        view.alpha_index_times_count=2; view.alpha_scalar_moments=4;
        view.radial_func_count=2; view.rb_size=2; view.max_mu=2;
        view.max_alpha_index_basic=1; view.min_dist=0; view.max_dist=5;
        view.cutoff=5; view.scaling=1; view.radial_basis=RadialBasis::Chebyshev;
        view.radial_coeffs=radial.data(); view.radial_coeffs_count=4;
        view.species_coeffs=species.data(); view.alpha_index_basic=basic.data();
        view.alpha_index_times=times.data(); view.alpha_moment_mapping=scalars.data();
        view.nu_arr_list=nu.data(); view.model=kind;
        const bool neural=kind==MTPGpuModel::MTPMNN1 || kind==MTPGpuModel::MTPMNN2;
        view.descriptor_count=kind==MTPGpuModel::MTPMLinear ? 1 : neural ? 3 : 2;
        view.model_hidden1=neural ? 4 : 0;
        view.model_hidden2=kind==MTPGpuModel::MTPMNN2 ? 3 : 0;
        const int q=view.descriptor_count, h=view.model_hidden1, h2=view.model_hidden2;
        const int count=kind==MTPGpuModel::MTPMQuadratic ? 5 : kind==MTPGpuModel::MTPMAsinh ? 1 :
            kind==MTPGpuModel::MTPMNN1 ? q*h+2*h : kind==MTPGpuModel::MTPMNN2 ? q*h+h+h*h2+2*h2 : 0;
        bias.resize(q); projection.resize(q*4); head.resize(count);
        for (int i=0;i<q;++i) bias[i]=0.03*(i+1);
        for (std::size_t i=0;i<projection.size();++i) projection[i]=0.02*std::cos(double(i));
        for (int i=0;i<count;++i) head[i]=0.1*std::sin(0.7*(i+1));
        view.descriptor_species_coeffs=bias.data(); view.descriptor_moment_coeffs=projection.data();
        view.model_coeffs=head.data(); view.model_coeffs_count=count;
    }
};

struct PartitionedModel : StandaloneModel {
    std::vector<int> graph_basic, graph_times, graph_scalars, graph_nu;
    std::vector<double> graph_radial;
    explicit PartitionedModel(MTPGpuModel kind) : StandaloneModel(kind) {
        // Independent basic operands make the reverse groups hit the same
        // 31/32/33 and 511/512/513 boundaries as the forward groups. The common
        // operand also receives a block-sized reverse group. Higher levels
        // use composites so these exact reverse boundary counts stay intact.
        constexpr int basic_count=11, moment_count=28;
        const std::array<int,9> counts{{31,32,33,63,64,65,511,512,513}};
        std::array<double,moment_count> scale{};
        scale.fill(1);
        for (int i=0;i<basic_count;++i) {
            graph_basic.insert(graph_basic.end(), {i,0,0,0});
            graph_nu.push_back(0);
            graph_radial.push_back(.03+.002*i);
            graph_radial.push_back(-.008+.001*i);
        }
        auto append = [&](int out, int lhs, int rhs, int count, int coefficient) {
            for (int i=0;i<count;++i)
                graph_times.insert(graph_times.end(), {lhs,rhs,coefficient,out});
            scale[out]=count*coefficient*scale[lhs]*scale[rhs];
        };
        for (int i=0;i<9;++i) append(11+i,i,9,counts[i],1+i%3);
        append(22,11,19,33,1);
        append(24,12,18,513,1);
        append(27,22,24,65,1);
        // A skip-level dependency, and unused IDs, catch level/barrier and
        // compact-slot mistakes. Projection scales keep every head nonsaturated.
        graph_times.insert(graph_times.end(), {13,10,1,27});
        // A higher-level square contributes twice to the reverse operand.
        graph_times.insert(graph_times.end(), {22,22,1,27});
        graph_scalars={0,1,11,12,13,14,15,16,17,18,19,22,24,27,19,27};
        const int scalars=static_cast<int>(graph_scalars.size());
        projection.resize(view.descriptor_count*scalars);
        for (int q=0;q<view.descriptor_count;++q)
            for (int i=0;i<scalars;++i)
                projection[q*scalars+i]=.02*std::cos(double(q*scalars+i))/scale[graph_scalars[i]];
        view.alpha_moments_count=moment_count;
        view.alpha_index_basic_count=basic_count;
        view.alpha_index_times_count=static_cast<int>(graph_times.size()/4);
        view.alpha_scalar_moments=scalars;
        view.radial_func_count=basic_count; view.max_mu=basic_count;
        view.radial_coeffs=graph_radial.data(); view.radial_coeffs_count=graph_radial.size();
        view.alpha_index_basic=graph_basic.data(); view.alpha_index_times=graph_times.data();
        view.alpha_moment_mapping=graph_scalars.data(); view.nu_arr_list=graph_nu.data();
        view.descriptor_moment_coeffs=projection.data();
    }
};

struct StandaloneAtoms {
    static constexpr int n=5, local=4;
    std::array<std::array<double,3>,n> x{{{{0,0,0}},{{1.1,.2,-.1}},{{-.4,1.3,.3}},{{.2,-.6,1.4}},{{1.5,.7,1.2}}}};
    std::array<std::array<double,3>,n> force{};
    std::array<std::array<double,6>,n> virial{};
    std::array<double,local> energy{};
    std::array<double,6> global{};
    double total=0;
    std::array<int,n> type{{1,1,1,1,1}}, counts{{4,4,4,4,4}};
    std::array<int,local> centers{{2,0,3,1}}; // non-identity owned-center order
    std::array<std::array<int,4>,n> neighbors{};
    std::array<double*,n> xp{}, fp{}, vp{};
    std::array<int*,n> np{};
    StandaloneAtoms() {
        for(int i=0;i<n;++i) {
            int k=0;
            for(int j=0;j<n;++j) if(i!=j) neighbors[i][k++]=j|0x40000000;
            xp[i]=x[i].data(); fp[i]=force[i].data(); vp[i]=virial[i].data(); np[i]=neighbors[i].data();
        }
    }
    MTPGpuComputeRequest request() {
        MTPGpuComputeRequest r;
        r.ilist=centers.data(); r.numneigh=counts.data(); r.firstneigh=np.data();
        r.inum=local; r.nlocal=local; r.ntotal=n; r.type=type.data();
        r.x=xp.data(); r.force=fp.data(); r.total_energy=&total;
        r.site_energy=energy.data(); r.site_virial=vp.data(); r.global_virial=global.data();
        r.neighbor_index_mask=0x3fffffff; r.accumulate_force=false;
        r.basic_tile=1;
        return r;
    }
};

void compare_standalone(const StandaloneAtoms &a, const StandaloneAtoms &b, double force_offset=0)
{
    REQUIRE(a.total==Catch::Approx(b.total).epsilon(1e-12).margin(1e-11));
    for(int i=0;i<a.n;++i) {
        for(int c=0;c<3;++c) REQUIRE(a.force[i][c]+force_offset==Catch::Approx(b.force[i][c]).epsilon(1e-12).margin(1e-11));
        for(int c=0;c<6;++c) REQUIRE(a.virial[i][c]==Catch::Approx(b.virial[i][c]).epsilon(1e-12).margin(1e-11));
    }
    for(int i=0;i<a.local;++i) REQUIRE(a.energy[i]==Catch::Approx(b.energy[i]).epsilon(1e-12).margin(1e-11));
    for(int c=0;c<6;++c) {
        REQUIRE(a.global[c]==Catch::Approx(b.global[c]).epsilon(1e-12).margin(1e-11));
        double sum=0;
        for(int i=0;i<a.n;++i) sum+=b.virial[i][c];
        REQUIRE(sum==Catch::Approx(b.global[c]).epsilon(1e-12).margin(1e-11));
    }
}
}

TEST_CASE("Native batch virial and optional force copies coexist with standalone inference", "[standalone-gpu][mtp][gpu][native-compat]")
{
    require_standalone_device();
    const auto kind=GENERATE(MTPGpuModel::MTPMLinear,MTPGpuModel::MTPMQuadratic,MTPGpuModel::MTPMAsinh,
                            MTPGpuModel::MTPMNN1,MTPGpuModel::MTPMNN2);
    const bool copy_force=GENERATE(true,false);
    const bool copy_site_virial=GENERATE(true,false);
    StandaloneModel model(kind);
    StandaloneAtoms expected, actual;
    MTPGpuDirect reference, batch;
    std::string error;
    REQUIRE(reference.initialize(model.view,&error));
    REQUIRE(batch.initialize(model.view,&error));
    auto r=expected.request(); r.inference_only=true;
    REQUIRE(reference.compute_lammps(r,&error));
    std::array<int,StandaloneAtoms::local+1> offsets{};
    std::vector<int> neighbors;
    for(int center=0;center<actual.local;++center) {
        for(int neighbor:actual.neighbors[actual.centers[center]])
            neighbors.push_back(neighbor&0x3fffffff);
        offsets[center+1]=static_cast<int>(neighbors.size());
    }
    for(auto &force:actual.force) force.fill(73.0);
    MTPGpuBatchRequest request;
    request.center_count=actual.local; request.nlocal=actual.local; request.ntotal=actual.n;
    request.centers=actual.centers.data(); request.offsets=offsets.data(); request.neighbors=neighbors.data();
    request.type=actual.type.data(); request.x=actual.x[0].data();
    request.force=copy_force?actual.force[0].data():nullptr;
    request.copy_forces_to_host=copy_force; request.accumulate_force=false;
    request.total_energy=&actual.total; request.site_energy=actual.energy.data();
    request.global_virial=actual.global.data();
    request.site_virial=copy_site_virial?actual.virial[0].data():nullptr;
    INFO(error);
    REQUIRE(batch.compute_batch(request,&error));
    REQUIRE(actual.total==Catch::Approx(expected.total).epsilon(1e-12).margin(1e-11));
    for(int i=0;i<actual.local;++i)
        REQUIRE(actual.energy[i]==Catch::Approx(expected.energy[i]).epsilon(1e-12).margin(1e-11));
    for(int c=0;c<6;++c)
        REQUIRE(actual.global[c]==Catch::Approx(expected.global[c]).epsilon(1e-12).margin(1e-11));
    for(int i=0;i<actual.n;++i) {
        for(int c=0;c<3;++c) {
            if(copy_force) REQUIRE(actual.force[i][c]==Catch::Approx(expected.force[i][c]).epsilon(1e-12).margin(1e-11));
            else REQUIRE(actual.force[i][c]==73.0);
        }
        if(copy_site_virial) for(int c=0;c<6;++c)
            REQUIRE(actual.virial[i][c]==Catch::Approx(expected.virial[i][c]).epsilon(1e-12).margin(1e-11));
    }
}

TEST_CASE("Standalone GPU partitioned inference covers graph tiers and dependencies", "[standalone-gpu][mtp][gpu][propagation]")
{
    require_standalone_device();
    const auto kind=GENERATE(MTPGpuModel::MTPMLinear,MTPGpuModel::MTPMQuadratic,MTPGpuModel::MTPMAsinh,
                            MTPGpuModel::MTPMNN1,MTPGpuModel::MTPMNN2);
    const auto kernel=GENERATE(MTPGpuForceKernel::BasicTile,MTPGpuForceKernel::CenterNeighbors);
    const int threads=GENERATE(1,16,32,64,128,256);
    const bool standard_layout=GENERATE(true,false);
    PartitionedModel model(kind);
    StandaloneAtoms expected, actual;
    MTPGpuDirect reference, partitioned;
    std::string error;
    REQUIRE(reference.initialize(model.view,&error));
    REQUIRE(partitioned.initialize(model.view,&error));
    REQUIRE(reference.potential_stats().forward_block_groups==2);
    REQUIRE(reference.potential_stats().reverse_block_groups>=2);
    REQUIRE(reference.potential_stats().active_moment_slots<model.view.alpha_moments_count);
    REQUIRE(reference.memory_stats().potential_bytes==partitioned.memory_stats().potential_bytes);
    auto r=expected.request(); r.propagate_threads=threads; r.force_kernel=kernel;
    r.basic_tile=5;
    if (!standard_layout) { r.neighbor_lanes=3; r.basic_lanes=7; }
    INFO(error);
    REQUIRE(reference.compute_lammps(r,&error)); // unpartitioned, full training cache
    auto c=actual.request(); c.propagate_threads=threads; c.force_kernel=kernel;
    c.basic_tile=r.basic_tile; c.neighbor_lanes=r.neighbor_lanes; c.basic_lanes=r.basic_lanes;
    c.inference_only=true; c.centers_per_chunk=3;
    REQUIRE(partitioned.compute_lammps(c,&error));
    compare_standalone(expected,actual);
    REQUIRE(partitioned.runtime_stats().last_chunk_count==2);
    if (threads==128 && standard_layout) {
        StandaloneAtoms fallback_atoms;
        MTPGpuDirect fallback;
        REQUIRE(fallback.initialize(model.view,&error));
        auto f=fallback_atoms.request();
        f.inference_only=true; f.propagate_threads=threads; f.force_kernel=kernel;
        f.basic_tile=5; f.neighbor_lanes=3; f.basic_lanes=7; f.centers_per_chunk=3;
        REQUIRE(fallback.compute_lammps(f,&error));
        // A different 2-D layout cannot enter the standard warp reduction.
        compare_standalone(actual,fallback_atoms);
    }
    c.reuse_topology=true; c.compute_site_energy=false; c.site_energy=nullptr; c.site_virial=nullptr;
    REQUIRE(partitioned.compute_lammps(c,&error));
    for (int k=0;k<6;++k) REQUIRE(actual.global[k]==Catch::Approx(expected.global[k]).epsilon(1e-12).margin(1e-11));
    c.global_virial=nullptr; c.compute_global_energy=false; c.total_energy=nullptr;
    REQUIRE(partitioned.compute_lammps(c,&error));
    for (int i=0;i<actual.n;++i) for (int k=0;k<3;++k)
        REQUIRE(actual.force[i][k]==Catch::Approx(expected.force[i][k]).epsilon(1e-12).margin(1e-11));
    c.compute_forces=false; c.force=nullptr; c.site_virial=actual.vp.data(); c.global_virial=actual.global.data();
    REQUIRE(partitioned.compute_lammps(c,&error));
    for (int i=0;i<actual.n;++i) for (int k=0;k<6;++k)
        REQUIRE(actual.virial[i][k]==Catch::Approx(expected.virial[i][k]).epsilon(1e-12).margin(1e-11));
    c.site_virial=nullptr; c.global_virial=nullptr; c.compute_global_energy=true; c.total_energy=&actual.total;
    REQUIRE(partitioned.compute_lammps(c,&error));
    REQUIRE(actual.total==Catch::Approx(expected.total).epsilon(1e-12).margin(1e-11));
    if (threads==128 && standard_layout) {
        // Energy-only never launches either force kernel: independently check
        // the optimized force, including an unowned/ghost atom contribution.
        const double h=1e-6;
        for (int atom: {0,4}) for (int axis=0;axis<3;++axis) {
            const double original=actual.x[atom][axis];
            actual.x[atom][axis]=original+h; REQUIRE(partitioned.compute_lammps(c,&error));
            const double plus=actual.total;
            actual.x[atom][axis]=original-h; REQUIRE(partitioned.compute_lammps(c,&error));
            const double minus=actual.total;
            actual.x[atom][axis]=original;
            REQUIRE(-(plus-minus)/(2*h)==Catch::Approx(expected.force[atom][axis]).epsilon(1e-9).margin(2e-8));
        }
    }
    if (threads==128 && standard_layout && kernel==MTPGpuForceKernel::BasicTile) {
        c=actual.request(); c.propagate_threads=threads; c.basic_tile=5; c.force_kernel=kernel;
        REQUIRE(partitioned.compute_lammps(c,&error)); // compact inference -> full training
        compare_standalone(expected,actual);
        std::array<int,StandaloneAtoms::local> structure{};
        std::array<double,StandaloneAtoms::n*3> df{};
        std::array<double,9> ds{{.03,.01,-.02,.02,-.04,.01,0,.02,.05}};
        for (std::size_t i=0;i<df.size();++i) df[i]=.01*std::sin(double(i));
        double de=.7;
        const int parameters=model.view.radial_coeffs_count+1+
            (model.view.alpha_scalar_moments+1)*model.view.descriptor_count+model.view.model_coeffs_count;
        std::vector<double> ga(parameters), gb(parameters);
        MTPGpuGradientRequest g;
        g.structure_count=1; g.center_structure=structure.data(); g.dloss_denergy=&de;
        g.dloss_dforce=df.data(); g.dloss_dstress=ds.data(); g.gradient_count=parameters;
        g.gradient=ga.data(); REQUIRE(reference.accumulate_gradient(g,&error));
        g.gradient=gb.data(); REQUIRE(partitioned.accumulate_gradient(g,&error));
        for (int i=0;i<parameters;++i) REQUIRE(ga[i]==Catch::Approx(gb[i]).epsilon(1e-11).margin(2e-10));
    }
}

TEST_CASE("Standalone GPU chunking and inference caches preserve five heads", "[standalone-gpu][mtp][gpu]")
{
    require_standalone_device();
    const auto kind=GENERATE(MTPGpuModel::MTPMLinear,MTPGpuModel::MTPMQuadratic,MTPGpuModel::MTPMAsinh,
                            MTPGpuModel::MTPMNN1,MTPGpuModel::MTPMNN2);
    const auto kernel=GENERATE(MTPGpuForceKernel::BasicTile,MTPGpuForceKernel::CenterNeighbors);
    StandaloneModel model(kind);
    StandaloneAtoms expected, actual;
    MTPGpuDirect full, chunked;
    std::string error;
    REQUIRE(full.initialize(model.view,&error));
    REQUIRE(chunked.initialize(model.view,&error));
    auto r=expected.request();
    r.force_kernel=kernel;
    INFO(error);
    REQUIRE(full.compute_lammps(r,&error)); // original training-compatible cache
    auto c=actual.request(); c.inference_only=true; c.centers_per_chunk=3;
    c.force_kernel=kernel; c.accumulate_force=true;
    for(auto &f:actual.force) f.fill(.25);
    REQUIRE(chunked.compute_lammps(c,&error));
    compare_standalone(expected,actual,.25);
    const auto stats=chunked.memory_stats();
    REQUIRE(stats.moment_workspace_bytes==3*4*sizeof(double));
    REQUIRE(stats.adjoint_workspace_bytes==3*4*sizeof(double));
    REQUIRE(chunked.runtime_stats().last_chunk_count==2);
    if (kind==MTPGpuModel::MTPMNN1 || kind==MTPGpuModel::MTPMNN2)
        REQUIRE(stats.neural_cache_bytes < full.memory_stats().neural_cache_bytes);
    c.reuse_topology=true; c.accumulate_force=false;
    const auto allocations=chunked.runtime_stats().buffer_reallocations;
    const auto uploads=chunked.runtime_stats().topology_uploads;
    REQUIRE(chunked.compute_lammps(c,&error));
    compare_standalone(expected,actual);
    REQUIRE(chunked.runtime_stats().buffer_reallocations==allocations);
    REQUIRE(chunked.runtime_stats().topology_uploads==uploads);
    // Switching back must restore full training layout, never compact HVP caches.
    c.inference_only=false; c.centers_per_chunk=0;
    REQUIRE(chunked.compute_lammps(c,&error));
    compare_standalone(expected,actual);
    REQUIRE(chunked.memory_stats().neural_cache_bytes>=full.memory_stats().neural_cache_bytes);
    // Force/stress loss derivatives exercise the training HVP cache after
    // switching from compact inference storage back to the original layout.
    std::array<int,StandaloneAtoms::local> structure{};
    std::array<double,StandaloneAtoms::n*3> df{};
    std::array<double,9> ds{{.03,.01,-.02,.02,-.04,.01,0,.02,.05}};
    for(std::size_t i=0;i<df.size();++i) df[i]=.01*std::sin(double(i));
    double de=.7;
    std::vector<double> grad_a(5+5*model.view.descriptor_count+model.view.model_coeffs_count), grad_b(grad_a.size());
    MTPGpuGradientRequest gradient;
    gradient.structure_count=1; gradient.center_structure=structure.data();
    gradient.dloss_denergy=&de; gradient.dloss_dforce=df.data(); gradient.dloss_dstress=ds.data();
    gradient.gradient_count=static_cast<int>(grad_a.size()); gradient.gradient=grad_a.data();
    REQUIRE(full.accumulate_gradient(gradient,&error));
    gradient.gradient=grad_b.data(); REQUIRE(chunked.accumulate_gradient(gradient,&error));
    for(std::size_t i=0;i<grad_a.size();++i)
        REQUIRE(grad_a[i]==Catch::Approx(grad_b[i]).epsilon(1e-11).margin(2e-10));
    // Global-only virial must not dereference or require a site output.
    c.inference_only=true; c.centers_per_chunk=1; c.site_virial=nullptr;
    c.compute_site_energy=false; c.site_energy=nullptr;
    REQUIRE(chunked.compute_lammps(c,&error));
    for(int k=0;k<6;++k) REQUIRE(actual.global[k]==Catch::Approx(expected.global[k]).epsilon(1e-12).margin(1e-11));
    c.global_virial=nullptr; c.total_energy=nullptr; c.compute_global_energy=false;
    REQUIRE(chunked.compute_lammps(c,&error));
    for(int i=0;i<actual.n;++i) for(int k=0;k<3;++k)
        REQUIRE(actual.force[i][k]==Catch::Approx(expected.force[i][k]).epsilon(1e-12).margin(1e-11));
    // Virial-only still differentiates the model but never writes force output.
    c.compute_forces=false; c.force=nullptr; c.global_virial=actual.global.data();
    c.site_virial=actual.vp.data();
    REQUIRE(chunked.compute_lammps(c,&error));
    for(int k=0;k<6;++k) REQUIRE(actual.global[k]==Catch::Approx(expected.global[k]).epsilon(1e-12).margin(1e-11));
    c.global_virial=nullptr; c.site_virial=nullptr;
    // Coordinate finite differences include the ghost force contribution.
    c.compute_forces=false; c.force=nullptr; c.compute_global_energy=true; c.total_energy=&actual.total;
    const double h=1e-6;
    for(int atom: {0,4}) for(int axis=0;axis<3;++axis) {
        const double original=actual.x[atom][axis];
        actual.x[atom][axis]=original+h; REQUIRE(chunked.compute_lammps(c,&error)); const double plus=actual.total;
        actual.x[atom][axis]=original-h; REQUIRE(chunked.compute_lammps(c,&error)); const double minus=actual.total;
        actual.x[atom][axis]=original;
        REQUIRE(-(plus-minus)/(2*h)==Catch::Approx(expected.force[atom][axis]).epsilon(1e-9).margin(2e-8));
    }
}

TEST_CASE("Standalone GPU budget includes fixed and double-buffered topology", "[standalone-gpu][mtp][gpu]")
{
    require_standalone_device();
    StandaloneModel model(MTPGpuModel::MTPMNN2);
    StandaloneAtoms expected, actual;
    MTPGpuDirect probe, bounded;
    std::string error;
    REQUIRE(probe.initialize(model.view,&error));
    auto p=expected.request(); p.inference_only=true;
    REQUIRE(probe.compute_lammps(p,&error));
    const auto memory=probe.memory_stats();
    const auto per_center=(memory.moment_workspace_bytes+memory.adjoint_workspace_bytes+memory.neural_cache_bytes)/expected.local;
    const auto budget=memory.potential_bytes+2*(memory.topology_bytes+memory.input_bytes)+memory.output_bytes+per_center;
    MTPGpuInitializationOptions options; options.memory_budget_bytes=budget;
    REQUIRE(bounded.initialize(model.view,options,&error));
    auto b=actual.request(); b.inference_only=true; b.memory_budget_bytes=budget;
    INFO(error);
    REQUIRE(bounded.compute_lammps(b,&error));
    REQUIRE(bounded.runtime_stats().last_centers_per_chunk==1);
    REQUIRE(bounded.memory_stats().tracked_device_bytes<=budget);
    compare_standalone(expected,actual);
    // A normal second rebuild allocates the other topology slot without exceeding the cap.
    REQUIRE(bounded.compute_lammps(b,&error));
    REQUIRE(bounded.memory_stats().tracked_device_bytes<=budget);
    compare_standalone(expected,actual);
    b.reuse_topology=true;
    const auto allocations=bounded.runtime_stats().buffer_reallocations;
    REQUIRE(bounded.compute_lammps(b,&error));
    REQUIRE(bounded.runtime_stats().buffer_reallocations==allocations);
    b.memory_budget_bytes=memory.potential_bytes;
    REQUIRE_FALSE(bounded.compute_lammps(b,&error));
    REQUIRE(error.find("memory_budget_bytes")!=std::string::npos);
    MTPGpuDirect too_small;
    options.memory_budget_bytes=memory.potential_bytes-1;
    REQUIRE_FALSE(too_small.initialize(model.view,options,&error));
    REQUIRE(too_small.memory_stats().tracked_device_bytes==0);
}

TEST_CASE("Standalone GPU force variants opt in to large shared memory", "[standalone-gpu][mtp][gpu]")
{
    require_standalone_device();
    const auto kernel=GENERATE(MTPGpuForceKernel::BasicTile,MTPGpuForceKernel::CenterNeighbors);
    StandaloneModel narrow(MTPGpuModel::MTPMQuadratic), wide(MTPGpuModel::MTPMQuadratic);
    // Zero coefficients for the higher Chebyshev polynomials preserve exactly
    // the narrow model while forcing the new force specialization above 48 KiB.
    constexpr int radial_size=400;
    std::vector<double> radial(2*radial_size,0);
    for (int mu=0;mu<2;++mu) for (int i=0;i<2;++i)
        radial[mu*radial_size+i]=wide.radial[mu*2+i];
    wide.view.rb_size=radial_size; wide.view.radial_coeffs=radial.data();
    wide.view.radial_coeffs_count=radial.size();
    StandaloneAtoms expected, actual;
    MTPGpuDirect reference, candidate;
    std::string error;
    REQUIRE(reference.initialize(narrow.view,&error));
    REQUIRE(candidate.initialize(wide.view,&error));
    auto r=expected.request(); r.force_kernel=kernel;
    REQUIRE(reference.compute_lammps(r,&error));
    auto c=actual.request(); c.inference_only=true; c.force_kernel=kernel;
    const bool computed=candidate.compute_lammps(c,&error);
    if (!computed && error.find("more shared memory than the GPU device provides")!=std::string::npos)
        SKIP("GPU cannot provide the synthetic model's shared-memory capacity");
    INFO(error);
    REQUIRE(computed);
    compare_standalone(expected,actual);
    c.site_virial=nullptr; c.global_virial=nullptr;
    REQUIRE(candidate.compute_lammps(c,&error));
    for (int i=0;i<actual.n;++i) for (int k=0;k<3;++k)
        REQUIRE(actual.force[i][k]==Catch::Approx(expected.force[i][k]).epsilon(1e-12).margin(1e-11));
    // The shared-memory attribute cache also covers switching to the original
    // training force specialization after an optimized inference invocation.
    c=actual.request(); c.force_kernel=kernel;
    REQUIRE(candidate.compute_lammps(c,&error));
    compare_standalone(expected,actual);
}

TEST_CASE("Standalone GPU force reduction includes every logical warp", "[standalone-gpu][mtp][gpu]")
{
    require_standalone_device();
    const auto kind=GENERATE(MTPGpuModel::MTPMLinear,MTPGpuModel::MTPMQuadratic,MTPGpuModel::MTPMAsinh,
                            MTPGpuModel::MTPMNN1,MTPGpuModel::MTPMNN2);
    const auto kernel=GENERATE(MTPGpuForceKernel::BasicTile,MTPGpuForceKernel::CenterNeighbors);
    StandaloneModel model(kind);
    constexpr int basics=20;
    std::vector<int> basic, nu(basics,0), scalar;
    std::vector<double> radial, projection(model.view.descriptor_count*basics);
    for (int i=0;i<basics;++i) {
        basic.insert(basic.end(), {i,0,0,0}); scalar.push_back(i);
        radial.push_back(.02+.001*i); radial.push_back(-.005-.0005*i);
        for (int q=0;q<model.view.descriptor_count;++q)
            projection[q*basics+i]=.003*std::cos(double(q*basics+i));
    }
    model.view.alpha_moments_count=basics; model.view.alpha_index_basic_count=basics;
    model.view.alpha_index_times_count=0; model.view.alpha_scalar_moments=basics;
    model.view.radial_func_count=basics; model.view.max_mu=basics;
    model.view.alpha_index_basic=basic.data(); model.view.alpha_moment_mapping=scalar.data();
    model.view.nu_arr_list=nu.data(); model.view.radial_coeffs=radial.data();
    model.view.radial_coeffs_count=radial.size(); model.view.descriptor_moment_coeffs=projection.data();
    StandaloneAtoms expected, actual;
    MTPGpuDirect reference, candidate;
    std::string error;
    REQUIRE(reference.initialize(model.view,&error));
    REQUIRE(candidate.initialize(model.view,&error));
    auto r=expected.request(); r.force_kernel=kernel; r.basic_tile=17;
    REQUIRE(reference.compute_lammps(r,&error));
    auto c=actual.request(); c.force_kernel=kernel; c.basic_tile=17; c.inference_only=true;
    // All 16 basic lanes have nonzero contributions, including ty=12..15.
    // BasicTile additionally handles a second, partial three-moment tile.
    REQUIRE(candidate.compute_lammps(c,&error));
    compare_standalone(expected,actual);
    c.neighbor_lanes=3; c.basic_lanes=7; c.reuse_topology=true;
    REQUIRE(candidate.compute_lammps(c,&error));
    compare_standalone(expected,actual);
}

TEST_CASE("Standalone GPU virial-only forward cannot prepare training loss", "[standalone-gpu][mtp][gpu]")
{
    require_standalone_device();
    const bool reused=GENERATE(false,true);
    StandaloneModel model(MTPGpuModel::MTPMNN2);
    StandaloneAtoms atoms;
    MTPGpuDirect gpu;
    std::string error;
    REQUIRE(gpu.initialize(model.view,&error));
    auto request=atoms.request();
    if (reused) REQUIRE(gpu.compute_lammps(request,&error));
    request.compute_forces=false; request.force=nullptr;
    REQUIRE(gpu.compute_lammps(request,&error));
    std::array<int,2> offsets{{0,StandaloneAtoms::n}};
    std::array<int,StandaloneAtoms::local> structure{};
    std::array<double,StandaloneAtoms::n*3> force{};
    std::array<double,StandaloneAtoms::n> force_weight{};
    std::array<double,9> stress{};
    std::array<double,MTPGpuLossRequest::metric_count> metrics{};
    double energy=0, weight=1;
    force_weight.fill(1);
    MTPGpuLossRequest loss;
    loss.structure_count=1; loss.structure_offsets=offsets.data(); loss.center_structure=structure.data();
    loss.target_energy=&energy; loss.energy_weight=&weight;
    loss.target_force=force.data(); loss.force_weight=force_weight.data();
    loss.target_stress=stress.data(); loss.stress_weight=&weight; loss.metrics=metrics.data();
    REQUIRE_FALSE(gpu.prepare_loss(loss,&error));
    REQUIRE(error.find("successful preceding")!=std::string::npos);
}

TEST_CASE("Standalone GPU refreshes changed species while reusing topology", "[standalone-gpu][mtp][gpu]")
{
    require_standalone_device();
    StandaloneModel model(MTPGpuModel::MTPMNN2);
    std::array<int,3> mapping{{-1,0,1}};
    std::array<double,2> species{{.3,-.2}};
    std::array<double,16> radial{};
    for(std::size_t i=0;i<radial.size();++i) radial[i]=.03*std::cos(.4*double(i));
    model.bias.resize(2*model.view.descriptor_count);
    for(std::size_t i=0;i<model.bias.size();++i) model.bias[i]=.03*(i+1);
    model.view.species_count=2; model.view.max_type=3; model.view.spmap=mapping.data();
    model.view.species_coeffs=species.data(); model.view.radial_coeffs=radial.data();
    model.view.radial_coeffs_count=radial.size(); model.view.descriptor_species_coeffs=model.bias.data();
    StandaloneAtoms expected,actual;
    MTPGpuDirect fresh,reused;
    std::string error;
    REQUIRE(fresh.initialize(model.view,&error)); REQUIRE(reused.initialize(model.view,&error));
    auto a=actual.request(); a.inference_only=true;
    REQUIRE(reused.compute_lammps(a,&error));
    actual.type[0]=actual.type[4]=expected.type[0]=expected.type[4]=2;
    auto e=expected.request(); e.inference_only=true;
    REQUIRE(fresh.compute_lammps(e,&error));
    const auto before=reused.runtime_stats();
    a.reuse_topology=true;
    REQUIRE(reused.compute_lammps(a,&error));
    compare_standalone(expected,actual);
    REQUIRE(reused.runtime_stats().topology_uploads==before.topology_uploads);
    REQUIRE(reused.runtime_stats().type_uploads==before.type_uploads+1);
}

TEST_CASE("Standalone GPU ordered chunks and empty ranks keep output contracts", "[standalone-gpu][mtp][gpu]")
{
    require_standalone_device();
    StandaloneModel model(MTPGpuModel::MTPMNN2);
    StandaloneAtoms expected, actual;
    MTPGpuDirect full, chunked;
    std::string error;
    REQUIRE(full.initialize(model.view,&error)); REQUIRE(chunked.initialize(model.view,&error));
    auto r=expected.request(); r.inference_only=true; r.deterministic_reduction=true;
    REQUIRE(full.compute_lammps(r,&error));
    auto c=actual.request(); c.inference_only=true; c.deterministic_reduction=true; c.centers_per_chunk=1;
    REQUIRE(chunked.compute_lammps(c,&error)); compare_standalone(expected,actual);
    c.compute_forces=false; c.global_virial=nullptr; c.site_virial=nullptr; c.force=nullptr;
    REQUIRE(chunked.compute_lammps(c,&error));
    REQUIRE(actual.total==Catch::Approx(expected.total).epsilon(1e-12).margin(1e-11));
    MTPGpuComputeRequest empty;
    double energy=7; std::array<double,6> virial; virial.fill(8);
    empty.inference_only=true; empty.total_energy=&energy; empty.global_virial=virial.data();
    REQUIRE(chunked.compute_lammps(empty,&error));
    REQUIRE(energy==0); for(double v:virial) REQUIRE(v==0);
}

TEST_CASE("Standalone reader transports every radial representation to device inference", "[standalone-gpu][radial]")
{
    require_standalone_device();
    namespace F=StandaloneRadialFixture;
    namespace R=MLIPCore::Radial;
    std::vector<std::pair<std::vector<std::string>,int>> cases;
    for(int size:{1,4}) {
        for(const auto *name:{"RBChebyshev","RBChebyCos","RBChebyshevPoly4","RBChebyshevPoly5","RBPolyB","RBPolyBCos"})cases.push_back({{F::block(name,size)},size});
        for(int cut=0;cut<12;++cut)for(int shift=0;shift<5;++shift)for(bool flip:{false,true})
            cases.push_back({{F::block("RBChebyshevConfigurable",size,cut,shift,flip)},size});
    }
    cases.push_back({{F::block("RBPolyB",4,0,0,false,.4,4.1),F::block("RBPolyBCos",4,0,0,false,.6,4.4),
        F::block("RBChebyshevConfigurable",4,11,4,true,.8,4.7),F::block("RBChebyshevPoly4")},4});
    for(const auto *name:{"RBPolyB","RBPolyBCos"})cases.push_back({{F::block(name,8)},8});
    for(const auto &item:cases)for(const std::string head:{"classic","li","q2","fs","n1","n2"}) {
        CAPTURE(item.first[0],head);
        int mapping[]={-1,0,1};double cutoff=0;
        const auto text=F::model(item.first,item.second,head);
        MLIPCore::Cpu::Model model(MLIPCore::setup_from_text(text,mapping,&cutoff,3));
        if(model->pair_radial_basis_count) {
            std::vector<MLIPCore::mlip::radial_basis_t *> pointers;
            for(int i=0;i<(item.first.size()==1?1:4);++i) {
                auto &b=model->pair_radial_bases[i];
                if(R::is_poly(b))for(auto &x:b.polynomial_matrix)x*=1.01;
                pointers.push_back(&b);
            }
            const auto deployed=text+R::cache_footer(pointers,item.first.size()!=1,2);
            model.reset(MLIPCore::setup_from_text(deployed,mapping,&cutoff,3));
        }
        StandaloneAtoms expected,actual;
        expected.type=actual.type={{1,2,1,2,1}};
        expected.x[1]=actual.x[1]={{.3,.1,0}}; // below min_dist
        expected.x[3]=actual.x[3]={{4.5,.2,0}}; // split-specific upper cutoffs
        MLIPCore::Cpu::Executor cpu(*model);
        MLIPCore::Cpu::Workspace workspace(*model);workspace.reserve(*model,4,MLIPCore::Cpu::DerivativeMode::Cached);
        for(int i:expected.centers) {
            const double energy=cpu.evaluate_site(i,4,expected.neighbors[i].data(),expected.type.data(),
                expected.xp.data(),mapping,true,true,workspace);
            expected.energy[i]=energy;expected.total+=energy;
            for(int e=0;e<workspace.edge_count();++e) {
                const int j=workspace.neighbor(e);const auto *gradient=workspace.gradient(e);
                for(int a=0;a<3;++a) {expected.force[i][a]+=gradient[a];expected.force[j][a]-=gradient[a];}
                double virial[6];MLIPOpt::standard_virial(workspace.displacement(e),gradient,virial);
                for(int a=0;a<6;++a)expected.global[a]+=virial[a];
                MLIPOpt::add_per_atom_virial(expected.virial[i].data(),expected.virial[j].data(),virial);
            }
        }
        MTPGpuDirect gpu;MTPGpuInitializationOptions options;std::string error;INFO(error);
        REQUIRE(gpu.initialize_from_mlip(*model,options,&error));
        // Also proves ownership: initialize copies matrices before this model goes away.
        model.reset();
        auto request=actual.request();request.inference_only=true;request.centers_per_chunk=2;
        REQUIRE(gpu.compute_lammps(request,&error));
        REQUIRE(actual.total==Catch::Approx(expected.total).margin(1e-9).epsilon(1e-9));
        for(int i=0;i<actual.n;++i) {
            for(int a=0;a<3;++a)REQUIRE(actual.force[i][a]==Catch::Approx(expected.force[i][a]).margin(1e-9).epsilon(1e-9));
            for(int a=0;a<6;++a)REQUIRE(actual.virial[i][a]==Catch::Approx(expected.virial[i][a]).margin(1e-9).epsilon(1e-9));
        }
        for(int i=0;i<actual.local;++i)REQUIRE(actual.energy[i]==Catch::Approx(expected.energy[i]).margin(1e-9).epsilon(1e-9));
        for(int a=0;a<6;++a)REQUIRE(actual.global[a]==Catch::Approx(expected.global[a]).margin(1e-9).epsilon(1e-9));
    }
}

#endif
