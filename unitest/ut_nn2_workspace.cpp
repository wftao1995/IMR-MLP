#include <catch2/catch_all.hpp>
#include "NNBase/nn2hidden.h"
#include <array>
#include <thread>

using namespace IMRMLPNN;
using Catch::Approx;
namespace {
Vector workspace_coefficients(const NN2lay &nn, double phase) {
    Vector result(nn.n_coeffs);
    for (int i=0; i<result.size(); ++i) result[i]=.21*std::sin(.37*i+phase);
    return result;
}
void workspace_network(NN2lay &nn) {
    nn.setup(3,5,2,1);
    nn.norm_mean=Vector::LinSpaced(3,-.23,.18);
    nn.norm_irstd=Vector::LinSpaced(3,.57,1.31);
    const auto owned=workspace_coefficients(nn,.7);
    nn.set_coeffs(owned.data());
}
struct WorkspaceResult {
    double energy=0.;
    Vector gradient, parameters;
    Matrix hessian,mixed;
    Tensor3 third;
};
WorkspaceResult workspace_evaluate(const NN2lay &nn, const Vector &input, const Vector &coefficients,
                                   NN2lay::Buffer &buffer, int order) {
    WorkspaceResult result; int tag=0;
    const auto view=nn.coefficient_view(coefficients.data());
    if(order==0) nn.propagate(input,result.energy,tag,view,buffer);
    if(order==1) nn.propagate(input,result.energy,result.gradient,tag,view,buffer);
    if(order==2) nn.propagate(input,result.energy,result.gradient,result.hessian,tag,view,buffer);
    if(order==4) nn.backpropgate(input,result.energy,result.parameters,tag,view,buffer);
    if(order==5) nn.backpropgate(input,result.energy,result.gradient,result.parameters,tag,view,buffer);
    if(order==6) nn.backpropgate(input,result.energy,result.parameters,result.mixed,tag,view,buffer);
    if(order==7) nn.backpropgate(input,result.energy,result.gradient,result.parameters,result.mixed,tag,view,buffer);
    if(order==8) nn.backpropgate(input,result.energy,result.gradient,result.parameters,result.mixed,result.hessian,tag,view,buffer);
    return result;
}
WorkspaceResult legacy_evaluate(NN2lay &nn,const Vector &input,int order) {
    WorkspaceResult result; int tag=0;
    if(order==0) nn.propagate(input,result.energy,tag,0);
    if(order==1) nn.propagate(input,result.energy,result.gradient,tag,0);
    if(order==2) nn.propagate(input,result.energy,result.gradient,result.hessian,tag,0);
    if(order==4) nn.backpropgate(input,result.energy,result.parameters,tag,0);
    if(order==5) nn.backpropgate(input,result.energy,result.gradient,result.parameters,tag,0);
    if(order==6) nn.backpropgate(input,result.energy,result.parameters,result.mixed,tag,0);
    if(order==7) nn.backpropgate(input,result.energy,result.gradient,result.parameters,result.mixed,tag,0);
    if(order==8) nn.backpropgate(input,result.energy,result.gradient,result.parameters,result.mixed,result.hessian,tag,0);
    return result;
}
void check_workspace_result(const WorkspaceResult &actual,const WorkspaceResult &expected) {
    REQUIRE(actual.energy==Approx(expected.energy).margin(1e-14));
    REQUIRE((actual.gradient-expected.gradient).norm()<1e-13);
    REQUIRE((actual.parameters-expected.parameters).norm()<1e-13);
    REQUIRE((actual.hessian-expected.hessian).norm()<1e-13);
    REQUIRE((actual.mixed-expected.mixed).norm()<1e-13);
    REQUIRE(actual.third.size()==expected.third.size());
    for (Eigen::Index i=0;i<actual.third.size();++i)
        REQUIRE(actual.third.data()[i]==Approx(expected.third.data()[i]).margin(1e-13));
}
}

TEST_CASE("NN2 explicit coefficient views preserve every legacy derivative overload", "[nn2][workspace]") {
    const auto activation=GENERATE(TANH,SIGMOID,RELU);
    NN2lay nn,reference; workspace_network(nn); workspace_network(reference);
    nn.acfunc=reference.acfunc=activation;
    const Vector input=Vector::LinSpaced(3,-.41,.32);
    auto coefficients=workspace_coefficients(nn,-.9);
    auto buffer=nn.make_buffer();
    Vector owned_before(nn.n_coeffs); nn.get_coeffs(owned_before.data());
    const auto mean=nn.norm_mean, inverse_std=nn.norm_irstd;
    for (int order=0;order<9;++order) {
        if (order == 3) continue;
        CAPTURE(activation,order);
        reference.set_coeffs(coefficients.data());
        check_workspace_result(workspace_evaluate(nn,input,coefficients,buffer,order),legacy_evaluate(reference,input,order));
    }
    // A view reads current backing coefficients without set_coeffs or postinit.
    coefficients[0]+=.17;
    reference.set_coeffs(coefficients.data());
    check_workspace_result(workspace_evaluate(nn,input,coefficients,buffer,8),legacy_evaluate(reference,input,8));
    Vector owned_after(nn.n_coeffs); nn.get_coeffs(owned_after.data());
    REQUIRE(owned_after==owned_before);
    REQUIRE(nn.norm_mean==mean); REQUIRE(nn.norm_irstd==inverse_std);
    REQUIRE(nn.buffers.size()==1);
    REQUIRE(nn.buffers[0].dh1di.size()==0);
    REQUIRE_THROWS(nn.coefficient_view(nullptr));
}

TEST_CASE("NN2 explicit input-gradient scratch remains linear and reuses warm storage", "[nn2][workspace][low-memory]") {
    NN2lay nn; nn.setup(128,64,32,0);
    auto coefficients=workspace_coefficients(nn,.4);
    auto buffer=nn.make_buffer();
    const Vector input=Vector::LinSpaced(128,-.3,.4);
    auto result=workspace_evaluate(nn,input,coefficients,buffer,1);
    const std::array<const double*,8> storage={buffer.input.data(),buffer.hiddenlay1.data(),buffer.hiddenlay2.data(),
        buffer.deris1.data(),buffer.deris2.data(),buffer.dedh1.data(),buffer.dedh1deris1.data(),buffer.dedh2deris2.data()};
    for(int iteration=0;iteration<12;++iteration) {
        workspace_evaluate(nn,input,coefficients,buffer,1);
        const std::array<const double*,8> current={buffer.input.data(),buffer.hiddenlay1.data(),buffer.hiddenlay2.data(),
            buffer.deris1.data(),buffer.deris2.data(),buffer.dedh1.data(),buffer.dedh1deris1.data(),buffer.dedh2deris2.data()};
        REQUIRE(current==storage);
        REQUIRE(buffer.dh1di.size()+buffer.dh2dh1.size()+buffer.dh2di.size()+buffer.dha2dq.size()==0);
        REQUIRE(buffer.d2h1didi.size()+buffer.d2h2dh1dh1.size()==0);
    }
    REQUIRE(buffer.input.size()+buffer.hiddenlay1.size()+buffer.hiddenlay2.size()+buffer.deris1.size()+buffer.deris2.size()
        +buffer.dedh1.size()+buffer.dedh1deris1.size()+buffer.dedh2deris2.size()==128+4*64+3*32);
    REQUIRE(nn.buffers.empty());
}

TEST_CASE("NN2 const model supports concurrent mixed and curvature requests with caller scratch", "[nn2][workspace]") {
    NN2lay nn; workspace_network(nn);
    const NN2lay &read_only=nn;
    const std::array<Vector,2> inputs={Vector::LinSpaced(3,-.3,.4),Vector::LinSpaced(3,.7,-.2)};
    const std::array<Vector,2> coefficients={workspace_coefficients(nn,.1),workspace_coefficients(nn,-.6)};
    std::array<std::array<WorkspaceResult,9>,2> expected,actual;
    for(int slot=0;slot<2;++slot) {
        auto buffer=nn.make_buffer();
        for(int order=0;order<9;++order) expected[slot][order]=workspace_evaluate(nn,inputs[slot],coefficients[slot],buffer,order);
    }
    auto evaluate=[&](int slot) {
        auto buffer=read_only.make_buffer();
        for(int iteration=0;iteration<8;++iteration)
            for(int order=0;order<9;++order)
                actual[slot][order]=workspace_evaluate(read_only,inputs[slot],coefficients[slot],buffer,order);
    };
    std::thread first(evaluate,0),second(evaluate,1); first.join();second.join();
    for(int slot=0;slot<2;++slot) for(int order=0;order<9;++order) check_workspace_result(actual[slot][order],expected[slot][order]);
}

TEST_CASE("NN2 external coefficient derivatives match independent input and parameter differences", "[nn2][workspace]") {
    NN2lay nn; workspace_network(nn);
    const NN2lay &read_only=nn;
    const Vector input=Vector::LinSpaced(3,-.41,.32);
    const auto coefficients=workspace_coefficients(nn,-.9);
    auto buffer=read_only.make_buffer();
    const auto full=workspace_evaluate(read_only,input,coefficients,buffer,8);
    for(double step : {2e-5,6e-6}) {
        CAPTURE(step);
        for(int axis=0;axis<3;++axis) {
            Vector plus=input,minus=input;plus[axis]+=step;minus[axis]-=step;
            const auto p=workspace_evaluate(read_only,plus,coefficients,buffer,2);
            const auto m=workspace_evaluate(read_only,minus,coefficients,buffer,2);
            REQUIRE(full.gradient[axis]==Approx((p.energy-m.energy)/(2*step)).margin(2e-9));
            REQUIRE((full.hessian.col(axis)-(p.gradient-m.gradient)/(2*step)).norm()<2e-8);
        }
        for(int parameter=0;parameter<coefficients.size();parameter+=7) {
            Vector plus=coefficients,minus=coefficients;plus[parameter]+=step;minus[parameter]-=step;
            const auto p=workspace_evaluate(read_only,input,plus,buffer,1);
            const auto m=workspace_evaluate(read_only,input,minus,buffer,1);
            REQUIRE(full.parameters[parameter]==Approx((p.energy-m.energy)/(2*step)).margin(2e-9));
            REQUIRE((full.mixed.col(parameter)-(p.gradient-m.gradient)/(2*step)).norm()<2e-8);
        }
    }
}
