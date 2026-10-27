#include <catch2/catch_all.hpp>
#include "NNBase/nn2hidden.h"
#include "potentials/potential_mtp_m.h"
#include <array>
#include <thread>

using namespace IMRMLPNN;
using Catch::Approx;
namespace {
void initialize(NN2lay &nn, int ni=5, int n1=4, int n2=3, int threads=2) {
    nn.setup(ni,n1,n2,threads);
    std::vector<double> coefficients(nn.n_coeffs);
    for (int i=0; i<nn.n_coeffs; ++i) coefficients[i]=.23*std::sin(.7*(i+1))+.08*std::cos(.2*(i+1));
    nn.set_coeffs(coefficients.data());
    for (int i=0; i<ni; ++i) { nn.norm_mean[i]=.13*std::cos(i+1.); nn.norm_irstd[i]=.4+.17*i; }
}
std::pair<double,Vector> jacobian_reference(const NN2lay &nn,const Vector &input) {
    const Vector normalized=(input.array()-nn.norm_mean.array())*nn.norm_irstd.array();
    Vector h1(nn.n_neuron1),h2(nn.n_neuron2);
    Matrix j1(nn.n_neuron1,nn.n_input),j2(nn.n_neuron2,nn.n_neuron1);
    for (int i=0; i<nn.n_neuron1; ++i) {
        const double value=nn.w_input.col(i).dot(normalized)+nn.b_hidden1[i];
        h1[i]=activation_function(value,nn.acfunc);
        const double derivative=activation_derivative(value,h1[i],nn.acfunc);
        j1.row(i)=(derivative*nn.w_input.col(i)).array()*nn.norm_irstd.array();
    }
    for (int i=0; i<nn.n_neuron2; ++i) {
        const double value=nn.w_hidden1.col(i).dot(h1)+nn.b_hidden2[i];
        h2[i]=activation_function(value,nn.acfunc);
        j2.row(i)=activation_derivative(value,h2[i],nn.acfunc)*nn.w_hidden1.col(i).transpose();
    }
    const Matrix input_jacobian=j2*j1;
    return {nn.w_hidden2.dot(h2),(nn.w_hidden2.transpose()*input_jacobian).transpose()};
}
Eigen::Index matrix_scratch(const NN2lay::Buffer &buffer) {
    return buffer.dh1di.size()+buffer.dh2dh1.size()+buffer.dh2di.size()+buffer.dha2dq.size()
        +buffer.d2h1didi.size()+buffer.d2h2dh1dh1.size();
}
Eigen::Index vector_scratch(const NN2lay::Buffer &b) {
    return b.input.size()+b.hiddenlay1.size()+b.hiddenlay2.size()+b.deris1.size()+b.deris2.size()
        +b.secderis1.size()+b.secderis2.size()+b.dedh1.size()+b.dedh1deris1.size()+b.dedh2deris2.size();
}
std::array<const double*,8> vector_storage(const NN2lay::Buffer &b) {
    return {b.input.data(),b.hiddenlay1.data(),b.hiddenlay2.data(),b.deris1.data(),b.deris2.data(),
            b.dedh1.data(),b.dedh1deris1.data(),b.dedh2deris2.data()};
}
}

TEST_CASE("NN2 reverse input gradient matches the matrix chain and independent energy differences", "[nn2][low-memory]") {
    const auto activation=GENERATE(TANH,RELU,SIGMOID);
    const bool wide=GENERATE(false,true);
    NN2lay nn; initialize(nn,wide ? 3 : 7,wide ? 7 : 4,wide ? 4 : 2); nn.acfunc=activation;
    Vector input(nn.n_input);
    for (int i=0; i<input.size(); ++i) input[i]=.31*std::sin(.4*(i+1));
    const auto expected=jacobian_reference(nn,input);
    int tag=0; double energy; Vector gradient;
    nn.propagate(input,energy,gradient,tag,0);
    REQUIRE(energy == Approx(expected.first).margin(1e-14));
    REQUIRE(gradient.isApprox(expected.second,2e-13));
    for (const double step : {1e-5,3e-6}) for (int i=0; i<input.size(); ++i) {
        auto plus=input,minus=input; plus[i]+=step; minus[i]-=step;
        double ep,em; nn.propagate(plus,ep,tag,1); nn.propagate(minus,em,tag,1);
        REQUIRE(gradient[i] == Approx((ep-em)/(2*step)).margin(2e-9));
    }
    // Input/output aliasing remains safe because normalization precedes contraction.
    Vector alias=input; nn.propagate(alias,energy,alias,tag,0);
    REQUIRE(alias.isApprox(expected.second,2e-13));
}

TEST_CASE("NN2 gradient-only thread scratch grows linearly without Jacobian or tensor storage", "[nn2][low-memory]") {
    NN2lay nn; initialize(nn,128,64,32,3);
    for (const auto &b : nn.buffers) REQUIRE(matrix_scratch(b)==0);
    Vector input=Vector::LinSpaced(128,-.3,.4),gradient;
    int tag=0; double energy; nn.propagate(input,energy,gradient,tag,0);
    const auto storage=vector_storage(nn.buffers[0]);
    const Eigen::Index resident=vector_scratch(nn.buffers[0]);
    REQUIRE(resident==nn.n_input+4*nn.n_neuron1+3*nn.n_neuron2);
    const Eigen::Index previous_matrices=nn.n_neuron1*nn.n_input+nn.n_neuron2*nn.n_neuron1+nn.n_neuron2*nn.n_input;
    REQUIRE(resident<previous_matrices/10);
    for (int i=0; i<20; ++i) {
        input[i]+=.01; nn.propagate(input,energy,gradient,tag,0);
        REQUIRE(vector_storage(nn.buffers[0])==storage);
        REQUIRE(vector_scratch(nn.buffers[0])==resident);
        REQUIRE(matrix_scratch(nn.buffers[0])==0);
    }
    nn.resize_omp_buffers(4);
    REQUIRE(matrix_scratch(nn.buffers[3])==0);
    REQUIRE(vector_scratch(nn.buffers[3])==nn.n_input+nn.n_neuron1+nn.n_neuron2);
}

TEST_CASE("NN2 higher derivatives and parameter mixed outputs allocate lazily with exact shapes", "[nn2][low-memory]") {
    NN2lay nn; initialize(nn,3,4,2);
    Vector input(3); input << .2,-.3,.4;
    int tag=0; double energy; Vector gradient; Matrix hessian; Tensor3 third;
    REQUIRE(matrix_scratch(nn.buffers[0])==0);
    nn.propagate(input,energy,gradient,hessian,tag,0);
    REQUIRE(hessian.rows()==3); REQUIRE(hessian.cols()==3);
    REQUIRE(nn.buffers[0].dh1di.rows()==4); REQUIRE(nn.buffers[0].dh1di.cols()==3);
    REQUIRE(nn.buffers[0].dh2dh1.rows()==2); REQUIRE(nn.buffers[0].dh2dh1.cols()==4);
    REQUIRE(matrix_scratch(nn.buffers[1])==0);
    for (int i=0; i<3; ++i) {
        const double step=1e-5; auto plus=input,minus=input; plus[i]+=step; minus[i]-=step;
        Vector gp,gm; Matrix hp,hm; double ep,em;
        nn.propagate(plus,ep,gp,hp,tag,1); nn.propagate(minus,em,gm,hm,tag,1);
        REQUIRE(hessian.col(i).isApprox((gp-gm)/(2*step),2e-7));
    }
    Vector parameters; Matrix mixed,combined_hessian; Vector combined_gradient;
    nn.backpropgate(input,energy,combined_gradient,parameters,mixed,combined_hessian,tag,0);
    REQUIRE(parameters.size()==nn.n_coeffs);
    REQUIRE(mixed.rows()==3); REQUIRE(mixed.cols()==nn.n_coeffs);
    REQUIRE(combined_hessian.isApprox(hessian,1e-13));
    REQUIRE(combined_gradient.isApprox(gradient,1e-13));
    Vector coefficients(nn.n_coeffs); nn.get_coeffs(coefficients.data());
    for (int i=0; i<nn.n_coeffs; i+=5) {
        const double step=1e-5; auto plus=coefficients,minus=coefficients; plus[i]+=step; minus[i]-=step;
        Vector gp,gm; double ep,em;
        nn.set_coeffs(plus.data()); nn.propagate(input,ep,gp,tag,1);
        nn.set_coeffs(minus.data()); nn.propagate(input,em,gm,tag,1);
        nn.set_coeffs(coefficients.data());
        REQUIRE(parameters[i]==Approx((ep-em)/(2*step)).margin(2e-9));
        REQUIRE(mixed.col(i).isApprox((gp-gm)/(2*step),2e-6));
    }
}

TEST_CASE("NN2 gradient buffers remain independent across concurrent and interleaved requests", "[nn2][low-memory]") {
    NN2lay nn; initialize(nn,9,7,5,2);
    Vector first=Vector::LinSpaced(9,-.7,.5),second=Vector::LinSpaced(9,.4,-.6);
    const auto expected1=jacobian_reference(nn,first),expected2=jacobian_reference(nn,second);
    std::array<Vector,2> gradients; std::array<double,2> energies;
    auto compute=[&](int slot) {
        int tag=slot;
        for (int i=0; i<30; ++i) nn.propagate(slot ? second : first,energies[slot],gradients[slot],tag,slot);
    };
    std::thread one(compute,0),two(compute,1); one.join(); two.join();
    REQUIRE(gradients[0].isApprox(expected1.second,2e-13));
    REQUIRE(gradients[1].isApprox(expected2.second,2e-13));
    REQUIRE(energies[0]==Approx(expected1.first).margin(1e-14));
    REQUIRE(energies[1]==Approx(expected2.first).margin(1e-14));
    REQUIRE(matrix_scratch(nn.buffers[0])==0); REQUIRE(matrix_scratch(nn.buffers[1])==0);
    Matrix hessian; int tag=0; double energy;
    nn.propagate(first,energy,gradients[0],hessian,tag,0);
    nn.propagate(second,energies[1],gradients[1],tag,1);
    REQUIRE(matrix_scratch(nn.buffers[1])==0);
    REQUIRE(gradients[1].isApprox(expected2.second,2e-13));
}

TEST_CASE("MTP-M nn2 wrapper uses the same low-memory input-gradient contraction", "[nn2][mtp_m][low-memory]") {
    Potentials::m_model_nn2 model; GPSettings settings;
    settings.Modify("n_input","5"); settings.Modify("n_neuron","4"); settings.Modify("n_neuron2","3");
    model.init(settings);
    std::vector<double> coefficients(model.get_ncoeffs());
    for (int i=0; i<coefficients.size(); ++i) coefficients[i]=.2*std::sin(.7*(i+1));
    model.coeffs=coefficients.data(); model.postinit();
    model.nn.norm_irstd=Vector::LinSpaced(5,.5,1.3);
    const Vector input=Vector::LinSpaced(5,-.3,.7); const auto expected=jacobian_reference(model.nn,input);
    int tag=0,thread=0; double energy; Vector gradient;
    model.propagate(input,energy,gradient,tag,thread);
    REQUIRE(energy==Approx(expected.first).margin(1e-14));
    REQUIRE(gradient.isApprox(expected.second,2e-13));
    REQUIRE(matrix_scratch(model.nn.buffers[0])==0);
}
