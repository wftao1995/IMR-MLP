// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#pragma once

#include "external/spglib_driver.h"
#include "potential_mtp.h"
#include "les.h"
#include "sog_model.h"
#include "NNBase/nn2hidden.h"
#include "utils/compressed_stream.h"

#include <algorithm>
#include <array>
#include <numeric>
#include <cctype>
#include <type_traits>

// ext_xxx won't be initialized to zero in this class !
// Supports optional LES/SOG long-range heads and their parameter gradients.

namespace Potentials {

/**
 * class T should have:
 * model.init(GPSettings &settings);
 * model.propagate(Qs, site_energy, nbh.centertype, ithread);
 * model.propagate(Qs, site_energy, ext_dUdQ, nbh.centertype, ithread);
 * model.backpropagate(Qs, site_energy, ext_dUdt, nbh.centertype, ithread);
 * model.backpropagate(Qs, site_energy, ext_dUdQ, ext_dUdt, nbh.centertype, ithread);
 * model.backpropagate(Qs, site_energy, ext_dUdt, ext_d2UdQdt, nbh.centertype, ithread);
 * model.backpropagate(Qs, site_energy, ext_dUdQ, ext_dUdt, ext_d2UdQdt, ext_d2UdQdQ, nbh.centertype, ithread);
 * int get_nprops();
 * int get_ncoeffs();
 */

// E = c1 * q1 + c2 * q2 + c11 * q1^2 + c22 * q2^2 + c12 * q1 * q2
// {c1, c2, c11, c22, c12}
struct m_model_mtpq2 {
    double* coeffs = nullptr; // coeffs handled externally
    m_model_mtpq2()
    {
        // for (int i = 0; i < 5; ++i)
        //     coeffs[i] = 1;
    }
    void init(GPSettings& settings)
    {
        ;
    }
    void postinit() {};
    int get_nprops()
    {
        return 2;
    }
    int get_ncoeffs()
    {
        return 5;
    }
    void add_loss_gard(double& loss, double* gard, int offset)
    {
        ;
    }
    void random_init(double lb, double ub)
    {
        coeffs[0] = 1;
        std::uniform_real_distribution<double> urd1(0, 0.1);
        std::uniform_real_distribution<double> urd2(-0.1, 0.1);
        coeffs[1] = urd2(RNG.mtd);
        coeffs[2] = urd1(RNG.mtd);
        coeffs[3] = urd1(RNG.mtd);
        coeffs[4] = urd2(RNG.mtd);
    }
    void propagate(const Eigen::VectorXd& Qs, double& site_energy, int& tag, int& ithread)
    {
        const double& q1 = Qs[0];
        const double& q2 = Qs[1];
        const double& c1 = coeffs[0];
        const double& c2 = coeffs[1];
        const double& c11 = coeffs[2];
        const double& c22 = coeffs[3];
        const double& c12 = coeffs[4];
        site_energy = c1 * q1 + c2 * q2 + c11 * q1 * q1 + c22 * q2 * q2 + c12 * q1 * q2;
    }
    void propagate(const Eigen::VectorXd& Qs, double& site_energy, Eigen::VectorXd& dUdQ, int& tag, int& ithread)
    {
        const double& q1 = Qs[0];
        const double& q2 = Qs[1];
        const double& c1 = coeffs[0];
        const double& c2 = coeffs[1];
        const double& c11 = coeffs[2];
        const double& c22 = coeffs[3];
        const double& c12 = coeffs[4];
        site_energy = c1 * q1 + c2 * q2 + c11 * q1 * q1 + c22 * q2 * q2 + c12 * q1 * q2;
        dUdQ[0] = c1 + 2 * c11 * q1 + c12 * q2;
        dUdQ[1] = c2 + 2 * c22 * q2 + c12 * q1;
    }
    void propagate(const Eigen::VectorXd& Qs, double& site_energy, Eigen::VectorXd& dUdQ, Eigen::MatrixXd& d2UdQdQ, int& tag, int& ithread)
    {
        const double& q1 = Qs[0];
        const double& q2 = Qs[1];
        const double& c1 = coeffs[0];
        const double& c2 = coeffs[1];
        const double& c11 = coeffs[2];
        const double& c22 = coeffs[3];
        const double& c12 = coeffs[4];
        site_energy = c1 * q1 + c2 * q2 + c11 * q1 * q1 + c22 * q2 * q2 + c12 * q1 * q2;
        dUdQ[0] = c1 + 2 * c11 * q1 + c12 * q2;
        dUdQ[1] = c2 + 2 * c22 * q2 + c12 * q1;
        d2UdQdQ(0, 0) = 2 * c11;
        d2UdQdQ(1, 1) = 2 * c22;
        d2UdQdQ(0, 1) = c12;
        d2UdQdQ(1, 0) = c12;
    }

    void backpropagate(const Eigen::VectorXd& Qs, double& site_energy, Eigen::VectorXd& dUdt, int& tag, int& ithread)
    {
        const double& q1 = Qs[0];
        const double& q2 = Qs[1];
        const double& c1 = coeffs[0];
        const double& c2 = coeffs[1];
        const double& c11 = coeffs[2];
        const double& c22 = coeffs[3];
        const double& c12 = coeffs[4];
        site_energy = c1 * q1 + c2 * q2 + c11 * q1 * q1 + c22 * q2 * q2 + c12 * q1 * q2;
        dUdt[0] = q1;
        dUdt[1] = q2;
        dUdt[2] = q1 * q1;
        dUdt[3] = q2 * q2;
        dUdt[4] = q1 * q2;
    }
    void backpropagate(const Eigen::VectorXd& Qs, double& site_energy, Eigen::VectorXd& dUdQ, Eigen::VectorXd& dUdt, int& tag, int& ithread)
    {
        const double& q1 = Qs[0];
        const double& q2 = Qs[1];
        const double& c1 = coeffs[0];
        const double& c2 = coeffs[1];
        const double& c11 = coeffs[2];
        const double& c22 = coeffs[3];
        const double& c12 = coeffs[4];
        site_energy = c1 * q1 + c2 * q2 + c11 * q1 * q1 + c22 * q2 * q2 + c12 * q1 * q2;
        dUdt[0] = q1;
        dUdt[1] = q2;
        dUdt[2] = q1 * q1;
        dUdt[3] = q2 * q2;
        dUdt[4] = q1 * q2;
        dUdQ[0] = c1 + 2 * c11 * q1 + c12 * q2;
        dUdQ[1] = c2 + 2 * c22 * q2 + c12 * q1;
    }
    void backpropagate(const Eigen::VectorXd& Qs, double& site_energy, Eigen::VectorXd& dUdt, Eigen::MatrixXd& d2UdQdt, int& tag, int& ithread)
    {
        const double& q1 = Qs[0];
        const double& q2 = Qs[1];
        const double& c1 = coeffs[0];
        const double& c2 = coeffs[1];
        const double& c11 = coeffs[2];
        const double& c22 = coeffs[3];
        const double& c12 = coeffs[4];
        site_energy = c1 * q1 + c2 * q2 + c11 * q1 * q1 + c22 * q2 * q2 + c12 * q1 * q2;
        dUdt[0] = q1;
        dUdt[1] = q2;
        dUdt[2] = q1 * q1;
        dUdt[3] = q2 * q2;
        dUdt[4] = q1 * q2;
        d2UdQdt.setZero();
        // (0, x) = 0
        // q1, c1 = 1
        d2UdQdt(0, 0) = 1;
        // q1, c11 = 2q1
        d2UdQdt(0, 2) = 2 * q1;
        // q1, c12 = q2
        d2UdQdt(0, 4) = q2;
        // q2, c2 = 1
        d2UdQdt(1, 1) = 1;
        d2UdQdt(1, 3) = 2 * q2;
        d2UdQdt(1, 4) = q1;
    }
    void backpropagate(const Eigen::VectorXd& Qs, double& site_energy, Eigen::VectorXd& dUdQ,
        Eigen::VectorXd& dUdt, Eigen::MatrixXd& d2UdQdt, Eigen::MatrixXd& d2UdQdQ,
        int& tag, int& ithread)
    {
        const double& q1 = Qs[0];
        const double& q2 = Qs[1];
        const double& c1 = coeffs[0];
        const double& c2 = coeffs[1];
        const double& c11 = coeffs[2];
        const double& c22 = coeffs[3];
        const double& c12 = coeffs[4];
        site_energy = c1 * q1 + c2 * q2 + c11 * q1 * q1 + c22 * q2 * q2 + c12 * q1 * q2;
        dUdt[0] = q1;
        dUdt[1] = q2;
        dUdt[2] = q1 * q1;
        dUdt[3] = q2 * q2;
        dUdt[4] = q1 * q2;
        d2UdQdt.setZero();
        // (0, x) = 0
        // q1, c1 = 1
        d2UdQdt(0, 0) = 1;
        // q1, c11 = 2q1
        d2UdQdt(0, 2) = 2 * q1;
        // q1, c12 = q2
        d2UdQdt(0, 4) = q2;
        // q2, c2 = 1
        d2UdQdt(1, 1) = 1;
        d2UdQdt(1, 3) = 2 * q2;
        d2UdQdt(1, 4) = q1;
        dUdQ[0] = c1 + 2 * c11 * q1 + c12 * q2;
        dUdQ[1] = c2 + 2 * c22 * q2 + c12 * q1;
        d2UdQdQ(0, 0) = 2 * c11;
        d2UdQdQ(1, 1) = 2 * c22;
        d2UdQdQ(0, 1) = c12;
        d2UdQdQ(1, 0) = c12;
    }
};
// E = q1 + c0 ArcSinh[ q2 ]
struct m_model_mtpfs {
    double* coeffs; // coeffs handled externally
    m_model_mtpfs()
    {
        // for (int i = 0; i < 2; ++i)
        //     coeffs[i] = 1;
    }
    void init(GPSettings& settings)
    {
        ;
    }
    void postinit() {};
    void random_init(double lb, double ub)
    {
        for (int i = 0; i < 1; ++i) {
            coeffs[i] = 1;
        }
    }
    int get_nprops()
    {
        return 2;
    }
    int get_ncoeffs()
    {
        return 1;
    }
    void add_loss_gard(double& loss, double* gard, int offset)
    {
        ;
    }
    void random_init() { };
    void propagate(const Eigen::VectorXd& Qs, double& site_energy, int& tag, int& ithread)
    {
        const double& q0 = Qs[0];
        const double& q1 = Qs[1];
        const double& c0 = coeffs[0];
        site_energy = q0 + c0 * asinh(q1);
    }
    void propagate(const Eigen::VectorXd& Qs, double& site_energy, Eigen::VectorXd& dUdQ, int& tag, int& ithread)
    {
        const double& q0 = Qs[0];
        const double& q1 = Qs[1];
        const double& c0 = coeffs[0];
        site_energy = q0 + c0 * asinh(q1);
        dUdQ[0] = 1;
        dUdQ[1] = c0 / sqrt(1. + q1 * q1);
    }
    void propagate(const Eigen::VectorXd& Qs, double& site_energy, Eigen::VectorXd& dUdQ, Eigen::MatrixXd& d2UdQdQ, int& tag, int& ithread)
    {
        const double& q0 = Qs[0];
        const double& q1 = Qs[1];
        const double& c0 = coeffs[0];
        double r = 1.0 / sqrt(1. + q1 * q1);
        site_energy = q0 + c0 * asinh(q1);
        dUdQ[0] = 1;
        dUdQ[1] = c0 * r;
        d2UdQdQ.setZero();
        d2UdQdQ(1, 1) = -c0 * q1 * r * r * r;
    }

    void backpropagate(const Eigen::VectorXd& Qs, double& site_energy, Eigen::VectorXd& dUdt, int& tag, int& ithread)
    {
        const double& q0 = Qs[0];
        const double& q1 = Qs[1];
        const double& c0 = coeffs[0];
        double f = asinh(q1);
        site_energy = q0 + c0 * f;
        dUdt[0] = f;
    }
    void backpropagate(const Eigen::VectorXd& Qs, double& site_energy, Eigen::VectorXd& dUdQ, Eigen::VectorXd& dUdt, int& tag, int& ithread)
    {
        const double& q0 = Qs[0];
        const double& q1 = Qs[1];
        const double& c0 = coeffs[0];
        double f = asinh(q1);
        double s = c0 / sqrt(1. + q1 * q1);
        site_energy = q0 + c0 * f;
        dUdt[0] = f;
        dUdQ[0] = 1;
        dUdQ[1] = s;
    }
    void backpropagate(const Eigen::VectorXd& Qs, double& site_energy, Eigen::VectorXd& dUdt, Eigen::MatrixXd& d2UdQdt, int& tag, int& ithread)
    {
        const double& q0 = Qs[0];
        const double& q1 = Qs[1];
        const double& c0 = coeffs[0];
        double f = asinh(q1);
        double r = 1. / sqrt(1. + q1 * q1);
        site_energy = q0 + c0 * f;
        dUdt[0] = f;
        d2UdQdt(0, 0) = 0;
        d2UdQdt(1, 0) = r;
    }
    void backpropagate(const Eigen::VectorXd& Qs, double& site_energy, Eigen::VectorXd& dUdQ,
        Eigen::VectorXd& dUdt, Eigen::MatrixXd& d2UdQdt, Eigen::MatrixXd& d2UdQdQ,
        int& tag, int& ithread)
    {
        const double& q0 = Qs[0];
        const double& q1 = Qs[1];
        const double& c0 = coeffs[0];
        double f = asinh(q1);
        double r = 1. / sqrt(1. + q1 * q1);
        double r3 = r * r * r;
        site_energy = q0 + c0 * f;
        dUdt[0] = f;
        d2UdQdt(0, 0) = 0;
        d2UdQdt(1, 0) = r;
        dUdQ[0] = 1;
        dUdQ[1] = c0 * r;
        d2UdQdQ.setZero();
        d2UdQdQ(1, 1) = -c0 * q1 * r3;
    }
};
// Linear model
struct m_model_mtpli {
    double* coeffs; // coeffs handled externally
    m_model_mtpli() {}
    void init(GPSettings& settings)
    {
        ;
    }
    void postinit() {};
    void random_init(double lb, double ub) {}
    int get_nprops()
    {
        return 1;
    }
    int get_ncoeffs()
    {
        return 0;
    }
    void add_loss_gard(double& loss, double* gard, int offset)
    {
        ;
    }
    void random_init() { };
    void propagate(const Eigen::VectorXd& Qs, double& site_energy, int& tag, int& ithread)
    {
        const double& q0 = Qs[0];
        site_energy = q0;
    }
    void propagate(const Eigen::VectorXd& Qs, double& site_energy, Eigen::VectorXd& dUdQ, int& tag, int& ithread)
    {
        const double& q0 = Qs[0];
        site_energy = q0;
        dUdQ[0] = 1;
    }
    void propagate(const Eigen::VectorXd& Qs, double& site_energy, Eigen::VectorXd& dUdQ, Eigen::MatrixXd& d2UdQdQ, int& tag, int& ithread)
    {
        const double& q0 = Qs[0];
        site_energy = q0;
        dUdQ[0] = 1;
        d2UdQdQ(0, 0) = 0;
    }


    void backpropagate(const Eigen::VectorXd& Qs, double& site_energy, Eigen::VectorXd& dUdt, int& tag, int& ithread)
    {
        const double& q0 = Qs[0];
        site_energy = q0;
    }
    void backpropagate(const Eigen::VectorXd& Qs, double& site_energy, Eigen::VectorXd& dUdQ, Eigen::VectorXd& dUdt, int& tag, int& ithread)
    {
        const double& q0 = Qs[0];
        site_energy = q0 ;
        dUdQ[0] = 1;
    }
    void backpropagate(const Eigen::VectorXd& Qs, double& site_energy, Eigen::VectorXd& dUdt, Eigen::MatrixXd& d2UdQdt, int& tag, int& ithread)
    {
        const double& q0 = Qs[0];
        site_energy = q0;
    }
    void backpropagate(const Eigen::VectorXd& Qs, double& site_energy, Eigen::VectorXd& dUdQ,
        Eigen::VectorXd& dUdt, Eigen::MatrixXd& d2UdQdt, Eigen::MatrixXd& d2UdQdQ,
        int& tag, int& ithread)
    {
        const double& q0 = Qs[0];
        site_energy = q0;
        dUdQ[0] = 1;
        d2UdQdQ.setZero();
    }
};
// NN 1 layer model
typedef double Scalar;
typedef Eigen::MatrixXd Matrix;
typedef Eigen::VectorXd Vector;
// --- Activation Function Definitions ---
// Using structs with static methods for clean grouping and performance.
struct TanhActivation {
    typedef double Scalar;
    static inline Scalar eval(Scalar x) { return tanh(x); }
    static inline Scalar derivative(Scalar, Scalar fx) { return 1.0 - fx * fx; }
    static inline Scalar second_derivative(Scalar, Scalar fx, Scalar dfx) { return -2.0 * fx * dfx; }
};

struct ReLUActivation {
    typedef double Scalar;
    static inline Scalar eval(Scalar x) { return std::max(Scalar(0), x); }
    static inline Scalar derivative(Scalar x, Scalar) { return x > 0 ? 1.0 : 0.0; }
    static inline Scalar second_derivative(Scalar, Scalar, Scalar) { return 0.0; } // Note: derivative is undefined at x=0
};

struct SigmoidActivation {
    typedef double Scalar;
    static inline Scalar eval(Scalar x) { return 1.0 / (1.0 + exp(-x)); }
    static inline Scalar derivative(Scalar, Scalar fx) { return fx * (1.0 - fx); }
    static inline Scalar second_derivative(Scalar, Scalar fx, Scalar dfx) { return dfx * (1.0 - 2.0 * fx); }
};
// --- End Activation Functions ---
struct m_model_nn1 {
    // --- Function Pointers for Activation ---
    // These pointers will be set during initialization to avoid switch statements in loops.
    using ActivationFuncPtr = Scalar (*)(Scalar);
    using DerivativeFuncPtr = Scalar (*)(Scalar, Scalar);
    using SecondDerivativeFuncPtr = Scalar (*)(Scalar, Scalar, Scalar);

    ActivationFuncPtr activation_eval = nullptr;
    DerivativeFuncPtr activation_deriv = nullptr;
    SecondDerivativeFuncPtr activation_deriv2 = nullptr;

    enum ActivationFunction {
        TANH,
        RELU,
        SIGMOID,
        LINEAR
    };

    double* coeffs; // coeffs handled externally
    ActivationFunction acfunc_enum = TANH;

    int n_coeffs; // Total number of coefficients (weights + biases)
    int n_weights; // Number of weights
    int n_bias; // Number of biases
    int n_neuron = 10; // Number of hidden neurons
    int n_input = 5; // Number of input elements

    // Using pointer for delaied initialization
    std::unique_ptr<Eigen::Map<Matrix>> w_input_map;
    std::unique_ptr<Eigen::Map<Vector>> w_hidden_map;
    std::unique_ptr<Eigen::Map<Vector>> b_hidden_map;
    
    // For quick reference
    Eigen::Map<Matrix>& w_input() { return *w_input_map; }
    Eigen::Map<Vector>& w_hidden() { return *w_hidden_map; }
    Eigen::Map<Vector>& b_hidden() { return *b_hidden_map; }

    // Buffers for intermediate calculations
    Vector hiddenlay;     // Hidden layer values after activation
    Vector hiddenlay_pre; // Hidden layer values before activation
    Vector hiddenlay_deriv; // Derivative of hidden layer w.r.t pre-activation
    Vector hiddenlay_deriv2;
    Vector delta_h;       // For backpropagation
    // Matrix dhdi;          // Derivative of hidden layer w.r.t input

    template<class V> struct WorkspaceStorage {
        V hiddenlay, hiddenlay_pre, hiddenlay_deriv, hiddenlay_deriv2, delta_h;
    };
    using Workspace = WorkspaceStorage<Vector>;
    using WorkspaceView = WorkspaceStorage<Vector&>;
    WorkspaceView workspace_view() {
        return {hiddenlay,hiddenlay_pre,hiddenlay_deriv,hiddenlay_deriv2,delta_h};
    }
    template<class Buffer> void prepare_workspace(Buffer& buffer) const {
        buffer.hiddenlay.resize(n_neuron);buffer.hiddenlay_pre.resize(n_neuron);
        buffer.hiddenlay_deriv.resize(n_neuron);buffer.hiddenlay_deriv2.resize(n_neuron);
        buffer.delta_h.resize(n_neuron);
    }

    m_model_nn1()
    {
        ;
    }
    void set_activation_function(ActivationFunction func_enum)
    {
        acfunc_enum = func_enum;
        switch (acfunc_enum) {
        case TANH:
            activation_eval = &TanhActivation::eval;
            activation_deriv = &TanhActivation::derivative;
            activation_deriv2 = &TanhActivation::second_derivative;
            break;
        case RELU:
            activation_eval = &ReLUActivation::eval;
            activation_deriv = &ReLUActivation::derivative;
            activation_deriv2 = &ReLUActivation::second_derivative;
            break;
        case SIGMOID:
            activation_eval = &SigmoidActivation::eval;
            activation_deriv = &SigmoidActivation::derivative;
            activation_deriv2 = &SigmoidActivation::second_derivative;
            break;
        default: // Linear
            activation_eval = [](Scalar x) { return x; };
            activation_deriv = [](Scalar, Scalar) { return 1.0; };
            activation_deriv2 = [](Scalar, Scalar, Scalar) { return 0.0; };
            break;
        }
    }
    void init(GPSettings& settings)
    {
        if (settings.count("n_neuron")) {
            n_neuron = std::stoi(settings["n_neuron"]);
        }
        if (settings.count("n_input")) {
            n_input = std::stoi(settings["n_input"]);
        }
        if (settings.count("activation_function")) {
             std::string func_name = settings["activation_function"];
             if (func_name == "TANH") acfunc_enum = TANH;
             else if (func_name == "RELU") acfunc_enum = RELU;
             else if (func_name == "SIGMOID") acfunc_enum = SIGMOID;
             else acfunc_enum = LINEAR;
        }
        n_weights = n_input * n_neuron + n_neuron;
        n_bias = n_neuron;
        n_coeffs = n_weights + n_bias;
        // w_input_map = std::make_unique<Eigen::Map<Matrix>>(coeffs, n_input, n_neuron);
        // w_hidden_map = std::make_unique<Eigen::Map<Vector>>(coeffs + n_input * n_neuron, n_neuron);
        // b_hidden_map = std::make_unique<Eigen::Map<Vector>>(coeffs + n_input * n_neuron + n_neuron, n_neuron);
        set_activation_function(acfunc_enum);
        hiddenlay.resize(n_neuron);
        hiddenlay_pre.resize(n_neuron);
        hiddenlay_deriv.resize(n_neuron);
        hiddenlay_deriv2.resize(n_neuron);
        delta_h.resize(n_neuron);
        // dhdi.resize(n_neuron, n_input);
    }
    void ensure_ready() const
    {
        if (coeffs == nullptr || !w_input_map || !w_hidden_map || !b_hidden_map)
            GPERROR("m_model_nn1 is not initialized");
    }
    void postinit() {
        if (coeffs == nullptr)
            GPERROR("m_model_nn1 coefficients are not initialized");
        w_input_map = std::make_unique<Eigen::Map<Matrix>>(coeffs, n_input, n_neuron);
        w_hidden_map = std::make_unique<Eigen::Map<Vector>>(coeffs + n_input * n_neuron, n_neuron);
        b_hidden_map = std::make_unique<Eigen::Map<Vector>>(coeffs + n_input * n_neuron + n_neuron, n_neuron);
    }
    void random_init(double lb, double ub)
    {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<double> dist(lb, ub);
        for (int i = 0; i < n_coeffs; ++i) {
            coeffs[i] = dist(gen);
        }
    }
    int get_nprops()
    {
        return n_input;
    }
    int get_ncoeffs()
    {
        return n_coeffs;
    }
    void add_loss_gard(double& loss, double* gard, int offset)
    {
        ;
    }
    void random_init() { random_init(-0.1, 0.1); };
    void propagate(const Eigen::VectorXd& Qs, double& site_energy, int& tag, int& ithread)
    {
        auto buffer = workspace_view();
        propagate(Qs, site_energy, tag, ithread, buffer);
    }

    template<class Buffer>
    void propagate(const Eigen::VectorXd& Qs, double& site_energy, int& tag, int& ithread, Buffer& buffer) const
    {
        prepare_workspace(buffer);
        auto& hiddenlay = buffer.hiddenlay;
        auto& hiddenlay_pre = buffer.hiddenlay_pre;
        auto& hiddenlay_deriv = buffer.hiddenlay_deriv;
        auto& hiddenlay_deriv2 = buffer.hiddenlay_deriv2;
        auto& delta_h = buffer.delta_h;
        const Eigen::Map<const Matrix> w_input(coeffs,n_input,n_neuron);
        const Eigen::Map<const Vector> w_hidden(coeffs+n_input*n_neuron,n_neuron);
        const Eigen::Map<const Vector> b_hidden(coeffs+n_input*n_neuron+n_neuron,n_neuron);

        ensure_ready();
        // Hidden layer pre-activation: z = W_in^T * Qs + b_h
        hiddenlay_pre = w_input.transpose() * Qs + b_hidden;
        
        // Hidden layer activation: h = f(z)
        hiddenlay = hiddenlay_pre.unaryExpr(activation_eval);

        // Output layer: site_energy = w_h . h
        site_energy = w_hidden.dot(hiddenlay);
    }
    void propagate(const Eigen::VectorXd& Qs, double& site_energy, Eigen::VectorXd& dUdQ, int& tag, int& ithread)
    {
        auto buffer = workspace_view();
        propagate(Qs, site_energy, dUdQ, tag, ithread, buffer);
    }

    template<class Buffer>
    void propagate(const Eigen::VectorXd& Qs, double& site_energy, Eigen::VectorXd& dUdQ, int& tag, int& ithread, Buffer& buffer) const
    {
        prepare_workspace(buffer);
        auto& hiddenlay = buffer.hiddenlay;
        auto& hiddenlay_pre = buffer.hiddenlay_pre;
        auto& hiddenlay_deriv = buffer.hiddenlay_deriv;
        auto& hiddenlay_deriv2 = buffer.hiddenlay_deriv2;
        auto& delta_h = buffer.delta_h;
        const Eigen::Map<const Matrix> w_input(coeffs,n_input,n_neuron);
        const Eigen::Map<const Vector> w_hidden(coeffs+n_input*n_neuron,n_neuron);
        const Eigen::Map<const Vector> b_hidden(coeffs+n_input*n_neuron+n_neuron,n_neuron);

        ensure_ready();
        // Hidden layer pre-activation: z = W_in^T * Qs + b_h
        hiddenlay_pre = w_input.transpose() * Qs + b_hidden;
        // Hidden layer activation: h = f(z)
        hiddenlay = hiddenlay_pre.unaryExpr(activation_eval);

        // Output layer: site_energy = w_h . h
        site_energy = w_hidden.dot(hiddenlay);

        // Derivative of activation function, h' = f'(z)
        for (int i = 0; i < n_neuron; ++i) {
            hiddenlay_deriv(i) = activation_deriv(hiddenlay_pre(i), hiddenlay(i));
        }

        // Gradient (chain rule): dE/dQs = W_in * (h' .* w_h)
        // where .* is element-wise product (Hadamard product)
        dUdQ = w_input * (hiddenlay_deriv.array() * w_hidden.array()).matrix();
    }
    void propagate(const Eigen::VectorXd& Qs, double& site_energy, Eigen::VectorXd& dUdQ, Eigen::MatrixXd& d2UdQdQ, int& tag, int& ithread)
    {
        auto buffer = workspace_view();
        propagate(Qs, site_energy, dUdQ, d2UdQdQ, tag, ithread, buffer);
    }

    template<class Buffer>
    void propagate(const Eigen::VectorXd& Qs, double& site_energy, Eigen::VectorXd& dUdQ, Eigen::MatrixXd& d2UdQdQ, int& tag, int& ithread, Buffer& buffer) const
    {
        prepare_workspace(buffer);
        auto& hiddenlay = buffer.hiddenlay;
        auto& hiddenlay_pre = buffer.hiddenlay_pre;
        auto& hiddenlay_deriv = buffer.hiddenlay_deriv;
        auto& hiddenlay_deriv2 = buffer.hiddenlay_deriv2;
        auto& delta_h = buffer.delta_h;
        const Eigen::Map<const Matrix> w_input(coeffs,n_input,n_neuron);
        const Eigen::Map<const Vector> w_hidden(coeffs+n_input*n_neuron,n_neuron);
        const Eigen::Map<const Vector> b_hidden(coeffs+n_input*n_neuron+n_neuron,n_neuron);

        ensure_ready();
        propagate(Qs, site_energy, dUdQ, tag, ithread, buffer); // Calculate energy and gradient

        // Calculate 1st and 2nd derivatives of activation
        for (int i = 0; i < n_neuron; ++i) {
            hiddenlay_deriv2(i) = activation_deriv2(hiddenlay_pre(i), hiddenlay(i), hiddenlay_deriv(i));
        }
        
        // Hessian (chain rule): d2E/dQs^2 = W_in * diag(h'' .* w_h) * W_in^T
        d2UdQdQ = w_input * (hiddenlay_deriv2.array() * w_hidden.array()).matrix().asDiagonal() * w_input.transpose();
    }

    void backpropagate(const Eigen::VectorXd& Qs, double& site_energy, Eigen::VectorXd& dUdt, int& tag, int& ithread)
    {
        auto buffer = workspace_view();
        backpropagate(Qs, site_energy, dUdt, tag, ithread, buffer);
    }

    template<class Buffer>
    void backpropagate(const Eigen::VectorXd& Qs, double& site_energy, Eigen::VectorXd& dUdt, int& tag, int& ithread, Buffer& buffer) const
    {
        prepare_workspace(buffer);
        auto& hiddenlay = buffer.hiddenlay;
        auto& hiddenlay_pre = buffer.hiddenlay_pre;
        auto& hiddenlay_deriv = buffer.hiddenlay_deriv;
        auto& hiddenlay_deriv2 = buffer.hiddenlay_deriv2;
        auto& delta_h = buffer.delta_h;
        const Eigen::Map<const Matrix> w_input(coeffs,n_input,n_neuron);
        const Eigen::Map<const Vector> w_hidden(coeffs+n_input*n_neuron,n_neuron);
        const Eigen::Map<const Vector> b_hidden(coeffs+n_input*n_neuron+n_neuron,n_neuron);
        dUdt.resize(n_coeffs);

        ensure_ready();
        propagate(Qs, site_energy, tag, ithread, buffer);

        for (int i = 0; i < n_neuron; ++i) {
            hiddenlay_deriv(i) = activation_deriv(hiddenlay_pre(i), hiddenlay(i));
        }
        
        delta_h = (hiddenlay_deriv.array() * w_hidden.array()).matrix();
        // dUdt.resize(n_coeffs);

        Eigen::Map<Matrix> dEdW_in(dUdt.data(), n_input, n_neuron);
        dEdW_in = Qs * delta_h.transpose();

        Eigen::Map<Vector> dEdw_h(dUdt.data() + n_input * n_neuron, n_neuron);
        dEdw_h = hiddenlay;

        Eigen::Map<Vector> dEdb_h(dUdt.data() + n_input * n_neuron + n_neuron, n_neuron);
        dEdb_h = delta_h;
    }
    void backpropagate(const Eigen::VectorXd& Qs, double& site_energy, Eigen::VectorXd& dUdQ, Eigen::VectorXd& dUdt, int& tag, int& ithread)
    {
        auto buffer = workspace_view();
        backpropagate(Qs, site_energy, dUdQ, dUdt, tag, ithread, buffer);
    }

    template<class Buffer>
    void backpropagate(const Eigen::VectorXd& Qs, double& site_energy, Eigen::VectorXd& dUdQ, Eigen::VectorXd& dUdt, int& tag, int& ithread, Buffer& buffer) const
    {
        prepare_workspace(buffer);
        auto& hiddenlay = buffer.hiddenlay;
        auto& hiddenlay_pre = buffer.hiddenlay_pre;
        auto& hiddenlay_deriv = buffer.hiddenlay_deriv;
        auto& hiddenlay_deriv2 = buffer.hiddenlay_deriv2;
        auto& delta_h = buffer.delta_h;
        const Eigen::Map<const Matrix> w_input(coeffs,n_input,n_neuron);
        const Eigen::Map<const Vector> w_hidden(coeffs+n_input*n_neuron,n_neuron);
        const Eigen::Map<const Vector> b_hidden(coeffs+n_input*n_neuron+n_neuron,n_neuron);
        dUdt.resize(n_coeffs);

        ensure_ready();
        propagate(Qs, site_energy, dUdQ, tag, ithread, buffer); // Calculate energy and gradient
        delta_h = (hiddenlay_deriv.array() * w_hidden.array()).matrix();
        // dUdt.resize(n_coeffs);

        Eigen::Map<Matrix> dEdW_in(dUdt.data(), n_input, n_neuron);
        dEdW_in = Qs * delta_h.transpose();

        Eigen::Map<Vector> dEdw_h(dUdt.data() + n_input * n_neuron, n_neuron);
        dEdw_h = hiddenlay;

        Eigen::Map<Vector> dEdb_h(dUdt.data() + n_input * n_neuron + n_neuron, n_neuron);
        dEdb_h = delta_h;
    }
    void backpropagate(const Eigen::VectorXd& Qs, double& site_energy,
        Eigen::VectorXd& dUdt, Eigen::MatrixXd& d2UdQdt, int& tag, int& ithread)
    {
        auto buffer = workspace_view();
        backpropagate(Qs, site_energy, dUdt, d2UdQdt, tag, ithread, buffer);
    }

    template<class Buffer>
    void backpropagate(const Eigen::VectorXd& Qs, double& site_energy,
        Eigen::VectorXd& dUdt, Eigen::MatrixXd& d2UdQdt, int& tag, int& ithread, Buffer& buffer) const
    {
        prepare_workspace(buffer);
        auto& hiddenlay = buffer.hiddenlay;
        auto& hiddenlay_pre = buffer.hiddenlay_pre;
        auto& hiddenlay_deriv = buffer.hiddenlay_deriv;
        auto& hiddenlay_deriv2 = buffer.hiddenlay_deriv2;
        auto& delta_h = buffer.delta_h;
        const Eigen::Map<const Matrix> w_input(coeffs,n_input,n_neuron);
        const Eigen::Map<const Vector> w_hidden(coeffs+n_input*n_neuron,n_neuron);
        const Eigen::Map<const Vector> b_hidden(coeffs+n_input*n_neuron+n_neuron,n_neuron);
        dUdt.resize(n_coeffs);
        d2UdQdt.resize(n_input,n_coeffs);

        ensure_ready();
        // 1. Perform a forward pass to compute energy and populate intermediate buffers (z, h)
        propagate(Qs, site_energy, tag, ithread, buffer);

        // 2. Compute first and second derivatives of the activation function for each neuron
        for (int i = 0; i < n_neuron; ++i) {
            hiddenlay_deriv(i) = activation_deriv(hiddenlay_pre(i), hiddenlay(i));
            hiddenlay_deriv2(i) = activation_deriv2(hiddenlay_pre(i), hiddenlay(i), hiddenlay_deriv(i));
        }

        // 3. Compute dUdt (the gradient of energy w.r.t. coefficients)
        // dUdt.resize(n_coeffs);
        delta_h = (hiddenlay_deriv.array() * w_hidden.array()).matrix();
        
        Eigen::Map<Matrix> dEdW_in(dUdt.data(), n_input, n_neuron);
        dEdW_in = Qs * delta_h.transpose();

        Eigen::Map<Vector> dEdw_h(dUdt.data() + n_input * n_neuron, n_neuron);
        dEdw_h = hiddenlay;

        Eigen::Map<Vector> dEdb_h(dUdt.data() + n_input * n_neuron + n_neuron, n_neuron);
        dEdb_h = delta_h;

        // 4. Compute d2UdQdt (the mixed second derivative matrix)
        // d2UdQdt.resize(n_input, n_coeffs);
        d2UdQdt.setZero();

        // d2U/dQ_i dW_in[j,m]
        // = δ_{ij} * w_h[m] * σ'(z_m) + w_h[m] * σ''(z_m) * W_in[i,m] * Q[j]
        int index = 0;
        for (int m = 0; m < n_neuron; ++m) {
            const double w_h_m = w_hidden(m);
            const double sigma_prime = hiddenlay_deriv(m);
            const double sigma_double_prime = hiddenlay_deriv2(m);

            for (int i = 0; i < n_input; ++i) {
                for (int j = 0; j < n_input; ++j) {
                    // when i == j
                    if (i == j) {
                        d2UdQdt(j, index) += w_h_m * sigma_prime;
                    }
                    // σ''(z_m) * w_h[m] * W_in[i,m] * Q[j]
                    d2UdQdt(j, index) += w_h_m * w_input(j, m) * Qs(i) * sigma_double_prime;
                }
                index++;
            }
        }

        // d2U/dQ_i dw_h[m]
        // = σ'(z_m) * W_in[i,m]
        for (int m = 0; m < n_neuron; ++m) {
            for (int i = 0; i < n_input; ++i) {
                d2UdQdt(i, index) += hiddenlay_deriv(m) * w_input(i, m);
            }
            index++;
        }

        // d2U/dQ_i db_h[m]
        // = w_h[m] * σ''(z_m) * W_in[i,m]
        for (int m = 0; m < n_neuron; ++m) {
            for (int i = 0; i < n_input; ++i) {
                d2UdQdt(i, index) += w_hidden(m) * hiddenlay_deriv2(m) * w_input(i, m);
            }
            ++index;
        }
    }
    void backpropagate(const Eigen::VectorXd& Qs, double& site_energy, Eigen::VectorXd& dUdQ,
        Eigen::VectorXd& dUdt, Eigen::MatrixXd& d2UdQdt, Eigen::MatrixXd& d2UdQdQ,
        int& tag, int& ithread)
    {
        auto buffer = workspace_view();
        backpropagate(Qs, site_energy, dUdQ, dUdt, d2UdQdt, d2UdQdQ, tag, ithread, buffer);
    }

    template<class Buffer>
    void backpropagate(const Eigen::VectorXd& Qs, double& site_energy, Eigen::VectorXd& dUdQ,
        Eigen::VectorXd& dUdt, Eigen::MatrixXd& d2UdQdt, Eigen::MatrixXd& d2UdQdQ,
        int& tag, int& ithread, Buffer& buffer) const
    {
        prepare_workspace(buffer);
        auto& hiddenlay = buffer.hiddenlay;
        auto& hiddenlay_pre = buffer.hiddenlay_pre;
        auto& hiddenlay_deriv = buffer.hiddenlay_deriv;
        auto& hiddenlay_deriv2 = buffer.hiddenlay_deriv2;
        auto& delta_h = buffer.delta_h;
        const Eigen::Map<const Matrix> w_input(coeffs,n_input,n_neuron);
        const Eigen::Map<const Vector> w_hidden(coeffs+n_input*n_neuron,n_neuron);
        const Eigen::Map<const Vector> b_hidden(coeffs+n_input*n_neuron+n_neuron,n_neuron);
        dUdt.resize(n_coeffs);
        d2UdQdt.resize(n_input,n_coeffs);

        ensure_ready();
        // Forward pass with all derivatives
        propagate(Qs, site_energy, dUdQ, d2UdQdQ, tag, ithread, buffer);
        
        // Compute parameter gradients
        delta_h = (hiddenlay_deriv.array() * w_hidden.array()).matrix();
        // dUdt.resize(n_coeffs);

        Eigen::Map<Matrix> dEdW_in(dUdt.data(), n_input, n_neuron);
        dEdW_in = Qs * delta_h.transpose();

        Eigen::Map<Vector> dEdw_h(dUdt.data() + n_input * n_neuron, n_neuron);
        dEdw_h = hiddenlay;

        Eigen::Map<Vector> dEdb_h(dUdt.data() + n_input * n_neuron + n_neuron, n_neuron);
        dEdb_h = delta_h;

        // Compute mixed derivatives d2UdQdt
        // d2UdQdt.resize(n_input, n_coeffs);
        d2UdQdt.setZero();

        // d2U/dQ_i dW_in[j,m]
        // = δ_{ij} * w_h[m] * σ'(z_m) + w_h[m] * σ''(z_m) * W_in[i,m] * Q[j]
        int index = 0;
        for (int m = 0; m < n_neuron; ++m) {
            const double w_h_m = w_hidden(m);
            const double sigma_prime = hiddenlay_deriv(m);
            const double sigma_double_prime = hiddenlay_deriv2(m);

            for (int i = 0; i < n_input; ++i) {
                for (int j = 0; j < n_input; ++j) {
                    // when i == j
                    if (i == j) {
                        d2UdQdt(j, index) += w_h_m * sigma_prime;
                    }
                    // σ''(z_m) * w_h[m] * W_in[i,m] * Q[j]
                    d2UdQdt(j, index) += w_h_m * w_input(j, m) * Qs(i) * sigma_double_prime;
                }
                index++;
            }
        }

        // d2U/dQ_i dw_h[m]
        // = σ'(z_m) * W_in[i,m]
        for (int m = 0; m < n_neuron; ++m) {
            for (int i = 0; i < n_input; ++i) {
                d2UdQdt(i, index) += hiddenlay_deriv(m) * w_input(i, m);
            }
            index++;
        }

        // d2U/dQ_i db_h[m]
        // = w_h[m] * σ''(z_m) * W_in[i,m]
        for (int m = 0; m < n_neuron; ++m) {
            for (int i = 0; i < n_input; ++i) {
                d2UdQdt(i, index) += w_hidden(m) * hiddenlay_deriv2(m) * w_input(i, m);
            }
            ++index;
        }
        }
};

struct m_model_nn2 {
    using Workspace = IMRMLPNN::NN2lay::Buffer;
    double* coeffs = nullptr; // coeffs handled externally

    IMRMLPNN::NN2lay nn;
    int n_input = 5;
    int n_neuron = 10;
    int n_neuron2 = 5;
    int n_coeffs = 0;
    int n_threads = 1;
    double reg_l1 = 0.0;
    double reg_l2 = 0.0;

    static std::string normalized_activation_name(std::string name)
    {
        std::transform(name.begin(), name.end(), name.begin(),
            [](unsigned char c) { return static_cast<char>(std::toupper(c)); });
        return name;
    }

    static IMRMLPNN::ActivationFunction parse_activation(GPSettings& settings)
    {
        if (!settings.count("activation_function"))
            return IMRMLPNN::TANH;

        const std::string name = normalized_activation_name(settings["activation_function"]);
        if (name == "RELU")
            return IMRMLPNN::RELU;
        if (name == "SIGMOID")
            return IMRMLPNN::SIGMOID;
        return IMRMLPNN::TANH;
    }

    void init(GPSettings& settings)
    {
        if (settings.count("n_input"))
            n_input = std::stoi(settings["n_input"]);
        if (settings.count("n_neuron"))
            n_neuron = std::stoi(settings["n_neuron"]);
        if (settings.count("n_neuron1"))
            n_neuron = std::stoi(settings["n_neuron1"]);
        if (settings.count("n_neuron2"))
            n_neuron2 = std::stoi(settings["n_neuron2"]);
        if (settings.count("n_hidden2"))
            n_neuron2 = std::stoi(settings["n_hidden2"]);
        if (settings.count("n_threads"))
            n_threads = std::max(1, std::stoi(settings["n_threads"]));
        if (settings.count("model_l1"))
            reg_l1 = std::stod(settings["model_l1"]);
        if (settings.count("model_l2"))
            reg_l2 = std::stod(settings["model_l2"]);

        nn.acfunc = parse_activation(settings);
        nn.setup(n_input, n_neuron, n_neuron2, n_threads);
        n_coeffs = nn.get_coeff_size();
    }

    void postinit()
    {
        if (coeffs != nullptr)
            nn.set_coeffs(coeffs);
    }

    void sync_from_coeffs(int ithread)
    {
        if (coeffs == nullptr)
            GPERROR("m_model_nn2 coefficients are not initialized");
        if (ithread >= static_cast<int>(nn.buffers.size()))
            nn.resize_omp_buffers(ithread + 1);
        nn.set_coeffs(coeffs);
    }

    int get_nprops()
    {
        return n_input;
    }

    int get_ncoeffs()
    {
        return n_coeffs;
    }

    void add_loss_gard(double& loss, double* gard, int offset)
    {
        if (reg_l1 <= 0.0 && reg_l2 <= 0.0)
            return;
        sync_from_coeffs(0);
        nn.add_pnal(reg_l1, reg_l2, loss, gard + offset);
    }

    void random_init(double lb, double ub)
    {
        if (coeffs == nullptr)
            GPERROR("m_model_nn2 coefficients are not initialized");
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<double> dist(lb, ub);
        for (int i = 0; i < n_coeffs; ++i)
            coeffs[i] = dist(gen);
        postinit();
    }

    void random_init()
    {
        if (coeffs == nullptr)
            GPERROR("m_model_nn2 coefficients are not initialized");
        nn.random_init();
        nn.get_coeffs(coeffs);
    }

    void propagate(const Eigen::VectorXd& Qs, double& site_energy, int& tag, int& ithread)
    {
        sync_from_coeffs(ithread);
        nn.propagate(Qs, site_energy, tag, ithread);
    }
    void propagate(const Eigen::VectorXd& Qs, double& site_energy, int& tag, int& ithread, Workspace& workspace) const
    {
        nn.propagate(Qs, site_energy, tag, nn.coefficient_view(coeffs), workspace);
    }


    void propagate(const Eigen::VectorXd& Qs, double& site_energy, Eigen::VectorXd& dUdQ, int& tag, int& ithread)
    {
        sync_from_coeffs(ithread);
        nn.propagate(Qs, site_energy, dUdQ, tag, ithread);
    }
    void propagate(const Eigen::VectorXd& Qs, double& site_energy, Eigen::VectorXd& dUdQ, int& tag, int& ithread, Workspace& workspace) const
    {
        nn.propagate(Qs, site_energy, dUdQ, tag, nn.coefficient_view(coeffs), workspace);
    }


    void propagate(const Eigen::VectorXd& Qs, double& site_energy, Eigen::VectorXd& dUdQ,
        Eigen::MatrixXd& d2UdQdQ, int& tag, int& ithread)
    {
        sync_from_coeffs(ithread);
        nn.propagate(Qs, site_energy, dUdQ, d2UdQdQ, tag, ithread);
    }
    void propagate(const Eigen::VectorXd& Qs, double& site_energy, Eigen::VectorXd& dUdQ, Eigen::MatrixXd& d2UdQdQ, int& tag, int& ithread, Workspace& workspace) const
    {
        nn.propagate(Qs, site_energy, dUdQ, d2UdQdQ, tag, nn.coefficient_view(coeffs), workspace);
    }






    void backpropagate(const Eigen::VectorXd& Qs, double& site_energy, Eigen::VectorXd& dUdt,
        int& tag, int& ithread)
    {
        sync_from_coeffs(ithread);
        nn.backpropgate(Qs, site_energy, dUdt, tag, ithread);
    }
    void backpropagate(const Eigen::VectorXd& Qs, double& site_energy, Eigen::VectorXd& dUdt, int& tag, int& ithread, Workspace& workspace) const
    {
        nn.backpropgate(Qs, site_energy, dUdt, tag, nn.coefficient_view(coeffs), workspace);
    }


    void backpropagate(const Eigen::VectorXd& Qs, double& site_energy, Eigen::VectorXd& dUdQ,
        Eigen::VectorXd& dUdt, int& tag, int& ithread)
    {
        sync_from_coeffs(ithread);
        nn.backpropgate(Qs, site_energy, dUdQ, dUdt, tag, ithread);
    }
    void backpropagate(const Eigen::VectorXd& Qs, double& site_energy, Eigen::VectorXd& dUdQ, Eigen::VectorXd& dUdt, int& tag, int& ithread, Workspace& workspace) const
    {
        nn.backpropgate(Qs, site_energy, dUdQ, dUdt, tag, nn.coefficient_view(coeffs), workspace);
    }


    void backpropagate(const Eigen::VectorXd& Qs, double& site_energy, Eigen::VectorXd& dUdt,
        Eigen::MatrixXd& d2UdQdt, int& tag, int& ithread)
    {
        sync_from_coeffs(ithread);
        nn.backpropgate(Qs, site_energy, dUdt, d2UdQdt, tag, ithread);
    }
    void backpropagate(const Eigen::VectorXd& Qs, double& site_energy, Eigen::VectorXd& dUdt, Eigen::MatrixXd& d2UdQdt, int& tag, int& ithread, Workspace& workspace) const
    {
        nn.backpropgate(Qs, site_energy, dUdt, d2UdQdt, tag, nn.coefficient_view(coeffs), workspace);
    }


    void backpropagate(const Eigen::VectorXd& Qs, double& site_energy, Eigen::VectorXd& dUdQ,
        Eigen::VectorXd& dUdt, Eigen::MatrixXd& d2UdQdt, Eigen::MatrixXd& d2UdQdQ,
        int& tag, int& ithread)
    {
        sync_from_coeffs(ithread);
        nn.backpropgate(Qs, site_energy, dUdQ, dUdt, d2UdQdt, d2UdQdQ, tag, ithread);
    }
    void backpropagate(const Eigen::VectorXd& Qs, double& site_energy, Eigen::VectorXd& dUdQ, Eigen::VectorXd& dUdt, Eigen::MatrixXd& d2UdQdt, Eigen::MatrixXd& d2UdQdQ, int& tag, int& ithread, Workspace& workspace) const
    {
        nn.backpropgate(Qs, site_energy, dUdQ, dUdt, d2UdQdt, d2UdQdQ, tag, nn.coefficient_view(coeffs), workspace);
    }

};
/**
 * @brief
 *
 * coeffs_layout: radical_coeffs - species_coeffs - moment_coeffs - model_coeffs
 *
 * E = E_species + F(q1, q2, ..., qn)
 * qi = bias_species + \sum_a c_a * B_a
 * TODO: remove SPL_CUTOFF flag
 * TODO: add_species()
 */
struct MTPMEmptyHeadWorkspace {};
template<class T> struct MTPMHeadWorkspace { using type = MTPMEmptyHeadWorkspace; };
template<> struct MTPMHeadWorkspace<m_model_nn1> { using type = m_model_nn1::Workspace; };
template<> struct MTPMHeadWorkspace<m_model_nn2> { using type = m_model_nn2::Workspace; };

template <class T>
class PotMTP_M : public PotMTP {
public:
    // Scratch belongs to one call or one execution thread. Model metadata and
    // coefficient storage remain on the potential and are read only by kernels.
    struct SiteWorkspace {
        ompbuffers moments;
        typename MTPMHeadWorkspace<T>::type head;
        Eigen::VectorXd ext_dUdt, q_direction, q_weighted;
        Eigen::MatrixXd ext_d2UdQdQ, ext_d2UdQdt;
        Eigen::Tensor<double,3> ext_d3UdQdQdQ;
        std::vector<double> dqdr;
        double q_les = 0.;
        std::array<int,8> shape = {-1,-1,-1,-1,-1,-1,-1,-1};
    };
    SiteWorkspace make_site_workspace() const {
        SiteWorkspace workspace;
        prepare_site_workspace(workspace);
        return workspace;
    }
    void prepare_site_workspace(SiteWorkspace& workspace) const {
        require_loaded_model();
        const std::array<int,8> shape = {max_alpha_index_basic,alpha_moments_count,
            rb_size,max_mu,alpha_index_basic_count,species_count,n_props,model_coeff_size};
        if (workspace.shape != shape) {
            workspace.moments = ompbuffers(max_alpha_index_basic,alpha_moments_count,rb_size,max_mu);
            workspace.shape = shape;
        }
        workspace.moments.Qs.resize(n_props);
        workspace.moments.ext_dUdQ.resize(n_props);
    }
    struct WorkspaceModel {
        PotMTP_M* owner;
        SiteWorkspace& workspace;
        template<class... Args>
        requires requires(T& head,Args&&... args) {head.propagate(std::forward<Args>(args)...);}
        void propagate(Args&&... args) {
            if constexpr (std::is_same_v<T,m_model_nn1> || std::is_same_v<T,m_model_nn2>)
                owner->model.propagate(std::forward<Args>(args)...,workspace.head);
            else owner->model.propagate(std::forward<Args>(args)...);
        }
        template<class... Args>
        requires requires(T& head,Args&&... args) {head.backpropagate(std::forward<Args>(args)...);}
        void backpropagate(Args&&... args) {
            if constexpr (std::is_same_v<T,m_model_nn1> || std::is_same_v<T,m_model_nn2>)
                owner->model.backpropagate(std::forward<Args>(args)...,workspace.head);
            else owner->model.backpropagate(std::forward<Args>(args)...);
        }
    };
    SiteWorkspace serial_workspace_;
    // Structure-sized responses and reciprocal caches are independent from the
    // site scratch, and belong to the caller for concurrent structure requests.
    // Explicit overloads cover EFS, parameter EFS contractions and Cartesian HVP.
    // Dense/cell Hessians also accept this workspace explicitly.
    struct EvaluationWorkspace {
        SiteWorkspace site;
        LES::Workspace reciprocal;
        Eigen::MatrixXd latents, descriptors;
        std::vector<int> atom_types;
        Eigen::VectorXd q, q_directional, dEdq, charge_vjp, moment_vjp, response,
            dLdq, direct_vjp, product;
        SiteCoordinateMatrix relative, local_gradient, local_product, local_combination;
        Eigen::Matrix<double,-1,3> zero_force;
        std::vector<double> kernel_gradient, q_der, mixed;
        LES::Evaluation evaluated;
        LES::CartesianResponse directional;
    };
    EvaluationWorkspace serial_evaluation_workspace_;
    void sync_evaluation_diagnostics() {
        if(lr_les) {
            LES_all_latents=serial_evaluation_workspace_.latents;
            LES_all_qs=LES_all_latents.col(0);
            model_les.publish_legacy_diagnostics(serial_evaluation_workspace_.reciprocal);
        }
    }
    EvaluationWorkspace make_evaluation_workspace() const {
        EvaluationWorkspace workspace;
        prepare_site_workspace(workspace.site);
        return workspace;
    }
    void getE(Atoms& atom,AnyNeighborhoods& nbhs,EvaluationWorkspace& workspace) {
        getEFS_impl<0>(atom,nbhs,workspace);
    }
    void getEFS(Atoms& atom,AnyNeighborhoods& nbhs,EvaluationWorkspace& workspace) {
        getEFS_impl<1>(atom,nbhs,workspace);
    }
    template<int REQ_F> void getEFS_impl(Atoms&,AnyNeighborhoods&,EvaluationWorkspace&);
    void compute_les_charges_and_descriptors(Atoms&,AnyNeighborhoods&,EvaluationWorkspace&);
    void charge_gradient_workspace(Neighborhood&,const double*,SiteWorkspace&,SiteCoordinateMatrix&);
    void accumulate_les_workspace(Atoms&,AnyNeighborhoods&,double,
        const Eigen::Matrix<double,-1,3>*,const Eigen::Matrix3d*,double*,EvaluationWorkspace&);
    void accumulate_charge_radial_workspace(Atoms&,AnyNeighborhoods&,const double*,
        const Eigen::VectorXd&,const Eigen::VectorXd&,const Eigen::Matrix<double,-1,3>&,
        const Eigen::Matrix3d&,double*,EvaluationWorkspace&);
    void AccumulateEFSCombinationGard(Atoms&,AnyNeighborhoods&,double,
        const Eigen::Matrix<double,-1,3>&,const Eigen::Matrix3d&,double*,EvaluationWorkspace&);
    void AccumulateEFSCombinationGard(Atoms&,AnyNeighborhoods&,double,double*,EvaluationWorkspace&);
    void applyHessianVectorProduct(Atoms&,AnyNeighborhoods&,const Eigen::VectorXd&,
        Eigen::VectorXd&,EvaluationWorkspace&);
    LatentCurvatureCallbacks curvature_callbacks(Atoms&,AnyNeighborhoods&,EvaluationWorkspace&);
    void getEFSHessian(Atoms&,AnyNeighborhoods&,HessianMatrix&,EvaluationWorkspace&);
    void getEFSHessianCell(Atoms&,AnyNeighborhoods&,HessianMatrix&,Eigen::MatrixXd&,Eigen::MatrixXd&,EvaluationWorkspace&);



    struct WorkspaceChargeHead {
        const double* species;
        const double* moments;
        int count;
        void propagate(const std::vector<double>& basis,double& charge,Eigen::VectorXd& gradient,
            Eigen::MatrixXd& hessian,int& type,int&) const {
            charge=species[type];gradient.setZero(basis.size());
            for(int a=0;a<count;++a) {charge+=moments[a]*basis[a+1];gradient[a+1]=moments[a];}
            hessian.resize(0,0);
        }
    };


    bool supports_species_append() const override {
        constexpr bool known=std::is_same_v<T,m_model_mtpli> || std::is_same_v<T,m_model_mtpq2>
            || std::is_same_v<T,m_model_mtpfs> || std::is_same_v<T,m_model_nn1> || std::is_same_v<T,m_model_nn2>;
        return known && typeid(*this)==typeid(PotMTP_M<T>) && !coeffs_handled_external && !hc_correction && !rR_correction;
    }
    // Maps the current req-masked nl_get_coeffs layout, including SOG tails.
    SpeciesAppendResult append_species(const std::vector<int>& clone_sources,
        const std::vector<std::string>& new_elements={}) override;

    bool supports_site_efs() const override { return !lr_les; }
    void CalcSiteEFS(Neighborhood &nbh, double &energy, SiteCoordinateMatrix &gradient) override {
        if (!supports_site_efs()) GPERROR("MTP-M with long-range terms has no local site EFS");
        energy = 0.0;
        CalcSiteEFS<1>(nbh, energy, gradient);
    }
    void CalcSiteE(Neighborhood &nbh, double &energy) override {
        if (lr_les) GPERROR("MTP-M with long-range terms has no local site energy");
        CalcSiteEFS<0>(nbh, energy, buff_site_energy_ders);
    }
    bool supports_site_parameter_gradients() const override { return !lr_les; }
    std::string metadata_model_type() const override {return "MTP_M";}
    MTPMetadata::Object model_metadata() const override {
        using MTPMetadata::Value;
        MTPMetadata::Object fields{{"type", Value("MTP_M")},
            {"model", Value(serialized_model_name())}, {"q", Value(double(n_props))},
            {"model_coeffs_count", Value(double(model_coeff_size))}};
        if constexpr (std::is_same_v<T, m_model_nn1>) {
            fields["nn"] = Value(MTPMetadata::Object{
                {"activation", Value(serialized_activation_name(model.acfunc_enum))},
                {"hidden_layers", Value(MTPMetadata::Array{Value(double(model.n_neuron))})}});
        } else if constexpr (std::is_same_v<T, m_model_nn2>) {
            fields["nn"] = Value(MTPMetadata::Object{
                {"activation", Value(serialized_activation_name(model.nn.acfunc))},
                {"hidden_layers", Value(MTPMetadata::Array{Value(double(model.n_neuron)), Value(double(model.n_neuron2))})}});
        }
        if (lr_les) fields["long_range"] = Value(MTPMetadata::Object{
            {"kernel", Value(uses_sog() ? "sog" : "coulomb")}, {"channels", Value(double(les_channels()))}});
        return fields;
    }
    void model_settings_from_tag(GPSettings &settings,const std::string &filename) {
        auto stream=open_ifile(filename);if(!stream)GPERROR("Cannot read potential metadata");
        auto header=MTPMetadata::read_header(*stream);
        auto fields=MTPMetadata::legacy_model_fields(header.tag);
        std::string line;
        while(std::getline(*stream,line)) {
            if(line.rfind("# mtp_",0)!=0) continue;
            const auto equals=line.find('=');if(equals==std::string::npos)continue;
            MTPMetadata::merge_legacy(fields,MTPMetadata::trim(line.substr(2,equals-2)),MTPMetadata::trim(line.substr(equals+1)));
        }
        if(header.tag.text("type") == "MTP_M" && fields["mtp_model"] != serialized_model_name()) GPERROR("MTP-M model differs from potential_tag");
        auto set=[&](const std::string &key,const std::string &value) {
            if(value.empty())return;
            const std::string alias=key=="n_neuron"?"n_neuron1":key=="n_neuron2"?"n_hidden2":"";
            if((settings.count(key)&&settings[key]!=value)||(!alias.empty()&&settings.count(alias)&&settings[alias]!=value)) GPERROR("MTP-M setting conflicts with potential_tag: "+key);
            settings.Modify(key,value);
        };
        if constexpr(std::is_same_v<T,m_model_nn1>||std::is_same_v<T,m_model_nn2>) {
            set("n_input",fields["mtp_descriptor_count"]);set("n_neuron",fields["mtp_hidden1"]);
            if constexpr(std::is_same_v<T,m_model_nn2>)set("n_neuron2",fields["mtp_hidden2"]);
            set("activation_function",fields["mtp_activation"]);
        }
    }
    bool supports_analytical_hessian() const override { return !lr_les || supports_analytical_hessian_vector_product(); }
    using PotMTP::applyHessianVectorProduct;
    bool supports_analytical_hessian_vector_product() const override {
        return typeid(*this) == typeid(PotMTP_M<T>) && !hc_correction && !rR_correction
            && (!lr_les || (les_species && les_moment));
    }
    bool supports_analytical_cell_hessian() const override {
        return !lr_les || (supports_analytical_hessian_vector_product()
            && model_les.cutoff_mode == LES::CutoffMode::Smooth);
    }
    // layout: radial coeffs; species_coeffs; moment_coeffs;
    // species_coeffs: global_species_coeffs; bias_species_coeffs
    // external: model coeffs
    bool req_radical = true;
    bool req_species = true;
    bool req_moment = true;
    bool req_model = true;

    int n_props = 2; // size of Qs vector
    int n_coeffs_fitting = 0;
    double* model_coeffs = nullptr;
    int model_coeff_size = 0;
    T model;

    // for LES part, this will add an extra Q 
    double *les_coeffs = nullptr; // les coeffs
    double *les_species = nullptr; // les species coeffs
    double *les_moment = nullptr;  // les moment coeffs

    bool lr_les = false; // LES long range model, this will request an extra Q, species and moment coeffs laied on the last section
    bool req_les = true; // request les part for fitting
    LES model_les;
    bool req_sog = true;
    int les_channels() const {return uses_sog() ? model_les.sog_channels : 1;}
    int les_head_count() const {return les_channels()*(species_count+alpha_scalar_moments);}
    Eigen::MatrixXd LES_all_latents;
    double sog_log_bandwidth_min=-4.0, sog_log_bandwidth_max=1.0;
private:
    bool pending_sog_definition_ = false;
    void clear_loaded_heads() noexcept {
        model_coeffs = les_coeffs = les_species = les_moment = nullptr;
        model.coeffs = nullptr;
        if constexpr (std::is_same_v<T, m_model_nn1>) {
            model.w_input_map.reset();
            model.w_hidden_map.reset();
            model.b_hidden_map.reset();
        }
        model_coeff_size = n_coeffs_fitting = 0;
    }
    void clear_failed_load() noexcept override {
        clear_loaded_heads();
        clear_loaded_model();
        lr_les = pending_sog_definition_ = false;
        LES_all_qs.resize(0);
        LES_all_latents.resize(0, 0);
    }
public:
    bool uses_sog() const { return lr_les && model_les.kernel==LES::Kernel::SOG; }
    int sog_fitting_offset() const {
        return les_fitting_offset()+(req_les ? les_head_count() : 0);
    }
    SOGModel sog_definition() const {
        SOGModel p;
        p.channels=model_les.sog_channels;
        p.weights=model_les.sog_weights;p.log_bandwidths=model_les.sog_log_bandwidths;
        p.lower=sog_log_bandwidth_min;p.upper=sog_log_bandwidth_max;
        p.dl=model_les.dl;p.taper=model_les.taper_fraction;
        p.smooth=model_les.cutoff_mode==LES::CutoffMode::Smooth;
        p.include_zero=model_les.sog_include_zero;p.onsite=model_les.remove_self_interaction;
        return p;
    }
    void set_sog_definition(const SOGModel& p, bool loading=false) {
        try { p.validate(); } catch (const std::exception& e) { GPERROR(e.what()); }
        if(!loading && regression_coeffs && (!lr_les || p.channels!=les_channels()))
            GPERROR("Set MTP-M channel layout before Load, or load a matching multichannel template");
        if (!loading) pending_sog_definition_ = true;
        lr_les=true;model_les.kernel=LES::Kernel::SOG;
        model_les.sog_channels=p.channels;model_les.sog_channel=0;
        model_les.sog_weights=p.weights;model_les.sog_log_bandwidths=p.log_bandwidths;
        sog_log_bandwidth_min=p.lower;sog_log_bandwidth_max=p.upper;
        model_les.dl=p.dl;model_les.taper_fraction=p.taper;
        model_les.cutoff_mode=p.smooth ? LES::CutoffMode::Smooth : LES::CutoffMode::ReferenceHard;
        model_les.sog_include_zero=p.include_zero;model_les.remove_self_interaction=p.onsite;
    }

    // buffers, TODO: remove the part, this should be handled externally
    Eigen::VectorXd LES_all_qs;
    double Q_les = 0;

    void normalize_les_request()
    {
        req_les = lr_les && req_les;
    }

    int calc_NL_coeffs_count() const
    {
        int count = 0;
        if (req_radical)
            count += radial_coeff_size;
        if (req_species)
            count += species_count;
        if (req_moment)
            count += moment_coeff_size + species_coeff_size - species_count;
        if (req_model)
            count += model_coeff_size;
        if (req_les)
            count += les_head_count();
        if (uses_sog() && req_sog) count+=2*model_les.sog_weights.size();
        return count;
    }

    int les_fitting_offset() const
    {
        int offset = 0;
        if (req_radical)
            offset += radial_coeff_size;
        if (req_species)
            offset += species_count;
        if (req_moment)
            offset += moment_coeff_size + species_coeff_size - species_count;
        if (req_model)
            offset += model_coeff_size;
        return offset;
    }

    static void require_read(bool ok, const std::string& what)
    {
        if (!ok)
            GPERROR("PotMTP_M failed to read " + what);
    }

    void load_configured_model(const std::string &filename, const GPSettings &settings) {
        const LES requested = model_les;
        const bool requested_lr = lr_les;
        Load(filename);
        try {
            if (settings.count("lr_les") && requested_lr != lr_les)
                GPERROR("MTP-M lr_les setting conflicts with the model file");
            if (lr_les && !uses_sog()) {
                const auto check = [&](const char *key, double expected, double actual) {
                    if (settings.count(key) && std::abs(expected - actual) >
                        1e-12 * std::max({1., std::abs(expected), std::abs(actual)}))
                        GPERROR(std::string("MTP-M setting conflicts with Coulomb definition: ") + key);
                };
                check("lr_sigma", requested.lr_sigma, model_les.lr_sigma);
                check("les_dl", requested.dl, model_les.dl);
                check("remove_self_interaction", requested.remove_self_interaction,
                      model_les.remove_self_interaction);
            }
        } catch (...) {
            clear_failed_load();
            throw;
        }
    }

    PotMTP_M(const std::string& potfile)
    {
        InitSettings();
        GPSettings settings;
        model_settings_from_tag(settings,potfile);
        model.init(settings);
        Load(potfile);
    }

    PotMTP_M(const std::string& potfile, GPSettings& settings)
    {
        InitSettings();
        LES::reject_obsolete_settings(settings);
        ApplySettings(settings);
        // must init model before load potential
        model_settings_from_tag(settings,potfile);
        model.init(settings);
        load_configured_model(potfile, settings);
        // must init model mappings after load potential for nn type
        model.postinit();
    }

    void InitBySettings(GPSettings& settings)
    {
        InitSettings();
        LES::reject_obsolete_settings(settings);
        ApplySettings(settings);
        model_settings_from_tag(settings,settings["potfile"]);
        model.init(settings);
        load_configured_model(settings["potfile"], settings);
        model.postinit();
    }

    PotMTP_M(GPSettings settings)
    {
        InitBySettings(settings);
    }
    // use this with causion!
    PotMTP_M() { set_empty(); };

    using PotMTP::inited;
    bool inited() override { return !empty; }
    void set_empty() override { empty=true;nl_inited=false; }
    void set_inited() override { init_coeffs(); }

    void InitSettings() {
        PotMTP::InitSettings();
        Connect(lr_les, "lr_les");
        Connect(req_les, "req_les");
        Connect(req_sog, "req_sog");
        Connect(model_les.use_recurrence, "les_recurrence");
        Connect(model_les.use_cache, "les_cache");
        Connect(req_radical, "req_radical");
        Connect(req_species, "req_species");
        Connect(req_moment, "req_moment");
        Connect(req_model, "req_model");
        Connect(model_les.lr_sigma, "lr_sigma");
        Connect(model_les.dl, "les_dl");
        Connect(model_les.remove_self_interaction, "remove_self_interaction");
    }

    virtual void Load(const std::string& potfile) try
    {
        if(load_gpm_if_requested(*this,potfile))return;
        read_flag_early_exit = true;
        auto ifs = open_ifile(potfile);
        if (!ifs) {
            GPERROR((std::string) "ERROR: can not open potential file " + potfile);
        }
        Load(*ifs);
    }
    catch (...) {
        clear_failed_load();
        throw;
    }
    void Load(std::istream& ifs) try
    {
        clear_loaded_heads();
        const auto start=ifs.tellg();
        bool has_sog=false, has_les_definition=false;
        std::map<std::string, std::string> footer_fields;
        std::string line;
        while (std::getline(ifs, line)) {
            line = MTPMetadata::trim(line);
            if (line.rfind("#MVS_v1.1", 0) == 0) break;
            if (!line.empty() && line[0] == '#') line = MTPMetadata::trim(line.substr(1));
            const auto equals = line.find('=');
            const auto field = MTPMetadata::trim(line.substr(0, equals));
            LES::reject_obsolete_parameter(field);
            if (field.rfind("mtp_", 0) == 0 && equals != std::string::npos)
                MTPMetadata::merge_legacy(footer_fields, field, MTPMetadata::trim(line.substr(equals + 1)));
            if (field=="les_kernel") {
                const auto value = MTPMetadata::trim(line.substr(equals + 1));
                if (value!="coulomb" && value!="sog") GPERROR("Unknown MTP-M long-range kernel");
                has_sog=has_sog || value=="sog";
            }
            has_sog=has_sog || field.rfind("sog_",0)==0;
            has_les_definition=has_les_definition || (field.rfind("les_",0)==0
                && field!="les_species_coeffs" && field!="les_moment_coeffs")
                || field=="lr_sigma" || field=="remove_self_interaction";
        }
        ifs.clear();ifs.seekg(start);
        if (has_sog) {
            try { set_sog_definition(SOGModel::read(ifs),true); }
            catch (const std::exception& e) { GPERROR(e.what()); }
            ifs.clear();ifs.seekg(start);
        } else if (has_les_definition) {
            model_les.load_coulomb_definition(ifs);
            lr_les=true;
            ifs.clear();ifs.seekg(start);
        } else if (model_les.kernel==LES::Kernel::SOG && !pending_sog_definition_) {
            model_les=LES();lr_les=false;
        }
        read_flag_early_exit = true;
        nl_inited = false;
        normalize_les_request();
        PotMTP::Load(ifs);
        auto header_fields = MTPMetadata::legacy_model_fields(potential_tag);
        for (const auto &field : footer_fields)
            MTPMetadata::merge_legacy(header_fields, field.first, field.second);
        n_props = model.get_nprops();
        Qs.resize(n_props);
        species_coeff_size = n_props * species_count + species_count;
        moment_coeff_size = alpha_scalar_moments * n_props;
        model_coeff_size = model.get_ncoeffs();
        if (potential_tag.text("type") == "MTP_M") {
            const auto expected = model_metadata();
            for (const auto &key : {"model", "q", "model_coeffs_count"}) {
                auto it = potential_tag.fields.find(key);
                if (it != potential_tag.fields.end()) {
                    auto wanted = expected.find(key);
                    MTPMetadata::require(wanted != expected.end() && it->second == wanted->second,
                        std::string(key) + " conflicts with the configured MTP_M model");
                }
            }
            if (potential_tag.fields.count("nn")) {
                MTPMetadata::require(expected.count("nn"), "nn conflicts with the configured MTP_M model");
                for (const auto &key : {"activation", "hidden_layers"})
                    MTPMetadata::require(potential_tag.fields.at("nn").get(key) == expected.at("nn").get(key),
                        std::string("nn.") + key + " conflicts with the configured MTP_M model");
            }
            const bool tagged_lr = potential_tag.fields.count("long_range");
            MTPMetadata::require(tagged_lr == lr_les, "long_range conflicts with MTP_M body");
            if (tagged_lr) for (const auto &key : {"kernel", "channels"})
                MTPMetadata::require(potential_tag.fields.at("long_range").get(key) == expected.at("long_range").get(key),
                    std::string("long_range.") + key + " conflicts with MTP_M body");
        }
        ext_dUdQ.resize(n_props);
        ext_dUdt.resize(model_coeff_size);
        ext_d2UdQdQ.resize(n_props, n_props);
        ext_d2UdQdt.resize(n_props, model_coeff_size);
        reg_coeff_size = radial_coeff_size + moment_coeff_size + species_coeff_size + model_coeff_size;
        if (lr_les) {
            reg_coeff_size += les_head_count();
        }
        allocate_loaded_coefficients();
        species_coeffs = regression_coeffs + radial_coeff_size;
        moment_coeffs = species_coeffs + species_coeff_size;
        model_coeffs = moment_coeffs + moment_coeff_size;
        if (lr_les) {
            les_species = model_coeffs + model_coeff_size;
            les_moment = les_species + les_channels()*species_count;
            les_coeffs = les_species;
        }
        for (int i = radial_coeff_size; i < reg_coeff_size; ++i) {
            regression_coeffs[i] = 1e-3;
        }

        std::string tmpstr;
        if (!(ifs >> tmpstr))
            tmpstr.clear();
        char foo;
        if (tmpstr == "species_coeffs") {
            ifs.ignore(4);
            for (int i = 0; i < species_coeff_size; ++i) {
                require_read(static_cast<bool>(ifs >> species_coeffs[i] >> foo), "species_coeffs");
            }
            ifs.ignore(1000, '\n');
            if (!(ifs >> tmpstr))
                tmpstr.clear();
        }
        if (tmpstr == "moment_coeffs") {
            empty = false;
            ifs.ignore(2);
            ifs.ignore(10, '{');
            for (int i = 0; i < alpha_scalar_moments * n_props; ++i) {
                require_read(static_cast<bool>(ifs >> moment_coeffs[i] >> foo), "moment_coeffs");
            }
            ifs.ignore(1000, '\n');
            if (!(ifs >> tmpstr))
                tmpstr.clear();
        } else {
            empty = true;
        }
        // model coeffsq
        model.coeffs = model_coeffs;
        if (tmpstr == "model_coeffs") {
            ifs.ignore(2);
            ifs.ignore(10, '{');
            for (int i = 0; i < model_coeff_size; ++i) {
                require_read(static_cast<bool>(ifs >> model_coeffs[i] >> foo), "model_coeffs");
            }
            ifs.ignore(1000, '\n');
            if (!(ifs >> tmpstr))
                tmpstr.clear();

            nl_inited = true;
        }
        model.postinit();
        if (lr_les) {
            const auto read_latent=[&](double* values,int count) {
                char c;
                const auto expect=[&](char wanted) {
                    require_read(static_cast<bool>(ifs>>c) && c==wanted,"LES coefficient delimiter");
                };
                expect('=');expect('{');
                for (int i=0;i<count;++i) {
                    if (i) expect(',');
                    require_read(static_cast<bool>(ifs>>values[i]) && std::isfinite(values[i]),"LES coefficient");
                }
                ifs>>std::ws;if (ifs.peek()==',') ifs.get();expect('}');
            };
            if (tmpstr != "les_species_coeffs") {
                GPERROR("PotMTP_M expected les_species_coeffs section");
            }
            read_latent(les_species,les_channels()*species_count);
            ifs.ignore(1000, '\n');
            require_read(static_cast<bool>(ifs >> tmpstr), "next section tag after les_species_coeffs");
            if (tmpstr != "les_moment_coeffs") {
                GPERROR("PotMTP_M expected les_moment_coeffs section");
            }
            read_latent(les_moment,les_channels()*alpha_scalar_moments);
            ifs.ignore(1000, '\n');
            if (!(ifs >> tmpstr))
                tmpstr.clear();
        } else if (tmpstr == "les_species_coeffs" || tmpstr == "les_moment_coeffs") {
            GPERROR("PotMTP_M file contains LES sections but lr_les is false");
        }

        // handle other things
        n_coeffs_fitting = calc_NL_coeffs_count();

        init_basic();

        if (!read_flag_immediate_return && tmpstr == "max_level") {
            ifs.ignore(2);
            require_read(static_cast<bool>(ifs >> max_level), "max_level");
            ifs.ignore(1000, '\n');
            require_read(static_cast<bool>(ifs >> tmpstr), "lev_scalar_nodes tag");
            if (tmpstr != "lev_scalar_nodes") {
                GPERROR("PotMTP_M expected lev_scalar_nodes section");
            }
            ifs.ignore(2);
            ifs.ignore(10, '{');
            lev_scalar_nodes.resize(max_level + 1);
            for (int i = 0; i < max_level + 1; ++i) {
                require_read(static_cast<bool>(ifs >> lev_scalar_nodes[i] >> foo), "lev_scalar_nodes");
            }
            ifs.ignore(1000, '\n');
            require_read(static_cast<bool>(ifs >> tmpstr), "lev_scalar_count tag");
            if (tmpstr != "lev_scalar_count") {
                GPERROR("PotMTP_M expected lev_scalar_count section");
            }
            ifs.ignore(2);
            ifs.ignore(10, '{');
            lev_scalar_count.resize(max_level + 1);
            for (int i = 0; i < max_level + 1; ++i) {
                require_read(static_cast<bool>(ifs >> lev_scalar_count[i] >> foo), "lev_scalar_count");
            }
            ifs.ignore(1000, '\n');
            require_read(static_cast<bool>(ifs >> tmpstr), "lev_times_count tag");
            if (tmpstr != "lev_times_count") {
                GPERROR("PotMTP_M expected lev_times_count section");
            }
            ifs.ignore(2);
            ifs.ignore(10, '{');
            lev_times_count.resize(max_level + 1);
            for (int i = 0; i < max_level + 1; ++i) {
                require_read(static_cast<bool>(ifs >> lev_times_count[i] >> foo), "lev_times_count");
            }
            lev_times_nodes.resize(max_level + 1);
            lev_times_nodes[0] = alpha_index_basic_count;
            for (int i = 1; i < max_level + 1; ++i) {
                lev_times_nodes[i] = lev_times_nodes[i - 1] + lev_times_count[i - 1];
            }
        }
        pending_sog_definition_ = false;
        nl_inited = !empty;
    }
    catch (...) {
        clear_failed_load();
        throw;
    }
    virtual void Save(const std::string& potfile)
    {
        if (save_gpm_if_requested(*this, potfile)) return;
        read_flag_early_exit = true;
        std::string c = potfile;
        auto ofs = open_ofile(c);
        if (!ofs) {
            throw std::runtime_error("Failed to open file: " + potfile);
        }
        Save(*ofs);
    }
    void Save(std::ostream& os)
    {
        read_flag_early_exit = true;
        write_header(os);
        write_radial_info(os);
        write_radial_coeff(os);
        write_rules(os);
        write_linear_coeff(os);
        os << "model_coeffs = {";
        for (int i = 0; i < model_coeff_size; ++i) {
            if (i != 0) {
                os << ", ";
            }
            os << model_coeffs[i];
        }
        os << "}\n";
        if (lr_les) {
            os << "les_species_coeffs = {";
            for (int i = 0; i < les_channels()*species_count; ++i) {
                if (i != 0) {
                    os << ", ";
                }
                os << les_species[i];
            }
            os << "}\n";
            os << "les_moment_coeffs = {";
            for (int i = 0; i < les_channels()*alpha_scalar_moments; ++i) {
                if (i != 0) {
                    os << ", ";
                }
                os << les_moment[i];
            }
            os << "}\n";
        }
        if (uses_sog()) {
            try { sog_definition().write(os); }
            catch (const std::exception& e) { GPERROR(e.what()); }
        } else if (lr_les) {
            model_les.save_coulomb_definition(os);
        }
        // These comments make a fitted short-range MTP-M potential
        // self-describing for the standalone LAMMPS reader while remaining
        // invisible to legacy IMRMLP readers.  Do not advertise LES potentials:
        // the standalone reader will then reject their model_coeffs section
        // instead of silently dropping the long-range contribution.
        if (!lr_les) {
            os << "# mtp_model = " << serialized_model_name() << "\n";
            os << "# mtp_descriptor_count = " << n_props << "\n";
            os << "# mtp_model_coeffs_count = " << model_coeff_size << "\n";
            if constexpr (std::is_same_v<T, m_model_nn1>) {
                os << "# mtp_activation = "
                   << serialized_activation_name(model.acfunc_enum) << "\n";
                os << "# mtp_hidden1 = " << model.n_neuron << "\n";
                os << "# mtp_hidden2 = 0\n";
            } else if constexpr (std::is_same_v<T, m_model_nn2>) {
                os << "# mtp_activation = "
                   << serialized_activation_name(model.nn.acfunc) << "\n";
                os << "# mtp_hidden1 = " << model.n_neuron << "\n";
                os << "# mtp_hidden2 = " << model.n_neuron2 << "\n";
            }
        }
    }

    static constexpr const char* serialized_model_name()
    {
        if constexpr (std::is_same_v<T, m_model_mtpq2>) return "mtp_m_q2";
        if constexpr (std::is_same_v<T, m_model_mtpfs>) return "mtp_m_fs";
        if constexpr (std::is_same_v<T, m_model_mtpli>) return "mtp_m_li";
        if constexpr (std::is_same_v<T, m_model_nn1>) return "mtp_m_n1";
        if constexpr (std::is_same_v<T, m_model_nn2>) return "mtp_m_n2";
        return "unknown";
    }

    static const char* serialized_activation_name(
        m_model_nn1::ActivationFunction activation)
    {
        switch (activation) {
        case m_model_nn1::TANH: return "TANH";
        case m_model_nn1::RELU: return "RELU";
        case m_model_nn1::SIGMOID: return "SIGMOID";
        case m_model_nn1::LINEAR: return "LINEAR";
        }
        return "LINEAR";
    }

    static const char* serialized_activation_name(
        IMRMLPNN::ActivationFunction activation)
    {
        switch (activation) {
        case IMRMLPNN::TANH: return "TANH";
        case IMRMLPNN::RELU: return "RELU";
        case IMRMLPNN::SIGMOID: return "SIGMOID";
        }
        return "TANH";
    }

    void getE(Atoms& atom) override {
        require_loaded_model();
        if (lr_les) LES::validate_periodic_geometry(atom);
        Neighborhoods nbhs(atom, cutoff);
        getE(atom, nbhs);
    }
    void getEFS(Atoms& atom) override {
        require_loaded_model();
        if (lr_les) LES::validate_periodic_geometry(atom);
        Neighborhoods nbhs(atom, cutoff);
        getEFS(atom, nbhs);
    }
    void getEOMP(Atoms& atom) override {
        require_loaded_model();
        if (lr_les) LES::validate_periodic_geometry(atom);
        Neighborhoods nbhs(atom, cutoff);
        getEOMP(atom, nbhs);
    }
    void getEOMP(Atoms& atom, AnyNeighborhoods& nbhs) override {
        if (lr_les) { getE(atom,nbhs); return; }
        get_short_range_efs_omp<0>(atom,nbhs);
    }
    void getEFSOMP(Atoms& atom) override {
        require_loaded_model();
        if (lr_les) LES::validate_periodic_geometry(atom);
        Neighborhoods nbhs(atom, cutoff);
        getEFSOMP(atom, nbhs);
    }
    void getEFSOMP(Atoms& atom, AnyNeighborhoods& nbhs) override {
        if (lr_les) getEFS(atom, nbhs);
        else get_short_range_efs_omp<1>(atom,nbhs);
    }
    using PotMTP::getEFSSymmOMP;
    void getEFSSymmOMP(Atoms& atom, double symprec = 1e-2) override {
        getEFSOMP(atom);
    }
    void getEFSOMPthread(Atoms& atom) override {
        require_loaded_model();
        if (lr_les) LES::validate_periodic_geometry(atom);
        Neighborhoods nbhs(atom, cutoff);
        getEFSOMPthread(atom, nbhs);
    }
    void getEFSOMPthread(Atoms& atom, AnyNeighborhoods& nbhs) override {
        if (!lr_les) {
            auto workspace=make_site_workspace();
            getEFS(atom,nbhs,workspace);
            return;
        }
        auto workspace=make_evaluation_workspace();
        getEFS(atom,nbhs,workspace);
    }

    // Explicit per-call structure evaluation; the reciprocal contribution has
    // its own structure workspace and is deliberately handled separately.
    void getE(Atoms& atom,AnyNeighborhoods& nbhs,SiteWorkspace& workspace) {
        get_short_range_efs<0>(atom,nbhs,workspace);
    }
    void getEFS(Atoms& atom,AnyNeighborhoods& nbhs,SiteWorkspace& workspace) {
        get_short_range_efs<1>(atom,nbhs,workspace);
    }
    template<int REQ_F>
    void get_short_range_efs(Atoms& atom,AnyNeighborhoods& nbhs,SiteWorkspace& workspace) {
        require_loaded_model();
        if (lr_les) GPERROR("MTP-M reciprocal evaluation requires a structure workspace");
        if (!add_efs) atom.resetEFS(); else atom.setEFSflag();
        for (int site=0;site<nbhs.size();++site) {
            auto& nbh=nbhs[site];
            auto& buffer=workspace.moments;
            CalcSiteEFS<REQ_F>(nbh,buffer.buff_site_energy,buffer.buff_site_energy_ders,workspace);
            atom.energy_+=buffer.buff_site_energy;
            atom.site_energies_[site]+=buffer.buff_site_energy;
            if constexpr(REQ_F) for(int neighbor=0;neighbor<nbh.nbhcount_;++neighbor) {
                atom.forces_[site]+=buffer.buff_site_energy_ders.row(neighbor);
                atom.forces_[nbh.indices_[neighbor]]-=buffer.buff_site_energy_ders.row(neighbor);
                for(int a=0;a<3;++a) for(int b=0;b<3;++b)
                    atom.stress_(a,b)-=buffer.buff_site_energy_ders(neighbor,a)*nbh.pos_[neighbor][b];
            }
        }
    }
    template<int REQ_F>
    void get_short_range_efs_omp(Atoms& atom,AnyNeighborhoods& nbhs) {
        require_loaded_model();
        const int count=std::min(get_omp_max_threads(),std::max(1,nbhs.size()));
        std::vector<SiteWorkspace> workers(count);
        for(auto& workspace:workers) {
            prepare_site_workspace(workspace);
            workspace.moments.energy=0.;
            workspace.moments.force.assign(atom.size(),Eigen::Vector3d::Zero());
            workspace.moments.stress.setZero();
        }
        if (!add_efs) atom.resetEFS(); else atom.setEFSflag();
#ifdef USE_OMP
#pragma omp parallel for num_threads(count) schedule(static)
#endif
        for(int site=0;site<nbhs.size();++site) {
            int thread=0;
#ifdef USE_OMP
            thread=omp_get_thread_num();
#endif
            auto& workspace=workers[thread];auto& buffer=workspace.moments;
            auto& nbh=nbhs[site];
            CalcSiteEFS<REQ_F>(nbh,buffer.buff_site_energy,buffer.buff_site_energy_ders,workspace,thread);
            buffer.energy+=buffer.buff_site_energy;
            atom.site_energies_[site]+=buffer.buff_site_energy;
            if constexpr(REQ_F) for(int neighbor=0;neighbor<nbh.nbhcount_;++neighbor) {
                buffer.force[site]+=buffer.buff_site_energy_ders.row(neighbor).transpose();
                buffer.force[nbh.indices_[neighbor]]-=buffer.buff_site_energy_ders.row(neighbor).transpose();
                for(int a=0;a<3;++a) for(int b=0;b<3;++b)
                    buffer.stress(a,b)-=buffer.buff_site_energy_ders(neighbor,a)*nbh.pos_[neighbor][b];
            }
        }
        for(const auto& workspace:workers) {
            const auto& buffer=workspace.moments;atom.energy_+=buffer.energy;
            if constexpr(REQ_F) {
                atom.stress_+=buffer.stress;
                for(int i=0;i<atom.size();++i) atom.forces_[i]+=buffer.force[i].transpose();
            }
        }
    }

    void getE(Atoms& atom, AnyNeighborhoods& nbhs)
    {
        getEFS_impl<0>(atom, nbhs);
    }

    void getEFS(Atoms& atom, AnyNeighborhoods& nbhs)
    {
        getEFS_impl<1>(atom, nbhs);
    }

    template <int REQ_F>
    void getEFS_impl(Atoms& atom, AnyNeighborhoods& nbhs)
    {
        getEFS_impl<REQ_F>(atom,nbhs,serial_evaluation_workspace_);
        sync_evaluation_diagnostics();
    }

    void applyHessianVectorProduct(
        Atoms &atom, AnyNeighborhoods &nbhs,
        const Eigen::VectorXd &direction,
        Eigen::VectorXd &product) override
    {
        applyHessianVectorProduct(atom,nbhs,direction,product,serial_evaluation_workspace_);
        if(lr_les) model_les.publish_legacy_diagnostics(serial_evaluation_workspace_.reciprocal);
    }

    // The base OMP kernels use the linear MTP model through static dispatch.
    // Use the model-aware serial implementations for these public entry points.
    void getEFSHessianOMP(Atoms& atom, AnyNeighborhoods& nbhs, HessianMatrix& hessian) override {
        getEFSHessian(atom, nbhs, hessian);
    }
    void getEFSHessianCellOMP(Atoms& atom, AnyNeighborhoods& nbhs, HessianMatrix& hessian,
        Eigen::MatrixXd& cellhess, Eigen::MatrixXd& lrhess) override {
        getEFSHessianCell(atom, nbhs, hessian, cellhess, lrhess);
    }
    void getEFSHessianCellSymmOMP(Atoms& atom, HessianMatrix& hessian,
        Eigen::MatrixXd& cellhess, Eigen::MatrixXd& lrhess, double symprec = 1e-4) override {
        getEFSHessianCellSymm(atom, hessian, cellhess, lrhess, symprec);
    }
    void getHessian(Atoms& atom, AnyNeighborhoods& nbhs,
        std::vector<std::pair<int,int>> inds, HessianSparse& hessian) override {
        if (lr_les) { assemble_requested_hessian(*this,atom,nbhs,inds,hessian); return; }
        if (nbhs.size() != atom.size()) GPERROR("Hessian neighborhoods have the wrong size");
        HessianSparse result;
        result.size = atom.size();
        for (auto [i, j] : inds) {
            if (i < 0 || j < 0 || i >= atom.size() || j >= atom.size())
                GPERROR("Sparse Hessian atom index is out of range");
            if (i > j) std::swap(i, j);
            result.hessian[{i, j}] = Eigen::Matrix3d::Zero();
        }
        // Scatter local relative-coordinate curvature only into requested
        // Cartesian blocks. Repeated periodic images retain their source atom
        // index; the center terms then also cancel periodic self images.
        for (int ii = 0; ii < nbhs.size() && !result.hessian.empty(); ++ii) {
            auto &nbh = nbhs[ii];
            CalcSiteHessLowMemoryM(nbh, buff_site_energy,
                buff_site_energy_ders, buff_site_hessian, 0);
            const auto add_block = [&](int i, int j, int ai, int aj, double sign) {
                if (i > j) return; // The transposed local pair supplies the other half.
                auto found = result.hessian.find({i, j});
                if (found == result.hessian.end()) return;
                for (int a = 0; a < 3; ++a)
                    for (int b = 0; b < 3; ++b)
                        found->second(a, b) += sign * buff_site_hessian(ai, a, aj, b);
            };
            for (int ai = 0; ai < nbh.nbhcount_; ++ai)
                for (int aj = 0; aj < nbh.nbhcount_; ++aj) {
                    add_block(nbh.indices_[ai], nbh.indices_[aj], ai, aj, 1.);
                    add_block(ii, nbh.indices_[aj], ai, aj, -1.);
                    add_block(nbh.indices_[ai], ii, ai, aj, -1.);
                    add_block(ii, ii, ai, aj, 1.);
                }
        }
        hessian = std::move(result);
    }
    void getEFSHessianSymmDiaOMP(Atoms& atom, std::vector<double>& diagonal, double symprec = 1e-4) override {
        if (lr_les) { Neighborhoods nbhs(atom,cutoff); assemble_cartesian_diagonal(*this,atom,nbhs,diagonal); return; }
        Neighborhoods nbhs(atom, cutoff);
        if (!add_efs) atom.resetEFS(); else atom.setEFSflag();
        diagonal.assign(3 * atom.size(), 0.);
        // The symmetry argument is an optional optimization only. Direct site
        // assembly preserves every repeated-image contribution without a
        // global atomic Hessian or symmetry-dependent atom mapping.
        for (int ii = 0; ii < nbhs.size(); ++ii) {
            auto &nbh = nbhs[ii];
            CalcSiteHessLowMemoryM(nbh, buff_site_energy,
                buff_site_energy_ders, buff_site_hessian, 0);
            atom.site_energies_[ii] += buff_site_energy;
            atom.energy_ += buff_site_energy;
            for (int ai = 0; ai < nbh.nbhcount_; ++ai) {
                atom.forces_[ii] += buff_site_energy_ders.row(ai);
                atom.forces_[nbh.indices_[ai]] -= buff_site_energy_ders.row(ai);
                atom.stress_ -= buff_site_energy_ders.row(ai).transpose() * nbh.pos_[ai];
                for (int aj = 0; aj < nbh.nbhcount_; ++aj)
                    for (int d = 0; d < 3; ++d) {
                        const double value = buff_site_hessian(ai, d, aj, d);
                        diagonal[3 * ii + d] += value;
                        if (nbh.indices_[ai] == nbh.indices_[aj])
                            diagonal[3 * nbh.indices_[ai] + d] += value;
                        if (nbh.indices_[ai] == ii) diagonal[3 * ii + d] -= value;
                        if (nbh.indices_[aj] == ii) diagonal[3 * ii + d] -= value;
                    }
            }
        }
    }

    void getEFSHessian(Atoms& atom, HessianMatrix& hessian)
    {
        Neighborhoods nbhs(atom, cutoff);
        getEFSHessian(atom, nbhs, hessian);
    }

    void getEFSHessian(Atoms& atom, AnyNeighborhoods& nbhs, HessianMatrix& hessian)
    {
        if (lr_les) { getEFSHessian(atom,nbhs,hessian,serial_evaluation_workspace_); sync_evaluation_diagnostics(); return; }
        if (!add_efs)
            atom.resetEFS();
        else
            atom.setEFSflag();
        hessian.resize(atom.size());

        for (int ii = 0; ii < nbhs.size(); ii++) {
            std::cout << "Currently on atom " << ii << " of " << nbhs.size() << " in total" << std::endl;
            Neighborhood& nbh = nbhs[ii];
            CalcSiteHessLowMemoryM(nbh, buff_site_energy,
                                   buff_site_energy_ders,
                                   buff_site_hessian, 0);
            // compute energy
            atom.site_energies_[ii] += buff_site_energy;
            atom.energy_ += buff_site_energy;
            // compute force and stress
            for (int i = 0; i < nbh.nbhcount_; ++i) {
                atom.forces_[ii](0) += buff_site_energy_ders(i, 0);
                atom.forces_[ii](1) += buff_site_energy_ders(i, 1);
                atom.forces_[ii](2) += buff_site_energy_ders(i, 2);
                atom.forces_[nbh.indices_[i]](0) -= buff_site_energy_ders(i, 0);
                atom.forces_[nbh.indices_[i]](1) -= buff_site_energy_ders(i, 1);
                atom.forces_[nbh.indices_[i]](2) -= buff_site_energy_ders(i, 2);

                for (int a = 0; a < 3; a++) {
                    for (int b = 0; b < 3; b++) {
                        atom.stress_(a, b) -= buff_site_energy_ders(i, a) * nbh.pos_[i](b);
                    }
                }
            }
            // compute hessian matrix
            /*
                1. i != j
                d2E/dxidxj = \sum_{m!=i,m!=j} d2Vm/d(xi-xm)d(xj-xm) -
                             \sum_{n} d2Vi/d(xn-xi)d(xj-xi) -
                             \sum_{n} d2Vj/d(xn-xj)d(xi-xj)
                2. i == j
                d2E/dxidxi = \sum_{m!=i} d2Vm/d(xi-xm)2 +
                             \sum_{n} d2Vi/d(xn-xi)2
            */
            // first term, \sum_{m!=i,m!=j} d2Vm/d(xi-xm)d(xj-xm)
            for (int ai = 0; ai < nbh.nbhcount_; ++ai) {
                int ati = nbh.indices_[ai];
                for (int aj = 0; aj < nbh.nbhcount_; ++aj) {
                    int atj = nbh.indices_[aj];
                    for (int a = 0; a < 3; ++a) {
                        for (int b = 0; b < 3; ++b) {
                            hessian(ati, a, atj, b) += buff_site_hessian(ai, a, aj, b);
                        }
                    }
                }
            }
            // second term, i != j, -\sum_{n} d2Vi/d(xn-xi)d(xj-xi) -\sum_{n} d2Vj/d(xi-xj)d(xn-xj)
            for (int jj = 0; jj < nbh.nbhcount_; ++jj) {
                int atj = nbh.indices_[jj];
                for (int a = 0; a < 3; ++a) {
                    for (int b = 0; b < 3; ++b) {
                        for (int kk = 0; kk < nbh.nbhcount_; ++kk) {
                            // int atk = nbh.indices_[kk];
                            hessian(ii, a, atj, b) -= buff_site_hessian(kk, a, jj, b);
                            hessian(atj, a, ii, b) -= buff_site_hessian(jj, a, kk, b);
                        }
                    }
                }
            }
            // second term, i == j, \sum_{m,n} d2Vi/d(xn-xi)d(xm-xi)
            for (int jj = 0; jj < nbh.nbhcount_; ++jj) {
                for (int kk = 0; kk < nbh.nbhcount_; ++kk) {
                    for (int a = 0; a < 3; ++a) {
                        for (int b = 0; b < 3; ++b) {
                            hessian(ii, a, ii, b) += buff_site_hessian(jj, a, kk, b);
                        }
                    }
                }
            }
        }
    }

    void getEFSHessianSymmOMP(Atoms& atom, HessianMatrix& hessian, double symprec = 1e-4) override
    {
        getEFSHessianSymm(atom, hessian, symprec);
    };

    void getEFSHessianSymm(Atoms& atom, HessianMatrix& hessian, double symprec = 1e-4)
    {
        if (lr_les) { getEFSHessian(atom,hessian); return; }
        if (!add_efs)
            atom.resetEFS();
        else
            atom.setEFSflag();
        Neighborhoods nbhs(atom, cutoff);
        hessian.resize(atom.size());
        // 1. find permitive cell and mapping relations, only translation symmetry
        auto dataset = Symmetry3D::get_sym_dataset(atom, symprec);
        auto& mapping_to_primitive = dataset.mapping_to_primitive;
        std::set<int> nonequat;
        std::vector<int> nonequatid;
        std::vector<int> nonequind;
        for (int i = 0; i < mapping_to_primitive.size(); ++i) {
            int primid = mapping_to_primitive[i];
            if (nonequat.count(primid) == 0) {
                nonequind.push_back(i);
                nonequat.insert(primid);
                nonequatid.push_back(primid);
            }
        }
        // 2. build neighbor list
        std::vector<std::pair<int, Eigen::RowVector3d>> sortind;
        for (int i = 0; i < nbhs.size(); ++i) {
            auto& nbh = nbhs[i];
            // sort
            sortind.resize(nbh.nbhcount_);
            for (int j = 0; j < nbh.nbhcount_; ++j) {
                sortind[j].first = j;
                sortind[j].second = nbh.pos_[j];
            }
            std::sort(sortind.begin(), sortind.end(), [&symprec](const std::pair<int, Eigen::RowVector3d>& a, const std::pair<int, Eigen::RowVector3d>& b) {
                if (a.second(0) < b.second(0) - symprec)
                    return true;
                if (a.second(0) > b.second(0) + symprec)
                    return false;

                if (a.second(1) < b.second(1) - symprec)
                    return true;
                if (a.second(1) > b.second(1) + symprec)
                    return false;

                if (a.second(2) < b.second(2) - symprec)
                    return true;
                if (a.second(2) > b.second(2) + symprec)
                    return false;

                return false;
            });
            // reorder
            auto nbhbck = nbh;
            for (int j = 0; j < nbh.nbhcount_; ++j) {
                nbh.chemmap_[j] = nbhbck.chemmap_[sortind[j].first];
                nbh.distances_[j] = nbhbck.distances_[sortind[j].first];
                nbh.indices_[j] = nbhbck.indices_[sortind[j].first];
                nbh.pos_[j] = nbhbck.pos_[sortind[j].first];
            }
        }
        // 3. compute
        for (int in = 0; in < nonequind.size(); ++in) {
            auto& nbhp = nbhs[nonequind[in]];
            int aid = nonequind[in];
            Message(fmt::format("Currently on atom {} of {} in total", in, nonequind.size()));
            CalcSiteHessLowMemoryM(nbhp, buff_site_energy,
                                   buff_site_energy_ders,
                                   buff_site_hessian, 0);

            double siteen = buff_site_energy;
            for (int ii = 0; ii < nbhs.size(); ++ii) {
                if (mapping_to_primitive[ii] != nonequatid[in])
                    continue;
                auto& nbh = nbhs[ii];
                // energy
                atom.site_energies_[ii] += siteen;
                atom.energy_ += siteen;
                // force
                for (int i = 0; i < nbh.nbhcount_; ++i) {
                    atom.forces_[ii](0) += buff_site_energy_ders(i, 0);
                    atom.forces_[ii](1) += buff_site_energy_ders(i, 1);
                    atom.forces_[ii](2) += buff_site_energy_ders(i, 2);
                    atom.forces_[nbh.indices_[i]](0) -= buff_site_energy_ders(i, 0);
                    atom.forces_[nbh.indices_[i]](1) -= buff_site_energy_ders(i, 1);
                    atom.forces_[nbh.indices_[i]](2) -= buff_site_energy_ders(i, 2);
                    // stress
                    for (int a = 0; a < 3; a++) {
                        for (int b = 0; b < 3; b++) {
                            atom.stress_(a, b) -= buff_site_energy_ders(i, a) * nbh.pos_[i](b);
                        }
                    }
                }
                // hessian
                for (int ai = 0; ai < nbh.nbhcount_; ++ai) {
                    int ati = nbh.indices_[ai];
                    for (int aj = 0; aj < nbh.nbhcount_; ++aj) {
                        int atj = nbh.indices_[aj];
                        for (int a = 0; a < 3; ++a) {
                            for (int b = 0; b < 3; ++b) {
                                hessian(ati, a, atj, b) += buff_site_hessian(ai, a, aj, b);
                            }
                        }
                    }
                }
                // second term, i != j, -\sum_{n} d2Vi/d(xn-xi)d(xj-xi) -\sum_{n} d2Vj/d(xi-xj)d(xn-xj)
                for (int jj = 0; jj < nbh.nbhcount_; ++jj) {
                    int atj = nbh.indices_[jj];
                    for (int a = 0; a < 3; ++a) {
                        for (int b = 0; b < 3; ++b) {
                            for (int kk = 0; kk < nbh.nbhcount_; ++kk) {
                                // int atk = nbh.indices_[kk];
                                hessian(ii, a, atj, b) -= buff_site_hessian(kk, a, jj, b);
                                hessian(atj, a, ii, b) -= buff_site_hessian(jj, a, kk, b);
                            }
                        }
                    }
                }
                // second term, i == j, \sum_{m,n} d2Vi/d(xn-xi)d(xm-xi)
                for (int jj = 0; jj < nbh.nbhcount_; ++jj) {
                    for (int kk = 0; kk < nbh.nbhcount_; ++kk) {
                        for (int a = 0; a < 3; ++a) {
                            for (int b = 0; b < 3; ++b) {
                                hessian(ii, a, ii, b) += buff_site_hessian(jj, a, kk, b);
                            }
                        }
                    }
                }
            }
        }
    }

    void getEFSHessianCell(Atoms& atom, HessianMatrix& hessian, Eigen::MatrixXd& cellhess, Eigen::MatrixXd& lrhess)
    {
        Neighborhoods nbhs(atom, cutoff);
        getEFSHessianCell(atom, nbhs, hessian, cellhess, lrhess);
    }

    void getEFSHessianCell(Atoms& atom, AnyNeighborhoods& nbhs, HessianMatrix& hessian, Eigen::MatrixXd& cellhess, Eigen::MatrixXd& lrhess)
    {
        if(lr_les) {
            getEFSHessianCell(atom,nbhs,hessian,cellhess,lrhess,serial_evaluation_workspace_);sync_evaluation_diagnostics();return;
        }
        if (!add_efs)
            atom.resetEFS();
        else
            atom.setEFSflag();
        auto rcell = atom.cell_.inverse();
        hessian.resize(atom.size());
        cellhess.resize(9, 9);
        cellhess.setZero();
        lrhess.resize(9, 3 * atom.size());
        lrhess.setZero();

        for (int ii = 0; ii < nbhs.size(); ii++) {
            std::cout << "Currently on atom " << ii << " of " << nbhs.size() << " in total" << std::endl;
            Neighborhood& nbh = nbhs[ii];
            buff_site_energy = 0;
            CalcSiteHess(nbh, buff_site_energy, buff_site_energy_ders, buff_site_hessian, 0);
            // compute energy
            atom.site_energies_[ii] += buff_site_energy;
            atom.energy_ += buff_site_energy;
            // compute force and stress
            for (int i = 0; i < nbh.nbhcount_; ++i) {
                /**
                dE/dxj = \sum_{i!=j}dVi/d(xj-xi) - \sum_{i}dVj/d(xi-xj)
                F = -dE/dxj
                **/
                // later term
                atom.forces_[ii](0) += buff_site_energy_ders(i, 0);
                atom.forces_[ii](1) += buff_site_energy_ders(i, 1);
                atom.forces_[ii](2) += buff_site_energy_ders(i, 2);
                // former term
                atom.forces_[nbh.indices_[i]](0) -= buff_site_energy_ders(i, 0);
                atom.forces_[nbh.indices_[i]](1) -= buff_site_energy_ders(i, 1);
                atom.forces_[nbh.indices_[i]](2) -= buff_site_energy_ders(i, 2);

                for (int a = 0; a < 3; a++) {
                    for (int b = 0; b < 3; b++) {
                        atom.stress_(a, b) -= buff_site_energy_ders(i, a) * nbh.pos_[i](b);
                    }
                }
                // 1st-order correction term applied on cellhess
                // d2r/dLdL
                // when do_realx = False, with this correction, equal to scale_atoms = True
                if (hc_correction) {
                    Eigen::RowVector3d ri = nbh.pos_[i] * rcell;
                    for (int a = 0; a < 3; ++a) {
                        for (int b = 0; b < 3; ++b) {
                            for (int l = 0; l < 3; ++l) {
                                for (int p = 0; p < 3; ++p) {
                                    cellhess(3 * a + b, 3 * l + p) += buff_site_energy_ders(i, b) * ri(l) * rcell(p, a) + buff_site_energy_ders(i, p) * ri(a) * rcell(b, l);
                                }
                            }
                        }
                    }
                }
                // assume dr/dR != 0, but what does this mean???
                int ati = ii;
                int atj = nbh.indices_[i];
                if (rR_correction) {
                    for (int a = 0; a < 3; ++a) {
                        for (int b = 0; b < 3; ++b) {
                            lrhess(3 * a + b, 3 * ati + a) -= rcell(a, a) * buff_site_energy_ders(i, b);
                            lrhess(3 * a + b, 3 * atj + a) += rcell(a, a) * buff_site_energy_ders(i, b);
                        }
                    }
                }
            }
            // compute hessian matrix
            /*
                1. i != j
                d2E/dxidxj = \sum_{m!=i,m!=j} d2Vm/d(xi-xm)d(xj-xm) -
                             \sum_{n} d2Vi/d(xn-xi)d(xj-xi) -
                             \sum_{n} d2Vj/d(xn-xj)d(xi-xj)
                2. i == j
                d2E/dxidxi = \sum_{m!=i} d2Vm/d(xi-xm)2 +
                             \sum_{n} d2Vi/d(xn-xi)2
            */
            // first term, \sum_{m!=i,m!=j} d2Vm/d(xi-xm)d(xj-xm)
            Eigen::RowVector3d ri, rj;
            for (int ai = 0; ai < nbh.nbhcount_; ++ai) {
                int ati = nbh.indices_[ai];
                ri = nbh.pos_[ai] * rcell;
                for (int aj = 0; aj < nbh.nbhcount_; ++aj) {
                    int atj = nbh.indices_[aj];
                    rj = nbh.pos_[aj] * rcell;
                    for (int a = 0; a < 3; ++a) {
                        for (int b = 0; b < 3; ++b) {
                            hessian(ati, a, atj, b) += buff_site_hessian(ai, a, aj, b);
                            for (int l = 0; l < 3; ++l) {
                                for (int p = 0; p < 3; ++p) {
                                    cellhess(3 * a + b, 3 * l + p) += buff_site_hessian(ai, b, aj, p) * ri(a) * rj(l);
                                }
                            }
                            for (int g = 0; g < 3; ++g) {
                                lrhess(3 * a + b, 3 * ii + g) -= ri(a) * buff_site_hessian(ai, b, aj, g);
                                lrhess(3 * a + b, 3 * atj + g) += ri(a) * buff_site_hessian(ai, b, aj, g);
                            }
                        }
                    }
                }
            }
            // second term, i != j, -\sum_{n} d2Vi/d(xn-xi)d(xj-xi) -\sum_{n} d2Vj/d(xi-xj)d(xn-xj)
            for (int jj = 0; jj < nbh.nbhcount_; ++jj) {
                int atj = nbh.indices_[jj];
                for (int a = 0; a < 3; ++a) {
                    for (int b = 0; b < 3; ++b) {
                        for (int kk = 0; kk < nbh.nbhcount_; ++kk) {
                            // int atk = nbh.indices_[kk];
                            hessian(ii, a, atj, b) -= buff_site_hessian(kk, a, jj, b);
                            hessian(atj, a, ii, b) -= buff_site_hessian(jj, a, kk, b);
                        }
                    }
                }
            }
            // second term, i == j, \sum_{m,n} d2Vi/d(xn-xi)d(xm-xi)
            for (int jj = 0; jj < nbh.nbhcount_; ++jj) {
                for (int kk = 0; kk < nbh.nbhcount_; ++kk) {
                    for (int a = 0; a < 3; ++a) {
                        for (int b = 0; b < 3; ++b) {
                            hessian(ii, a, ii, b) += buff_site_hessian(jj, a, kk, b);
                        }
                    }
                }
            }
        }
    }

    void getEFSHessianCellSymm(Atoms& atom, HessianMatrix& hessian, Eigen::MatrixXd& cellhess, Eigen::MatrixXd& lrhess, double symprec)
    {
        if (lr_les) { getEFSHessianCell(atom,hessian,cellhess,lrhess); return; }
        if (!add_efs)
            atom.resetEFS();
        else
            atom.setEFSflag();
        auto rcell = atom.cell_.inverse();
        Neighborhoods nbhs(atom, cutoff);
        hessian.resize(atom.size());
        cellhess.resize(9, 9);
        cellhess.setZero();
        lrhess.resize(9, 3 * atom.size());
        lrhess.setZero();
        // 1. find permitive cell and mapping relations, only translation symmetry
        auto dataset = Symmetry3D::get_sym_dataset(atom, symprec);
        auto& mapping_to_primitive = dataset.mapping_to_primitive;
        std::set<int> nonequat;
        std::vector<int> nonequatid;
        std::vector<int> nonequind;
        for (int i = 0; i < mapping_to_primitive.size(); ++i) {
            int primid = mapping_to_primitive[i];
            if (nonequat.count(primid) == 0) {
                nonequind.push_back(i);
                nonequat.insert(primid);
                nonequatid.push_back(primid);
            }
        }
        // 2. build neighbor list
        std::vector<std::pair<int, Eigen::RowVector3d>> sortind;
        for (int i = 0; i < nbhs.size(); ++i) {
            auto& nbh = nbhs[i];
            // sort
            sortind.resize(nbh.nbhcount_);
            for (int j = 0; j < nbh.nbhcount_; ++j) {
                sortind[j].first = j;
                sortind[j].second = nbh.pos_[j];
            }
            std::sort(sortind.begin(), sortind.end(), [&symprec](const std::pair<int, Eigen::RowVector3d>& a, const std::pair<int, Eigen::RowVector3d>& b) {
                if (a.second(0) < b.second(0) - symprec)
                    return true;
                if (a.second(0) > b.second(0) + symprec)
                    return false;

                if (a.second(1) < b.second(1) - symprec)
                    return true;
                if (a.second(1) > b.second(1) + symprec)
                    return false;

                if (a.second(2) < b.second(2) - symprec)
                    return true;
                if (a.second(2) > b.second(2) + symprec)
                    return false;

                return false;
            });
            // reorder
            auto nbhbck = nbh;
            for (int j = 0; j < nbh.nbhcount_; ++j) {
                nbh.chemmap_[j] = nbhbck.chemmap_[sortind[j].first];
                nbh.distances_[j] = nbhbck.distances_[sortind[j].first];
                nbh.indices_[j] = nbhbck.indices_[sortind[j].first];
                nbh.pos_[j] = nbhbck.pos_[sortind[j].first];
            }
        }
        // 3. compute
        for (int in = 0; in < nonequind.size(); ++in) {
            auto& nbhp = nbhs[nonequind[in]];
            int aid = nonequind[in];
            Message(fmt::format("Currently on atom {} of {} in total", in, nonequind.size()));
            CalcSiteHess(nbhp, buff_site_energy, buff_site_energy_ders, buff_site_hessian, 0);

            double siteen = buff_site_energy;
            for (int ii = 0; ii < nbhs.size(); ++ii) {
                if (mapping_to_primitive[ii] != nonequatid[in])
                    continue;
                auto& nbh = nbhs[ii];
                // energy
                atom.site_energies_[ii] += siteen;
                atom.energy_ += siteen;
                // force
                for (int i = 0; i < nbh.nbhcount_; ++i) {
                    atom.forces_[ii](0) += buff_site_energy_ders(i, 0);
                    atom.forces_[ii](1) += buff_site_energy_ders(i, 1);
                    atom.forces_[ii](2) += buff_site_energy_ders(i, 2);
                    atom.forces_[nbh.indices_[i]](0) -= buff_site_energy_ders(i, 0);
                    atom.forces_[nbh.indices_[i]](1) -= buff_site_energy_ders(i, 1);
                    atom.forces_[nbh.indices_[i]](2) -= buff_site_energy_ders(i, 2);
                    // stress
                    for (int a = 0; a < 3; a++) {
                        for (int b = 0; b < 3; b++) {
                            atom.stress_(a, b) -= buff_site_energy_ders(i, a) * nbh.pos_[i](b);
                        }
                    }
                    // 1st-order correction term applied on cellhess
                    if (hc_correction) {
                        Eigen::RowVector3d ri = nbh.pos_[i] * rcell;
                        for (int a = 0; a < 3; ++a) {
                            for (int b = 0; b < 3; ++b) {
                                for (int l = 0; l < 3; ++l) {
                                    for (int p = 0; p < 3; ++p) {
                                        cellhess(3 * a + b, 3 * l + p) += buff_site_energy_ders(i, b) * ri(l) * rcell(p, a) + buff_site_energy_ders(i, p) * ri(a) * rcell(b, l);
                                    }
                                }
                            }
                        }
                    }
                    int ati = ii;
                    int atj = nbh.indices_[i];
                    if (rR_correction) {
                        for (int a = 0; a < 3; ++a) {
                            for (int b = 0; b < 3; ++b) {
                                lrhess(3 * a + b, 3 * ati + a) -= rcell(a, a) * buff_site_energy_ders(i, b);
                                lrhess(3 * a + b, 3 * atj + a) += rcell(a, a) * buff_site_energy_ders(i, b);
                            }
                        }
                    }
                }
                // hessian
                Eigen::RowVector3d ri, rj;
                for (int ai = 0; ai < nbh.nbhcount_; ++ai) {
                    int ati = nbh.indices_[ai];
                    ri = nbh.pos_[ai] * rcell;
                    for (int aj = 0; aj < nbh.nbhcount_; ++aj) {
                        int atj = nbh.indices_[aj];
                        rj = nbh.pos_[aj] * rcell;
                        for (int a = 0; a < 3; ++a) {
                            for (int b = 0; b < 3; ++b) {
                                hessian(ati, a, atj, b) += buff_site_hessian(ai, a, aj, b);
                                for (int l = 0; l < 3; ++l) {
                                    for (int p = 0; p < 3; ++p) {
                                        cellhess(3 * a + b, 3 * l + p) += buff_site_hessian(ai, b, aj, p) * ri(a) * rj(l);
                                    }
                                }
                                for (int g = 0; g < 3; ++g) {
                                    lrhess(3 * a + b, 3 * ii + g) -= ri(a) * buff_site_hessian(ai, b, aj, g);
                                    lrhess(3 * a + b, 3 * atj + g) += ri(a) * buff_site_hessian(ai, b, aj, g);
                                }
                            }
                        }
                    }
                }
                // second term, i != j, -\sum_{n} d2Vi/d(xn-xi)d(xj-xi) -\sum_{n} d2Vj/d(xi-xj)d(xn-xj)
                for (int jj = 0; jj < nbh.nbhcount_; ++jj) {
                    int atj = nbh.indices_[jj];
                    for (int a = 0; a < 3; ++a) {
                        for (int b = 0; b < 3; ++b) {
                            for (int kk = 0; kk < nbh.nbhcount_; ++kk) {
                                // int atk = nbh.indices_[kk];
                                hessian(ii, a, atj, b) -= buff_site_hessian(kk, a, jj, b);
                                hessian(atj, a, ii, b) -= buff_site_hessian(jj, a, kk, b);
                            }
                        }
                    }
                }
                // second term, i == j, \sum_{m,n} d2Vi/d(xn-xi)d(xm-xi)
                for (int jj = 0; jj < nbh.nbhcount_; ++jj) {
                    for (int kk = 0; kk < nbh.nbhcount_; ++kk) {
                        for (int a = 0; a < 3; ++a) {
                            for (int b = 0; b < 3; ++b) {
                                hessian(ii, a, ii, b) += buff_site_hessian(jj, a, kk, b);
                            }
                        }
                    }
                }
            }
        }
    }
    // NEED TEST
    template <int REQ_F = 1>
    void CalcSiteEFS(Neighborhood& nbh, double& site_energy,
        Eigen::Matrix<double, -1, 3, Eigen::RowMajor>& site_energy_ders, int ithread = 0)
    {
        CalcSiteEFS<REQ_F>(nbh,site_energy,site_energy_ders,serial_workspace_,ithread);
        Q_les=serial_workspace_.q_les;
    }

    template <int REQ_F = 1, class Site = Neighborhood>
    void CalcSiteEFS(Site& nbh, double& site_energy,
        Eigen::Matrix<double, -1, 3, Eigen::RowMajor>& site_energy_ders, SiteWorkspace& workspace, int ithread = 0)
    {
        prepare_site_workspace(workspace);
        auto& moment_vals = workspace.moments.moment_vals;
        auto& moment_jacobian = workspace.moments.moment_jacobian;
        auto& site_energy_ders_wrt_moments_ = workspace.moments.site_energy_ders_wrt_moments_;
        auto& Qs = workspace.moments.Qs;
        auto& ext_dUdQ = workspace.moments.ext_dUdQ;
        WorkspaceModel model{this,workspace};

        // Delegate descriptor computation to shared implementation
        if constexpr (REQ_F) {
            CalcSiteDescriptorsOMP(nbh,workspace.moments);
            site_energy_ders.resize(nbh.nbhcount_, 3);
            site_energy_ders.setZero();
        } else {
            CalcSiteBasisOMP(nbh,workspace.moments);
        }

        // Build Q-vector: Q_j = species_bias + Σ moment_coeffs[j,a] * B_a
        ext_dUdQ.setZero();
        for (int i = 0; i < n_props; ++i) {
            Qs[i] = species_coeffs[species_count + i * species_count + nbh.centertype];
            for (int a = 0; a < alpha_scalar_moments; ++a) {
                Qs[i] += moment_vals[alpha_moment_mapping[a]] * moment_coeffs[i * alpha_scalar_moments + a];
            }
        }
        if (lr_les) {
            workspace.q_les = les_species[nbh.centertype];
            for (int a = 0; a < alpha_scalar_moments; ++a) {
                workspace.q_les += moment_vals[alpha_moment_mapping[a]] * les_moment[a];
            }
        }

        if constexpr (REQ_F) {
            model.propagate(Qs, site_energy, ext_dUdQ, nbh.centertype, ithread);

            // Backpropagation step 1: dE/dB = Σ_j moment_coeffs[j,a] * dE/dQ_j
            std::fill(site_energy_ders_wrt_moments_.begin(), site_energy_ders_wrt_moments_.end(), 0);
            for (int i = 0; i < alpha_scalar_moments; ++i) {
                for (int j = 0; j < n_props; ++j) {
                    site_energy_ders_wrt_moments_[alpha_moment_mapping[i]] += moment_coeffs[j * alpha_scalar_moments + i] * ext_dUdQ(j);
                }
            }

            // Backpropagation step 2: reverse traversal through product rules
            for (int index=alpha_index_times_count-1;index>=0;--index) {
                const int left=alpha_index_times(index,0),right=alpha_index_times(index,1),out=alpha_index_times(index,3);
                const double weight=alpha_index_times(index,2)*site_energy_ders_wrt_moments_[out];
                site_energy_ders_wrt_moments_[right]+=weight*moment_vals[left];
                site_energy_ders_wrt_moments_[left]+=weight*moment_vals[right];
            }

            // Backpropagation step 3: contract with Jacobian → site_energy_ders
            for (int i = 0; i < alpha_index_basic_count; ++i)
                for (int j = 0; j < nbh.nbhcount_; ++j)
                    for (int a = 0; a < 3; a++) {
                        site_energy_ders(j, a) += site_energy_ders_wrt_moments_[i] * moment_jacobian(i, j, a);
                    }
        } else {
            model.propagate(Qs, site_energy, nbh.centertype, ithread);
        }
        // add global species coeffs
        site_energy += species_coeffs[nbh.centertype];
    }

    struct MFC2Model {
        PotMTP_M *owner;
        SiteWorkspace& workspace;
        Eigen::VectorXd& dUdQ;
        Eigen::MatrixXd& d2UdQdQ;
        MFC2Model(PotMTP_M* potential, SiteWorkspace& scratch)
            : owner(potential),workspace(scratch),dUdQ(scratch.moments.ext_dUdQ),
              d2UdQdQ(scratch.ext_d2UdQdQ) {}


        void propagate(const std::vector<double> &basis, double &site_energy,
                       Eigen::VectorXd &gradient, Eigen::MatrixXd &basis_hessian,
                       int &tag, int &ithread) {
            auto& qs=workspace.moments.Qs;
            qs.resize(owner->n_props);
            for (int p = 0; p < owner->n_props; ++p) {
                qs[p] = owner->species_coeffs[
                    owner->species_count + p * owner->species_count + tag];
                for (int a = 0; a < owner->alpha_scalar_moments; ++a)
                    qs[p] += owner->moment_coeffs[
                        p * owner->alpha_scalar_moments + a] * basis[a + 1];
            }

            dUdQ.resize(owner->n_props);
            d2UdQdQ.resize(owner->n_props, owner->n_props);
            WorkspaceModel head{owner,workspace};
            head.propagate(qs, site_energy, dUdQ, d2UdQdQ,
                                   tag, ithread);
            site_energy += owner->species_coeffs[tag];

            gradient.setZero(basis.size());
            gradient[0] = 1.0;
            for (int a = 0; a < owner->alpha_scalar_moments; ++a)
                for (int p = 0; p < owner->n_props; ++p)
                    gradient[a + 1] += dUdQ[p] * owner->moment_coeffs[
                        p * owner->alpha_scalar_moments + a];
            basis_hessian.resize(0, 0);
        }

        void add_coordinate_hessian(
            const Eigen::Tensor<double, 3, Eigen::RowMajor> &moment_jacobian,
            const int *mapping, int scalar_count, int neighbour_count,
            HessianMatrix &site_hessian) const {
            auto& q_jacobian=workspace.moments.Q_jacobian;
            q_jacobian.resize(owner->n_props,neighbour_count,3);
            q_jacobian.setZero();
            for (int p = 0; p < owner->n_props; ++p)
                for (int k = 0; k < scalar_count; ++k)
                    for (int i = 0; i < neighbour_count; ++i)
                        for (int a = 0; a < 3; ++a)
                            q_jacobian(p, i, a) += owner->moment_coeffs[
                                p * scalar_count + k] *
                                moment_jacobian(mapping[k], i, a);

            for (int p = 0; p < owner->n_props; ++p)
                for (int q = 0; q < owner->n_props; ++q) {
                    const double weight = d2UdQdQ(p, q);
                    if (weight == 0.0) continue;
                    for (int i = 0; i < neighbour_count; ++i)
                        for (int j = 0; j < neighbour_count; ++j)
                            for (int a = 0; a < 3; ++a)
                                for (int b = 0; b < 3; ++b)
                                    site_hessian(i, a, j, b) += weight *
                                        q_jacobian(p, i, a) *
                                        q_jacobian(q, j, b);
                }
        }

        void add_coordinate_hessian_vector(
            const Eigen::Tensor<double, 3, Eigen::RowMajor> &moment_jacobian,
            const int *mapping, int scalar_count, int neighbour_count,
            const Eigen::Matrix<double, -1, 3, Eigen::RowMajor> &direction,
            Eigen::Matrix<double, -1, 3, Eigen::RowMajor> &product) const {
            auto& q_jacobian=workspace.moments.Q_jacobian;
            q_jacobian.resize(owner->n_props,neighbour_count,3);
            q_jacobian.setZero();
            auto& q_direction=workspace.q_direction;
            q_direction.setZero(owner->n_props);
            for (int p=0;p<owner->n_props;++p)
                for (int k=0;k<scalar_count;++k)
                    for (int i=0;i<neighbour_count;++i)
                        for (int axis=0;axis<3;++axis)
                            q_jacobian(p,i,axis)+=owner->moment_coeffs[p*scalar_count+k]*moment_jacobian(mapping[k],i,axis);
            for (int p=0;p<owner->n_props;++p)
                for (int i=0;i<neighbour_count;++i)
                    for (int axis=0;axis<3;++axis)
                        q_direction[p]+=q_jacobian(p,i,axis)*direction(i,axis);
            workspace.q_weighted.noalias()=d2UdQdQ*q_direction;
            for (int p=0;p<owner->n_props;++p)
                for (int i=0;i<neighbour_count;++i)
                    for (int axis=0;axis<3;++axis)
                        product(i,axis)+=workspace.q_weighted[p]*q_jacobian(p,i,axis);
        }
    };

    void CalcSiteHessLowMemoryM(
        Neighborhood &nbh, double &site_energy,
        SiteCoordinateMatrix &site_energy_ders, HessianMatrix &site_hessian, int ithread = 0) {
        CalcSiteHessLowMemoryM(nbh,site_energy,site_energy_ders,site_hessian,serial_workspace_,ithread);
    }
    void CalcSiteHessLowMemoryM(
        Neighborhood &nbh, double &site_energy, SiteCoordinateMatrix &site_energy_ders,
        HessianMatrix &site_hessian, SiteWorkspace& workspace, int ithread = 0) {
        prepare_site_workspace(workspace);
        MFC2Model fc2_model{this,workspace};
        CalcSiteHessImpl<true,MFC2Model,0>(nbh,workspace.moments,site_energy,
            site_energy_ders,site_hessian,fc2_model,ithread);
    }
    void CalcSiteHessLowMemoryMOMP(Neighborhood &nbh, ompbuffers &buffer, int ithread) {
        auto workspace=make_site_workspace();
        MFC2Model fc2_model{this,workspace};
        CalcSiteHessImpl<true,MFC2Model,0>(nbh,buffer,buffer.buff_site_energy,
            buffer.buff_site_energy_ders,buffer.buff_site_hessian,fc2_model,ithread);
    }
    void CalcSiteHessianVectorProductM(Neighborhood &nbh,const SiteCoordinateMatrix &direction,
                                     SiteCoordinateMatrix &product) {
        CalcSiteHessianVectorProductM(nbh,direction,product,serial_workspace_);
    }
    void CalcSiteHessianVectorProductM(Neighborhood &nbh,const SiteCoordinateMatrix &direction,
                                     SiteCoordinateMatrix &product,SiteWorkspace& workspace) {
        prepare_site_workspace(workspace);
        MFC2Model fc2_model{this,workspace};
        auto& buffer=workspace.moments;
        CalcSiteHessImpl<true,MFC2Model,0>(nbh,buffer,buffer.buff_site_energy,
            buffer.buff_site_energy_ders,buffer.buff_site_hessian,fc2_model,0,&direction,&product);
    }

    void CalcSiteHess(Neighborhood& nbh, double& site_energy, Eigen::Matrix<double, -1, 3, Eigen::RowMajor>& site_energy_ders, HessianMatrix& site_hessian, int ithread = 0)
    {
        CalcSiteHess(nbh,site_energy,site_energy_ders,site_hessian,serial_workspace_,ithread);
    }

    void CalcSiteHess(Neighborhood& nbh, double& site_energy, Eigen::Matrix<double, -1, 3, Eigen::RowMajor>& site_energy_ders, HessianMatrix& site_hessian, SiteWorkspace& workspace, int ithread = 0)
    {
        prepare_site_workspace(workspace);
        auto& coords_powers = workspace.moments.coords_powers;
        auto& moment_vals = workspace.moments.moment_vals;
        auto& moment_jacobian = workspace.moments.moment_jacobian;
        auto& moment_hessian = workspace.moments.moment_hessian;
        auto& Q_jacobian = workspace.moments.Q_jacobian;
        auto& Q_hessian = workspace.moments.Q_hessian;
        auto& dist_powers = workspace.moments.dist_powers;
        auto& rb_vals = workspace.moments.rb_vals;
        auto& rb_ders = workspace.moments.rb_ders;
        auto& rb_derders = workspace.moments.rb_derders;
        auto& val_mu = workspace.moments.val_mu;
        auto& der_mu = workspace.moments.der_mu;
        auto& derder_mu = workspace.moments.derder_mu;
        auto& site_energy_ders_wrt_moments_ = workspace.moments.site_energy_ders_wrt_moments_;
        auto& dfdra = workspace.moments.dfdra;
        auto& dpdra = workspace.moments.dpdra;
        auto& dmdra = workspace.moments.dmdra;
        auto& d2fdrab = workspace.moments.d2fdrab;
        auto& d2pdrab = workspace.moments.d2pdrab;
        auto& d2mdrab = workspace.moments.d2mdrab;
        auto& Qs = workspace.moments.Qs;
        auto& ext_dUdQ = workspace.moments.ext_dUdQ;
        auto& buff_site_hessian = workspace.moments.buff_site_hessian;
        auto& ext_d2UdQdQ = workspace.ext_d2UdQdQ;
        WorkspaceModel model{this,workspace};
        ext_d2UdQdQ.resize(n_props,n_props);

        // int ns = alpha_scalar_moments + species_count;
        site_energy_ders.resize(nbh.nbhcount_, 3);
        site_energy_ders.setZero();
        site_hessian.resize(nbh.nbhcount_);
        site_hessian.setzero();
        // basis_vals.resize(alpha_scalar_moments + 1);
        // b i alpha
        moment_jacobian.resize(alpha_moments_count, nbh.nbhcount_, 3);
        moment_jacobian.setZero();
        Q_jacobian.resize(n_props, nbh.nbhcount_, 3);
        Q_jacobian.setZero();
        // b i alpha j beta
        moment_hessian.resize(alpha_moments_count);
        Q_hessian.resize(n_props);
        for (int i = 0; i < moment_hessian.size(); ++i) {
            moment_hessian[i].resize(nbh.nbhcount_);
            // moment_hessian[i].setzero();
        }
        for (int i = 0; i < n_props; ++i) {
            Q_hessian[i].resize(nbh.nbhcount_);
        }

        std::fill(site_energy_ders_wrt_moments_.begin(),
            site_energy_ders_wrt_moments_.end(), 0);
        std::fill(moment_vals.begin(), moment_vals.end(), 0);
        std::fill(rb_vals.begin(), rb_vals.end(), 0);
        std::fill(rb_ders.begin(), rb_ders.end(), 0);
        std::fill(rb_derders.begin(), rb_derders.end(), 0);
        std::fill(val_mu.begin(), val_mu.end(), 0);
        std::fill(der_mu.begin(), der_mu.end(), 0);

        const int itype = nbh.centertype;
        for (int jj = 0; jj < nbh.nbhcount_; ++jj) {
            const double& rij = nbh.distances_[jj];
            const int &jtype = nbh.chemmap_[jj];
            if (rij > cutoffs(itype, jtype)) continue;
            const double rijinv = 1.0 / rij;
            rb_calcderder_omp(itype, jtype, rij,workspace.moments);

            dist_powers[0] = 1;
            for (int k = 1; k < max_alpha_index_basic + 4; ++k) {
                dist_powers[k] = dist_powers[k - 1] * rijinv;
            }
            for (int k = 1; k < max_alpha_index_basic; ++k) {
                for (int a = 0; a < 3; a++) {
                    coords_powers(k + 1, a) = coords_powers(k, a) * nbh.pos_[jj][a];
                }
            }
            // dist_powers[max_alpha_index_basic - 1] = 1.0 / dist_powers[max_alpha_index_basic - 1];

            for (int mu = 0; mu < max_mu; ++mu) {
                val_mu[mu] = 0;
                der_mu[mu] = 0;
                derder_mu[mu] = 0;
                int idxoffset = (nbh.centertype * species_count + nbh.chemmap_[jj]) * radial_func_count * rb_size + mu * rb_size;
                for (int xi = 0; xi < rb_size; xi++) {
                    val_mu[mu] += radial_coeffs[idxoffset + xi] * rb_vals[xi];
                    der_mu[mu] += radial_coeffs[idxoffset + xi] * rb_ders[xi];
                    derder_mu[mu] += radial_coeffs[idxoffset + xi] * rb_derders[xi];
                }
            }

            for (int i = 0; i < alpha_index_basic_count; ++i) {
                int mu = alpha_index_basic(i, 0);
                double val = val_mu[mu];
                double der = der_mu[mu];
                double derder = derder_mu[mu]; // d2f/dr2
                const auto& r = nbh.pos_[jj];
                // int k = alpha_index_basic(i, 1) + alpha_index_basic(i, 2) + alpha_index_basic(i, 3); // nv
                int k = nu_arr_list[i];
                double& powk = dist_powers[k]; // pw
                double pow0 = coords_powers(alpha_index_basic(i, 1) + 1, 0);
                double pow1 = coords_powers(alpha_index_basic(i, 2) + 1, 1);
                double pow2 = coords_powers(alpha_index_basic(i, 3) + 1, 2);
                double mult0 = pow0 * pow1 * pow2; // element of tensor r ^ ("x" nv), mt
                double multa1[3] = { coords_powers(alpha_index_basic(i, 1), 0) * pow1 * pow2,
                    pow0 * coords_powers(alpha_index_basic(i, 2), 1) * pow2,
                    pow0 * pow1 * coords_powers(alpha_index_basic(i, 3), 2) };
                double multa2[3][3];

                if (alpha_index_basic(i, 1) < 2)
                    multa2[0][0] = 0;
                else
                    multa2[0][0] = coords_powers(alpha_index_basic(i, 1) - 1, 0) * pow1 * pow2;
                if (alpha_index_basic(i, 2) < 2)
                    multa2[1][1] = 0;
                else
                    multa2[1][1] = pow0 * coords_powers(alpha_index_basic(i, 2) - 1, 1) * pow2;
                if (alpha_index_basic(i, 3) < 2)
                    multa2[2][2] = 0;
                else
                    multa2[2][2] = pow0 * pow1 * coords_powers(alpha_index_basic(i, 3) - 1, 2);
                // ab
                if (alpha_index_basic(i, 1) < 1 || alpha_index_basic(i, 2) < 1)
                    multa2[0][1] = 0;
                else
                    multa2[0][1] = coords_powers(alpha_index_basic(i, 1), 0) * coords_powers(alpha_index_basic(i, 2), 1) * pow2;
                // ac
                if (alpha_index_basic(i, 1) < 1 || alpha_index_basic(i, 3) < 1)
                    multa2[0][2] = 0;
                else
                    multa2[0][2] = coords_powers(alpha_index_basic(i, 1), 0) * pow1 * coords_powers(alpha_index_basic(i, 3), 2);
                // bc
                if (alpha_index_basic(i, 2) < 1 || alpha_index_basic(i, 3) < 1)
                    multa2[1][2] = 0;
                else
                    multa2[1][2] = pow0 * coords_powers(alpha_index_basic(i, 2), 1) * coords_powers(alpha_index_basic(i, 3), 2);
                multa2[1][0] = multa2[0][1];
                multa2[2][0] = multa2[0][2];
                multa2[2][1] = multa2[1][2];

                double dptmp1 = -k * dist_powers[k + 2];
                double dptmp2 = k * (k + 2) * dist_powers[k + 4];
                double dertmp = der * dist_powers[1];
                double der2tmp = (derder - dist_powers[1] * der) * dist_powers[2];
                double ftmpr = der * dist_powers[1];
                for (int a = 0; a < 3; ++a) {
                    dfdra[a] = dertmp * r[a];
                    dpdra[a] = dptmp1 * r[a];
                    dmdra[a] = alpha_index_basic(i, a + 1) * multa1[a];
                    // a == b
                    d2fdrab[a][a] = r[a] * r[a] * der2tmp + ftmpr;
                    d2mdrab[a][a] = (alpha_index_basic(i, a + 1) - 1) * alpha_index_basic(i, a + 1) * multa2[a][a];
                    d2pdrab[a][a] = dptmp2 * r[a] * r[a] + dptmp1;
                    for (int b = a + 1; b < 3; ++b) {
                        d2fdrab[a][b] = r[a] * r[b] * der2tmp;
                        d2fdrab[b][a] = d2fdrab[a][b];
                        d2pdrab[a][b] = dptmp2 * r[a] * r[b];
                        d2pdrab[b][a] = d2pdrab[a][b];
                        d2mdrab[a][b] = alpha_index_basic(i, b + 1) * alpha_index_basic(i, a + 1) * multa2[a][b];
                        d2mdrab[b][a] = d2mdrab[a][b];
                    }
                }
                // b = f * mt * pw
                // db = df * mt * pw + f * dmt * pw + f * mt * dpw
                moment_vals[i] += val * mult0 * powk;
                for (int a = 0; a < 3; ++a) {
                    moment_jacobian(i, jj, a) += dfdra[a] * mult0 * powk + val * dmdra[a] * powk + val * mult0 * dpdra[a];
                    for (int b = a; b < 3; ++b) {
                        moment_hessian[i](jj, a, jj, b) += d2fdrab[a][b] * mult0 * powk + val * d2mdrab[a][b] * powk + val * mult0 * d2pdrab[a][b] + dfdra[a] * (dmdra[b] * powk + mult0 * dpdra[b]) + dfdra[b] * (dmdra[a] * powk + mult0 * dpdra[a]) + val * (dmdra[a] * dpdra[b] + dmdra[b] * dpdra[a]);
                    }
                }
            }
        }
        // fill symmetryc part of moment_hessian basic diongal terms
        for (int i = 0; i < alpha_index_basic_count; ++i) {
            for (int jj = 0; jj < nbh.nbhcount_; ++jj) {
                for (int a = 0; a < 3; ++a) {
                    for (int b = 0; b < a; ++b) {
                        moment_hessian[i](jj, a, jj, b) = moment_hessian[i](jj, b, jj, a);
                    }
                }
            }
        }

        // non-elementary b_i
        for (int i = 0; i < alpha_index_times_count; ++i) {
            // bk = c * bi * bj
            const int& bi = alpha_index_times(i, 0);
            const int& bj = alpha_index_times(i, 1);
            const int& bk = alpha_index_times(i, 3);
            double vbi = moment_vals[bi];
            double vbj = moment_vals[bj];
            double times = alpha_index_times(i, 2);
            moment_vals[bk] += times * vbi * vbj;
            for (int j = 0; j < nbh.nbhcount_; ++j) {
                for (int a = 0; a < 3; ++a) {
                    moment_jacobian(bk, j, a) += times * (moment_jacobian(bi, j, a) * vbj + vbi * moment_jacobian(bj, j, a));
                }
            }
        }
        // moment_hessians, requires moment_jacobian calculated first
        for (int i = 0; i < alpha_index_times_count; ++i) {
            const int& bi = alpha_index_times(i, 0);
            const int& bj = alpha_index_times(i, 1);
            const int& bk = alpha_index_times(i, 3);
            double vbi = moment_vals[bi];
            double vbj = moment_vals[bj];
            double times = alpha_index_times(i, 2);
            for (int a = 0; a < 3; ++a) {
                for (int b = 0; b < 3; ++b) {
                    for (int ati = 0; ati < nbh.nbhcount_; ++ati) {
                        for (int atj = ati; atj < nbh.nbhcount_; ++atj) {
                            moment_hessian[bk](ati, a, atj, b) += times * (moment_hessian[bi](ati, a, atj, b) * vbj + moment_jacobian(bi, ati, a) * moment_jacobian(bj, atj, b) + moment_jacobian(bi, atj, b) * moment_jacobian(bj, ati, a) + vbi * moment_hessian[bj](ati, a, atj, b));
                        }
                    }
                }
            }
        }
        // fill symmetric part of moment_hessian for non-basic tensors
        for (int i = alpha_index_basic_count; i < alpha_moments_count; ++i) {
            for (int a = 0; a < 3; ++a) {
                for (int b = 0; b < 3; ++b) {
                    for (int ati = 0; ati < nbh.nbhcount_; ++ati) {
                        for (int atj = 0; atj < ati; ++atj) {
                            moment_hessian[i](ati, a, atj, b) = moment_hessian[i](atj, b, ati, a);
                        }
                    }
                }
            }
        }
        // Call external model
        Qs.setZero();
        for (int i = 0; i < n_props; ++i) {
            Qs[i] = species_coeffs[species_count + i * species_count + nbh.centertype];
            for (int a = 0; a < alpha_scalar_moments; ++a) {
                Qs[i] += moment_vals[alpha_moment_mapping[a]] * moment_coeffs[i * alpha_scalar_moments + a];
            }
        }
        model.propagate(Qs, site_energy, ext_dUdQ, ext_d2UdQdQ, nbh.centertype, ithread);
        site_energy += species_coeffs[nbh.centertype];
        // compute dq/dr
        for (int a = 0; a < 3; ++a) {
            for (int j = 0; j < nbh.nbhcount_; ++j) {
                for (int m = 0; m < n_props; ++m) {
                    for (int k = 0; k < alpha_scalar_moments; ++k) {
                        Q_jacobian(m, j, a) += moment_coeffs[m * alpha_scalar_moments + k] * moment_jacobian(alpha_moment_mapping[k], j, a);
                    }
                }
            }
        }

        for (int a = 0; a < 3; ++a) {
            for (int j = 0; j < nbh.nbhcount_; ++j) {
                for (int m = 0; m < n_props; ++m) {
                    site_energy_ders(j, a) += ext_dUdQ[m] * Q_jacobian(m, j, a);
                }
            }
        }
        // compute Q_hessian
        for (int i = 0; i < n_props; ++i) {
            for (int k = 0; k < alpha_scalar_moments; ++k) {
                for (int a = 0; a < 3; ++a) {
                    for (int b = 0; b < 3; ++b) {
                        for (int ai = 0; ai < nbh.nbhcount_; ++ai) {
                            for (int aj = 0; aj < nbh.nbhcount_; ++aj) {
                                Q_hessian[i](ai, a, aj, b) += moment_coeffs[i * alpha_scalar_moments + k] * moment_hessian[alpha_moment_mapping[k]](ai, a, aj, b);
                            }
                        }
                    }
                }
            }
        }
        // d2U/dxdy = \sum_i dU/dBi * d2Bi/dxdy + \sum_i,j d2U/dBiBj * dBi/dx * dBj/dy
        // first term
        for (int a = 0; a < 3; ++a)
            for (int b = 0; b < 3; ++b)
                for (int i = 0; i < n_props; ++i)
                    for (int ai = 0; ai < nbh.nbhcount_; ++ai)
                        for (int aj = ai; aj < nbh.nbhcount_; ++aj)
                            site_hessian(ai, a, aj, b) += ext_dUdQ[i] * Q_hessian[i](ai, a, aj, b);
        // second term
        for (int a = 0; a < 3; ++a)
            for (int b = 0; b < 3; ++b)
                for (int i = 0; i < n_props; ++i)
                    for (int j = 0; j < n_props; ++j)
                        for (int ai = 0; ai < nbh.nbhcount_; ++ai)
                            for (int aj = ai; aj < nbh.nbhcount_; ++aj)
                                site_hessian(ai, a, aj, b) += ext_d2UdQdQ(i, j) * Q_jacobian(i, ai, a) * Q_jacobian(j, aj, b);
        // fill symmetric part of buff_site_hessian
        for (int a = 0; a < 3; ++a) {
            for (int b = 0; b < 3; ++b) {
                for (int ai = 0; ai < nbh.nbhcount_; ++ai) {
                    for (int aj = 0; aj < ai; ++aj) {
                        site_hessian(ai, a, aj, b) = site_hessian(aj, b, ai, a);
                    }
                }
            }
        }
    }

    virtual void AccumulateEFSCombinationGard(Atoms& atom,
        AnyNeighborhoods& nbhs,
        double dLdE,
        Eigen::Matrix<double, -1, 3>& dLdF,
        Eigen::Matrix3d& dLdS,
        Eigen::Matrix<double, -1, 3, Eigen::RowMajor>& wgt_d2UdRdx, // buffer
        double* out_gard_accumulator)
    {
        AccumulateEFSCombinationGard(atom,nbhs,dLdE,dLdF,dLdS,out_gard_accumulator,serial_evaluation_workspace_);
        wgt_d2UdRdx=serial_evaluation_workspace_.site.moments.tmp_se_ders_wgts;
    }
    // For energy-only case
    virtual void AccumulateEFSCombinationGard(Atoms& atom,
        AnyNeighborhoods& nbhs,
        double dLdE,
        double* out_gard_accumulator)
    {
        AccumulateEFSCombinationGard(atom,nbhs,dLdE,out_gard_accumulator,serial_evaluation_workspace_);
    }

    void compute_les_charges_and_descriptors(Atoms& atom,
        AnyNeighborhoods& nbhs,
        Eigen::VectorXd& q_latent,
        Eigen::MatrixXd& descriptors,
        std::vector<int>& atom_types, int channel=0)
    {
        compute_les_charges_and_descriptors(atom,nbhs,serial_evaluation_workspace_);
        q_latent=serial_evaluation_workspace_.latents.col(channel);
        descriptors=serial_evaluation_workspace_.descriptors;
        atom_types=serial_evaluation_workspace_.atom_types;
    }

    void AccumulateLESGard(Atoms& atom,
        AnyNeighborhoods& nbhs,
        double dLdE,
        Eigen::Matrix<double, -1, 3>* dLdF,
        Eigen::Matrix3d* dLdS,
        double* out_gard_accumulator)
    {
        accumulate_les_workspace(atom,nbhs,dLdE,dLdF,dLdS,out_gard_accumulator,serial_evaluation_workspace_);
    }

    void AccumulateFPGard(Neighborhood& nbh,
        double* out_gard_accumulator,
        double se_weight,
        Eigen::Matrix<double, -1, 3, Eigen::RowMajor>* f_weight)
    {
        int ithread = 0;
        if (f_weight == nullptr) {
            AccumulateFPGard_impl<0>(nbh, out_gard_accumulator, se_weight, f_weight, ithread);
        } else {
            AccumulateFPGard_impl<1>(nbh, out_gard_accumulator, se_weight, f_weight, ithread);
        }
    }

    void AccumulateFPGard(Neighborhood& nbh,
        double* out_gard_accumulator,
        double se_weight,
        Eigen::Matrix<double, -1, 3, Eigen::RowMajor>* f_weight,
        int ithread = 0)
    {
        if (f_weight == nullptr) {
            AccumulateFPGard_impl<0>(nbh, out_gard_accumulator, se_weight, f_weight, ithread);
        } else {
            AccumulateFPGard_impl<1>(nbh, out_gard_accumulator, se_weight, f_weight, ithread);
        }
    }
    // NEED TEST
    // radical force part is still incorrect
    // REQ_RADICAL: request calculate MTP radial coeffs
    // REQ_MODEL: request calculate external model coeffs
    // REQ_MOMENT: request calculate moment coeffs
    // REQ_F: request force related terms
    void AccumulateFPGard(Neighborhood& nbh,double* gradient,double energy_weight,
        SiteCoordinateMatrix* force_weight,SiteWorkspace& workspace,int ithread=0) {
        if(force_weight) AccumulateFPGard_impl<1>(nbh,gradient,energy_weight,force_weight,workspace,ithread);
        else AccumulateFPGard_impl<0>(nbh,gradient,energy_weight,force_weight,workspace,ithread);
    }

    template <int REQ_F = 1>
    void AccumulateFPGard_impl(Neighborhood& nbh,
        double* out_gard_accumulator,
        double se_weight,
        Eigen::Matrix<double, -1, 3, Eigen::RowMajor>* f_weight,
        int ithread = 0)
    {
        AccumulateFPGard_impl<REQ_F>(nbh,out_gard_accumulator,se_weight,f_weight,serial_workspace_,ithread);
    }

    template <int REQ_F = 1>
    void AccumulateFPGard_impl(Neighborhood& nbh,
        double* out_gard_accumulator,
        double se_weight,
        Eigen::Matrix<double, -1, 3, Eigen::RowMajor>* f_weight,
        SiteWorkspace& workspace, int ithread = 0)
    {
        prepare_site_workspace(workspace);
        auto& coords_powers = workspace.moments.coords_powers;
        auto& moment_vals = workspace.moments.moment_vals;
        auto& moment_jacobian = workspace.moments.moment_jacobian;
        auto& dist_powers = workspace.moments.dist_powers;
        auto& rb_vals = workspace.moments.rb_vals;
        auto& rb_ders = workspace.moments.rb_ders;
        auto& val_mu = workspace.moments.val_mu;
        auto& der_mu = workspace.moments.der_mu;
        auto& site_energy_ders_wrt_moments_ = workspace.moments.site_energy_ders_wrt_moments_;
        auto& moment_jacobian_coeffs = workspace.moments.moment_jacobian_coeffs;
        auto& moment_jacobian_coeff_coord = workspace.moments.moment_jacobian_coeff_coord;
        auto& dloss_dmom = workspace.moments.dloss_dmom;
        auto& dloss_dsenders = workspace.moments.dloss_dsenders;
        auto& Qs = workspace.moments.Qs;
        auto& ext_dUdQ = workspace.moments.ext_dUdQ;
        auto& buff_site_energy = workspace.moments.buff_site_energy;
        auto& ext_dUdt = workspace.ext_dUdt;
        auto& ext_d2UdQdQ = workspace.ext_d2UdQdQ;
        auto& ext_d2UdQdt = workspace.ext_d2UdQdt;
        WorkspaceModel model{this,workspace};
        workspace.moments.moment_jacobian_coeffs.resize(alpha_index_basic_count,species_count,radial_func_count*rb_size);
        workspace.moments.moment_jacobian_coeff_coord.resize(alpha_index_basic_count,species_count,radial_func_count*rb_size);
        workspace.moments.dloss_dmom.resize(alpha_moments_count);
        workspace.moments.dloss_dsenders.resize(alpha_moments_count);
        ext_dUdt.resize(model_coeff_size);
        ext_d2UdQdQ.resize(n_props,n_props);
        ext_d2UdQdt.resize(n_props,model_coeff_size);
        ext_dUdQ.setZero();

        if constexpr (REQ_F) {
            moment_jacobian.resize(alpha_index_basic_count, nbh.nbhcount_, 3);
            moment_jacobian.setZero();
            moment_jacobian_coeff_coord.setZero();
            std::fill(rb_ders.begin(), rb_ders.end(), 0);
            std::fill(der_mu.begin(), der_mu.end(), 0);
        }
        moment_jacobian_coeffs.setZero();
        // buff_site_energy = 0;
        // basis_vals.resize(alpha_scalar_moments + 1, 0);

        std::fill(moment_vals.begin(), moment_vals.end(), 0);
        std::fill(rb_vals.begin(), rb_vals.end(), 0);
        std::fill(val_mu.begin(), val_mu.end(), 0);
        std::fill(dloss_dmom.begin(), dloss_dmom.end(), 0);
        std::fill(dloss_dsenders.begin(), dloss_dsenders.end(), 0);
        std::fill(site_energy_ders_wrt_moments_.begin(), site_energy_ders_wrt_moments_.end(), 0);
        auto& dqdr=workspace.dqdr;
        dqdr.assign(n_props,0.);
        const int &itype = nbh.centertype;
        for (int jj = 0; jj < nbh.nbhcount_; ++jj) {
            const double& rij = nbh.distances_[jj];
            const int &jtype = nbh.chemmap_[jj];
            if (rij > cutoffs(itype, jtype)) continue;
            const double rijinv = 1.0 / rij;
            if constexpr (REQ_F) {
                rb_calcder_omp(itype, jtype, rij,workspace.moments);
            }
            else {
                rb_calcval_omp(itype, jtype, rij,workspace.moments);
            }

            dist_powers[0] = 1;
            for (int k = 1; k < max_alpha_index_basic; ++k) {
                dist_powers[k] = dist_powers[k - 1] * rijinv;
                for (int a = 0; a < 3; a++) {
                    coords_powers(k + 1, a) = coords_powers(k, a) * nbh.pos_[jj][a]; // off set by 1 to aviod if later
                }
            }

            for (int mu = 0; mu < max_mu; ++mu) {
                val_mu[mu] = 0;
                if constexpr (REQ_F)
                    der_mu[mu] = 0;
                int idxoffset = (nbh.centertype * species_count + nbh.chemmap_[jj]) * radial_func_count * rb_size + mu * rb_size;
                for (int xi = 0; xi < rb_size; ++xi) {
                    val_mu[mu] += radial_coeffs[idxoffset + xi] * rb_vals[xi];
                    if constexpr (REQ_F)
                        der_mu[mu] += radial_coeffs[idxoffset + xi] * rb_ders[xi];
                }
            }

            for (int i = 0; i < alpha_index_basic_count; ++i) {
                // @OPTIMIZE: init mu-array while loading potentials, do not need offset search
                int mu = alpha_index_basic(i, 0);
                double val = val_mu[mu];
                double der;
                if constexpr (REQ_F)
                    der = der_mu[mu];
                // int k = alpha_index_basic(i, 1) + alpha_index_basic(i, 2) + alpha_index_basic(i, 3); // nv
                int k = nu_arr_list[i];
                double& powk = dist_powers[k];
                // TODO: test weather move this out can accelearte computation by defining val_mu_nv[:] and der_mu_nv[:]
                // since some (mu, nv) pair may not exist.
                val *= powk; // normalize, all tenser element is devided by rij^\nv to avoid extremely large values
                if constexpr (REQ_F)
                    der = der * powk - k * val / rij;
                // offset by 1
                double pow0 = coords_powers(alpha_index_basic(i, 1) + 1, 0); // x
                double pow1 = coords_powers(alpha_index_basic(i, 2) + 1, 1); // y
                double pow2 = coords_powers(alpha_index_basic(i, 3) + 1, 2); // z

                double mult0 = pow0 * pow1 * pow2; // element of tensor r ^ ("x" nv)

                moment_vals[i] += val * mult0;
                for (int xi = 0; xi < rb_size; xi++) {
                    // dbm / dc
                    moment_jacobian_coeffs(i, nbh.chemmap_[jj], mu * rb_size + xi) += rb_vals[xi] * powk * mult0;
                    // force related
                    if constexpr (REQ_F)
                        if (f_weight != nullptr) {
                            // here, derx = d2bm / dxdc
                            double tail = (rb_ders[xi] * powk * mult0 - rb_vals[xi] * k * powk * mult0 / nbh.distances_[jj]);
                            double derx = (nbh.pos_[jj][0] / nbh.distances_[jj]) * tail;
                            double dery = (nbh.pos_[jj][1] / nbh.distances_[jj]) * tail;
                            double derz = (nbh.pos_[jj][2] / nbh.distances_[jj]) * tail;
                            derx += rb_vals[xi] * powk * alpha_index_basic(i, 1) * coords_powers(alpha_index_basic(i, 1), 0) * pow1 * pow2;
                            dery += rb_vals[xi] * powk * alpha_index_basic(i, 2) * coords_powers(alpha_index_basic(i, 2), 1) * pow0 * pow2;
                            derz += rb_vals[xi] * powk * alpha_index_basic(i, 3) * coords_powers(alpha_index_basic(i, 3), 2) * pow0 * pow1;
                            // this is w_i \cdot d2bm/drdc
                            moment_jacobian_coeff_coord(i, nbh.chemmap_[jj], mu * rb_size + xi) += (*f_weight)(jj, 0) * derx + (*f_weight)(jj, 1) * dery + (*f_weight)(jj, 2) * derz;
                        }
                }
                if constexpr (REQ_F) {
                    mult0 *= der / rij;
                    moment_jacobian(i, jj, 0) += mult0 * nbh.pos_[jj][0];
                    moment_jacobian(i, jj, 1) += mult0 * nbh.pos_[jj][1];
                    moment_jacobian(i, jj, 2) += mult0 * nbh.pos_[jj][2];
                    // coords_powers is shifted by 3 to avoid if statement
                    // note that 0 * Inf = NaN
                    moment_jacobian(i, jj, 0) += alpha_index_basic(i, 1) * val * coords_powers(alpha_index_basic(i, 1), 0) * pow1 * pow2;
                    moment_jacobian(i, jj, 1) += alpha_index_basic(i, 2) * val * coords_powers(alpha_index_basic(i, 2), 1) * pow0 * pow2;
                    moment_jacobian(i, jj, 2) += alpha_index_basic(i, 3) * val * coords_powers(alpha_index_basic(i, 3), 2) * pow0 * pow1;
                }
            }
        }

        // non-elementary b_i
        for (int i = 0; i < alpha_index_times_count; ++i) {
            double val0 = moment_vals[alpha_index_times(i, 0)];
            double val1 = moment_vals[alpha_index_times(i, 1)];
            double val2 = alpha_index_times(i, 2);
            moment_vals[alpha_index_times(i, 3)] += val2 * val0 * val1;
        }

        // TODO: remove this?
        // ext_dUdQ.setZero();
        // TODO: test order
        for (int i = 0; i < n_props; ++i) {
            Qs[i] = species_coeffs[species_count + i * species_count + nbh.centertype];
            for (int a = 0; a < alpha_scalar_moments; ++a) {
                Qs[i] += moment_vals[alpha_moment_mapping[a]] * moment_coeffs[i * alpha_scalar_moments + a];
            }
        }
        double site_energy = 0;
        /**
         * Request Force True:
         *                 dUdQ     dUdt     d2UdQdt     d2UdQdQ
         * REQ_RADICAL      T         F         F            T
         * REQ_SPECIES      T         F         F            T
         * REQ_MOMENT       T         F         F            T
         * REQ_MODEL        F         T         T            F
         * Request Froce False:
         *                 dUdQ     dUdt     d2UdQdt     d2UdQdQ
         * REQ_RADICAL      T         F         F            F
         * REQ_SPECIES      T         F         F            F
         * REQ_MOMENT       T         F         F            F
         * REQ_MODEL        F         T         F            F
         */
        if constexpr (REQ_F) {
            if ((req_radical || req_species || req_moment) && req_model) {
                model.backpropagate(Qs, site_energy, ext_dUdQ, ext_dUdt, ext_d2UdQdt, ext_d2UdQdQ, nbh.centertype, ithread);
            } else if (req_radical || req_species || req_moment) {
                model.propagate(Qs, site_energy, ext_dUdQ, ext_d2UdQdQ, nbh.centertype, ithread);
            } else if (req_model) {
                model.backpropagate(Qs, site_energy, ext_dUdt, ext_d2UdQdt, nbh.centertype, ithread);
            } else {
                GPERROR("PotMTP_M invalid fit request: all req_* flags are false");
            }
        } else {
            if ((req_radical || req_species || req_moment) && req_model) {
                model.backpropagate(Qs, site_energy, ext_dUdQ, ext_dUdt, nbh.centertype, ithread);
            } else if (req_radical || req_species || req_moment) {
                model.propagate(Qs, site_energy, ext_dUdQ, nbh.centertype, ithread);
            } else if (req_model) {
                model.backpropagate(Qs, site_energy, ext_dUdt, nbh.centertype, ithread);
            } else {
                GPERROR("PotMTP_M invalid fit request: all req_* flags are false");
            }
        }
        site_energy += species_coeffs[nbh.centertype];

        // Backpropagation step 1: site energy derivative is the corresponding linear combination
        std::fill(site_energy_ders_wrt_moments_.begin(), site_energy_ders_wrt_moments_.end(), 0);
        for (int i = 0; i < alpha_scalar_moments; ++i) {
            for (int j = 0; j < n_props; ++j) {
                site_energy_ders_wrt_moments_[alpha_moment_mapping[i]] += moment_coeffs[j * alpha_scalar_moments + i] * ext_dUdQ[j];
            }
        }

        // Backpropagation step 2: expressing through basic moments:
        for (int i = alpha_index_times_count - 1; i >= 0; --i) {
            double val0 = moment_vals[alpha_index_times(i, 0)];
            double val1 = moment_vals[alpha_index_times(i, 1)];
            double val2 = alpha_index_times(i, 2);

            site_energy_ders_wrt_moments_[alpha_index_times(i, 1)] += site_energy_ders_wrt_moments_[alpha_index_times(i, 3)] * val2 * val0;
            site_energy_ders_wrt_moments_[alpha_index_times(i, 0)] += site_energy_ders_wrt_moments_[alpha_index_times(i, 3)] * val2 * val1;
        }

        for (int i = 0; i < alpha_index_basic_count; ++i)
            dloss_dmom[i] = se_weight * site_energy_ders_wrt_moments_[i]; // currently this is weight_eng * dUi/dbm, these basic term won't be inkoved during BP process later if force is requesed
        if constexpr (REQ_F) {
            if (f_weight != nullptr) {
                // Backpropagation step 3: multiply by the Jacobian:
                for (int i = 0; i < alpha_index_basic_count; ++i)
                    for (int j = 0; j < nbh.nbhcount_; ++j)
                        for (int a = 0; a < 3; a++)
                            dloss_dsenders[i] += (*f_weight)(j, a) * moment_jacobian(i, j, a); // this is wgt_fceij \cdot dbm/drij
                // f_weight * db_j/dr
                for (int i = 0; i < alpha_index_times_count; ++i) {
                    double val2 = alpha_index_times(i, 2);
                    double val0 = moment_vals[alpha_index_times(i, 0)];
                    double val1 = moment_vals[alpha_index_times(i, 1)];
                    dloss_dsenders[alpha_index_times(i, 3)] += dloss_dsenders[alpha_index_times(i, 0)] * val2 * val1 + dloss_dsenders[alpha_index_times(i, 1)] * val2 * val0;
                }

                // now compute dq_i/dr
                for (int i = 0; i < n_props; ++i) {
                    for (int a = 0; a < alpha_scalar_moments; ++a) {
                        dqdr[i] += moment_coeffs[i * alpha_scalar_moments + a] * dloss_dsenders[alpha_moment_mapping[a]];
                    }
                }

                if (req_radical) {
                    // fill d/dr(dU/dB)
                    for (int i = 0; i < alpha_scalar_moments; ++i) {
                        // TODO: use upper only
                        for (int m = 0; m < n_props; ++m) {
                            for (int n = 0; n < n_props; ++n) {
                                dloss_dmom[alpha_moment_mapping[i]] += ext_d2UdQdQ(m, n) * dqdr[m] * moment_coeffs[n * alpha_scalar_moments + i];
                            }
                        }
                    }

                    // d^2E/db_jdr,
                    /**
                     * Note1. Backpropagation only ADD things to basic moments index terms, thus dloss_dmom still valid for energy der components.
                     * Note2. This loop
                     */
                    // for local energy U_i, we have dU/dbj += c * bi * dU/dbk over rule {i, j, c, k}
                    // then we have d/dr(dU/dbj) = c dbi/dr dU/dbk + c bi d/dr(dU/dbk), this loop add the first term
                    for (int i = 0; i < alpha_index_times_count; ++i) {
                        double val2 = alpha_index_times(i, 2);

                        dloss_dmom[alpha_index_times(i, 1)] += dloss_dsenders[alpha_index_times(i, 0)] * site_energy_ders_wrt_moments_[alpha_index_times(i, 3)] * val2;
                        dloss_dmom[alpha_index_times(i, 0)] += dloss_dsenders[alpha_index_times(i, 1)] * site_energy_ders_wrt_moments_[alpha_index_times(i, 3)] * val2;
                    }

                    // Backpropagation, calculate contribution of d/dr(dU/dbm), bm is basic
                    for (int i = alpha_index_times_count - 1; i >= 0; --i) {
                        double val0 = moment_vals[alpha_index_times(i, 0)];
                        double val1 = moment_vals[alpha_index_times(i, 1)];
                        double val2 = alpha_index_times(i, 2);

                        dloss_dmom[alpha_index_times(i, 0)] += dloss_dmom[alpha_index_times(i, 3)] * val2 * val1;
                        dloss_dmom[alpha_index_times(i, 1)] += dloss_dmom[alpha_index_times(i, 3)] * val2 * val0;
                    }
                }
            }
        }
        int coeff_offset = 0;
        if (req_radical) {
            // radial coeffs
            for (int i = 0; i < alpha_index_basic_count; ++i) {
                for (int j = 0; j < species_count; ++j) {
                    for (int k = 0; k < radial_func_count * rb_size; ++k) {
                        int idx = (nbh.centertype * species_count + j) * rb_size * radial_func_count + k;
                        out_gard_accumulator[idx] += dloss_dmom[i] * moment_jacobian_coeffs(i, j, k);
                        if constexpr (REQ_F) {
                            if (f_weight != nullptr) {
                                // dE/db_i*d^2b/(dxdc) term, the d^2E/(db_mdrij)*db_m/dc term is included in the dloss_dmom array
                                out_gard_accumulator[idx] += site_energy_ders_wrt_moments_[i] * moment_jacobian_coeff_coord(i, j, k);
                            }
                        }
                    }
                }
            }
            coeff_offset = radial_coeff_size;
        }
        // Species coeffs
        if (req_species) {
            out_gard_accumulator[coeff_offset + nbh.centertype] += se_weight; // energy part
            coeff_offset += species_count;
        }
        if (req_moment) {
            // energy part
            for (int i = 0; i < n_props; ++i) {
                out_gard_accumulator[coeff_offset + i * species_count + nbh.centertype] += se_weight * ext_dUdQ[i];
            }
            if constexpr (REQ_F) {
                if (f_weight != nullptr) {
                    for (int m = 0; m < n_props; ++m) {
                        for (int n = 0; n < n_props; ++n) {
                            out_gard_accumulator[coeff_offset + m * species_count + nbh.centertype] += ext_d2UdQdQ(m, n) * dqdr[n];
                        }
                    }
                }
            }
            coeff_offset += species_coeff_size - species_count;
        }
        // moment coeffs
        if (req_moment) {
            // energy part
            for (int j = 0; j < n_props; ++j) {
                for (int a = 0; a < alpha_scalar_moments; ++a) {
                    out_gard_accumulator[coeff_offset + j * alpha_scalar_moments + a] += se_weight * ext_dUdQ[j] * moment_vals[alpha_moment_mapping[a]];
                }
            }
            if constexpr (REQ_F) {
                if (f_weight != nullptr) {
                    for (int j = 0; j < n_props; ++j) {
                        for (int a = 0; a < alpha_scalar_moments; ++a) {
                            out_gard_accumulator[coeff_offset + j * alpha_scalar_moments + a] += dloss_dsenders[alpha_moment_mapping[a]] * ext_dUdQ[j];
                            for (int n = 0; n < n_props; ++n) {
                                out_gard_accumulator[coeff_offset + j * alpha_scalar_moments + a] += ext_d2UdQdQ(n, j) * dqdr[n] * moment_vals[alpha_moment_mapping[a]];
                            }
                        }
                    }
                }
            }
            coeff_offset += n_props * alpha_scalar_moments;
        }
        // model coeffs. Energy part is simple, just replace B to dU/dt, force part also simple
        if (req_model) {
            for (int i = 0; i < ext_dUdt.size(); ++i) {
                out_gard_accumulator[coeff_offset + i] += se_weight * ext_dUdt[i];
                if constexpr (REQ_F) {
                    if (f_weight != nullptr) {
                        for (int m = 0; m < n_props; ++m) {
                            out_gard_accumulator[coeff_offset + i] += ext_d2UdQdt(m, i) * dqdr[m];
                        }
                    }
                }
            }
            coeff_offset += model_coeff_size;
        }
    }

    virtual void random_init()
    {
        std::uniform_real_distribution<double> uniform(random_lower, random_upper); // -1e-3 - 1e-3 ??
        if (req_radical) {
            for (int i = 0; i < radial_coeff_size; ++i) {
                radial_coeffs[i] = uniform(RNG.mtd);
            }
            Orthogonalize();
        }
        if (req_species) {
            for (int i = 0; i < species_count; ++i) {
                species_coeffs[i] = uniform(RNG.mtd);
            }
        }
        if (req_moment) {
            for (int i = species_count; i < species_coeff_size; ++i) {
                species_coeffs[i] = uniform(RNG.mtd);
            }
            for (int i = 0; i < moment_coeff_size; ++i) {
                moment_coeffs[i] = uniform(RNG.mtd);
            }
        }
        if (req_model) {
            for (int i = 0; i < model_coeff_size; ++i) {
                model_coeffs[i] = uniform(RNG.mtd);
            }
            model.random_init(random_lower, random_upper);
        }
        if (req_les) {
            for (int i = 0; i < les_head_count(); ++i) {
                les_coeffs[i] = uniform(RNG.mtd);
            }
        }
    }
    void init_coeffs() override
    {
        require_loaded_model();
        if (!regression_coeffs)
            GPERROR("MTP-M potential has no loaded coefficients");
        // Load supplies defaults for omitted sections. Initialization retains
        // those values (and fitted coefficients) while reconnecting the head.
        radial_coeffs = regression_coeffs;
        species_coeffs = radial_coeffs + radial_coeff_size;
        moment_coeffs = species_coeffs + species_coeff_size;
        model_coeffs = moment_coeffs + moment_coeff_size;
        model.coeffs = model_coeffs;
        if (lr_les) {
            les_species = model_coeffs + model_coeff_size;
            les_moment = les_species + les_channels() * species_count;
            les_coeffs = les_species;
        }
        model.postinit();
        empty = false;
        nl_inited = true;
    }
    virtual void train_init()
    {
        PotMTP::train_init();
    }
    virtual void set_scale(double s) { PotMTP::set_scale(s); };
    virtual int get_NL_coeffs_count() const { return calc_NL_coeffs_count(); };
    virtual void nl_get_coeffs(double* x) const
    {
        int offset = 0;
        if (req_radical) {
            for (int i = 0; i < radial_coeff_size; ++i) {
                x[i] = radial_coeffs[i];
            }
            offset += radial_coeff_size;
        }
        if (req_species) {
            for (int i = 0; i < species_count; ++i) {
                x[offset + i] = species_coeffs[i];
            }
            offset += species_count;
        }
        if (req_moment) {
            for (int i = species_count; i < species_coeff_size; ++i) {
                x[offset + i - species_count] = species_coeffs[i];
            }
            offset += species_coeff_size - species_count;
            for (int i = 0; i < moment_coeff_size; ++i) {
                x[offset + i] = moment_coeffs[i];
            }
            offset += moment_coeff_size;
        }
        if (req_model) {
            for (int i = 0; i < model_coeff_size; ++i) {
                x[offset + i] = model_coeffs[i];
            }
            offset += model_coeff_size;
        }
        if (req_les) {
            for (int i = 0; i < les_head_count(); ++i) {
                x[offset + i] = les_coeffs[i];
            }
        }
        if (uses_sog() && req_sog) {
            const int off=sog_fitting_offset(), terms=model_les.sog_weights.size();
            const double span=sog_log_bandwidth_max-sog_log_bandwidth_min;
            const double mid=.5*(sog_log_bandwidth_max+sog_log_bandwidth_min);
            for(int m=0;m<terms;++m){x[off+m]=model_les.sog_weights[m];x[off+terms+m]=std::tan(M_PI*(model_les.sog_log_bandwidths[m]-mid)/span);}
        }
    }
    virtual void nl_set_coeffs(const double* x)
    {
        int offset = 0;
        if (req_radical) {
            for (int i = 0; i < radial_coeff_size; ++i) {
                radial_coeffs[i] = x[i];
            }
            offset += radial_coeff_size;
        }
        if (req_species) {
            for (int i = 0; i < species_count; ++i) {
                species_coeffs[i] = x[offset + i];
            }
            offset += species_count;
        }
        if (req_moment) {
            for (int i = species_count; i < species_coeff_size; ++i) {
                species_coeffs[i] = x[offset + i - species_count];
            }
            offset += species_coeff_size - species_count;
            for (int i = 0; i < moment_coeff_size; ++i) {
                moment_coeffs[i] = x[offset + i];
            }
            offset += moment_coeff_size;
        }
        if (req_model) {
            for (int i = 0; i < model_coeff_size; ++i) {
                model_coeffs[i] = x[offset + i];
            }
            offset += model_coeff_size;
        }
        if (req_les) {
            for (int i = 0; i < les_head_count(); ++i) {
                les_coeffs[i] = x[offset + i];
            }
        }
        if (uses_sog() && req_sog) {
            const int off=sog_fitting_offset(), terms=model_les.sog_weights.size();
            const double span=sog_log_bandwidth_max-sog_log_bandwidth_min;
            const double mid=.5*(sog_log_bandwidth_max+sog_log_bandwidth_min);
            for(int m=0;m<terms;++m){
                if(!std::isfinite(x[off+m]) || !std::isfinite(x[off+terms+m])) GPERROR("Nonfinite SOG optimizer parameter");
                model_les.sog_weights[m]=x[off+m];model_les.sog_log_bandwidths[m]=mid+span/M_PI*std::atan(x[off+terms+m]);
            }
            try { sog_definition().validate(); } catch(const std::exception& e) { GPERROR(e.what()); }
        }
    }
    std::vector<NLParameterGroup> nl_parameter_groups() const override {
        std::vector<NLParameterGroup> groups;
        int off=0;
        const auto add=[&](const char* name,int count){groups.push_back({name,off,off+count});off+=count;};
        if(req_radical)add("sr_radial",radial_coeff_size);
        if(req_species)add("sr_species",species_count);
        if(req_moment)add("sr_moment",species_coeff_size-species_count+moment_coeff_size);
        if(req_model)add("sr_model",model_coeff_size);
        if(req_les){add("les_species",les_channels()*species_count);add("les_moment",les_channels()*alpha_scalar_moments);}
        if(uses_sog() && req_sog){add("sog_weights",model_les.sog_weights.size());add("sog_log_bandwidths",model_les.sog_weights.size());}
        return groups;
    }
    std::vector<ConditionalLinearBlock> conditional_linear_blocks(double penalty_weight) const override {
        if (typeid(*this) != typeid(PotMTP_M<T>)) return {};
        if constexpr (!(std::is_same_v<T,m_model_mtpli> || std::is_same_v<T,m_model_mtpq2> ||
                        std::is_same_v<T,m_model_mtpfs> || std::is_same_v<T,m_model_nn1> ||
                        std::is_same_v<T,m_model_nn2>)) return {};
        ConditionalLinearBlock block; block.name="output";
        int offset=req_radical ? radial_coeff_size : 0;
        const auto append_regularized=[&](int index) {
            block.append(index,index<int(reg_L1_vector.size()) ? reg_L1_vector[index] : 0.,
                index<int(reg_L2_vector.size()) ? reg_L2_vector[index] : 0.);
        };
        if (req_species) {
            block.composition_count=species_count;
            for (int i=0;i<species_count;++i) append_regularized(offset+i);
            offset+=species_count;
        }
        if (req_moment) {
            const int count=species_coeff_size-species_count+moment_coeff_size;
            // Only the linear model is jointly affine in its property heads.
            if constexpr (std::is_same_v<T,m_model_mtpli>) {
                block.composition_count+=species_coeff_size-species_count;
                for (int i=0;i<count;++i) append_regularized(offset+i);
            }
            offset+=count;
        }
        if (req_model) {
            if constexpr (std::is_same_v<T,m_model_mtpq2> || std::is_same_v<T,m_model_mtpfs>) {
                for (int i=0;i<model_coeff_size;++i) block.append(offset+i);
            } else if constexpr (std::is_same_v<T,m_model_nn1>) {
                // This model packs [Win, Wout, bias], unlike standalone NN1.
                const int first=model.n_input*model.n_neuron;
                for (int i=0;i<model.n_neuron;++i) block.append(offset+first+i);
            } else if constexpr (std::is_same_v<T,m_model_nn2>) {
                for (int i=model_coeff_size-model.n_neuron2;i<model_coeff_size;++i)
                    block.append(offset+i,model.reg_l1>0. || model.reg_l2>0. ? model.reg_l1 : 0.,
                        model.reg_l1>0. || model.reg_l2>0. ? model.reg_l2 : 0.);
            }
            offset+=model_coeff_size;
        }
        if (req_les) offset+=les_channels()*(species_count+alpha_scalar_moments);
        std::vector<ConditionalLinearBlock> blocks;
        if (!block.indices.empty()) blocks.push_back(std::move(block));
        if (uses_sog() && req_sog) {
            ConditionalLinearBlock sog; sog.name="sog";
            for (int i=0;i<int(model_les.sog_weights.size());++i) sog.append(offset+i,0.,penalty_weight);
            blocks.push_back(std::move(sog));
        }
        return blocks;
    }
    void conditional_linear_set_coeffs(const std::vector<int>& indices, const Eigen::VectorXd& values) override {
        if (values.size()!=static_cast<Eigen::Index>(indices.size()) || !values.allFinite())
            GPERROR("Invalid conditional linear MTP-M update");
        // Map enabled packed groups directly to physical storage. In particular,
        // do not round-trip frozen SOG bandwidths through atan/tan.
        std::vector<double*> packed;
        const auto append=[&](double* data,int count) {
            for (int i=0;i<count;++i) packed.push_back(data+i);
        };
        if (req_radical) append(radial_coeffs,radial_coeff_size);
        if (req_species) append(species_coeffs,species_count);
        if (req_moment) {
            append(species_coeffs+species_count,species_coeff_size-species_count);
            append(moment_coeffs,moment_coeff_size);
        }
        if (req_model) append(model_coeffs,model_coeff_size);
        if (req_les) append(les_coeffs,les_head_count());
        if (uses_sog() && req_sog) append(model_les.sog_weights.data(),model_les.sog_weights.size());
        // Only advertised affine indices are accepted, including when the class
        // is inherited by an unproved potential or a custom model template.
        std::vector<bool> allowed(packed.size(),false);
        for (const auto &block : conditional_linear_blocks(0.))
            for (int index : block.indices) allowed[index]=true;
        for (int index : indices)
            if (index<0 || index>=int(packed.size()) || !allowed[index])
                GPERROR("Invalid conditional linear MTP-M index");
        for (int j=0;j<values.size();++j) *packed[indices[j]]=values[j];
    }
    // virtual void AccumulateFPGard(Neighborhood& nbh,
    //     double* out_gard_accumulator,
    //     double se_weight, // 2*wgt_eng*(E_ref - E_efs) for MSE loss
    //     Eigen::Matrix<double, -1, 3, Eigen::RowMajor>* f_weight);
    virtual void AddPenaltyGard(double weight, double& loss, double* loss_gard)
    {
        // this normalizes the sum of squares of coefficients in C*C*R dimensions to 1 for each K
        int offset = 0;
        if (req_radical) {
            if (radial_mode == 0) {
                for (int k = 0; k < radial_func_count; k++) {
                    double norm = 0;

                    for (int i = 0; i < species_count; i++)
                        for (int j = 0; j < species_count; j++)
                            for (int l = 0; l < rb_size; l++) {
                                int idx = (i * species_count + j) * rb_size * radial_func_count + k * rb_size + l;
                                norm += radial_coeffs[idx] * radial_coeffs[idx];
                            }

                    loss += weight * (norm - 1) * (norm - 1);

                    for (int i = 0; i < species_count; i++)
                        for (int j = 0; j < species_count; j++)
                            for (int l = 0; l < rb_size; l++) {
                                int idx = (i * species_count + j) * rb_size * radial_func_count + k * rb_size + l;
                                loss_gard[idx] += weight * 4 * (norm - 1) * radial_coeffs[idx];
                            }
                }
                for (int k1 = 0; k1 < radial_func_count; k1++)
                    for (int k2 = k1 + 1; k2 < radial_func_count; k2++) {
                        double scal = 0;

                        for (int i = 0; i < species_count; i++)
                            for (int j = 0; j < species_count; j++)
                                for (int l = 0; l < rb_size; l++) {
                                    int idx1 = (i * species_count + j) * rb_size * radial_func_count + k1 * rb_size + l;
                                    int idx2 = (i * species_count + j) * rb_size * radial_func_count + k2 * rb_size + l;
                                    scal += radial_coeffs[idx1] * radial_coeffs[idx2];
                                }

                        loss += weight * scal * scal;

                        for (int i = 0; i < species_count; i++)
                            for (int j = 0; j < species_count; j++)
                                for (int l = 0; l < rb_size; l++) {
                                    int idx1 = (i * species_count + j) * rb_size * radial_func_count + k1 * rb_size + l;
                                    int idx2 = (i * species_count + j) * rb_size * radial_func_count + k2 * rb_size + l;
                                    loss_gard[idx1] += weight * 2 * scal * radial_coeffs[idx2];
                                    loss_gard[idx2] += weight * 2 * scal * radial_coeffs[idx1];
                                }
                    }
            } else {
                // if radial mode is 1, change weight
                double interval = cutoff;
                double dr = interval / quad_point;
                double pref = dr / interval / interval;
                double r0 = 0;
                bool have_w0 = false;
                bool have_w1 = false;
                bool have_w2 = false;
                if (weight_rad_w2 > 0) have_w2 = true;
                if (weight_rad_w1 > 0) have_w1 = true;
                if (weight_rad_w0 > 0) have_w0 = true;
                if (have_w1 || have_w2 || have_w0)
                    for (int n = 0; n < quad_point; ++n) {
                        double r = r0 + n * dr;
                        for (int ifunc = 0; ifunc < radial_func_count; ++ifunc) {
                            for (int typei = 0; typei < species_count; ++typei) {
                                for (int typej = 0; typej < species_count; ++typej) {
                                    if (have_w2)
                                        rb_calcderder_safe(typei, typej, r);
                                    else if (have_w1)
                                        rb_calcder_safe(typei, typej, r);
                                    else
                                        rb_calcval_safe(typei, typej, r);
                                    double Qsum = 0, dQsum = 0, ddQsum = 0;
                                    int idxoffset = (typei * species_count + typej) * rb_size * radial_func_count + ifunc * rb_size;
                                    for (int k = 0; k < rb_size; ++k) {
                                        if (have_w0)
                                            Qsum += radial_coeffs[k + idxoffset] * rb_vals[k];
                                        if (have_w1)
                                            dQsum += radial_coeffs[k + idxoffset] * rb_ders[k];
                                        if (have_w2)
                                            ddQsum += radial_coeffs[k + idxoffset] * rb_derders[k];
                                    }
                                    if (weight_rad_w0 > 0) {
                                        loss += pref * weight_rad_w0 * r * r * fabs(Qsum);
                                        for (int k = 0; k < rb_size; ++k) {
                                            double sgn = 1;
                                            if (Qsum < 0)
                                                sgn = -1;
                                            loss_gard[k + idxoffset] += pref * weight_rad_w0 * r * r * rb_vals[k] * sgn;
                                        }
                                    }
                                    if (weight_rad_w1 > 0) {
                                        loss += pref * weight_rad_w1 * r * r * fabs(dQsum);
                                        for (int k = 0; k < rb_size; ++k) {
                                            double sgn = 1;
                                            if (dQsum < 0)
                                                sgn = -1;
                                            loss_gard[k + idxoffset] += pref * weight_rad_w1 * r * r * rb_ders[k] * sgn;
                                        }
                                    }
                                    if (weight_rad_w2 > 0) {
                                        loss += pref * weight_rad_w2 * r * r * fabs(ddQsum);
                                        for (int k = 0; k < rb_size; ++k) {
                                            double sgn = 1;
                                            if (ddQsum < 0)
                                                sgn = -1;
                                            loss_gard[k + idxoffset] += pref * weight_rad_w2 * r * r * rb_derders[k] * sgn;
                                        }
                                    }
                                }
                            }
                        }
                    }
            }
            offset = radial_coeff_size;
        }

        // L1/L2 regs
        if (req_species) {
            for (int i = 0; i < species_count; ++i) {
                // L2 part
                loss += species_coeffs[i] * species_coeffs[i] * reg_L2_vector[offset + i];
                loss_gard[offset + i] += 2 * species_coeffs[i] * reg_L2_vector[offset + i];
                // L1 part
                loss += fabs(species_coeffs[i]) * reg_L1_vector[offset + i];
                loss_gard[offset + i] += species_coeffs[i] > 0 ? reg_L1_vector[offset + i] : -reg_L1_vector[offset + i];
            }
            offset += species_count;
        }
        if (req_moment) {
            for (int i = species_count; i < species_coeff_size; ++i) {
                const int fit_idx = offset + i - species_count;
                loss += species_coeffs[i] * species_coeffs[i] * reg_L2_vector[fit_idx];
                loss_gard[fit_idx] += 2 * species_coeffs[i] * reg_L2_vector[fit_idx];
                loss += fabs(species_coeffs[i]) * reg_L1_vector[fit_idx];
                loss_gard[fit_idx] += species_coeffs[i] > 0 ? reg_L1_vector[fit_idx] : -reg_L1_vector[fit_idx];
            }
            offset += species_coeff_size - species_count;
            for (int i = 0; i < moment_coeff_size; ++i) {
                // L2 part
                loss += moment_coeffs[i] * moment_coeffs[i] * reg_L2_vector[offset + i];
                loss_gard[offset + i] += 2 * moment_coeffs[i] * reg_L2_vector[offset + i];
                // L1 part
                loss += fabs(moment_coeffs[i]) * reg_L1_vector[offset + i];
                loss_gard[offset + i] += moment_coeffs[i] > 0 ? reg_L1_vector[offset + i] : -reg_L1_vector[offset + i];
            }
            offset += moment_coeff_size;
        }
        // model coeffs: ???
        if (req_model) {
            model.add_loss_gard(loss, loss_gard, offset);
            offset += model_coeff_size;
        }
        if (req_les) {
            for (int i = 0; i < les_head_count(); ++i) {
                loss += les_coeffs[i] * les_coeffs[i] * reg_L2_vector[offset + i];
                loss_gard[offset + i] += 2 * les_coeffs[i] * reg_L2_vector[offset + i];
                loss += fabs(les_coeffs[i]) * reg_L1_vector[offset + i];
                loss_gard[offset + i] += les_coeffs[i] > 0 ? reg_L1_vector[offset + i] : -reg_L1_vector[offset + i];
            }
        }
        if (uses_sog() && req_sog) {
            const int off=sog_fitting_offset();
            for(size_t m=0;m<model_les.sog_weights.size();++m) {
                const double w=model_les.sog_weights[m];
                loss+=weight*w*w;loss_gard[off+m]+=2*weight*w;
            }
        }
    }

    // hacked, no les part
    virtual void get_n_props(int& nprops)
    {
        nprops = n_props;
    };
    virtual void get_n_coeffs(int& n_radical_coeffs, int& n_species_coeffs, int& n_moment_coeffs, int& n_model_coeffs)
    {
        n_radical_coeffs = radial_coeff_size;
        n_species_coeffs = species_coeff_size;
        n_moment_coeffs = moment_coeff_size;
        n_model_coeffs = model_coeff_size;
    }
    virtual void get_pointers(double*& radical_coeffs_ptr, double*& species_coeffs_ptr, double*& moment_coeffs_ptr, double*& model_coeffs_ptr)
    {
        radical_coeffs_ptr = radial_coeffs;
        species_coeffs_ptr = species_coeffs;
        moment_coeffs_ptr = moment_coeffs;
        model_coeffs_ptr = model_coeffs;
    }

};


#include "potential_mtp_m_evaluation.h"
#include "potential_mtp_m_append.h"
#include "potential_mtp_m_curvature_workspace.h"

AnyPotential* create_mtp_m_potential(GPSettings& settings);

}
