// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#pragma once
#include <Eigen/Dense>
#include <fstream>
#include <iostream>
#include <random>
#include "nn1hidden.h"
#include <unsupported/Eigen/CXX11/Tensor>
namespace IMRMLPNN {
    typedef Eigen::Tensor<double, 3> Tensor3;
    /// @brief Two-hidden-layer neural network.
    /// @ingroup nnbase
    class NN2lay {
    public:
        ActivationFunction acfunc = TANH;
        int n_neuron1;        // Number of hidden neurons lay 1
        int n_neuron2;        // Number of hidden neurons lay 2 
        int n_input;          // Number of input elements

        int n_coeffs;         // Total number of coefficients (weights + biases)
        int n_weights;        // Number of weights in total
        int n_bias;           // Number of biases in total

        Matrix w_input;       // Input to hidden weights (n_input x n_neuron1)
        Vector b_hidden1;     // Hidden layer 1 biases (n_neuron1)
        Matrix w_hidden1;     // Hidden1 to hidden2 weights (n_neuron1 x n_neuron 2)
        Vector b_hidden2;     // Hidden layer 2 biases (n_neuron2)
        Vector w_hidden2;     // Hidden2 to output weights (n_neuron2)
        // Scalar b_output;      // Output layer bias, we do not use output bais here since outer model will hold an constant energy term.

        Vector norm_mean;     // Input normalization, shift
        Vector norm_irstd;    // Input normalization, inverse standard deviation

        struct Buffer {
            Vector input;     // Transformed input, x' = (x-mean)*irstd
            Vector hiddenlay1;// Hidden layer1 values
            Vector hiddenlay2;// Hidden layer2 values
            Matrix dh1di;     // Lazily allocated for higher derivatives. Derivative of hidden layer 1 w.r.t input (n_neuron1 * n_input)
            Matrix dh2dh1;    // Lazily allocated for higher derivatives. Derivative of hidden layer 2 w.r.t hidden layer 1 (n_neuron2 * n_neuron1)
            Vector deris1;
            Vector secderis1;
            Vector secderis2;
            Vector deris2;
            Matrix dh2di;
            Vector dedh1;
            Vector dedh1deris1;
            Vector dedh2deris2;
            Matrix dha2dq;
            Tensor3 d2h2dh1dh1;
            Tensor3 d2h1didi;
        };

        // Read-only views into either the network-owned matrices or the caller's
        // current serialized coefficient array. The backing storage and input
        // normalization must outlive the view and remain fixed during evaluation.
        struct CoeffView {
            Eigen::Map<const Matrix> w_input;
            Eigen::Map<const Vector> b_hidden1;
            Eigen::Map<const Matrix> w_hidden1;
            Eigen::Map<const Vector> b_hidden2, w_hidden2;
            const Vector &norm_mean, &norm_irstd;
            CoeffView(const double* wi, const double* b1, const double* wh,
                      const double* b2, const double* wo, int ni, int n1, int n2,
                      const Vector& mean, const Vector& inverse_std)
                : w_input(wi,ni,n1), b_hidden1(b1,n1), w_hidden1(wh,n1,n2),
                  b_hidden2(b2,n2), w_hidden2(wo,n2), norm_mean(mean), norm_irstd(inverse_std) {}
        };
        CoeffView coefficient_view() const;
        CoeffView coefficient_view(const double* coefficients) const;
        Buffer make_buffer() const;
        void prepare_buffer(Buffer& buffer) const;

        // Explicit scratch overloads never access buffers[] or mutate weights.
        void propagate(const Vector &input, Scalar &output, int &tag , const CoeffView &weights, Buffer &buffer) const;
        void propagate(const Vector &input, Scalar &output, Vector &dedq, int &tag , const CoeffView &weights, Buffer &buffer) const;
        void propagate(const Vector& input, Scalar& output, Vector& dedq, Matrix& d2edqdq, int &tag , const CoeffView &weights, Buffer &buffer) const;

        void backpropgate(const Vector &input, Scalar &output, Vector &dedq, Vector &dedt, Matrix &d2edqdt, int &tag , const CoeffView &weights, Buffer &buffer) const;
        void backpropgate(const Vector& input, Scalar& output, Vector& dedt, int &tag , const CoeffView &weights, Buffer &buffer) const;
        void backpropgate(const Vector &input, Scalar &output, Vector &dedq, Vector &dedt, Matrix &d2edqdt, Matrix &d2edqdq, int &tag , const CoeffView &weights, Buffer &buffer) const;
        void backpropgate(const Vector &input, Scalar &output, Vector &dedt, Matrix &d2edqdt, int &tag , const CoeffView &weights, Buffer &buffer) const;
        void backpropgate(const Vector& input, Scalar& output, Vector& dedq, Vector& dedt, int &tag , const CoeffView &weights, Buffer &buffer) const;

        NN2lay(){};

        NN2lay(int _n_input, int _n_neuron1, int _n_neuron2, int _n_threads) {
            setup(_n_input, _n_neuron1, _n_neuron2, _n_threads);
        };

        NN2lay(std::string fnm) { load_coeffs(fnm); };

        void resize_omp_buffers(int n_threads);

        inline int get_coeff_size() {return n_coeffs;};

        inline Vector normalize(const Vector &i) { return (i.array() - norm_mean.array()) * norm_irstd.array(); };

        void get_coeffs(double *x) const;

        void set_coeffs(const double *x);

        // add L1 and L2 regulation
        void add_pnal(const double &l1, const double &l2, double &loss, double *pgard);

        void random_init();

        // Initialize buffers and dimensions
        void setup(int _n_input, int _n_neuron1, int _n_neuron2, int _n_threads = 1);

        // for threads
        std::vector<Buffer> buffers;
        /**
         * @brief Forward propagation to compute energy
         *
         * @param input {Q}
         * @param output E
         * @param tag    Species tag (atom type)
         * @param ibuffer thread-id
         */
        void propagate(const Vector &input, Scalar &output, int &tag , int ibuffer);
        void propagate_test(const Vector &input, Scalar &output, int ibuffer);
        void propagate_test2(int ibuffer);
        /**
         * @brief Compute energy and dU/dQ
         *
         * @param input {Q} array
         * @param output E
         * @param outder dE/dQ array
         * @param tag    Species tag (atom type)
         * @param ibuffer buffer id for multi-thread
         */
        void propagate(const Vector &input, Scalar &output, Vector &outder, int &tag , int ibuffer);
        /**
         * @brief Compute energy, dU/dQ and d2U/dQ1dQ2
         *
         * @param input      {Q} array
         * @param output     Energy scalar
         * @param outder     dE/dQ array
         * @param d2edqdq    d2E/dQ1dQ2 matrix
         * @param tag        Species tag (atom type)
         * @param ibuffer    Thread buffer index
         */
        void propagate(const Vector &input, Scalar &output, Vector &outder, Matrix &d2edqdq, int &tag , int ibuffer);
        /**
         * @brief BP process, actually forwrad since there is only one hidden lay
         * t is parameters of NN (weight, bais, etc)
         * @param input {Q}
         * @param output E
         * @param dedq dE/dQ array
         * @param dedt dE/dt array
         * @param d2edqdt d2E/dQdt array
         * @param ibuffer thread-id
         * @return * void
         */
        void backpropgate(const Vector &input,
                          Scalar &output,
                          Vector &dedq,
                          Vector &dedt,
                          Matrix &d2edqdt,
                          int &tag , 
                          int ibuffer);

        void backpropgate(const Vector &input,
                          Scalar &output,
                          Vector &dedt,
                          int &tag , 
                          int ibuffer);

        void backpropgate(const Vector &input,
                          Scalar &output,
                          Vector &dedq,
                          Vector &dedt,
                          Matrix &d2edqdt,
                          Matrix &d2edqdq,
                          int &tag , 
                          int ibuffer);

        void backpropgate(const Vector &input,
                          Scalar &output,
                          Vector &dedt,
                          Matrix &d2edqdt,
                          int &tag , 
                          int ibuffer);

        void backpropgate(const Vector &input,
                          Scalar &output,
                          Vector &dedq,
                          Vector &dedt,
                          int &tag , 
                          int ibuffer);

        // Save network coefficients to a binary file
        void dump_coeffs(const std::string &outfile);

        void dump_coeffs_txt(const std::string &outfile);

        // Load network coefficients from a binary file
        void load_coeffs(const std::string &infile);
    };

};
