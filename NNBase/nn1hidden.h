// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#pragma once
#include <Eigen/Dense>
#include <fstream>
#include <iostream>
#include <random>
namespace IMRMLPNN {
    typedef double Scalar;
    typedef Eigen::MatrixXd Matrix;
    typedef Eigen::VectorXd Vector;
    enum ActivationFunction {
        TANH, // RECOMMEND
        RELU, // DO NOT USE THIS SINCE WE REQUEST 2ND-DERIVATIVE
        SIGMOID // NOT FULLY TESTED, SHOULD BE OK
    };
    inline Scalar activation_function(Scalar x, ActivationFunction func) {
        switch (func) {
            case TANH: return tanh(x);
            case RELU: return std::max(Scalar(0), x);
            case SIGMOID: return 1 / (1 + exp(-x));
            default: return x; // Default to linear
        }
    }
    inline Scalar activation_derivative(Scalar x, Scalar fx, ActivationFunction func) {
        switch (func) {
            case TANH: return 1 - fx * fx; // tanh derivative
            case RELU: return x > 0 ? 1 : 0; // ReLU derivative
            case SIGMOID: return fx * (1 - fx); // sigmoid derivative
            default: return 1; // Default to linear
        }
    }
    inline Scalar activation_2nd_derivative(Scalar x, Scalar fx, Scalar dfx, ActivationFunction func) {
        switch (func) {
            case TANH: return -2 * fx * dfx; // tanh derivative
            case RELU: return 0; // ReLU derivative, actually break at 0 (non-exist)
            case SIGMOID: return dfx * (1 - 2.0 * fx); // sigmoid derivative
            default: return 0; // Default to linear
        }
    }
    /// @brief Single-hidden-layer neural network with activation functions.
    /// @ingroup nnbase
    class NN1lay {
    public:
        ActivationFunction acfunc = TANH;
        int n_coeffs;         // Total number of coefficients (weights + biases)
        int n_weights;        // Number of weights
        int n_bias;           // Number of biases
        int n_neuron;         // Number of hidden neurons
        int n_input;          // Number of input elements

        Matrix w_input;       // Input to hidden weights (n_input x n_neuron)
        Vector w_hidden;      // Hidden to output weights (n_neuron)
        Vector b_hidden;      // Hidden layer biases (n_neuron)
        // Scalar b_output;      // Output layer bias, we do not use output bais here since outer model will hold an constant energy term.

        Vector norm_mean;     // Input normalization, shift
        Vector norm_irstd;    // Input normalization, inverse standard deviation

        struct Buffer {
            Vector input;
            Vector hiddenlay; // Hidden layer values
            Matrix dhdi;      // Derivative of hidden layer w.r.t input
        };

        NN1lay(){};

        NN1lay(int _n_input, int _n_neuron, int _n_threads) {
            setup(_n_input, _n_neuron, _n_threads);
        };

        NN1lay(std::string fnm) { load_coeffs(fnm); };

        void resize_omp_buffers(int n_threads);

        inline int get_coeff_size() {return n_coeffs;};

        inline Vector normalize(const Vector &i) { return (i.array() - norm_mean.array()) * norm_irstd.array(); };

        // [w_input (column-major), b_hidden, w_hidden, b_output]
        void get_coeffs(double *x) const {
            // Start by copying w_input (column-major order)
            // w_input : n_input x n_neuron
            for (int col = 0; col < w_input.cols(); ++col) {
                for (int row = 0; row < w_input.rows(); ++row) {
                    *x++ = w_input(row, col);
                }
            }

            // Copy b_hidden
            for (int i = 0; i < b_hidden.size(); ++i) {
                *x++ = b_hidden(i);
            }

            // Copy w_hidden
            for (int i = 0; i < w_hidden.size(); ++i) {
                *x++ = w_hidden(i);
            }

            // // Copy b_output
            // *x = b_output;
        }

        void set_coeffs(const double *x) {
            // Populate w_input (column-major order)
            for (int col = 0; col < w_input.cols(); ++col) {
                for (int row = 0; row < w_input.rows(); ++row) {
                    w_input(row, col) = *x++;
                }
            }

            // Populate b_hidden
            for (int i = 0; i < b_hidden.size(); ++i) {
                b_hidden(i) = *x++;
            }

            // Populate w_hidden
            for (int i = 0; i < w_hidden.size(); ++i) {
                w_hidden(i) = *x++;
            }

            // Populate b_output
            // b_output = *x;
        }

        void add_pnal(const double &l1, const double &l2, double &loss, double *pgard);

        void random_init();

        // Initialize buffers and dimensions
        void setup(int _n_input, int _n_neuron, int _n_threads = 1);

        // for threads
        std::vector<Buffer> buffers;
        /**
         * @brief Forward propagation to compute energy
         *
         * @param input {Q}
         * @param output E
         * @param tag flag center type, will not referenced in this class, just for compability
         * @param ibuffer thread-id
         */
        void propagate(const Vector &input, Scalar &output, int &tag , int ibuffer);
        /**
         * @brief Compute energy and dU/dQ
         *
         * @param input {Q} array
         * @param output E
         * @param outder dE/dQ array
         * @param ibuffer buffer id for multi-thread
         */
        void propagate(const Vector &input, Scalar &output, Vector &outder, int &tag , int ibuffer);
        /**
         * @brief Compute energy, dU/dQ and d2U/dQ1dQ2
         * 
         * @param input 
         * @param output 
         * @param outder 
         * @param d2edqdq 
         * @param ibuffer 
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
                          Vector &dedq,
                          Vector &dedt,
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

        // Save network coefficients to a file
        void dump_coeffs(const std::string &outfile);

        void dump_coeffs_txt(const std::string &outfile);

        // Load network coefficients from a file
        void load_coeffs(const std::string &infile);
    };

};