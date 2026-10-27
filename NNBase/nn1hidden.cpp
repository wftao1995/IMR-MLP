// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#include "nn1hidden.h"
#include "coeff_io.h"
#include "tools/tools.h"
#include <fmt/format.h>
#include <fmt/ranges.h>

void IMRMLPNN::NN1lay::resize_omp_buffers(int n_threads) {
    if (buffers.size() < n_threads) {
        buffers.resize(n_threads);
        for (int i = 1; i < n_threads; ++i) {
            auto &buf = buffers[i];
            buf.hiddenlay = Vector::Zero(n_neuron);
            buf.dhdi = Matrix::Zero(n_neuron, n_input);
        }
    }
}
void IMRMLPNN::NN1lay::add_pnal(const double &l1, const double &l2, double &loss, double *pgard)
{
    // if (x[i] > 0) pgard[i] -= reg_L1vec[i];
    //     else if (x[i] < 0) pgard[i] += reg_L1vec[i];
    //     pgard[i] += 2 * x[i] * reg_L2vec[i];
    int m = 0;
    if (l1 > 0 || l2 > 0) {
        // w_input : n_input x n_neuron
        for (int col = 0; col < w_input.cols(); ++col) {
            for (int row = 0; row < w_input.rows(); ++row) {
                double &x = w_input(row, col);
                loss += l1 * fabs(x) + l2 * x * x;
                pgard[m++] += (x > 0 ? l1 : -l1) + 2 * x * l2;
            }
        }

        // Copy b_hidden
        for (int i = 0; i < b_hidden.size(); ++i) {
            double &x = b_hidden(i);
            loss += l1 * fabs(x) + l2 * x * x;
            pgard[m++] += (x > 0 ? l1 : -l1) + 2 * x * l2;
        }

        // Copy w_hidden
        for (int i = 0; i < w_hidden.size(); ++i) {
            double &x = w_hidden(i);
            loss += l1 * fabs(x) + l2 * x * x;
            pgard[m++] += (x > 0 ? l1 : -l1) + 2 * x * l2;
        }
    }

    // // Copy b_output
    // *x = b_output;
}
// LeCun Init, one should ensure that setup() is called before.
void IMRMLPNN::NN1lay::random_init()
{
    // Random number generator
    std::random_device rd;
    std::mt19937 gen(rd());
    double sig = 1.0 / n_input;
    double sig2 = 1.0 / n_neuron;
    std::normal_distribution<Scalar> dist(0, sig);
    std::normal_distribution<Scalar> dist2(0, sig2);

    // Initialize weights with random values
    w_input = Matrix::NullaryExpr(n_input, n_neuron, [&]() { return dist(gen); });
    w_hidden = Vector::NullaryExpr(n_neuron, [&]() { return dist2(gen); });
    // b_hidden = Vector::NullaryExpr(n_neuron, [&]() { return dist(gen); });
    b_hidden.setZero();
    // b_output = dist(gen);
}

void IMRMLPNN::NN1lay::setup(int _n_input, int _n_neuron, int _n_threads) {
    // n_input(n_input), n_neuron(n_neuron) 
    n_input = _n_input;
    n_neuron = _n_neuron;
    // Compute dimensions
    n_weights = n_input * n_neuron + n_neuron;
    n_bias = n_neuron; // + 1;
    n_coeffs = n_weights + n_bias;
    // Allocate arrays
    w_input.resize(n_input, n_neuron);
    w_hidden.resize(n_neuron);
    b_hidden.resize(n_neuron);
    norm_irstd.setOnes(n_input);
    norm_mean.setZero(n_input);
    // random_init();

    // Allocate buffers for threading
    buffers.resize(_n_threads);
    for (auto &buf : buffers) {
        buf.hiddenlay = Vector::Zero(n_neuron);
        buf.dhdi = Matrix::Zero(n_neuron, n_input);
    }
}

void IMRMLPNN::NN1lay::propagate(const Vector &input, Scalar &output, int &tag , int ibuffer) {
    Buffer &buf = buffers[ibuffer];
    buf.input = normalize(input);
    buf.hiddenlay = (w_input.transpose() * buf.input + b_hidden).unaryExpr([&](Scalar x) { return activation_function(x, acfunc); });
    // for (int i = 0; i < n_neuron; ++i) {
    //     // one change activation function here
    //     Scalar activation = w_input.col(i).dot(buf.input) + b_hidden(i);
    //     buf.hiddenlay(i) = activation_function(activation, acfunc);
    // }
    output = w_hidden.dot(buf.hiddenlay); // + b_output;
}

void IMRMLPNN::NN1lay::propagate(const Vector &input, Scalar &output, Vector &outder, int &tag , int ibuffer)
{
    Buffer &buf = buffers[ibuffer];
    buf.input = normalize(input);
    output = 0;
    // outder.resize(n_input); // assume already resized
    // buf.hiddenlay.setZero();
    // buf.dhdi.resize(n_neuron, n_input);
    // buf.dhdi.setZero();
    for (int i = 0; i < n_neuron; ++i) {
        // one change activation function here
        Scalar activation = w_input.col(i).dot(buf.input) + b_hidden(i);
        buf.hiddenlay(i) = activation_function(activation, acfunc);
        Scalar derivative = activation_derivative(activation, buf.hiddenlay(i), acfunc);
        buf.dhdi.row(i) = derivative * w_input.col(i).transpose();
    }
    output = w_hidden.dot(buf.hiddenlay); // + b_output;
    // for (int i = 0; i < n_input; ++i) {
    //     outder(i) = w_hidden.dot(buf.dhdi.col(i)) * norm_irstd(i);
    // }
    outder = (w_hidden.transpose() * buf.dhdi).transpose().array() * norm_irstd.array();
}

void IMRMLPNN::NN1lay::propagate(const Vector& input, Scalar& output, Vector& dedq, Matrix& d2edqdq, int &tag , int ibuffer)
{
    Buffer &buf = buffers[ibuffer];
    buf.input = normalize(input);
    output = 0;
    d2edqdq.setZero(n_input, n_input); // takes upper
    int index = 0;
    // 1. Forward propagate to get output and dedq
    std::vector<double> actis(n_neuron, 0), deris(n_neuron, 0), der2nd(n_neuron, 0);
    for (int i = 0; i < n_neuron; ++i)
    {
        // one change activation function here
        Scalar activation = w_input.col(i).dot(buf.input) + b_hidden(i);
        buf.hiddenlay(i) = activation_function(activation, acfunc);
        Scalar derivative = activation_derivative(activation, buf.hiddenlay(i), acfunc);
        Scalar derivative_2nd = activation_2nd_derivative(activation, buf.hiddenlay(i), derivative, acfunc);
        actis[i] = activation;
        deris[i] = derivative;
        der2nd[i] = derivative_2nd;
        buf.dhdi.row(i) = (derivative * w_input.col(i)).array() * norm_irstd.array();
    }

    output = w_hidden.dot(buf.hiddenlay); // + b_output;
    dedq = w_hidden.transpose() * buf.dhdi;

    // step 4, compute d2udqdq, fill lower part
    for (int i = 0; i < n_neuron; ++i) {
        for (int m = 0; m < n_input; ++m) {
            d2edqdq(m, m) += w_hidden(i) * w_input(m, i) * w_input(m, i) * der2nd[i] * norm_irstd(m) * norm_irstd(m);
            for (int n = m + 1; n < n_input; ++n) {
                double c = w_hidden(i) * w_input(n, i) * w_input(m, i) * der2nd[i] * norm_irstd(m) * norm_irstd(n);
                d2edqdq(n, m) += c;
                d2edqdq(m, n) += c;
            }
        }
    }
}

void IMRMLPNN::NN1lay::backpropgate(const Vector &_input, Scalar &output, Vector &dedq, Vector &dedt, Matrix &d2edqdt, int &tag , int ibuffer)
{
    Buffer &buf = buffers[ibuffer];
    buf.input = normalize(_input);
    output = 0;
    // dedq.resize(n_input); // assume already resized
    // `dedt` holds derivatives for all parameters in the order:
    // [w_input (column-major), b_hidden, w_hidden, b_output]
    // dedt.setZero(n_coeffs); // assume already resized
    dedt.resize(n_coeffs);
    d2edqdt.setZero(n_input, n_coeffs);
    int index = 0;
    // 1. Forward propagate to get output and dedq
    std::vector<double> actis(n_neuron, 0), deris(n_neuron, 0), der2nd(n_neuron, 0);
    for (int i = 0; i < n_neuron; ++i)
    {
        // one change activation function here
        Scalar activation = w_input.col(i).dot(buf.input) + b_hidden(i);
        buf.hiddenlay(i) = activation_function(activation, acfunc);
        ;
        Scalar derivative = activation_derivative(activation, buf.hiddenlay(i), acfunc);
        Scalar derivative_2nd = activation_2nd_derivative(activation, buf.hiddenlay(i), derivative, acfunc);
        actis[i] = activation;
        deris[i] = derivative;
        der2nd[i] = derivative_2nd;
        buf.dhdi.row(i) = derivative * w_input.col(i).transpose();
        // Gradients w.r.t. weights in `w_input`
        for (int j = 0; j < n_input; ++j)
        {
            Scalar grad = w_hidden(i) * derivative * buf.input(j);
            dedt(index++) = grad;
        }
    }

    output = w_hidden.dot(buf.hiddenlay); // + b_output;
    dedq = (w_hidden.transpose() * buf.dhdi).array() * norm_irstd.array();

    // Gradients w.r.t. biases in `b_hidden`
    for (int i = 0; i < n_neuron; ++i)
    {
        dedt(index++) = w_hidden(i) * deris[i];
    }

    // Gradients w.r.t. weights in `w_hidden`
    for (int i = 0; i < n_neuron; ++i)
    {
        dedt(index++) = buf.hiddenlay(i);
    }

    // Gradient w.r.t. `b_output`
    // dedt(index++) = 1.0;

    // Step 3: Compute second-order derivatives (`d2edqdt`)
    // Contribution of weights and biases in `w_input` and `b_hidden`
    index = 0;
    for (int i = 0; i < n_neuron; ++i)
    {
        // Scalar activation = actis[i];
        Scalar derivative = deris[i];
        Scalar derivative_2nd = der2nd[i];

        for (int j = 0; j < n_input; ++j)
        {
            for (int k = 0; k < n_input; ++k)
            {
                if (k == j)
                {
                    d2edqdt(k, index) += w_hidden(i) * derivative * norm_irstd(k);
                }
                d2edqdt(k, index) += w_hidden(i) * w_input(k, i) * buf.input(j) * derivative_2nd * norm_irstd(k);
            }
            index++;
        }
    }

    for (int i = 0; i < n_neuron; ++i)
    {
        // Bias contribution
        for (int k = 0; k < n_input; ++k)
        {
            d2edqdt(k, index) += w_hidden(i) * w_input(k, i) * der2nd[i] * norm_irstd(k);
        }
        index++;
    }

    // Contribution from `w_hidden` (linear, no higher-order effects)
    for (int j = 0; j < n_neuron; ++j)
    {
        for (int k = 0; k < n_input; ++k)
        {
            d2edqdt(k, index) = buf.dhdi(j, k) * norm_irstd(k);
        }
        index++;
    }

    // Contribution from `b_output` (constant)
    // No second-order contribution needed, keep zero
}

void IMRMLPNN::NN1lay::backpropgate(const Vector& input, Scalar& output, Vector& dedq, Vector& dedt, int &tag , int ibuffer)
{
    Buffer &buf = buffers[ibuffer];
    buf.input = normalize(input);
    output = 0;
    dedt.resize(n_coeffs);
    int index = 0;
    // 1. Forward propagate to get output and dedq
    std::vector<double> deris(n_neuron, 0);
    for (int i = 0; i < n_neuron; ++i)
    {
        // one change activation function here
        Scalar activation = w_input.col(i).dot(buf.input) + b_hidden(i);
        buf.hiddenlay(i) = activation_function(activation, acfunc);
        Scalar derivative = activation_derivative(activation, buf.hiddenlay(i), acfunc);
        deris[i] = derivative;
        buf.dhdi.row(i) = (derivative * w_input.col(i)).array() * norm_irstd.array();
        // Gradients w.r.t. weights in `w_input`
        for (int j = 0; j < n_input; ++j)
        {
            Scalar grad = w_hidden(i) * derivative * buf.input(j);
            dedt(index++) = grad;
        }
    }

    output = w_hidden.dot(buf.hiddenlay); // + b_output;
    dedq = w_hidden.transpose() * buf.dhdi;

    // Gradients w.r.t. biases in `b_hidden`
    for (int i = 0; i < n_neuron; ++i)
    {
        dedt(index++) = w_hidden(i) * deris[i];
    }

    // Gradients w.r.t. weights in `w_hidden`
    for (int i = 0; i < n_neuron; ++i)
    {
        dedt(index++) = buf.hiddenlay(i);
    }
}

void IMRMLPNN::NN1lay::backpropgate(const Vector& input, Scalar& output, Vector& dedt, int &tag , int ibuffer)
{
    Buffer &buf = buffers[ibuffer];
    buf.input = normalize(input);
    output = 0;
    dedt.resize(n_coeffs);
    int index = 0;
    // 1. Forward propagate to get output and dedq
    std::vector<double> deris(n_neuron, 0);
    for (int i = 0; i < n_neuron; ++i)
    {
        // one change activation function here
        Scalar activation = w_input.col(i).dot(buf.input) + b_hidden(i);
        buf.hiddenlay(i) = activation_function(activation, acfunc);
        Scalar derivative = activation_derivative(activation, buf.hiddenlay(i), acfunc);
        deris[i] = derivative;
        // Gradients w.r.t. weights in `w_input`
        for (int j = 0; j < n_input; ++j)
        {
            Scalar grad = w_hidden(i) * derivative * buf.input(j);
            dedt(index++) = grad;
        }
    }

    output = w_hidden.dot(buf.hiddenlay); // + b_output;

    // Gradients w.r.t. biases in `b_hidden`
    for (int i = 0; i < n_neuron; ++i)
    {
        dedt(index++) = w_hidden(i) * deris[i];
    }

    // Gradients w.r.t. weights in `w_hidden`
    for (int i = 0; i < n_neuron; ++i)
    {
        dedt(index++) = buf.hiddenlay(i);
    }
}

void IMRMLPNN::NN1lay::backpropgate(const Vector &_input, 
        Scalar &output, 
        Vector &dedq, 
        Vector &dedt, 
        Matrix &d2edqdt, 
        Matrix &d2edqdq, 
        int &tag , 
        int ibuffer)
{
    Buffer &buf = buffers[ibuffer];
    buf.input = normalize(_input);
    output = 0;
    // dedq.resize(n_input); // assume already resized
    // `dedt` holds derivatives for all parameters in the order:
    // [w_input (column-major), b_hidden, w_hidden, b_output]
    // dedt.setZero(n_coeffs); // assume already resized
    dedt.resize(n_coeffs);
    d2edqdt.setZero(n_input, n_coeffs);
    d2edqdq.setZero(n_input, n_input); // takes upper
    int index = 0;
    // 1. Forward propagate to get output and dedq
    std::vector<double> actis(n_neuron, 0), deris(n_neuron, 0), der2nd(n_neuron, 0);
    for (int i = 0; i < n_neuron; ++i)
    {
        // one change activation function here
        Scalar activation = w_input.col(i).dot(buf.input) + b_hidden(i);
        buf.hiddenlay(i) = activation_function(activation, acfunc);
        Scalar derivative = activation_derivative(activation, buf.hiddenlay(i), acfunc);
        Scalar derivative_2nd = activation_2nd_derivative(activation, buf.hiddenlay(i), derivative, acfunc);
        actis[i] = activation;
        deris[i] = derivative;
        der2nd[i] = derivative_2nd;
        buf.dhdi.row(i) = (derivative * w_input.col(i)).array() * norm_irstd.array();
        // Gradients w.r.t. weights in `w_input`
        for (int j = 0; j < n_input; ++j)
        {
            Scalar grad = w_hidden(i) * derivative * buf.input(j);
            dedt(index++) = grad;
        }
    }

    output = w_hidden.dot(buf.hiddenlay); // + b_output;
    dedq = w_hidden.transpose() * buf.dhdi;

    // Gradients w.r.t. biases in `b_hidden`
    for (int i = 0; i < n_neuron; ++i)
    {
        dedt(index++) = w_hidden(i) * deris[i];
    }

    // Gradients w.r.t. weights in `w_hidden`
    for (int i = 0; i < n_neuron; ++i)
    {
        dedt(index++) = buf.hiddenlay(i);
    }

    // Gradient w.r.t. `b_output`
    // dedt(index++) = 1.0;

    // Step 3: Compute second-order derivatives (`d2edqdt`)
    // Contribution of weights and biases in `w_input` and `b_hidden`
    index = 0;
    for (int i = 0; i < n_neuron; ++i)
    {
        // Scalar activation = actis[i];
        Scalar derivative = deris[i];
        Scalar derivative_2nd = der2nd[i];

        for (int j = 0; j < n_input; ++j)
        {
            for (int k = 0; k < n_input; ++k)
            {
                if (k == j)
                {
                    d2edqdt(k, index) += w_hidden(i) * derivative * norm_irstd(k);
                }
                d2edqdt(k, index) += w_hidden(i) * w_input(k, i) * buf.input(j) * derivative_2nd * norm_irstd(k);
            }
            index++;
        }
    }

    for (int i = 0; i < n_neuron; ++i)
    {
        // Bias contribution
        for (int k = 0; k < n_input; ++k)
        {
            d2edqdt(k, index) += w_hidden(i) * w_input(k, i) * der2nd[i] * norm_irstd(k);
        }
        index++;
    }

    // Contribution from `w_hidden` (linear, no higher-order effects)
    for (int j = 0; j < n_neuron; ++j)
    {
        for (int k = 0; k < n_input; ++k)
        {
            d2edqdt(k, index) = buf.dhdi(j, k);
        }
        index++;
    }

    // Contribution from `b_output` (constant)
    // No second-order contribution needed, keep zero

    // step 4, compute d2udqdq, fill lower part
    for (int i = 0; i < n_neuron; ++i) {
        for (int m = 0; m < n_input; ++m) {
            d2edqdq(m, m) += w_hidden(i) * w_input(m, i) * w_input(m, i) * der2nd[i] * norm_irstd(m) * norm_irstd(m);
            for (int n = m + 1; n < n_input; ++n) {
                double c = w_hidden(i) * w_input(n, i) * w_input(m, i) * der2nd[i] * norm_irstd(m) * norm_irstd(n);
                d2edqdq(n, m) += c;
                d2edqdq(m, n) += c;
            }
        }
    }
}

void IMRMLPNN::NN1lay::backpropgate(const Vector &_input, Scalar &output, Vector &dedt, Matrix &d2edqdt, int &tag , int ibuffer)
{
    Buffer &buf = buffers[ibuffer];
    buf.input = normalize(_input);
    output = 0;
    // dedq.resize(n_input); // assume already resized
    // `dedt` holds derivatives for all parameters in the order:
    // [w_input (column-major), b_hidden, w_hidden, b_output]
    // dedt.setZero(n_coeffs); // assume already resized
    dedt.resize(n_coeffs);
    d2edqdt.setZero(n_input, n_coeffs);
    int index = 0;
    // 1. Forward propagate to get output and dedq
    std::vector<double> actis(n_neuron, 0), deris(n_neuron, 0), der2nd(n_neuron, 0);
    for (int i = 0; i < n_neuron; ++i)
    {
        // one change activation function here
        Scalar activation = w_input.col(i).dot(buf.input) + b_hidden(i);
        buf.hiddenlay(i) = activation_function(activation, acfunc);
        Scalar derivative = activation_derivative(activation, buf.hiddenlay(i), acfunc);
        Scalar derivative_2nd = activation_2nd_derivative(activation, buf.hiddenlay(i), derivative, acfunc);
        actis[i] = activation;
        deris[i] = derivative;
        der2nd[i] = derivative_2nd;
        buf.dhdi.row(i) = derivative * w_input.col(i).transpose();
        // Gradients w.r.t. weights in `w_input`
        for (int j = 0; j < n_input; ++j)
        {
            Scalar grad = w_hidden(i) * derivative * buf.input(j);
            dedt(index++) = grad;
        }
    }

    output = w_hidden.dot(buf.hiddenlay); // + b_output;

    // Gradients w.r.t. biases in `b_hidden`
    for (int i = 0; i < n_neuron; ++i)
    {
        dedt(index++) = w_hidden(i) * deris[i];
    }

    // Gradients w.r.t. weights in `w_hidden`
    for (int i = 0; i < n_neuron; ++i)
    {
        dedt(index++) = buf.hiddenlay(i);
    }

    // Gradient w.r.t. `b_output`
    // dedt(index++) = 1.0;

    // Step 3: Compute second-order derivatives (`d2edqdt`)
    // Contribution of weights and biases in `w_input` and `b_hidden`
    index = 0;
    for (int i = 0; i < n_neuron; ++i)
    {
        // Scalar activation = actis[i];
        Scalar derivative = deris[i];
        Scalar derivative_2nd = der2nd[i];

        for (int j = 0; j < n_input; ++j)
        {
            for (int k = 0; k < n_input; ++k)
            {
                if (k == j)
                {
                    d2edqdt(k, index) += w_hidden(i) * derivative * norm_irstd(k);
                }
                d2edqdt(k, index) += w_hidden(i) * w_input(k, i) * buf.input(j) * derivative_2nd * norm_irstd(k);
            }
            index++;
        }
    }

    for (int i = 0; i < n_neuron; ++i)
    {
        // Bias contribution
        for (int k = 0; k < n_input; ++k)
        {
            d2edqdt(k, index) += w_hidden(i) * w_input(k, i) * der2nd[i] * norm_irstd(k);
        }
        index++;
    }

    // Contribution from `w_hidden` (linear, no higher-order effects)
    for (int j = 0; j < n_neuron; ++j)
    {
        for (int k = 0; k < n_input; ++k)
        {
            d2edqdt(k, index) = buf.dhdi(j, k) * norm_irstd(k);
        }
        index++;
    }

    // Contribution from `b_output` (constant)
    // No second-order contribution needed, keep zero
}

void IMRMLPNN::NN1lay::dump_coeffs(const std::string &outfile) {
    std::ofstream ofs(outfile, std::ios::binary);
    if (!ofs) {
        throw std::ios_base::failure("Failed to open file for writing: " + outfile);
    }

    CoeffIO::write_header(ofs, 1, acfunc);
    ofs.write(reinterpret_cast<const char *>(&n_neuron), sizeof(n_neuron));
    ofs.write(reinterpret_cast<const char *>(&n_input), sizeof(n_input));

    ofs.write(reinterpret_cast<const char *>(w_input.data()), w_input.size() * sizeof(Scalar));
    ofs.write(reinterpret_cast<const char *>(w_hidden.data()), w_hidden.size() * sizeof(Scalar));
    ofs.write(reinterpret_cast<const char *>(b_hidden.data()), b_hidden.size() * sizeof(Scalar));
    // ofs.write(reinterpret_cast<const char *>(&b_output), sizeof(b_output));
    ofs.write(reinterpret_cast<const char *>(norm_mean.data()), norm_mean.size() * sizeof(Scalar));
    ofs.write(reinterpret_cast<const char *>(norm_irstd.data()), norm_irstd.size() * sizeof(Scalar));
    CoeffIO::finish_output(ofs, outfile);
}

void IMRMLPNN::NN1lay::dump_coeffs_txt(const std::string &outfile)
{
    std::ofstream ofs(outfile);
    if (!ofs) {
        throw std::ios_base::failure("Failed to open file for writing: " + outfile);
    }
    ofs << fmt::format("{} {}\n", n_neuron, n_input);
    for (int i = 0; i < w_input.rows(); ++i) {
        ofs << fmt::format("{}\n", fmt::join(w_input.row(i), " "));
    }
    ofs << fmt::format("{}\n{}\n{}\n{}\n", fmt::join(w_hidden, " "), fmt::join(b_hidden, " "), fmt::join(norm_mean, " "), fmt::join(norm_irstd, " "));
    CoeffIO::finish_output(ofs, outfile);
}

void IMRMLPNN::NN1lay::load_coeffs(const std::string &infile)
{
    std::ifstream ifs(infile, std::ios::binary);
    if (!ifs) throw std::ios_base::failure("Failed to open file for reading: " + infile);
    const auto header = CoeffIO::read_header(ifs, infile, 1);
    if (header.layers != 1) CoeffIO::fail(infile);
    int hidden = 0, input = 0;
    ifs.read(reinterpret_cast<char *>(&hidden), sizeof(hidden));
    ifs.read(reinterpret_cast<char *>(&input), sizeof(input));
    const bool normalized = CoeffIO::check_payload(ifs, infile, header.legacy, input, hidden);
    NN1lay loaded;
    loaded.setup(input, hidden);
    loaded.acfunc = static_cast<ActivationFunction>(header.activation);
    CoeffIO::read_array(ifs, loaded.w_input, infile);
    CoeffIO::read_array(ifs, loaded.w_hidden, infile);
    CoeffIO::read_array(ifs, loaded.b_hidden, infile);
    if (normalized) {
        CoeffIO::read_array(ifs, loaded.norm_mean, infile);
        CoeffIO::read_array(ifs, loaded.norm_irstd, infile);
    }
    *this = std::move(loaded);
}
