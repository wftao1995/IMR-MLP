// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#include "nn2hidden.h"
#include "coeff_io.h"
#include <fmt/format.h>
#include <fmt/ranges.h>
#include "tools/tools.h"
#include <stdexcept>

IMRMLPNN::NN2lay::CoeffView IMRMLPNN::NN2lay::coefficient_view() const
{
    return {w_input.data(), b_hidden1.data(), w_hidden1.data(), b_hidden2.data(),
            w_hidden2.data(), n_input, n_neuron1, n_neuron2, norm_mean, norm_irstd};
}

IMRMLPNN::NN2lay::CoeffView IMRMLPNN::NN2lay::coefficient_view(const double* coefficients) const
{
    if (!coefficients) throw std::invalid_argument("NN2 coefficient view requires initialized coefficients");
    const double* b1 = coefficients + n_input*n_neuron1;
    const double* wh = b1 + n_neuron1;
    const double* b2 = wh + n_neuron1*n_neuron2;
    const double* wo = b2 + n_neuron2;
    return {coefficients,b1,wh,b2,wo,n_input,n_neuron1,n_neuron2,norm_mean,norm_irstd};
}

void IMRMLPNN::NN2lay::prepare_buffer(Buffer& buffer) const
{
    buffer.input.resize(n_input);
    buffer.hiddenlay1.resize(n_neuron1);
    buffer.hiddenlay2.resize(n_neuron2);
}

IMRMLPNN::NN2lay::Buffer IMRMLPNN::NN2lay::make_buffer() const
{
    Buffer buffer;
    prepare_buffer(buffer);
    return buffer;
}

void IMRMLPNN::NN2lay::resize_omp_buffers(int n_threads)
{
    buffers.resize(n_threads);
    for (auto &buffer : buffers) {
        buffer.input.resize(n_input);
        buffer.hiddenlay1.resize(n_neuron1);
        buffer.hiddenlay2.resize(n_neuron2);
    }
}

// [w_input (col-major), b_hidden1, w_hidden1, b_hidden2, w_hidden2]
void IMRMLPNN::NN2lay::get_coeffs(double *x) const
{
    // w_input
    for (int col = 0; col < w_input.cols(); ++col)
    {
        for (int row = 0; row < w_input.rows(); ++row)
        {
            *x++ = w_input(row, col);
        }
    }
    // b_hidden1
    for (int i = 0; i < b_hidden1.size(); ++i)
    {
        *x++ = b_hidden1(i);
    }
    // w_hidden1
    for (int col = 0; col < w_hidden1.cols(); ++col)
    {
        for (int row = 0; row < w_hidden1.rows(); ++row)
        {
            *x++ = w_hidden1(row, col);
        }
    }
    // b_hidden2
    for (int i = 0; i < b_hidden2.size(); ++i)
    {
        *x++ = b_hidden2(i);
    }
    //
    for (int i = 0; i < w_hidden2.size(); ++i)
    {
        *x++ = w_hidden2(i);
    }
}

void IMRMLPNN::NN2lay::set_coeffs(const double *x)
{
    // w_input
    for (int col = 0; col < w_input.cols(); ++col)
    {
        for (int row = 0; row < w_input.rows(); ++row)
        {
            w_input(row, col) = *x++;
        }
    }
    // b_hidden1
    for (int i = 0; i < b_hidden1.size(); ++i)
    {
        b_hidden1(i) = *x++;
    }
    // w_hidden1
    for (int col = 0; col < w_hidden1.cols(); ++col)
    {
        for (int row = 0; row < w_hidden1.rows(); ++row)
        {
            w_hidden1(row, col) = *x++;
        }
    }
    // b_hidden2
    for (int i = 0; i < b_hidden2.size(); ++i)
    {
        b_hidden2(i) = *x++;
    }
    // w_hidden2
    for (int i = 0; i < w_hidden2.size(); ++i)
    {
        w_hidden2(i) = *x++;
    }
}

void IMRMLPNN::NN2lay::add_pnal(const double &l1, const double &l2, double &loss, double *pgard)
{
    int m = 0;
    if (l1 > 0 || l2 > 0)
    {
        // w_input
        for (int col = 0; col < w_input.cols(); ++col)
        {
            for (int row = 0; row < w_input.rows(); ++row)
            {
                double &x = w_input(row, col);
                loss += l1 * fabs(x) + l2 * x * x;
                pgard[m++] += (x > 0 ? l1 : -l1) + 2 * x * l2;
            }
        }
        // b_hidden1
        for (int i = 0; i < b_hidden1.size(); ++i)
        {
            double &x = b_hidden1(i);
            loss += l1 * fabs(x) + l2 * x * x;
            pgard[m++] += (x > 0 ? l1 : -l1) + 2 * x * l2;
        }
        // w_hidden1
        for (int col = 0; col < w_hidden1.cols(); ++col)
        {
            for (int row = 0; row < w_hidden1.rows(); ++row)
            {
                double &x = w_hidden1(row, col);
                loss += l1 * fabs(x) + l2 * x * x;
                pgard[m++] += (x > 0 ? l1 : -l1) + 2 * x * l2;
            }
        }
        // b_hidden2
        for (int i = 0; i < b_hidden2.size(); ++i)
        {
            double &x = b_hidden2(i);
            loss += l1 * fabs(x) + l2 * x * x;
            pgard[m++] += (x > 0 ? l1 : -l1) + 2 * x * l2;
        }
        // w_hidden2
        for (int i = 0; i < w_hidden2.size(); ++i)
        {
            double &x = w_hidden2(i);
            loss += l1 * fabs(x) + l2 * x * x;
            pgard[m++] += (x > 0 ? l1 : -l1) + 2 * x * l2;
        }
    }
}

void IMRMLPNN::NN2lay::random_init()
{
    std::random_device rd;
    std::mt19937 gen(rd());
    double sig1 = 1.0 / n_input;
    double sig2 = 1.0 / n_neuron1;
    double sig3 = 1.0 / n_neuron2;
    std::normal_distribution<Scalar> dis1(0, sig1);
    std::normal_distribution<Scalar> dis2(0, sig2);
    std::normal_distribution<Scalar> dis3(0, sig3);

    w_input = Matrix::NullaryExpr(n_input, n_neuron1, [&]()
                                  { return dis1(gen); });
    b_hidden1.setZero();
    w_hidden1 = Matrix::NullaryExpr(n_neuron1, n_neuron2, [&]()
                                    { return dis2(gen); });
    b_hidden2.setZero();
    w_hidden2 = Vector::NullaryExpr(n_neuron2, [&]()
                                    { return dis3(gen); });
}

void IMRMLPNN::NN2lay::setup(int _n_input, int _n_neuron1, int _n_neuron2, int _n_threads)
{
    n_input = _n_input;
    n_neuron1 = _n_neuron1;
    n_neuron2 = _n_neuron2;

    w_input = Matrix::Random(n_input, n_neuron1);
    b_hidden1 = Vector::Random(n_neuron1);
    w_hidden1 = Matrix::Random(n_neuron1, n_neuron2);
    b_hidden2 = Vector::Random(n_neuron2);
    w_hidden2 = Vector::Random(n_neuron2);

    norm_mean = Vector::Zero(n_input);
    norm_irstd = Vector::Ones(n_input);
    // random_init();

    n_weights = w_input.size() + w_hidden1.size() + w_hidden2.size();
    n_bias = b_hidden1.size() + b_hidden2.size();
    n_coeffs = n_weights + n_bias;

    resize_omp_buffers(_n_threads);
}

void IMRMLPNN::NN2lay::propagate(const Vector &input, Scalar &output, int &tag , int ibuffer)
{
    propagate(input, output, tag, coefficient_view(), buffers[ibuffer]);
}

void IMRMLPNN::NN2lay::propagate(const Vector &input, Scalar &output, int &tag , const CoeffView &weights, Buffer &buffer) const
{
    prepare_buffer(buffer);
    const auto &w_input = weights.w_input;
    const auto &b_hidden1 = weights.b_hidden1;
    const auto &w_hidden1 = weights.w_hidden1;
    const auto &b_hidden2 = weights.b_hidden2;
    const auto &w_hidden2 = weights.w_hidden2;
    const auto &norm_mean = weights.norm_mean;
    const auto &norm_irstd = weights.norm_irstd;

    buffer.input.array() = (input.array() - norm_mean.array()) * norm_irstd.array();
    buffer.hiddenlay1 = (w_input.transpose() * buffer.input + b_hidden1)
                .unaryExpr([&](Scalar x) { return activation_function(x, acfunc); });
    buffer.hiddenlay2 = (w_hidden1.transpose() * buffer.hiddenlay1 + b_hidden2)
                .unaryExpr([&](Scalar x) { return activation_function(x, acfunc); });
    output = w_hidden2.dot(buffer.hiddenlay2);
}

void IMRMLPNN::NN2lay::propagate(const Vector &input, Scalar &output, Vector &dedq, int &tag , int ibuffer)
{
    propagate(input, output, dedq, tag, coefficient_view(), buffers[ibuffer]);
}

void IMRMLPNN::NN2lay::propagate(const Vector &input, Scalar &output, Vector &dedq, int &tag , const CoeffView &weights, Buffer &buffer) const
{
    prepare_buffer(buffer);
    const auto &w_input = weights.w_input;
    const auto &b_hidden1 = weights.b_hidden1;
    const auto &w_hidden1 = weights.w_hidden1;
    const auto &b_hidden2 = weights.b_hidden2;
    const auto &w_hidden2 = weights.w_hidden2;
    const auto &norm_mean = weights.norm_mean;
    const auto &norm_irstd = weights.norm_irstd;

    buffer.input.array() = (input.array() - norm_mean.array()) * norm_irstd.array();
    buffer.deris1.resize(n_neuron1);
    buffer.deris2.resize(n_neuron2);
    for (int i = 0; i < n_neuron1; ++i) {
        const Scalar value = w_input.col(i).dot(buffer.input) + b_hidden1(i);
        buffer.hiddenlay1[i] = activation_function(value, acfunc);
        buffer.deris1[i] = activation_derivative(value, buffer.hiddenlay1[i], acfunc);
    }
    for (int i = 0; i < n_neuron2; ++i) {
        const Scalar value = w_hidden1.col(i).dot(buffer.hiddenlay1) + b_hidden2(i);
        buffer.hiddenlay2[i] = activation_function(value, acfunc);
        buffer.deris2[i] = activation_derivative(value, buffer.hiddenlay2[i], acfunc);
    }
    output = w_hidden2.dot(buffer.hiddenlay2);
    // Contract backwards from the scalar output. No hidden/input Jacobian is
    // needed: these scratch vectors scale only with the two hidden widths.
    buffer.dedh2deris2 = w_hidden2.array() * buffer.deris2.array();
    buffer.dedh1.noalias() = w_hidden1 * buffer.dedh2deris2;
    buffer.dedh1deris1 = buffer.dedh1.array() * buffer.deris1.array();
    dedq.noalias() = w_input * buffer.dedh1deris1;
    dedq.array() *= norm_irstd.array();
}

void IMRMLPNN::NN2lay::propagate(const Vector& input, Scalar& output, Vector& dedq, Matrix& d2edqdq, int &tag , int ibuffer)
{
    propagate(input, output, dedq, d2edqdq, tag, coefficient_view(), buffers[ibuffer]);
}

void IMRMLPNN::NN2lay::propagate(const Vector& input, Scalar& output, Vector& dedq, Matrix& d2edqdq, int &tag , const CoeffView &weights, Buffer &buffer) const
{
    prepare_buffer(buffer);
    const auto &w_input = weights.w_input;
    const auto &b_hidden1 = weights.b_hidden1;
    const auto &w_hidden1 = weights.w_hidden1;
    const auto &b_hidden2 = weights.b_hidden2;
    const auto &w_hidden2 = weights.w_hidden2;
    const auto &norm_mean = weights.norm_mean;
    const auto &norm_irstd = weights.norm_irstd;

    buffer.dh1di.resize(n_neuron1, n_input);
    buffer.dh2dh1.resize(n_neuron2, n_neuron1);
    buffer.input.array() = (input.array() - norm_mean.array()) * norm_irstd.array();
    buffer.d2h1didi.resize(n_neuron1, n_input, n_input);
    buffer.d2h2dh1dh1.resize(n_neuron2, n_neuron1, n_neuron1);
    d2edqdq.setZero(n_input, n_input); // takes upper

    // input -> hidden1  dh1/di    d2h1/didi
    int index = 0;
    for (int i = 0; i < n_neuron1; ++i) {
        Scalar activation = w_input.col(i).dot(buffer.input) + b_hidden1(i);
        buffer.hiddenlay1(i) = activation_function(activation, acfunc);
        Scalar derivative = activation_derivative(activation, buffer.hiddenlay1(i), acfunc);
        buffer.dh1di.row(i) = (derivative * w_input.col(i)).array() * norm_irstd.array();
        Scalar derivative2nd = activation_2nd_derivative(activation, buffer.hiddenlay1(i), derivative, acfunc);
        for (int m = 0; m < n_input; ++m) {
            buffer.d2h1didi(i, m, m) = w_input(m, i) * w_input(m, i) * derivative2nd * norm_irstd(m) * norm_irstd(m);
            for (int n = m+1; n < n_input; ++n) {
                double c = w_input(m, i) * w_input(n, i) * derivative2nd * norm_irstd(m) * norm_irstd(n);
                buffer.d2h1didi(i, m, n) = c;
                buffer.d2h1didi(i, n, m) = c;
            }
        }
    }
    // hidden1 -> hidden2   dh2/dh1
    for (int i = 0; i < n_neuron2; ++i) {
        Scalar activation = w_hidden1.col(i).dot(buffer.hiddenlay1) + b_hidden2(i);
        buffer.hiddenlay2(i) = activation_function(activation, acfunc);
        Scalar derivative = activation_derivative(activation, buffer.hiddenlay2(i), acfunc);
        buffer.dh2dh1.row(i) = derivative * w_hidden1.col(i).transpose();
        Scalar derivative2nd = activation_2nd_derivative(activation, buffer.hiddenlay2(i), derivative, acfunc);
        for (int m = 0; m < n_neuron1; ++m) {
            buffer.d2h2dh1dh1(i, m, m) = w_hidden1(m, i) * w_hidden1(m, i) * derivative2nd;
            for (int n = m+1; n < n_neuron1; ++n) {
                double c = w_hidden1(m, i) * w_hidden1(n, i) * derivative2nd;
                buffer.d2h2dh1dh1(i, m, n) = c;
                buffer.d2h2dh1dh1(i, n, m) = c;
            }
        }
    }
    // hidden2 -> output
    output = w_hidden2.dot(buffer.hiddenlay2);
    // dout/di  TODO: reduce 
    buffer.dh2di = buffer.dh2dh1 * buffer.dh1di;
    buffer.dedh1 = (w_hidden2.transpose() * buffer.dh2dh1).transpose();
    dedq = (w_hidden2.transpose() * buffer.dh2di).transpose();
    for (int m = 0; m < n_input; ++m) {
        for (int n = m; n < n_input; ++n) {
            for (int i = 0; i < n_neuron2; ++i) {
                for (int j = 0; j < n_neuron1; ++j) {
                    d2edqdq(m, n) += w_hidden2(i) * buffer.dh2dh1(i, j) * buffer.d2h1didi(j, m, n);
                    for (int k = 0; k < n_neuron1; ++k) {
                        d2edqdq(m, n) += w_hidden2(i) * buffer.d2h2dh1dh1(i, j, k) * buffer.dh1di(k, n) * buffer.dh1di(j, m);
                    }
                }
            }
        }
    }
    // fill symm part
    for (int m = 1; m < n_input; ++m) {
        for (int n = 0; n < m; ++n) {
            d2edqdq(m, n) = d2edqdq(n, m);
        }
    }
}

// dedt d2edqdt
void IMRMLPNN::NN2lay::propagate_test(const Vector &input, Scalar &output, int ibuffer)
{
    Buffer &buffer = buffers[ibuffer];
    buffer.dh1di.resize(n_neuron1, n_input);
    buffer.dh2dh1.resize(n_neuron2, n_neuron1);
    buffer.input = normalize(input);
    buffer.d2h1didi.resize(n_neuron1, n_input, n_input);
    buffer.d2h2dh1dh1.resize(n_neuron2, n_neuron1, n_neuron1);
    buffer.deris1.resize(n_neuron1);
    buffer.secderis1.resize(n_neuron1);
    buffer.deris2.resize(n_neuron2);
    buffer.secderis2.resize(n_neuron2);
    int index = 0;
    for (int i = 0; i < n_neuron1; ++i) {
        Scalar activation = w_input.col(i).dot(buffer.input) + b_hidden1(i);
        buffer.hiddenlay1(i) = activation_function(activation, acfunc);
        Scalar derivative = activation_derivative(activation, buffer.hiddenlay1(i), acfunc);
        buffer.deris1[i] = derivative;
        buffer.dh1di.row(i) = (derivative * w_input.col(i)).array() * norm_irstd.array();
        Scalar derivative2nd = activation_2nd_derivative(activation, buffer.hiddenlay1(i), derivative, acfunc);
        buffer.secderis1[i] = derivative2nd;
        for (int m = 0; m < n_input; ++m) {
            buffer.d2h1didi(i, m, m) = w_input(m, i) * w_input(m, i) * derivative2nd * norm_irstd(m) * norm_irstd(m);
            for (int n = m+1; n < n_input; ++n) {
                double c = w_input(m, i) * w_input(n, i) * derivative2nd * norm_irstd(m) * norm_irstd(n);
                buffer.d2h1didi(i, m, n) = c;
                buffer.d2h1didi(i, n, m) = c;
            }
        }
    }
    for (int i = 0; i < n_neuron2; ++i) {
        Scalar activation = w_hidden1.col(i).dot(buffer.hiddenlay1) + b_hidden2(i);
        buffer.hiddenlay2(i) = activation_function(activation, acfunc);
        Scalar derivative = activation_derivative(activation, buffer.hiddenlay2(i), acfunc);
        buffer.deris2[i] = derivative;
        buffer.dh2dh1.row(i) = derivative * w_hidden1.col(i).transpose();
        Scalar derivative2nd = activation_2nd_derivative(activation, buffer.hiddenlay2(i), derivative, acfunc);
        buffer.secderis2[i] = derivative2nd;
        for (int m = 0; m < n_neuron1; ++m) {
            buffer.d2h2dh1dh1(i, m, m) = w_hidden1(m, i) * w_hidden1(m, i) * derivative2nd;
            for (int n = m+1; n < n_neuron1; ++n) {
                double c = w_hidden1(m, i) * w_hidden1(n, i) * derivative2nd;
                buffer.d2h2dh1dh1(i, m, n) = c;
                buffer.d2h2dh1dh1(i, n, m) = c;
            }
        }
    }
    output = w_hidden2.dot(buffer.hiddenlay2);
    buffer.dh2di = buffer.dh2dh1 * buffer.dh1di;
    buffer.dedh1 = (w_hidden2.transpose() * buffer.dh2dh1).transpose();
}

void IMRMLPNN::NN2lay::propagate_test2(int ibuffer)
{
    Buffer &buffer = buffers[ibuffer];
    buffer.dh1di.resize(n_neuron1, n_input);
    buffer.dh2dh1.resize(n_neuron2, n_neuron1);
    for (int i = 0; i < n_neuron2; ++i) {
        Scalar activation = w_hidden1.col(i).dot(buffer.hiddenlay1) + b_hidden2(i);
        buffer.hiddenlay2(i) = activation_function(activation, acfunc);
        Scalar derivative = activation_derivative(activation, buffer.hiddenlay2(i), acfunc);
        buffer.deris2[i] = derivative;
        buffer.dh2dh1.row(i) = derivative * w_hidden1.col(i).transpose();
        Scalar derivative2nd = activation_2nd_derivative(activation, buffer.hiddenlay2(i), derivative, acfunc);
        buffer.secderis2[i] = derivative2nd;
        for (int m = 0; m < n_neuron1; ++m) {
            buffer.d2h2dh1dh1(i, m, m) = w_hidden1(m, i) * w_hidden1(m, i) * derivative2nd;
            for (int n = m+1; n < n_neuron1; ++n) {
                double c = w_hidden1(m, i) * w_hidden1(n, i) * derivative2nd;
                buffer.d2h2dh1dh1(i, m, n) = c;
                buffer.d2h2dh1dh1(i, n, m) = c;
            }
        }
    }
    buffer.dh2di = buffer.dh2dh1 * buffer.dh1di;
    buffer.dedh1 = (w_hidden2.transpose() * buffer.dh2dh1).transpose();
}

void IMRMLPNN::NN2lay::backpropgate(const Vector &input, Scalar &output, Vector &dedq, Vector &dedt, Matrix &d2edqdt, int &tag , int ibuffer)
{
    backpropgate(input, output, dedq, dedt, d2edqdt, tag, coefficient_view(), buffers[ibuffer]);
}

void IMRMLPNN::NN2lay::backpropgate(const Vector &input, Scalar &output, Vector &dedq, Vector &dedt, Matrix &d2edqdt, int &tag , const CoeffView &weights, Buffer &buffer) const
{
    prepare_buffer(buffer);
    const auto &w_input = weights.w_input;
    const auto &b_hidden1 = weights.b_hidden1;
    const auto &w_hidden1 = weights.w_hidden1;
    const auto &b_hidden2 = weights.b_hidden2;
    const auto &w_hidden2 = weights.w_hidden2;
    const auto &norm_mean = weights.norm_mean;
    const auto &norm_irstd = weights.norm_irstd;

    buffer.dh1di.resize(n_neuron1, n_input);
    buffer.dh2dh1.resize(n_neuron2, n_neuron1);
    buffer.input.array() = (input.array() - norm_mean.array()) * norm_irstd.array();
    buffer.d2h1didi.resize(n_neuron1, n_input, n_input);
    buffer.d2h2dh1dh1.resize(n_neuron2, n_neuron1, n_neuron1);
    buffer.deris1.resize(n_neuron1);
    buffer.secderis1.resize(n_neuron1);
    buffer.deris2.resize(n_neuron2);
    buffer.secderis2.resize(n_neuron2);
    dedt.setZero(n_coeffs);
    d2edqdt.setZero(n_input, n_coeffs);
    // input -> hidden1  dh1/di    d2h1/didi
    int index = 0;
    for (int i = 0; i < n_neuron1; ++i) {
        Scalar activation = w_input.col(i).dot(buffer.input) + b_hidden1(i);
        buffer.hiddenlay1(i) = activation_function(activation, acfunc);
        Scalar derivative = activation_derivative(activation, buffer.hiddenlay1(i), acfunc);
        buffer.deris1[i] = derivative;
        buffer.dh1di.row(i) = (derivative * w_input.col(i)).array() * norm_irstd.array();
        Scalar derivative2nd = activation_2nd_derivative(activation, buffer.hiddenlay1(i), derivative, acfunc);
        buffer.secderis1[i] = derivative2nd;
        for (int m = 0; m < n_input; ++m) {
            buffer.d2h1didi(i, m, m) = w_input(m, i) * w_input(m, i) * derivative2nd * norm_irstd(m) * norm_irstd(m);
            for (int n = m+1; n < n_input; ++n) {
                double c = w_input(m, i) * w_input(n, i) * derivative2nd * norm_irstd(m) * norm_irstd(n);
                buffer.d2h1didi(i, m, n) = c;
                buffer.d2h1didi(i, n, m) = c;
            }
        }
    }
    // hidden1 -> hidden2   dh2/dh1
    for (int i = 0; i < n_neuron2; ++i) {
        Scalar activation = w_hidden1.col(i).dot(buffer.hiddenlay1) + b_hidden2(i);
        buffer.hiddenlay2(i) = activation_function(activation, acfunc);
        Scalar derivative = activation_derivative(activation, buffer.hiddenlay2(i), acfunc);
        buffer.deris2[i] = derivative;
        buffer.dh2dh1.row(i) = derivative * w_hidden1.col(i).transpose();
        Scalar derivative2nd = activation_2nd_derivative(activation, buffer.hiddenlay2(i), derivative, acfunc);
        buffer.secderis2[i] = derivative2nd;
        for (int m = 0; m < n_neuron1; ++m) {
            buffer.d2h2dh1dh1(i, m, m) = w_hidden1(m, i) * w_hidden1(m, i) * derivative2nd;
            for (int n = m+1; n < n_neuron1; ++n) {
                double c = w_hidden1(m, i) * w_hidden1(n, i) * derivative2nd;
                buffer.d2h2dh1dh1(i, m, n) = c;
                buffer.d2h2dh1dh1(i, n, m) = c;
            }
        }
    }
    // hidden2 -> output
    output = w_hidden2.dot(buffer.hiddenlay2);
    // dout/di  TODO: reduce 
    buffer.dh2di = buffer.dh2dh1 * buffer.dh1di;
    buffer.dedh1 = (w_hidden2.transpose() * buffer.dh2dh1).transpose();
    dedq = (w_hidden2.transpose() * buffer.dh2di).transpose();
    // back propagation. 
    // dedt
    // w_inpt, b_hidden1 w_hidden1 b_hidden2 w_hidden2
    // w_input
    buffer.dedh1deris1 = buffer.dedh1.array() * buffer.deris1.array();
    for (int i = 0; i < n_neuron1; ++i) {
        for (int k = 0; k < n_input; ++k) {
            dedt(index++) = buffer.dedh1deris1(i) * buffer.input(k);
        }
    }
    // b_hidden1
    for (int i = 0; i < n_neuron1; ++i) {
        dedt(index++) = buffer.dedh1deris1(i);
    }
    // w_hidden1
    buffer.dedh2deris2 = w_hidden2.array() * buffer.deris2.array();
    for (int i = 0; i < n_neuron2; ++i) {
        for (int k = 0; k < n_neuron1; ++k) {
            dedt(index++) = buffer.dedh2deris2(i) * buffer.hiddenlay1(k);
        }
    }
    // b_hidden2
    for (int i = 0; i < n_neuron2; ++i) {
        dedt(index++) = buffer.dedh2deris2(i);
    }
    // w_hidden2
    for (int i = 0; i < n_neuron2; ++i) {
        dedt(index++) = buffer.hiddenlay2(i);
    }
    // d2edqdt
    // w_inpt, b_hidden1 w_hidden1 b_hidden2 w_hidden2
    // w_input
    index = 0;
    for (int i = 0; i < n_neuron1; ++i) {
        for (int j = 0; j < n_input; ++j) {
            for (int k = 0; k < n_input; ++k) {
                d2edqdt(k, index) += buffer.dedh1(i) * buffer.secderis1(i) * w_input(k, i) * buffer.input(j) * norm_irstd(k);
                if (j == k) d2edqdt(k, index) += buffer.dedh1(i) * buffer.deris1(i) * norm_irstd(k);
                for (int m = 0; m < n_neuron1; ++m) {
                    for (int p = 0; p < n_neuron2; ++p) {
                        d2edqdt(k, index) += w_hidden2(p) * buffer.d2h2dh1dh1(p, m, i) * buffer.deris1(m) * buffer.deris1(i) * w_input(k, m) * buffer.input(j) * norm_irstd(k);
                    }
                }
            }
            index++;
        }
    }
    // b-hidden1
    for (int i = 0; i < n_neuron1; ++i) {
        for (int k = 0; k < n_input; ++k) {
            d2edqdt(k, index) += buffer.secderis1(i) * w_input(k, i) * buffer.dedh1(i) * norm_irstd(k);
            for (int m = 0; m < n_neuron1; ++m) {
                for (int p = 0; p < n_neuron2; ++p) {
                    d2edqdt(k, index) += w_hidden2(p) * buffer.d2h2dh1dh1(p, m, i) * buffer.deris1(m) * buffer.deris1(i) * w_input(k, m) * norm_irstd(k);
                }
            }
        }
        index++;
    }
    // w-hidden1
    buffer.dha2dq = w_hidden1.transpose() * buffer.dh1di;
    // buffer.dha2dq = buffer.dh1di * w_hidden1;
    for (int i = 0; i < n_neuron2; ++i) {
        for (int j = 0; j < n_neuron1; ++j) {
            for (int k = 0; k < n_input; ++k) {
                d2edqdt(k, index) += w_hidden2(i) * (
                    buffer.secderis2[i] * buffer.dha2dq(i, k) *  buffer.hiddenlay1(j)
                    + buffer.deris2(i) * buffer.dh1di(j, k));
            }
            index++;
        }
    }
    // b-hidden2
    for (int i = 0; i < n_neuron2; ++i) {
        for (int k = 0; k < n_input; ++k) {
            d2edqdt(k, index) += w_hidden2(i) * buffer.secderis2[i] * buffer.dha2dq(i, k);
        }
        index++;
    }
    // w-hidden2
    for (int i = 0; i < n_neuron2; ++i) {
        for (int k = 0; k < n_input; ++k) {
            d2edqdt(k, index) += buffer.dh2di(i, k);
        }
        index++;
    }
}

void IMRMLPNN::NN2lay::backpropgate(const Vector& input, Scalar& output, Vector& dedt, int &tag , int ibuffer)
{
    backpropgate(input, output, dedt, tag, coefficient_view(), buffers[ibuffer]);
}

void IMRMLPNN::NN2lay::backpropgate(const Vector& input, Scalar& output, Vector& dedt, int &tag , const CoeffView &weights, Buffer &buffer) const
{
    prepare_buffer(buffer);
    const auto &w_input = weights.w_input;
    const auto &b_hidden1 = weights.b_hidden1;
    const auto &w_hidden1 = weights.w_hidden1;
    const auto &b_hidden2 = weights.b_hidden2;
    const auto &w_hidden2 = weights.w_hidden2;
    const auto &norm_mean = weights.norm_mean;
    const auto &norm_irstd = weights.norm_irstd;

    buffer.dh1di.resize(n_neuron1, n_input);
    buffer.dh2dh1.resize(n_neuron2, n_neuron1);
    buffer.input.array() = (input.array() - norm_mean.array()) * norm_irstd.array();
    buffer.deris1.resize(n_neuron1);
    buffer.deris2.resize(n_neuron2);
    dedt.setZero(n_coeffs);
    // input -> hidden1  dh1/di    d2h1/didi
    int index = 0;
    for (int i = 0; i < n_neuron1; ++i) {
        Scalar activation = w_input.col(i).dot(buffer.input) + b_hidden1(i);
        buffer.hiddenlay1(i) = activation_function(activation, acfunc);
        Scalar derivative = activation_derivative(activation, buffer.hiddenlay1(i), acfunc);
        buffer.deris1[i] = derivative;
        buffer.dh1di.row(i) = (derivative * w_input.col(i)).array() * norm_irstd.array();
    }
    // hidden1 -> hidden2   dh2/dh1
    for (int i = 0; i < n_neuron2; ++i) {
        Scalar activation = w_hidden1.col(i).dot(buffer.hiddenlay1) + b_hidden2(i);
        buffer.hiddenlay2(i) = activation_function(activation, acfunc);
        Scalar derivative = activation_derivative(activation, buffer.hiddenlay2(i), acfunc);
        buffer.deris2[i] = derivative;
        buffer.dh2dh1.row(i) = derivative * w_hidden1.col(i).transpose();
    }
    // hidden2 -> output
    output = w_hidden2.dot(buffer.hiddenlay2);
    // dout/di  TODO: reduce 
    buffer.dh2di = buffer.dh2dh1 * buffer.dh1di;
    buffer.dedh1 = (w_hidden2.transpose() * buffer.dh2dh1).transpose();
    // back propagation. 
    // dedt
    // w_inpt, b_hidden1 w_hidden1 b_hidden2 w_hidden2
    // w_input
    buffer.dedh1deris1 = buffer.dedh1.array() * buffer.deris1.array();
    for (int i = 0; i < n_neuron1; ++i) {
        for (int k = 0; k < n_input; ++k) {
            dedt(index++) = buffer.dedh1deris1(i) * buffer.input(k);
        }
    }
    // b_hidden1
    for (int i = 0; i < n_neuron1; ++i) {
        dedt(index++) = buffer.dedh1deris1(i);
    }
    // w_hidden1
    buffer.dedh2deris2 = w_hidden2.array() * buffer.deris2.array();
    for (int i = 0; i < n_neuron2; ++i) {
        for (int k = 0; k < n_neuron1; ++k) {
            dedt(index++) = buffer.dedh2deris2(i) * buffer.hiddenlay1(k);
        }
    }
    // b_hidden2
    for (int i = 0; i < n_neuron2; ++i) {
        dedt(index++) = buffer.dedh2deris2(i);
    }
    // w_hidden2
    for (int i = 0; i < n_neuron2; ++i) {
        dedt(index++) = buffer.hiddenlay2(i);
    }
}

// dedt d2edqdt
void IMRMLPNN::NN2lay::backpropgate(const Vector &input, 
        Scalar &output, 
        Vector &dedq, // correct
        Vector &dedt, // correct
        Matrix &d2edqdt, 
        Matrix &d2edqdq, 
        int &tag , 
        int ibuffer)
{
    backpropgate(input, output, dedq, dedt, d2edqdt, d2edqdq, tag, coefficient_view(), buffers[ibuffer]);
}

void IMRMLPNN::NN2lay::backpropgate(const Vector &input, Scalar &output, Vector &dedq, Vector &dedt, Matrix &d2edqdt, Matrix &d2edqdq, int &tag , const CoeffView &weights, Buffer &buffer) const
{
    prepare_buffer(buffer);
    const auto &w_input = weights.w_input;
    const auto &b_hidden1 = weights.b_hidden1;
    const auto &w_hidden1 = weights.w_hidden1;
    const auto &b_hidden2 = weights.b_hidden2;
    const auto &w_hidden2 = weights.w_hidden2;
    const auto &norm_mean = weights.norm_mean;
    const auto &norm_irstd = weights.norm_irstd;

    buffer.dh1di.resize(n_neuron1, n_input);
    buffer.dh2dh1.resize(n_neuron2, n_neuron1);
    buffer.input.array() = (input.array() - norm_mean.array()) * norm_irstd.array();
    buffer.d2h1didi.resize(n_neuron1, n_input, n_input);
    buffer.d2h2dh1dh1.resize(n_neuron2, n_neuron1, n_neuron1);
    buffer.deris1.resize(n_neuron1);
    buffer.secderis1.resize(n_neuron1);
    buffer.deris2.resize(n_neuron2);
    buffer.secderis2.resize(n_neuron2);
    dedt.setZero(n_coeffs);
    d2edqdt.setZero(n_input, n_coeffs);
    d2edqdq.setZero(n_input, n_input); // takes upper
    // propagate(input, output, ibuffer);
    // Derivative of hidden layer 2 w.r.t hidden layer 1
    // Vector d_hidden2_dhidden1 = buffer.hiddenlay2.unaryExpr([&](Scalar x) { return activation_derivative(x, acfunc); });
    // input -> hidden1  dh1/di    d2h1/didi
    int index = 0;
    for (int i = 0; i < n_neuron1; ++i) {
        Scalar activation = w_input.col(i).dot(buffer.input) + b_hidden1(i);
        buffer.hiddenlay1(i) = activation_function(activation, acfunc);
        Scalar derivative = activation_derivative(activation, buffer.hiddenlay1(i), acfunc);
        buffer.deris1[i] = derivative;
        buffer.dh1di.row(i) = (derivative * w_input.col(i)).array() * norm_irstd.array();
        Scalar derivative2nd = activation_2nd_derivative(activation, buffer.hiddenlay1(i), derivative, acfunc);
        buffer.secderis1[i] = derivative2nd;
        for (int m = 0; m < n_input; ++m) {
            buffer.d2h1didi(i, m, m) = w_input(m, i) * w_input(m, i) * derivative2nd * norm_irstd(m) * norm_irstd(m);
            for (int n = m+1; n < n_input; ++n) {
                double c = w_input(m, i) * w_input(n, i) * derivative2nd * norm_irstd(m) * norm_irstd(n);
                buffer.d2h1didi(i, m, n) = c;
                buffer.d2h1didi(i, n, m) = c;
            }
        }
    }
    // hidden1 -> hidden2   dh2/dh1
    for (int i = 0; i < n_neuron2; ++i) {
        Scalar activation = w_hidden1.col(i).dot(buffer.hiddenlay1) + b_hidden2(i);
        buffer.hiddenlay2(i) = activation_function(activation, acfunc);
        Scalar derivative = activation_derivative(activation, buffer.hiddenlay2(i), acfunc);
        buffer.deris2[i] = derivative;
        buffer.dh2dh1.row(i) = derivative * w_hidden1.col(i).transpose();
        Scalar derivative2nd = activation_2nd_derivative(activation, buffer.hiddenlay2(i), derivative, acfunc);
        buffer.secderis2[i] = derivative2nd;
        for (int m = 0; m < n_neuron1; ++m) {
            buffer.d2h2dh1dh1(i, m, m) = w_hidden1(m, i) * w_hidden1(m, i) * derivative2nd;
            for (int n = m+1; n < n_neuron1; ++n) {
                double c = w_hidden1(m, i) * w_hidden1(n, i) * derivative2nd;
                buffer.d2h2dh1dh1(i, m, n) = c;
                buffer.d2h2dh1dh1(i, n, m) = c;
            }
        }
    }
    // hidden2 -> output
    output = w_hidden2.dot(buffer.hiddenlay2);
    // dout/di  TODO: reduce 
    buffer.dh2di = buffer.dh2dh1 * buffer.dh1di;
    buffer.dedh1 = (w_hidden2.transpose() * buffer.dh2dh1).transpose();
    dedq = (w_hidden2.transpose() * buffer.dh2di).transpose();
    for (int m = 0; m < n_input; ++m) {
        for (int n = m; n < n_input; ++n) {
            for (int i = 0; i < n_neuron2; ++i) {
                for (int j = 0; j < n_neuron1; ++j) {
                    d2edqdq(m, n) += w_hidden2(i) * buffer.dh2dh1(i, j) * buffer.d2h1didi(j, m, n);
                    for (int k = 0; k < n_neuron1; ++k) {
                        d2edqdq(m, n) += w_hidden2(i) * buffer.d2h2dh1dh1(i, j, k) * buffer.dh1di(k, n) * buffer.dh1di(j, m);
                    }
                }
            }
        }
    }
    // fill symm part
    for (int m = 1; m < n_input; ++m) {
        for (int n = 0; n < m; ++n) {
            d2edqdq(m, n) = d2edqdq(n, m);
        }
    }
    // back propagation. 
    // dedt
    // w_inpt, b_hidden1 w_hidden1 b_hidden2 w_hidden2
    // w_input
    buffer.dedh1deris1 = buffer.dedh1.array() * buffer.deris1.array();
    for (int i = 0; i < n_neuron1; ++i) {
        for (int k = 0; k < n_input; ++k) {
            dedt(index++) = buffer.dedh1deris1(i) * buffer.input(k);
        }
    }
    // b_hidden1
    for (int i = 0; i < n_neuron1; ++i) {
        dedt(index++) = buffer.dedh1deris1(i);
    }
    // w_hidden1
    buffer.dedh2deris2 = w_hidden2.array() * buffer.deris2.array();
    for (int i = 0; i < n_neuron2; ++i) {
        for (int k = 0; k < n_neuron1; ++k) {
            dedt(index++) = buffer.dedh2deris2(i) * buffer.hiddenlay1(k);
        }
    }
    // b_hidden2
    for (int i = 0; i < n_neuron2; ++i) {
        dedt(index++) = buffer.dedh2deris2(i);
    }
    // w_hidden2
    for (int i = 0; i < n_neuron2; ++i) {
        dedt(index++) = buffer.hiddenlay2(i);
    }
    // d2edqdt
    // w_inpt, b_hidden1 w_hidden1 b_hidden2 w_hidden2
    // w_input
    index = 0;
    for (int i = 0; i < n_neuron1; ++i) {
        for (int j = 0; j < n_input; ++j) {
            for (int k = 0; k < n_input; ++k) {
                d2edqdt(k, index) += buffer.dedh1(i) * buffer.secderis1(i) * w_input(k, i) * buffer.input(j) * norm_irstd(k);;
                if (j == k) d2edqdt(k, index) += buffer.dedh1(i) * buffer.deris1(i) * norm_irstd(k);
                for (int m = 0; m < n_neuron1; ++m) {
                    for (int p = 0; p < n_neuron2; ++p) {
                        d2edqdt(k, index) += w_hidden2(p) * buffer.d2h2dh1dh1(p, m, i) * buffer.deris1(m) * buffer.deris1(i) * w_input(k, m) * buffer.input(j) * norm_irstd(k);;
                    }
                }
            }
            index++;
        }
    }
    // b-hidden1
    for (int i = 0; i < n_neuron1; ++i) {
        for (int k = 0; k < n_input; ++k) {
            d2edqdt(k, index) += buffer.secderis1(i) * w_input(k, i) * buffer.dedh1(i) * norm_irstd(k);;
            for (int m = 0; m < n_neuron1; ++m) {
                for (int p = 0; p < n_neuron2; ++p) {
                    d2edqdt(k, index) += w_hidden2(p) * buffer.d2h2dh1dh1(p, m, i) * buffer.deris1(m) * buffer.deris1(i) * w_input(k, m) * norm_irstd(k);;
                }
            }
        }
        index++;
    }
    // w-hidden1
    buffer.dha2dq = w_hidden1.transpose() * buffer.dh1di;
    // buffer.dha2dq = buffer.dh1di * w_hidden1;
    for (int i = 0; i < n_neuron2; ++i) {
        for (int j = 0; j < n_neuron1; ++j) {
            for (int k = 0; k < n_input; ++k) {
                d2edqdt(k, index) += w_hidden2(i) * (
                    buffer.secderis2[i] * buffer.dha2dq(i, k) *  buffer.hiddenlay1(j)
                    + buffer.deris2(i) * buffer.dh1di(j, k));
            }
            index++;
        }
    }
    // b-hidden2
    for (int i = 0; i < n_neuron2; ++i) {
        for (int k = 0; k < n_input; ++k) {
            d2edqdt(k, index) += w_hidden2(i) * buffer.secderis2[i] * buffer.dha2dq(i, k);
        }
        index++;
    }
    // w-hidden2
    for (int i = 0; i < n_neuron2; ++i) {
        for (int k = 0; k < n_input; ++k) {
            d2edqdt(k, index) += buffer.dh2di(i, k);
        }
        index++;
    }
}

void IMRMLPNN::NN2lay::backpropgate(const Vector &input, Scalar &output, Vector &dedt, Matrix &d2edqdt, int &tag , int ibuffer)
{
    backpropgate(input, output, dedt, d2edqdt, tag, coefficient_view(), buffers[ibuffer]);
}

void IMRMLPNN::NN2lay::backpropgate(const Vector &input, Scalar &output, Vector &dedt, Matrix &d2edqdt, int &tag , const CoeffView &weights, Buffer &buffer) const
{
    prepare_buffer(buffer);
    const auto &w_input = weights.w_input;
    const auto &b_hidden1 = weights.b_hidden1;
    const auto &w_hidden1 = weights.w_hidden1;
    const auto &b_hidden2 = weights.b_hidden2;
    const auto &w_hidden2 = weights.w_hidden2;
    const auto &norm_mean = weights.norm_mean;
    const auto &norm_irstd = weights.norm_irstd;

    buffer.dh1di.resize(n_neuron1, n_input);
    buffer.dh2dh1.resize(n_neuron2, n_neuron1);
    buffer.input.array() = (input.array() - norm_mean.array()) * norm_irstd.array();
    buffer.d2h1didi.resize(n_neuron1, n_input, n_input);
    buffer.d2h2dh1dh1.resize(n_neuron2, n_neuron1, n_neuron1);
    buffer.deris1.resize(n_neuron1);
    buffer.secderis1.resize(n_neuron1);
    buffer.deris2.resize(n_neuron2);
    buffer.secderis2.resize(n_neuron2);
    dedt.setZero(n_coeffs);
    d2edqdt.setZero(n_input, n_coeffs);
    // input -> hidden1  dh1/di    d2h1/didi
    int index = 0;
    for (int i = 0; i < n_neuron1; ++i) {
        Scalar activation = w_input.col(i).dot(buffer.input) + b_hidden1(i);
        buffer.hiddenlay1(i) = activation_function(activation, acfunc);
        Scalar derivative = activation_derivative(activation, buffer.hiddenlay1(i), acfunc);
        buffer.deris1[i] = derivative;
        buffer.dh1di.row(i) = (derivative * w_input.col(i)).array() * norm_irstd.array();
        Scalar derivative2nd = activation_2nd_derivative(activation, buffer.hiddenlay1(i), derivative, acfunc);
        buffer.secderis1[i] = derivative2nd;
        for (int m = 0; m < n_input; ++m) {
            buffer.d2h1didi(i, m, m) = w_input(m, i) * w_input(m, i) * derivative2nd * norm_irstd(m) * norm_irstd(m);
            for (int n = m+1; n < n_input; ++n) {
                double c = w_input(m, i) * w_input(n, i) * derivative2nd * norm_irstd(m) * norm_irstd(n);
                buffer.d2h1didi(i, m, n) = c;
                buffer.d2h1didi(i, n, m) = c;
            }
        }
    }
    // hidden1 -> hidden2   dh2/dh1
    for (int i = 0; i < n_neuron2; ++i) {
        Scalar activation = w_hidden1.col(i).dot(buffer.hiddenlay1) + b_hidden2(i);
        buffer.hiddenlay2(i) = activation_function(activation, acfunc);
        Scalar derivative = activation_derivative(activation, buffer.hiddenlay2(i), acfunc);
        buffer.deris2[i] = derivative;
        buffer.dh2dh1.row(i) = derivative * w_hidden1.col(i).transpose();
        Scalar derivative2nd = activation_2nd_derivative(activation, buffer.hiddenlay2(i), derivative, acfunc);
        buffer.secderis2[i] = derivative2nd;
        for (int m = 0; m < n_neuron1; ++m) {
            buffer.d2h2dh1dh1(i, m, m) = w_hidden1(m, i) * w_hidden1(m, i) * derivative2nd;
            for (int n = m+1; n < n_neuron1; ++n) {
                double c = w_hidden1(m, i) * w_hidden1(n, i) * derivative2nd;
                buffer.d2h2dh1dh1(i, m, n) = c;
                buffer.d2h2dh1dh1(i, n, m) = c;
            }
        }
    }
    // hidden2 -> output
    output = w_hidden2.dot(buffer.hiddenlay2);
    // dout/di  TODO: reduce 
    buffer.dh2di = buffer.dh2dh1 * buffer.dh1di;
    buffer.dedh1 = (w_hidden2.transpose() * buffer.dh2dh1).transpose();
    // back propagation. 
    // dedt
    // w_inpt, b_hidden1 w_hidden1 b_hidden2 w_hidden2
    // w_input
    buffer.dedh1deris1 = buffer.dedh1.array() * buffer.deris1.array();
    for (int i = 0; i < n_neuron1; ++i) {
        for (int k = 0; k < n_input; ++k) {
            dedt(index++) = buffer.dedh1deris1(i) * buffer.input(k);
        }
    }
    // b_hidden1
    for (int i = 0; i < n_neuron1; ++i) {
        dedt(index++) = buffer.dedh1deris1(i);
    }
    // w_hidden1
    buffer.dedh2deris2 = w_hidden2.array() * buffer.deris2.array();
    for (int i = 0; i < n_neuron2; ++i) {
        for (int k = 0; k < n_neuron1; ++k) {
            dedt(index++) = buffer.dedh2deris2(i) * buffer.hiddenlay1(k);
        }
    }
    // b_hidden2
    for (int i = 0; i < n_neuron2; ++i) {
        dedt(index++) = buffer.dedh2deris2(i);
    }
    // w_hidden2
    for (int i = 0; i < n_neuron2; ++i) {
        dedt(index++) = buffer.hiddenlay2(i);
    }
    // d2edqdt
    // w_inpt, b_hidden1 w_hidden1 b_hidden2 w_hidden2
    // w_input
    index = 0;
    for (int i = 0; i < n_neuron1; ++i) {
        for (int j = 0; j < n_input; ++j) {
            for (int k = 0; k < n_input; ++k) {
                d2edqdt(k, index) += buffer.dedh1(i) * buffer.secderis1(i) * w_input(k, i) * buffer.input(j) * norm_irstd(k);;
                if (j == k) d2edqdt(k, index) += buffer.dedh1(i) * buffer.deris1(i) * norm_irstd(k);
                for (int m = 0; m < n_neuron1; ++m) {
                    for (int p = 0; p < n_neuron2; ++p) {
                        d2edqdt(k, index) += w_hidden2(p) * buffer.d2h2dh1dh1(p, m, i) * buffer.deris1(m) * buffer.deris1(i) * w_input(k, m) * buffer.input(j) * norm_irstd(k);;
                    }
                }
            }
            index++;
        }
    }
    // b-hidden1
    for (int i = 0; i < n_neuron1; ++i) {
        for (int k = 0; k < n_input; ++k) {
            d2edqdt(k, index) += buffer.secderis1(i) * w_input(k, i) * buffer.dedh1(i) * norm_irstd(k);;
            for (int m = 0; m < n_neuron1; ++m) {
                for (int p = 0; p < n_neuron2; ++p) {
                    d2edqdt(k, index) += w_hidden2(p) * buffer.d2h2dh1dh1(p, m, i) * buffer.deris1(m) * buffer.deris1(i) * w_input(k, m) * norm_irstd(k);;
                }
            }
        }
        index++;
    }
    // w-hidden1
    buffer.dha2dq = w_hidden1.transpose() * buffer.dh1di;
    // buffer.dha2dq = buffer.dh1di * w_hidden1;
    for (int i = 0; i < n_neuron2; ++i) {
        for (int j = 0; j < n_neuron1; ++j) {
            for (int k = 0; k < n_input; ++k) {
                d2edqdt(k, index) += w_hidden2(i) * (
                    buffer.secderis2[i] * buffer.dha2dq(i, k) *  buffer.hiddenlay1(j)
                    + buffer.deris2(i) * buffer.dh1di(j, k));
            }
            index++;
        }
    }
    // b-hidden2
    for (int i = 0; i < n_neuron2; ++i) {
        for (int k = 0; k < n_input; ++k) {
            d2edqdt(k, index) += w_hidden2(i) * buffer.secderis2[i] * buffer.dha2dq(i, k);
        }
        index++;
    }
    // w-hidden2
    for (int i = 0; i < n_neuron2; ++i) {
        for (int k = 0; k < n_input; ++k) {
            d2edqdt(k, index) += buffer.dh2di(i, k);
        }
        index++;
    }
}

void IMRMLPNN::NN2lay::backpropgate(const Vector& input, Scalar& output, Vector& dedq, Vector& dedt, int &tag , int ibuffer)
{
    backpropgate(input, output, dedq, dedt, tag, coefficient_view(), buffers[ibuffer]);
}

void IMRMLPNN::NN2lay::backpropgate(const Vector& input, Scalar& output, Vector& dedq, Vector& dedt, int &tag , const CoeffView &weights, Buffer &buffer) const
{
    prepare_buffer(buffer);
    const auto &w_input = weights.w_input;
    const auto &b_hidden1 = weights.b_hidden1;
    const auto &w_hidden1 = weights.w_hidden1;
    const auto &b_hidden2 = weights.b_hidden2;
    const auto &w_hidden2 = weights.w_hidden2;
    const auto &norm_mean = weights.norm_mean;
    const auto &norm_irstd = weights.norm_irstd;

    buffer.dh1di.resize(n_neuron1, n_input);
    buffer.dh2dh1.resize(n_neuron2, n_neuron1);
    buffer.input.array() = (input.array() - norm_mean.array()) * norm_irstd.array();
    buffer.deris1.resize(n_neuron1);
    buffer.deris2.resize(n_neuron2);
    dedt.setZero(n_coeffs);
    // input -> hidden1  dh1/di    d2h1/didi
    int index = 0;
    for (int i = 0; i < n_neuron1; ++i) {
        Scalar activation = w_input.col(i).dot(buffer.input) + b_hidden1(i);
        buffer.hiddenlay1(i) = activation_function(activation, acfunc);
        Scalar derivative = activation_derivative(activation, buffer.hiddenlay1(i), acfunc);
        buffer.deris1[i] = derivative;
        buffer.dh1di.row(i) = (derivative * w_input.col(i)).array() * norm_irstd.array();
    }
    // hidden1 -> hidden2   dh2/dh1
    for (int i = 0; i < n_neuron2; ++i) {
        Scalar activation = w_hidden1.col(i).dot(buffer.hiddenlay1) + b_hidden2(i);
        buffer.hiddenlay2(i) = activation_function(activation, acfunc);
        Scalar derivative = activation_derivative(activation, buffer.hiddenlay2(i), acfunc);
        buffer.deris2[i] = derivative;
        buffer.dh2dh1.row(i) = derivative * w_hidden1.col(i).transpose();
    }
    // hidden2 -> output
    output = w_hidden2.dot(buffer.hiddenlay2);
    // dout/di  TODO: reduce 
    buffer.dh2di = buffer.dh2dh1 * buffer.dh1di;
    buffer.dedh1 = (w_hidden2.transpose() * buffer.dh2dh1).transpose();
    dedq = (w_hidden2.transpose() * buffer.dh2di).transpose();
    // back propagation. 
    // dedt
    // w_inpt, b_hidden1 w_hidden1 b_hidden2 w_hidden2
    // w_input
    buffer.dedh1deris1 = buffer.dedh1.array() * buffer.deris1.array();
    for (int i = 0; i < n_neuron1; ++i) {
        for (int k = 0; k < n_input; ++k) {
            dedt(index++) = buffer.dedh1deris1(i) * buffer.input(k);
        }
    }
    // b_hidden1
    for (int i = 0; i < n_neuron1; ++i) {
        dedt(index++) = buffer.dedh1deris1(i);
    }
    // w_hidden1
    buffer.dedh2deris2 = w_hidden2.array() * buffer.deris2.array();
    for (int i = 0; i < n_neuron2; ++i) {
        for (int k = 0; k < n_neuron1; ++k) {
            dedt(index++) = buffer.dedh2deris2(i) * buffer.hiddenlay1(k);
        }
    }
    // b_hidden2
    for (int i = 0; i < n_neuron2; ++i) {
        dedt(index++) = buffer.dedh2deris2(i);
    }
    // w_hidden2
    for (int i = 0; i < n_neuron2; ++i) {
        dedt(index++) = buffer.hiddenlay2(i);
    }
}

void IMRMLPNN::NN2lay::dump_coeffs(const std::string &outfile)
{
    std::ofstream ofs(outfile, std::ios::binary);
    if (!ofs) {
        throw std::ios_base::failure("Failed to open file for writing: " + outfile);
    }

    CoeffIO::write_header(ofs, 2, acfunc);
    ofs.write(reinterpret_cast<const char *>(&n_neuron1), sizeof(n_neuron1));
    ofs.write(reinterpret_cast<const char *>(&n_neuron2), sizeof(n_neuron2));
    ofs.write(reinterpret_cast<const char *>(&n_input), sizeof(n_input));

    ofs.write(reinterpret_cast<const char *>(w_input.data()), w_input.size() * sizeof(Scalar));
    ofs.write(reinterpret_cast<const char *>(b_hidden1.data()), b_hidden1.size() * sizeof(Scalar));
    ofs.write(reinterpret_cast<const char *>(w_hidden1.data()), w_hidden1.size() * sizeof(Scalar));
    ofs.write(reinterpret_cast<const char *>(b_hidden2.data()), b_hidden2.size() * sizeof(Scalar));
    ofs.write(reinterpret_cast<const char *>(w_hidden2.data()), w_hidden2.size() * sizeof(Scalar));
    ofs.write(reinterpret_cast<const char *>(norm_mean.data()), norm_mean.size() * sizeof(Scalar));
    ofs.write(reinterpret_cast<const char *>(norm_irstd.data()), norm_irstd.size() * sizeof(Scalar));
    CoeffIO::finish_output(ofs, outfile);
}

void IMRMLPNN::NN2lay::dump_coeffs_txt(const std::string &outfile)
{
    std::ofstream ofs(outfile);
    if (!ofs) {
        throw std::ios_base::failure("Failed to open file for writing: " + outfile);
    }
    ofs << fmt::format("{} {} {}\n", n_input, n_neuron1, n_neuron2);
    for (int i = 0; i < w_input.rows(); ++i) {
        ofs << fmt::format("{}\n", fmt::join(w_input.row(i), " "));
    }
    for (int i = 0; i < w_hidden1.rows(); ++i) {
        ofs << fmt::format("{}\n", fmt::join(w_hidden1.row(i), " "));
    }
    ofs << fmt::format("{}\n{}\n{}\n{}\n{}\n", fmt::join(b_hidden1, " "), fmt::join(w_hidden2, " "), fmt::join(b_hidden2, " "), fmt::join(norm_mean, " "), fmt::join(norm_irstd, " "));
    CoeffIO::finish_output(ofs, outfile);
}

void IMRMLPNN::NN2lay::load_coeffs(const std::string &infile)
{
    std::ifstream ifs(infile, std::ios::binary);
    if (!ifs) throw std::ios_base::failure("Failed to open file for reading: " + infile);
    const auto header = CoeffIO::read_header(ifs, infile, 2);
    if (header.layers != 2) CoeffIO::fail(infile);
    int hidden1 = 0, hidden2 = 0, input = 0;
    ifs.read(reinterpret_cast<char *>(&hidden1), sizeof(hidden1));
    ifs.read(reinterpret_cast<char *>(&hidden2), sizeof(hidden2));
    ifs.read(reinterpret_cast<char *>(&input), sizeof(input));
    if (hidden2 <= 0) CoeffIO::fail(infile);
    const bool normalized = CoeffIO::check_payload(ifs, infile, header.legacy, input, hidden1, hidden2);
    NN2lay loaded;
    loaded.setup(input, hidden1, hidden2);
    loaded.acfunc = static_cast<ActivationFunction>(header.activation);
    CoeffIO::read_array(ifs, loaded.w_input, infile);
    CoeffIO::read_array(ifs, loaded.b_hidden1, infile);
    CoeffIO::read_array(ifs, loaded.w_hidden1, infile);
    CoeffIO::read_array(ifs, loaded.b_hidden2, infile);
    CoeffIO::read_array(ifs, loaded.w_hidden2, infile);
    if (normalized) {
        CoeffIO::read_array(ifs, loaded.norm_mean, infile);
        CoeffIO::read_array(ifs, loaded.norm_irstd, infile);
    }
    *this = std::move(loaded);
}
