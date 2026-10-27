// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#include <catch2/catch_all.hpp>

#include "potentials/potential_mtp_m.h"

#include <cmath>
#include <fstream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

using Catch::Approx;

namespace {

void init_nn2_model(Potentials::m_model_nn2& model, std::vector<double>& coeffs)
{
    GPSettings settings;
    settings.Modify("n_input", "3");
    settings.Modify("n_neuron", "4");
    settings.Modify("n_neuron2", "3");
    settings.Modify("activation_function", "TANH");

    model.init(settings);
    coeffs.resize(model.get_ncoeffs());
    for (int i = 0; i < model.get_ncoeffs(); ++i)
        coeffs[i] = 0.11 * std::sin(0.37 * (i + 1)) + 0.03 * std::cos(0.19 * (i + 2));

    model.coeffs = coeffs.data();
    model.postinit();
}

double model_energy(Potentials::m_model_nn2& model, const Eigen::VectorXd& qs)
{
    double energy = 0.0;
    int tag = 0;
    int ithread = 0;
    model.propagate(qs, energy, tag, ithread);
    return energy;
}

std::string make_mtp_m_template_file()
{
    const std::string path = "/tmp/ut_mtp_m_template.mtp";
    std::ifstream in;
    for (const std::string& candidate : {
             "tests/test-nnmtp/_Trained_Potential_.mtp",
             "unitest/test-data/potential.mtp",
             "../../unitest/test-data/potential.mtp",
             "../../../unitest/test-data/potential.mtp",
         }) {
        in.open(candidate);
        if (in.good())
            break;
        in.close();
    }
    std::ofstream out(path);
    REQUIRE(in.good());
    REQUIRE(out.good());

    std::string line;
    while (std::getline(in, line)) {
        if (line.rfind("species_coeffs", 0) == 0)
            break;
        out << line << '\n';
    }
    return path;
}

} // namespace

TEST_CASE("PotMTP_M two-layer NN model derivatives match finite differences", "[mtp_m][nn2]")
{
    Potentials::m_model_nn2 model;
    std::vector<double> coeffs;
    init_nn2_model(model, coeffs);

    Eigen::VectorXd qs(3);
    qs << -0.31, 0.27, 0.63;

    double energy = 0.0;
    Eigen::VectorXd dUdQ(3);
    Eigen::VectorXd dUdt(model.get_ncoeffs());
    Eigen::MatrixXd d2UdQdt(3, model.get_ncoeffs());
    Eigen::MatrixXd d2UdQdQ(3, 3);
    int tag = 0;
    int ithread = 0;
    model.backpropagate(qs, energy, dUdQ, dUdt, d2UdQdt, d2UdQdQ, tag, ithread);

    const double q_eps = 1e-6;
    for (int iq = 0; iq < qs.size(); ++iq) {
        Eigen::VectorXd qp = qs;
        Eigen::VectorXd qm = qs;
        qp(iq) += q_eps;
        qm(iq) -= q_eps;
        const double finite = (model_energy(model, qp) - model_energy(model, qm)) / (2.0 * q_eps);
        REQUIRE(dUdQ(iq) == Approx(finite).margin(1e-7));
    }

    const double c_eps = 1e-6;
    for (int ip = 0; ip < model.get_ncoeffs(); ip += 3) {
        const double orig = coeffs[ip];
        coeffs[ip] = orig + c_eps;
        const double ep = model_energy(model, qs);
        coeffs[ip] = orig - c_eps;
        const double em = model_energy(model, qs);
        coeffs[ip] = orig;

        const double finite = (ep - em) / (2.0 * c_eps);
        REQUIRE(dUdt(ip) == Approx(finite).margin(1e-7));
    }

    Eigen::VectorXd dUdt_plus(model.get_ncoeffs());
    Eigen::VectorXd dUdt_minus(model.get_ncoeffs());
    for (int iq = 0; iq < qs.size(); ++iq) {
        Eigen::VectorXd qp = qs;
        Eigen::VectorXd qm = qs;
        qp(iq) += q_eps;
        qm(iq) -= q_eps;

        double ep = 0.0;
        double em = 0.0;
        model.backpropagate(qp, ep, dUdt_plus, tag, ithread);
        model.backpropagate(qm, em, dUdt_minus, tag, ithread);

        for (int ip = iq; ip < model.get_ncoeffs(); ip += 5) {
            const double finite = (dUdt_plus(ip) - dUdt_minus(ip)) / (2.0 * q_eps);
            REQUIRE(d2UdQdt(iq, ip) == Approx(finite).margin(2e-6));
        }
    }
}

TEST_CASE("PotMTP_M two-layer NN model supports scalar regularization", "[mtp_m][nn2][regularization]")
{
    Potentials::m_model_nn2 model;
    std::vector<double> coeffs;
    GPSettings settings;
    settings.Modify("n_input", "2");
    settings.Modify("n_neuron", "3");
    settings.Modify("n_neuron2", "2");
    settings.Modify("model_l1", "0.5");
    settings.Modify("model_l2", "0.25");
    model.init(settings);
    coeffs.resize(model.get_ncoeffs());
    for (int i = 0; i < model.get_ncoeffs(); ++i)
        coeffs[i] = (i % 2 == 0 ? 0.1 : -0.2) * (i + 1);
    model.coeffs = coeffs.data();
    model.postinit();

    std::vector<double> grad(model.get_ncoeffs() + 4, 0.0);
    constexpr int offset = 4;
    double loss = 0.0;
    model.add_loss_gard(loss, grad.data(), offset);

    double expected_loss = 0.0;
    for (int i = 0; i < model.get_ncoeffs(); ++i) {
        expected_loss += 0.5 * std::abs(coeffs[i]) + 0.25 * coeffs[i] * coeffs[i];
        REQUIRE(grad[offset + i] == Approx((coeffs[i] > 0.0 ? 0.5 : -0.5) + 0.5 * coeffs[i]));
    }
    for (int i = 0; i < offset; ++i)
        REQUIRE(grad[i] == Approx(0.0));
    REQUIRE(loss == Approx(expected_loss));
}

TEST_CASE("PotMTP_M factory registers two-layer NN model", "[mtp_m][nn2][factory]")
{
    GPSettings settings;
    settings.Modify("potfile", make_mtp_m_template_file());
    settings.Modify("extmodel", "mtp_m_n2");
    settings.Modify("n_input", "3");
    settings.Modify("n_neuron", "4");
    settings.Modify("n_neuron2", "2");

    std::unique_ptr<Potentials::AnyPotential> pot(Potentials::create_mtp_m_potential(settings));
    REQUIRE(pot != nullptr);
}

TEST_CASE("PotMTP_M saves LAMMPS GPU model metadata", "[mtp_m][nn2][io]")
{
    GPSettings settings;
    settings.Modify("n_input", "3");
    settings.Modify("n_neuron", "4");
    settings.Modify("n_neuron2", "2");
    settings.Modify("activation_function", "SIGMOID");
    Potentials::PotMTP_M<Potentials::m_model_nn2> potential(
        make_mtp_m_template_file(), settings);

    std::ostringstream output;
    potential.Save(output);
    const std::string text = output.str();
    REQUIRE(text.find("# mtp_model = mtp_m_n2\n") != std::string::npos);
    REQUIRE(text.find("# mtp_descriptor_count = 3\n") != std::string::npos);
    REQUIRE(text.find("# mtp_model_coeffs_count = 28\n") != std::string::npos);
    REQUIRE(text.find("# mtp_activation = SIGMOID\n") != std::string::npos);
    REQUIRE(text.find("# mtp_hidden1 = 4\n") != std::string::npos);
    REQUIRE(text.find("# mtp_hidden2 = 2\n") != std::string::npos);

    REQUIRE(std::string(Potentials::PotMTP_M<
                Potentials::m_model_mtpq2>::serialized_model_name()) == "mtp_m_q2");
    REQUIRE(std::string(Potentials::PotMTP_M<
                Potentials::m_model_mtpfs>::serialized_model_name()) == "mtp_m_fs");
    REQUIRE(std::string(Potentials::PotMTP_M<
                Potentials::m_model_nn1>::serialized_model_name()) == "mtp_m_n1");
}

TEST_CASE("PotMTP_M NN models reject use before coefficient mapping", "[mtp_m][nn]")
{
    Potentials::m_model_nn1 nn1;
    GPSettings settings;
    settings.Modify("n_input", "2");
    settings.Modify("n_neuron", "3");
    nn1.init(settings);

    Eigen::VectorXd qs(2);
    qs << 0.2, -0.1;
    double energy = 0.0;
    int tag = 0;
    int ithread = 0;
    REQUIRE_THROWS_AS(nn1.propagate(qs, energy, tag, ithread), GPException);

    Potentials::m_model_nn2 nn2;
    settings.Modify("n_neuron2", "2");
    nn2.init(settings);
    REQUIRE_THROWS_AS(nn2.propagate(qs, energy, tag, ithread), GPException);
}
