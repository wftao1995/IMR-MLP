// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#include "basis.h"
#include <array>
#include <cmath>
#include <Eigen/Eigenvalues>
#include <utility>

BasisType basis_name_to_code(const std::string& rbasis_type)
{
    if (rbasis_type == "RBChebyshevConfigurable")
        return BasisType::RB_ChebyshevConfigurable;
    if (rbasis_type == "RBChebyshev")
        return BasisType::RB_Chebyshev;
    if (rbasis_type == "RBChebyshevPoly4")
        return BasisType::RB_ChebyshevPoly4;
    if (rbasis_type == "RBChebyshevPoly5")
        return BasisType::RB_ChebyshevPoly5;
    if (rbasis_type == "RBPolyB")
        return BasisType::RB_PolyB;
    if (rbasis_type == "RBPolyBCos")
        return BasisType::RB_PolyBCos;
    if (rbasis_type == "RBChebyCos")
        return BasisType::RB_ChebyCos;
    return BasisType::RB_Unknown;
}

AnyBasis* create_basis_by_code(const BasisType& rbasis_type, std::istream& is, double _scaling)
{
    switch (rbasis_type)
    {
        case BasisType::RB_ChebyshevConfigurable:
            return new RBasis_ChebyshevConfigurable(is, _scaling);
        case BasisType::RB_Chebyshev:
            return new RBasis_Chebyshev(is, _scaling);
        case BasisType::RB_ChebyshevPoly4:
            return new RBasis_ChebyshevPoly4(is, _scaling);
        case BasisType::RB_ChebyshevPoly5:
            return new RBasis_ChebyshevPoly5(is, _scaling);
        case BasisType::RB_PolyB:
            return new RBasis_PolyB(is, _scaling);
        case BasisType::RB_PolyBCos:
            return new RBasis_PolyBCos(is, _scaling);
        case BasisType::RB_ChebyCos:
            return new RBasis_ChebyCos(is, _scaling);
        default:
            throw std::runtime_error("Unknown basis type");
    }
}

AnyBasis * create_basis_by_name(const std::string & rbasis_type, std::istream & is, double _scaling)
{
    BasisType code = basis_name_to_code(rbasis_type);
    if (code == BasisType::RB_Unknown) {
        throw std::runtime_error("Unknown basis type: " + rbasis_type);
    }
    return create_basis_by_code(code, is, _scaling);
}

AnyBasis::AnyBasis(double _min_val, double _max_val, int _size, double _scaling) : basis_size(_size), min_val(_min_val), max_val(_max_val), scaling(_scaling) {
    if (basis_size <= 0)
        throw std::invalid_argument("radial_basis_size must be positive");
    vals.resize(basis_size);
    ders.resize(basis_size);
    derders.resize(basis_size);
}

// read min_dist, max_dist and radial_basis_size. Other tags must present afterward
void AnyBasis::LoadFromStream(std::istream& is) {
    std::string tmpstr;
    is >> tmpstr; 
    if (tmpstr != "min_dist") {
        throw std::runtime_error("Error in loading basis: " + tmpstr);
    }
    is.ignore(2);
    is >> min_val;
    if (is.fail()) {
        throw std::runtime_error("Error in loading basis: " + tmpstr);
    }
    is >> tmpstr;
    if (tmpstr != "max_dist") {
        throw std::runtime_error("Error in loading basis: " + tmpstr);
    }
    is.ignore(2);
    is >> max_val;
    if (is.fail()) {
        throw std::runtime_error("Error in loading basis: " + tmpstr);
    }
    is >> tmpstr;
    if (tmpstr != "radial_basis_size") {
        throw std::runtime_error("Error in loading basis: " + tmpstr);
    }
    is.ignore(2);
    is >> basis_size;
    if (is.fail()) {
        throw std::runtime_error("Error in loading basis: " + tmpstr);
    }
    if (basis_size <= 0) {
        throw std::runtime_error("Error in loading basis: radial_basis_size must be positive");
    }

    vals.resize(basis_size);
    ders.resize(basis_size);
    derders.resize(basis_size);
};

void RBasis_Chebyshev::setup()
{
    mult = 2.0 / (max_val - min_val);
    su = max_val - min_val;
    pu = max_val + min_val;
    name = "RBChebyshev";
}

RBasis_Chebyshev::RBasis_Chebyshev(double _min_val, double _max_val, int _size, double _scaling)
    : AnyBasis(_min_val, _max_val, _size, _scaling)
{
    setup();
}

void RBasis_Chebyshev::calc_val(double value, std::vector<double> &val) {
    double ksi = (2 * value - (min_val + max_val)) / (max_val - min_val);
    double ru = value - max_val;

    val[0] = scaling * (1 * ru * ru);
    if (basis_size == 1) return;
    val[1] = scaling * (ksi * ru * ru);
    for (int i = 2; i < basis_size; i++)
    {
        val[i] = 2 * ksi * val[i - 1] - val[i - 2];
    }
}

void RBasis_Chebyshev::calc_der(double value, std::vector<double> &val, std::vector<double> &der) {
    double ksi = (2 * value - pu) / su;
    double ru = value - max_val;

    val[0] = scaling * (1 * ru * ru);
    der[0] = scaling * (0 * ru * ru + 2 * ru);
    if (basis_size == 1) return;
    val[1] = scaling * (ksi * ru * ru);
    der[1] = scaling * (mult * ru * ru + 2 * ksi * ru);
    for (int i = 2; i < basis_size; i++)
    {
        val[i] = 2 * ksi * val[i - 1] - val[i - 2];
        der[i] = 2 * (mult * val[i - 1] + ksi * der[i - 1]) - der[i - 2];
    }
}

void RBasis_Chebyshev::calc_derder(double value, std::vector<double> &val, std::vector<double> &der, std::vector<double> &derder) {
    double ksi = (2 * value - pu) / su;
    double ru = value - max_val;

    val[0] = scaling * (1 * ru * ru);
    der[0] = scaling * (2 * ru);
    derder[0] = scaling * 2;
    if (basis_size == 1) return;
    val[1] = scaling * (ksi * ru * ru);
    der[1] = scaling * (mult * ru * ru + 2 * ksi * ru);
    derder[1] = scaling * (4 * mult * ru + 2 * ksi); 
    for (int i = 2; i < basis_size; i++)
    {
        val[i] = 2 * ksi * val[i - 1] - val[i - 2];
        der[i] = 2 * (mult * val[i - 1] + ksi * der[i - 1]) - der[i - 2];
        derder[i] = 2 * (2 * mult * der[i-1] + ksi * derder[i-1]) - derder[i-2];
    }
}

void RBasis_Chebyshev::calc_val(double value, double rmin, double rmax, std::vector<double> &val) {
    double ksi = (2 * value - (rmin + rmax)) / (rmin - rmax);
    double ru = value - rmax;

    val[0] = scaling * (1 * ru * ru);
    if (basis_size == 1) return;
    val[1] = scaling * (ksi * ru * ru);
    for (int i = 2; i < basis_size; i++)
    {
        val[i] = 2 * ksi * val[i - 1] - val[i - 2];
    }
}

void RBasis_Chebyshev::calc_der(double value, double rmin, double rmax, std::vector<double> &val, std::vector<double> &der) {
    double rpu = rmin + rmax;
    double rsu = rmin - rmax;
    double rmult = 2.0 / rsu;
    double ksi = (2 * value - rpu) / rsu;
    double ru = value - rmax;

    val[0] = scaling * (1 * ru * ru);
    der[0] = scaling * (0 * ru * ru + 2 * ru);
    if (basis_size == 1) return;
    val[1] = scaling * (ksi * ru * ru);
    der[1] = scaling * (rmult * ru * ru + 2 * ksi * ru);
    for (int i = 2; i < basis_size; i++)
    {
        val[i] = 2 * ksi * val[i - 1] - val[i - 2];
        der[i] = 2 * (rmult * val[i - 1] + ksi * der[i - 1]) - der[i - 2];
    }
}

void RBasis_Chebyshev::calc_derder(double value, double rmin, double rmax, std::vector<double> &val, std::vector<double> &der, std::vector<double> &derder) {
    double rpu = rmin + rmax;
    double rsu = rmin - rmax;
    double rmult = 2.0 / rsu;
    double ksi = (2 * value - rpu) / rsu;
    double ru = value - rmax;

    val[0] = scaling * (1 * ru * ru);
    der[0] = scaling * (2 * ru);
    derder[0] = scaling * 2;
    if (basis_size == 1) return;
    val[1] = scaling * (ksi * ru * ru);
    der[1] = scaling * (rmult * ru * ru + 2 * ksi * ru);
    derder[1] = scaling * (4 * rmult * ru + 2 * ksi); 
    for (int i = 2; i < basis_size; i++)
    {
        val[i] = 2 * ksi * val[i - 1] - val[i - 2];
        der[i] = 2 * (rmult * val[i - 1] + ksi * der[i - 1]) - der[i - 2];
        derder[i] = 2 * (2 * rmult * der[i-1] + ksi * derder[i-1]) - derder[i-2];
    }
}

RBasis_ChebyCos::RBasis_ChebyCos(double _min_val, double _max_val, int _size, double _scaling) : AnyBasis(_min_val, _max_val, _size, _scaling)
{
    setup();
}

void RBasis_ChebyCos::setup()
{
    mult = 2.0 / (max_val - min_val);
    su = max_val - min_val;
    pu = max_val + min_val;
    irpi = M_PI / max_val;
    name = "RBChebyCos";
}

void RBasis_ChebyCos::calc_der(double value, std::vector<double> &val, std::vector<double> &der)
{
    double ksi = (2 * value - pu) / su;
    double rc = 0.5 * (1.0 + cos(value * irpi ));
    double drc = -0.5 * irpi * sin(value * irpi);

    val[0] = scaling * (1 * rc);
    der[0] = scaling * (0 * rc + drc);
    val[1] = scaling * (ksi * rc);
    der[1] = scaling * (mult * rc + ksi * drc);
    for (int i = 2; i < basis_size; i++)
    {
        val[i] = 2 * ksi * val[i - 1] - val[i - 2];
        der[i] = 2 * (mult * val[i - 1] + ksi * der[i - 1]) - der[i - 2];
    }
}

void RBasis_ChebyCos::calc_derder(double value, std::vector<double> &val, std::vector<double> &der, std::vector<double> &derder) {
    double ksi = (2 * value - pu) / su;
    double rc = 0.5 * (1.0 + cos(value * irpi ));
    double drc = -0.5 * irpi * sin(value * irpi);
    double ddrc = -0.5 * irpi * irpi * cos(value * irpi);

    val[0] = scaling * (1 * rc);
    der[0] = scaling * (0 * rc + drc);
    derder[0] = scaling * ddrc;

    val[1] = scaling * (ksi * rc);
    der[1] = scaling * (mult * rc + ksi * drc);
    derder[1] = scaling * (2.0 * mult * drc + ksi * ddrc);
    for (int i = 2; i < basis_size; i++)
    {
        val[i] = 2 * ksi * val[i - 1] - val[i - 2];
        der[i] = 2 * (mult * val[i - 1] + ksi * der[i - 1]) - der[i - 2];
        derder[i] = 2 * (2.0 * mult * der[i-1] + ksi * derder[i-1]) - derder[i-2];
    }
}

void RBasis_ChebyCos::calc_val(double value, std::vector<double> &val)
{
    double ksi = (2 * value - pu) / su;
    double rc = 0.5 * (1.0 + cos(value * irpi ));

    val[0] = scaling * (1 * rc);
    val[1] = scaling * (ksi * rc);
    for (int i = 2; i < basis_size; i++)
    {
        val[i] = 2 * ksi * val[i - 1] - val[i - 2];
    }
}

RBasis_PolyB::RBasis_PolyB(double _min_val, double _max_val, int _size, double _scaling) : AnyBasis(_min_val, _max_val, _size, _scaling)
{
    setup();
}

void RBasis_PolyB::calc_val(double value, std::vector<double> &val)
{
    double r = max_val - value;
    std::vector<double> phival(basis_size);
    for (int i = 0; i < basis_size; ++i)
    {
        phival[i] = scaling * invNalpha[i] * pow(r, i + 2);
    }
    for (int i = 0; i < basis_size; ++i)
    {
        val[i] = 0;
        for (int j = 0; j < basis_size; ++j)
        {
            val[i] += coeffs(i, j) * phival[j];
        }
        val[i] *= r;
    }
}

void RBasis_PolyB::calc_der(double value, std::vector<double> &val, std::vector<double> &der)
{
    double r = max_val - value;
    // double fc = r;
    // double dfc = 1;
    rpows[0] = scaling * r * r;
    for (int i = 1; i < basis_size + 1; ++i) {
        rpows[i] = rpows[i-1] * r;
    }
    for (int i = 0; i < basis_size; ++i)
    {
        phivals[i] = invNalpha[i] * rpows[i+1];
        dphivals[i] = -dinv[i] * rpows[i];
        // ddphivals[i] = ddinv[i] * rpows[i];
    }
    for (int i = 0; i < basis_size; ++i)
    {
        val[i] = 0;
        der[i] = 0;
        for (int j = 0; j < basis_size; ++j)
        {
            val[i] += coeffs(i, j) * phivals[j];
            der[i] += coeffs(i, j) * dphivals[j];
        }
    }
}

void RBasis_PolyB::calc_derder(double value, std::vector<double> &val, std::vector<double> &der, std::vector<double> &derder)
{
    // The Hessian stage of FC3 OMP shares this basis between threads.
    std::vector<double> rpows(basis_size + 2), phivals(basis_size),
                        dphivals(basis_size), ddphivals(basis_size);
    double r = max_val - value;
    rpows[0] = scaling * r;
    for (int i = 1; i < basis_size + 2; ++i) {
        rpows[i] = rpows[i-1] * r;
    }
    for (int i = 0; i < basis_size; ++i)
    {
        phivals[i] = invNalpha[i] * rpows[i+2];
        dphivals[i] = -dinv[i] * rpows[i+1];
        ddphivals[i] = ddinv[i] * rpows[i];
    }
    for (int i = 0; i < basis_size; ++i)
    {
        val[i] = 0;
        der[i] = 0;
        derder[i] = 0;
        for (int j = 0; j < basis_size; ++j)
        {
            val[i] += coeffs(i, j) * phivals[j];
            der[i] += coeffs(i, j) * dphivals[j];
            derder[i] += coeffs(i, j) * ddphivals[j];
        }
    }
}



void RBasis_PolyB::setup()
{
    invNalpha.resize(basis_size);
    dinv.resize(basis_size);
    ddinv.resize(basis_size);
    phivals.resize(basis_size);
    dphivals.resize(basis_size);
    ddphivals.resize(basis_size);
    rpows.resize(basis_size + 2);

    for (int i = 0; i < basis_size; ++i)
    {
        int bpower = i+1;
        invNalpha[i] = sqrt((2.0 * bpower + 5.0) / pow(max_val - min_val, 2 * bpower + 5));
        dinv[i] = invNalpha[i] * (bpower + 2);
        ddinv[i] = invNalpha[i] * (bpower + 2) * (bpower + 1);
    }
    Eigen::MatrixXd Sab(basis_size, basis_size);
    for (int i = 0; i < basis_size; ++i)
    {
        int ipower = i + 1;
        for (int j = i; j < basis_size; ++j)
        {
            int jpower = j + 1;
            Sab(i, j) = sqrt((2.0 * ipower + 5.0) * (2.0 * jpower + 5.0)) / (ipower + jpower + 5.0);
            Sab(j, i) = Sab(i, j);
        }
    }

    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigensolver(Sab, Eigen::ComputeEigenvectors);
    Eigen::MatrixXd matv = eigensolver.eigenvectors();
    Eigen::MatrixXd ww = Eigen::MatrixXd::Identity(basis_size, basis_size);
    auto w = eigensolver.eigenvalues();
    for (int i = 0; i < basis_size; ++i)
    {
        ww(i, i) = 1.0 / sqrt(w(i));
    }
    coeffs = matv * ww * matv.transpose();

    name = "RBPolyB";
}

void RBasis_PolyBCos::setup()
{
    invNalpha.resize(basis_size);
    dinv.resize(basis_size);
    ddinv.resize(basis_size);
    phivals.resize(basis_size);
    dphivals.resize(basis_size);
    ddphivals.resize(basis_size);
    rpows.resize(basis_size + 2);
    irpi = M_PI / max_val;

    for (int i = 0; i < basis_size; ++i)
    {
        int bpower = i+1;
        invNalpha[i] = sqrt((2.0 * bpower + 5.0) / pow(max_val - min_val, 2 * bpower + 5));
        dinv[i] = invNalpha[i] * (bpower + 2);
        ddinv[i] = invNalpha[i] * (bpower + 2) * (bpower + 1);
    }
    Eigen::MatrixXd Sab(basis_size, basis_size);
    for (int i = 0; i < basis_size; ++i)
    {
        int ipower = i + 1;
        for (int j = i; j < basis_size; ++j)
        {
            int jpower = j + 1;
            Sab(i, j) = sqrt((2.0 * ipower + 5.0) * (2.0 * jpower + 5.0)) / (ipower + jpower + 5.0);
            Sab(j, i) = Sab(i, j);
        }
    }

    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigensolver(Sab, Eigen::ComputeEigenvectors);
    Eigen::MatrixXd matv = eigensolver.eigenvectors();
    Eigen::MatrixXd ww = Eigen::MatrixXd::Identity(basis_size, basis_size);
    auto w = eigensolver.eigenvalues();
    for (int i = 0; i < basis_size; ++i)
    {
        ww(i, i) = 1.0 / sqrt(w(i));
    }
    coeffs = matv * ww * matv.transpose();

    name = "RBPolyBCos";
}

RBasis_PolyBCos::RBasis_PolyBCos(double _min_val, double _max_val, int _size, double _scaling)
    : AnyBasis(_min_val, _max_val, _size, _scaling)
{
    setup();
}

void RBasis_PolyBCos::calc_val(double value, std::vector<double> &val)
{
    double fc = 0.5 * (1.0 + cos(value * irpi ));
    double r = max_val - value;
    std::vector<double> phival(basis_size);
    for (int i = 0; i < basis_size; ++i)
    {
        phival[i] = scaling * invNalpha[i] * pow(r, i + 2) * fc;
    }
    for (int i = 0; i < basis_size; ++i)
    {
        val[i] = 0;
        for (int j = 0; j < basis_size; ++j)
        {
            val[i] += coeffs(i, j) * phival[j];
        }
    }
}

void RBasis_PolyBCos::calc_der(double value, std::vector<double> &val, std::vector<double> &der)
{
    double r = max_val - value;
    double fc = 0.5 * (1.0 + cos(value * irpi ));
    double dfc = -0.5 * irpi * sin(value * irpi);
    rpows[0] = scaling * r;
    for (int i = 1; i < basis_size + 1; ++i) {
        rpows[i] = rpows[i-1] * r;
    }
    for (int i = 0; i < basis_size; ++i)
    {
        phivals[i] = invNalpha[i] * rpows[i+1];
        dphivals[i] = -(i + 2) * invNalpha[i] * rpows[i];
    }
    for (int i = 0; i < basis_size; ++i)
    {
        val[i] = 0;
        der[i] = 0;
        for (int j = 0; j < basis_size; ++j)
        {
            val[i] += coeffs(i, j) * phivals[j];
            der[i] += coeffs(i, j) * dphivals[j];
        }
        der[i] = der[i] * fc + val[i] * dfc;
        val[i] *= fc;
    }
}

void RBasis_PolyBCos::calc_derder(double value, std::vector<double> &val, std::vector<double> &der, std::vector<double> &derder)
{
    // The Hessian stage of FC3 OMP shares this basis between threads.
    std::vector<double> rpows(basis_size + 2), phivals(basis_size),
                        dphivals(basis_size), ddphivals(basis_size);
    double r = max_val - value;
    double fc = 0.5 * (1.0 + cos(value * irpi ));
    double dfc = -0.5 * irpi * sin(value * irpi);
    double ddfc = -0.5 * irpi * irpi * cos(value * irpi);
    rpows[0] = scaling;
    for (int i = 1; i < basis_size + 2; ++i) {
        rpows[i] = rpows[i-1] * r;
    }
    for (int i = 0; i < basis_size; ++i)
    {
        phivals[i] = invNalpha[i] * rpows[i+2];
        dphivals[i] = -(i + 2) * invNalpha[i] * rpows[i+1];
        ddphivals[i] = (i + 2) * (i + 1) * invNalpha[i] * rpows[i];
    }
    for (int i = 0; i < basis_size; ++i)
    {
        val[i] = 0;
        der[i] = 0;
        derder[i] = 0;
        for (int j = 0; j < basis_size; ++j)
        {
            val[i] += coeffs(i, j) * phivals[j];
            der[i] += coeffs(i, j) * dphivals[j];
            derder[i] += coeffs(i, j) * ddphivals[j];
        }
        derder[i] = derder[i] * fc + 2.0 * der[i] * dfc + val[i] * ddfc;
        der[i] = der[i] * fc + val[i] * dfc;
        val[i] *= fc;
    }
}



void RBasis_ChebyshevPoly4::setup()
{
    mult = 2.0 / (max_val - min_val);
    su = max_val - min_val;
    pu = max_val + min_val;
    dxdr = 1.0 / su;
    name = "RBChebyshevPoly4";
}

void RBasis_ChebyshevPoly4::LoadFromStream(std::istream& is)
{
    AnyBasis::LoadFromStream(is);
    std::string tmpstr;
    is >> tmpstr;
    if (tmpstr != "rb_b") {
        throw std::runtime_error("Error in loading basis: " + tmpstr);
    }
    is.ignore(2);
    is >> b;
    if (is.fail()) {
        throw std::runtime_error("Error in loading basis: " + tmpstr);
    }
}

void RBasis_ChebyshevPoly4::SaveToStream(std::ostream& os)
{
    AnyBasis::SaveToStream(os);
    os << "\trb_b = " << b << '\n';
}

// Poly4
RBasis_ChebyshevPoly4::RBasis_ChebyshevPoly4(double _min_val, double _max_val, int _size, double _b, double _scaling) : AnyBasis(_min_val, _max_val, _size, _scaling) {
    b = _b;
    setup();
}

void RBasis_ChebyshevPoly4::calc_val(double value, std::vector<double> &val) {
    if (value < min_val || value > max_val) {
        val.assign(basis_size, 0);
        return;
    }
    double ksi = (2 * value - pu) / su;
    double x = (value - min_val) / su;
    double vx = 1.0 - x;
    // double xx = x * x;
    double rc = vx * vx * vx * (1 + (3 + b) * x);

    val[0] = scaling * rc;
    if (basis_size == 1) return;
    val[1] = scaling * (ksi * rc);
    for (int i = 2; i < basis_size; i++)
    {
        val[i] = 2 * ksi * val[i - 1] - val[i - 2];
    }
}

void RBasis_ChebyshevPoly4::calc_der(double value, std::vector<double> &val, std::vector<double> &der) {
    if (value < min_val || value > max_val) {
        val.assign(basis_size, 0);
        der.assign(basis_size, 0);
        return;
    }
    double ksi = (2 * value - pu) / su;
    double x = (value - min_val) / su;
    // double d2xdr2 = 0.0;
    double vx = 1.0 - x;
    double vxx = vx * vx;
    double vxxx = vxx * vx;
    double rc = vxxx * (1 + (3 + b) * x);
    double drc = vxx * (b*(1. - 4.0*x) - 12.0 * x) * dxdr;

    val[0] = scaling * rc;
    der[0] = scaling * drc;
    if (basis_size == 1) return;
    val[1] = scaling * (ksi * rc);
    der[1] = scaling * (mult * rc + ksi * drc);
    for (int i = 2; i < basis_size; i++)
    {
        val[i] = 2 * ksi * val[i - 1] - val[i - 2];
        der[i] = 2 * (mult * val[i - 1] + ksi * der[i - 1]) - der[i - 2];
    }
}

void RBasis_ChebyshevPoly4::calc_derder(double value, std::vector<double> &val, std::vector<double> &der, std::vector<double> &derder) {
    if (value < min_val || value > max_val) {
        val.assign(basis_size, 0);
        der.assign(basis_size, 0);
        derder.assign(basis_size, 0);
        return;
    }
    double ksi = (2 * value - pu) / su;
    double x = (value - min_val) / su;
    // double d2xdr2 = 0.0;
    double vx = 1.0 - x;
    double vxx = vx * vx;
    double vxxx = vxx * vx;
    double rc = vxxx * (1 + (3 + b) * x);
    double drc = vxx * (b*(1. - 4.0*x) - 12.0 * x) * dxdr;
    double ddrc = 6.0 * vx * (-2.0 + 6.0 * x + b * (2.0 * x - 1.0)) * dxdr * dxdr;

    val[0] = scaling * rc;
    der[0] = scaling * drc;
    derder[0] = scaling * ddrc;
    if (basis_size == 1) return;
    val[1] = scaling * (ksi * rc);
    der[1] = scaling * (mult * rc + ksi * drc);
    derder[1] = scaling * (2.0 * mult * drc + ksi * ddrc);
    for (int i = 2; i < basis_size; i++)
    {
        val[i] = 2 * ksi * val[i - 1] - val[i - 2];
        der[i] = 2 * (mult * val[i - 1] + ksi * der[i - 1]) - der[i - 2];
        derder[i] = 2 * (2 * mult * der[i-1] + ksi * derder[i-1]) - derder[i-2];
    }
}

void RBasis_ChebyshevPoly5::setup()
{
    mult = 2.0 / (max_val - min_val);
    su = max_val - min_val;
    pu = max_val + min_val;
    dxdr = 1.0 / su;
    name = "RBChebyshevPoly5";
}

void RBasis_ChebyshevPoly5::LoadFromStream(std::istream& is)
{
    AnyBasis::LoadFromStream(is);
    std::string tmpstr;
    is >> tmpstr;
    if (tmpstr != "rb_b") {
        throw std::runtime_error("Error in loading basis: " + tmpstr);
    }
    is.ignore(2);
    is >> b;
    if (is.fail()) {
        throw std::runtime_error("Error in loading basis: " + tmpstr);
    }
}

void RBasis_ChebyshevPoly5::SaveToStream(std::ostream& os)
{
    AnyBasis::SaveToStream(os);
    os << "\trb_b = " << b << '\n';
}

// Poly5
RBasis_ChebyshevPoly5::RBasis_ChebyshevPoly5(double _min_val, double _max_val, int _size, double _b, double _scaling) : AnyBasis(_min_val, _max_val, _size, _scaling) {
    b = _b;
    setup();
}

void RBasis_ChebyshevPoly5::calc_val(double value, std::vector<double> &val) {
    if (value < min_val || value > max_val) {
        val.assign(basis_size, 0);
        return;
    }
    double ksi = (2 * value - pu) / su;
    double x = (value - min_val) / su;
    double vx = 1.0 - x;
    double vxx = vx * vx;
    double vxxxx = vxx * vxx;
    double rc = vxxxx * (1 + (4 + b) * x);

    val[0] = scaling * rc;
    if (basis_size == 1) return;
    val[1] = scaling * (ksi * rc);
    for (int i = 2; i < basis_size; i++)
    {
        val[i] = 2 * ksi * val[i - 1] - val[i - 2];
    }
}

void RBasis_ChebyshevPoly5::calc_der(double value, std::vector<double> &val, std::vector<double> &der) {
    if (value < min_val || value > max_val) {
        val.assign(basis_size, 0);
        der.assign(basis_size, 0);
        return;
    }
    double ksi = (2 * value - pu) / su;
    double x = (value - min_val) / su;
    // double d2xdr2 = 0.0;
    double vx = 1.0 - x;
    double vxx = vx * vx;
    double vxxx = vxx * vx;
    double vxxxx = vxx * vxx;
    double rc = vxxxx * (1 + (4 + b) * x);
    double drc = -vxxx * (20 * x + b * (5*x - 1)) * dxdr;

    val[0] = scaling * rc;
    der[0] = scaling * drc;
    if (basis_size == 1) return;
    val[1] = scaling * (ksi * rc);
    der[1] = scaling * (mult * rc + ksi * drc);
    for (int i = 2; i < basis_size; i++)
    {
        val[i] = 2 * ksi * val[i - 1] - val[i - 2];
        der[i] = 2 * (mult * val[i - 1] + ksi * der[i - 1]) - der[i - 2];
    }
}

void RBasis_ChebyshevPoly5::calc_derder(double value, std::vector<double> &val, std::vector<double> &der, std::vector<double> &derder) {
    if (value < min_val || value > max_val) {
        val.assign(basis_size, 0);
        der.assign(basis_size, 0);
        derder.assign(basis_size, 0);
        return;
    }
    double ksi = (2 * value - pu) / su;
    double x = (value - min_val) / su;
    // double d2xdr2 = 0.0;
    double vx = 1.0 - x;
    double vxx = vx * vx;
    double vxxx = vxx * vx;
    double vxxxx = vxx * vxx;
    double rc = vxxxx * (1 + (4 + b) * x);
    double drc = -vxxx * (20 * x + b * (5*x - 1)) * dxdr;
    double ddrc = 4.0 * vxx * (-5.0 + 20.0 * x + b * (5.0 * x - 2.0)) * dxdr * dxdr;

    val[0] = scaling * rc;
    der[0] = scaling * drc;
    derder[0] = scaling * ddrc;
    if (basis_size == 1) return;
    val[1] = scaling * (ksi * rc);
    der[1] = scaling * (mult * rc + ksi * drc);
    derder[1] = scaling * (2.0 * mult * drc + ksi * ddrc);
    for (int i = 2; i < basis_size; i++)
    {
        val[i] = 2 * ksi * val[i - 1] - val[i - 2];
        der[i] = 2 * (mult * val[i - 1] + ksi * der[i - 1]) - der[i - 2];
        derder[i] = 2 * (2 * mult * der[i-1] + ksi * derder[i-1]) - derder[i-2];
    }
}

const char *basis_function_type_name(BasisFunctionType type) noexcept
{
    switch (type) {
    case BasisFunctionType::Chebyshev: return "chebyshev";
    case BasisFunctionType::OneMinusChebyshev:
        return "one_minus_chebyshev";
    }
    return "unknown";
}

BasisFunctionType basis_function_type_from_name(const std::string &name)
{
    if (name == "chebyshev") return BasisFunctionType::Chebyshev;
    if (name == "one_minus_chebyshev")
        return BasisFunctionType::OneMinusChebyshev;
    throw std::runtime_error(
        "Unknown configurable radial basis function type: " + name);
}

RBasis_ChebyshevConfigurable::RBasis_ChebyshevConfigurable(
    double _min_val, double _max_val, int _size,
    std::unique_ptr<Cutoff::AnyCutoff> cutoff,
    std::unique_ptr<XTransForm::AnyTrans> shift, double _scaling,
    BasisFunctionType basis_function_type)
    : AnyBasis(_min_val, _max_val, _size, _scaling),
      cutoff_(std::move(cutoff)), shift_(std::move(shift)),
      basis_function_type_(basis_function_type)
{
    setup();
}

void RBasis_ChebyshevConfigurable::setup()
{
    if (!std::isfinite(min_val) || !std::isfinite(max_val) ||
        !(min_val < max_val))
        throw std::invalid_argument(
            "RBChebyshevConfigurable requires finite min_dist < max_dist");
    if (basis_size <= 0)
        throw std::invalid_argument(
            "RBChebyshevConfigurable requires radial_basis_size > 0");
    if (!std::isfinite(scaling))
        throw std::invalid_argument(
            "RBChebyshevConfigurable requires finite scaling");
    if (!cutoff_ || !shift_)
        throw std::invalid_argument(
            "RBChebyshevConfigurable requires cutoff and shift functions");
    if (cutoff_->min_value() != min_val ||
        cutoff_->max_value() != max_val ||
        shift_->min_value() != min_val ||
        shift_->max_value() != max_val)
        throw std::invalid_argument(
            "RBChebyshevConfigurable component intervals must match the "
            "basis interval");
    name = "RBChebyshevConfigurable";
}

void RBasis_ChebyshevConfigurable::LoadFromStream(std::istream &is)
{
    AnyBasis::LoadFromStream(is);
    basis_function_type_ = BasisFunctionType::Chebyshev;

    is >> std::ws;
    if (is.peek() == 'b') {
        std::string key;
        std::string equals;
        std::string value;
        if (!(is >> key >> equals >> value) ||
            key != "basis_function_type" || equals != "=")
            throw std::runtime_error(
                "Error loading configurable radial basis: expected "
                "basis_function_type = <value>");
        basis_function_type_ = basis_function_type_from_name(value);
    }

    cutoff_ = Cutoff::load_cutoff(is, min_val, max_val);
    shift_ = XTransForm::load_shift(is, min_val, max_val);
}

void RBasis_ChebyshevConfigurable::SaveToStream(std::ostream &os)
{
    AnyBasis::SaveToStream(os);
    os << "\tbasis_function_type = "
       << basis_function_type_name(basis_function_type_) << '\n';
    cutoff_->SaveToStream(os);
    shift_->SaveToStream(os);
}

bool RBasis_ChebyshevConfigurable::outside_interval(
    double value) const noexcept
{
    return value < min_val || value > max_val;
}

void RBasis_ChebyshevConfigurable::zero_outputs(
    std::vector<double> &val, std::vector<double> *der,
    std::vector<double> *derder) const
{
    val.assign(basis_size, 0.0);
    if (der) der->assign(basis_size, 0.0);
    if (derder) derder->assign(basis_size, 0.0);
}

void RBasis_ChebyshevConfigurable::calculate(
    double value, std::vector<double> &val, std::vector<double> *der,
    std::vector<double> *derder)
{
    if (outside_interval(value)) {
        zero_outputs(val, der, derder);
        return;
    }

    const int order = derder ? 2 : (der ? 1 : 0);
    double x = 0.0, dx = 0.0, ddx = 0.0;
    double fc = 0.0, dfc = 0.0, ddfc = 0.0;
    if (order == 0) {
        x = shift_->compute(value);
        fc = cutoff_->compute(value);
    } else if (order == 1) {
        shift_->compute_der(value, x, dx);
        cutoff_->compute_der(value, fc, dfc);
    } else {
        shift_->compute_derder(value, x, dx, ddx);
        cutoff_->compute_derder(value, fc, dfc, ddfc);
    }

    const bool flipped =
        basis_function_type_ == BasisFunctionType::OneMinusChebyshev;
    const int polynomial_count = basis_size + (flipped ? 1 : 0);
    val.assign(polynomial_count, 0.0);
    if (der) der->assign(polynomial_count, 0.0);
    if (derder) derder->assign(polynomial_count, 0.0);

    val[0] = 1.0;
    if (polynomial_count > 1) {
        val[1] = x;
        if (der) (*der)[1] = dx;
        if (derder) (*derder)[1] = ddx;
    }
    for (int i = 2; i < polynomial_count; ++i) {
        val[i] = 2.0 * x * val[i - 1] - val[i - 2];
        if (der)
            (*der)[i] =
                2.0 * (dx * val[i - 1] + x * (*der)[i - 1]) -
                (*der)[i - 2];
        if (derder)
            (*derder)[i] =
                2.0 * (ddx * val[i - 1] +
                       2.0 * dx * (*der)[i - 1] +
                       x * (*derder)[i - 1]) -
                (*derder)[i - 2];
    }

    for (int i = 0; i < basis_size; ++i) {
        const int polynomial_index = i + (flipped ? 1 : 0);
        const double raw =
            flipped ? 1.0 - val[polynomial_index]
                    : val[polynomial_index];
        const double raw_der =
            der
                ? (flipped ? -(*der)[polynomial_index]
                           : (*der)[polynomial_index])
                : 0.0;
        const double raw_derder =
            derder
                ? (flipped ? -(*derder)[polynomial_index]
                           : (*derder)[polynomial_index])
                : 0.0;

        val[i] = scaling * raw * fc;
        if (der)
            (*der)[i] = scaling * (raw_der * fc + raw * dfc);
        if (derder)
            (*derder)[i] =
                scaling * (raw_derder * fc +
                           2.0 * raw_der * dfc + raw * ddfc);
    }
    val.resize(basis_size);
    if (der) der->resize(basis_size);
    if (derder) derder->resize(basis_size);
}

void RBasis_ChebyshevConfigurable::calc_val(
    double value, std::vector<double> &val)
{
    calculate(value, val, nullptr, nullptr);
}

void RBasis_ChebyshevConfigurable::calc_der(
    double value, std::vector<double> &val, std::vector<double> &der)
{
    calculate(value, val, &der, nullptr);
}

void RBasis_ChebyshevConfigurable::calc_derder(
    double value, std::vector<double> &val, std::vector<double> &der,
    std::vector<double> &derder)
{
    calculate(value, val, &der, &derder);
}
