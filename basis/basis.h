// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#pragma once

#include <iosfwd>
#include <ostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <Eigen/Dense>

#include "cutoff.h"
#include "xtrans.h"
#include "utils/gputils.h"

enum BasisType {
    RB_ChebyshevConfigurable,
    RB_Chebyshev,
    RB_ChebyCos,
    RB_PolyB,
    RB_PolyBCos,
    RB_ChebyshevPoly4,
    RB_ChebyshevPoly5,
    RB_Unknown,
};

BasisType basis_name_to_code(const std::string &bname);

class AnyBasis : public ConnectSettings {
public:
    int basis_size;
    double min_val;
    double max_val;
    double scaling = 1.0;
    std::vector<double> vals;
    std::vector<double> ders;
    std::vector<double> derders;
    std::string name;

    AnyBasis(double min_val, double max_val, int size,
             double scaling = 1.0);
    AnyBasis() = default;
    virtual ~AnyBasis() = default;

    virtual void LoadFromStream(std::istream &is);
    virtual void SaveToStream(std::ostream &os)
    {
        os << "radial_basis_type = " << name << "\n";
        os << "\tmin_dist = " << min_val << "\n";
        os << "\tmax_dist = " << max_val << "\n";
        os << "\tradial_basis_size = " << basis_size << "\n";
    }

    // Internal-buffer interface used by existing potentials.
    virtual void calc_val(double value) { calc_val(value, vals); }
    virtual void calc_der(double value) { calc_der(value, vals, ders); }
    virtual void calc_derder(double value)
    {
        calc_derder(value, vals, ders, derders);
    }
    virtual void calc_val(double value, std::vector<double> &val) = 0;
    virtual void calc_der(double value, std::vector<double> &val,
                          std::vector<double> &der) = 0;
    virtual void calc_derder(double value, std::vector<double> &val,
                             std::vector<double> &der,
                             std::vector<double> &derder)
    {
    }
    virtual void calc_val(double value, double, double,
                          std::vector<double> &val)
    {
        calc_val(value, val);
    }
    virtual void calc_der(double value, double, double,
                          std::vector<double> &val,
                          std::vector<double> &der)
    {
        calc_der(value, val, der);
    }
    virtual void calc_derder(double value, double, double,
                             std::vector<double> &val,
                             std::vector<double> &der,
                             std::vector<double> &derder)
    {
        calc_derder(value, val, der, derder);
    }
};

AnyBasis *create_basis_by_code(const BasisType &type, std::istream &is,
                               double scaling = 1.0);
AnyBasis *create_basis_by_name(const std::string &type, std::istream &is,
                               double scaling = 1.0);

class RBasis_Chebyshev : public AnyBasis {
    double mult;
    double su;
    double pu;

public:
    void setup();
    RBasis_Chebyshev(double min_val, double max_val, int size,
                     double scaling = 1.0);
    RBasis_Chebyshev(std::istream &is, double scaling = 1.0)
    {
        this->scaling = scaling;
        LoadFromStream(is);
        setup();
    }

    void calc_val(double value, std::vector<double> &val) override;
    void calc_der(double value, std::vector<double> &val,
                  std::vector<double> &der) override;
    void calc_derder(double value, std::vector<double> &val,
                     std::vector<double> &der,
                     std::vector<double> &derder) override;
    void calc_val(double value, double rmin, double rmax,
                  std::vector<double> &val) override;
    void calc_der(double value, double rmin, double rmax,
                  std::vector<double> &val,
                  std::vector<double> &der) override;
    void calc_derder(double value, double rmin, double rmax,
                     std::vector<double> &val,
                     std::vector<double> &der,
                     std::vector<double> &derder) override;
};

/** Legacy cosine basis retained for compatibility with its r/max formula. */
class RBasis_ChebyCos : public AnyBasis {
    double mult;
    double su;
    double pu;
    double irpi;

public:
    void setup();
    RBasis_ChebyCos(double min_val, double max_val, int size,
                    double scaling = 1.0);
    RBasis_ChebyCos(std::istream &is, double scaling = 1.0)
    {
        this->scaling = scaling;
        LoadFromStream(is);
        setup();
    }

    void calc_val(double value, std::vector<double> &val) override;
    void calc_der(double value, std::vector<double> &val,
                  std::vector<double> &der) override;
    void calc_derder(double value, std::vector<double> &val,
                     std::vector<double> &der,
                     std::vector<double> &derder) override;
};

// Evaluates the polynomial continuation without interval clipping. At max_val,
// the third derivative is the interior value; zero extension is only C2 there.
class RBasis_PolyB : public AnyBasis {
public:
    Eigen::MatrixXd coeffs;
    std::vector<double> invNalpha;
    std::vector<double> dinv;
    std::vector<double> ddinv;
    std::vector<double> phivals;
    std::vector<double> dphivals;
    std::vector<double> ddphivals;
    std::vector<double> rpows;

    void setup();
    RBasis_PolyB(double min_val, double max_val, int size,
                 double scaling = 1.0);
    RBasis_PolyB(std::istream &is, double scaling = 1.0)
    {
        this->scaling = scaling;
        LoadFromStream(is);
        setup();
    }

    void calc_val(double value, std::vector<double> &val) override;
    void calc_der(double value, std::vector<double> &val,
                  std::vector<double> &der) override;
    void calc_derder(double value, std::vector<double> &val,
                     std::vector<double> &der,
                     std::vector<double> &derder) override;

};

// Like PolyB, evaluation does not clip the interval. The cosine factor makes
// this basis and its first three derivatives vanish at max_val.
class RBasis_PolyBCos : public AnyBasis {
public:
    Eigen::MatrixXd coeffs;
    std::vector<double> invNalpha;
    std::vector<double> dinv;
    std::vector<double> ddinv;
    std::vector<double> phivals;
    std::vector<double> dphivals;
    std::vector<double> ddphivals;
    std::vector<double> rpows;
    double irpi;

    void setup();
    RBasis_PolyBCos(double min_val, double max_val, int size,
                    double scaling = 1.0);
    RBasis_PolyBCos(std::istream &is, double scaling = 1.0)
    {
        this->scaling = scaling;
        LoadFromStream(is);
        setup();
    }

    void calc_val(double value, std::vector<double> &val) override;
    void calc_der(double value, std::vector<double> &val,
                  std::vector<double> &der) override;
    void calc_derder(double value, std::vector<double> &val,
                     std::vector<double> &der,
                     std::vector<double> &derder) override;

};

class RBasis_ChebyshevPoly4 : public AnyBasis {
public:
    double b = -4.0 / 3.0;

private:
    double mult;
    double su;
    double pu;
    double dxdr;

public:
    void setup();
    void LoadFromStream(std::istream &is) override;
    void SaveToStream(std::ostream &os) override;
    RBasis_ChebyshevPoly4(double min_val, double max_val, int size,
                          double b = -1.3333333,
                          double scaling = 1.0);
    RBasis_ChebyshevPoly4(std::istream &is, double scaling = 1.0)
    {
        this->scaling = scaling;
        LoadFromStream(is);
        setup();
    }

    void calc_val(double value, std::vector<double> &val) override;
    void calc_der(double value, std::vector<double> &val,
                  std::vector<double> &der) override;
    void calc_derder(double value, std::vector<double> &val,
                     std::vector<double> &der,
                     std::vector<double> &derder) override;
};

class RBasis_ChebyshevPoly5 : public AnyBasis {
public:
    double b = -1.0 / 3.0;

private:
    double mult;
    double su;
    double pu;
    double dxdr;

public:
    void setup();
    void LoadFromStream(std::istream &is) override;
    void SaveToStream(std::ostream &os) override;
    RBasis_ChebyshevPoly5(double min_val, double max_val, int size,
                          double b = -1.3333333,
                          double scaling = 1.0);
    RBasis_ChebyshevPoly5(std::istream &is, double scaling = 1.0)
    {
        this->scaling = scaling;
        LoadFromStream(is);
        setup();
    }

    void calc_val(double value, std::vector<double> &val) override;
    void calc_der(double value, std::vector<double> &val,
                  std::vector<double> &der) override;
    void calc_derder(double value, std::vector<double> &val,
                     std::vector<double> &der,
                     std::vector<double> &derder) override;
};

enum class BasisFunctionType {
    Chebyshev,
    OneMinusChebyshev,
};

const char *basis_function_type_name(BasisFunctionType type) noexcept;
BasisFunctionType basis_function_type_from_name(const std::string &name);

class RBasis_ChebyshevConfigurable : public AnyBasis {
    std::unique_ptr<Cutoff::AnyCutoff> cutoff_;
    std::unique_ptr<XTransForm::AnyTrans> shift_;
    BasisFunctionType basis_function_type_ = BasisFunctionType::Chebyshev;

    bool outside_interval(double value) const noexcept;
    void zero_outputs(std::vector<double> &val,
                      std::vector<double> *der = nullptr,
                      std::vector<double> *derder = nullptr) const;
    void calculate(double value, std::vector<double> &val,
                   std::vector<double> *der,
                   std::vector<double> *derder);

public:
    void setup();
    void LoadFromStream(std::istream &is) override;
    void SaveToStream(std::ostream &os) override;
    RBasis_ChebyshevConfigurable(
        double min_val, double max_val, int size,
        std::unique_ptr<Cutoff::AnyCutoff> cutoff,
        std::unique_ptr<XTransForm::AnyTrans> shift,
        double scaling = 1.0,
        BasisFunctionType basis_function_type =
            BasisFunctionType::Chebyshev);
    RBasis_ChebyshevConfigurable(std::istream &is, double scaling = 1.0)
    {
        this->scaling = scaling;
        LoadFromStream(is);
        setup();
    }

    const Cutoff::AnyCutoff &cutoff_function() const noexcept
    {
        return *cutoff_;
    }
    const XTransForm::AnyTrans &shift_function() const noexcept
    {
        return *shift_;
    }
    BasisFunctionType basis_function_type() const noexcept
    {
        return basis_function_type_;
    }

    void calc_val(double value, std::vector<double> &val) override;
    void calc_der(double value, std::vector<double> &val,
                  std::vector<double> &der) override;
    void calc_derder(double value, std::vector<double> &val,
                     std::vector<double> &der,
                     std::vector<double> &derder) override;
};
