// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#ifndef IMRMLP_ANGULAR_FUNCS_H
#define IMRMLP_ANGULAR_FUNCS_H

#include <complex>
#include <unordered_map>
#include <vector>


#define MAX_TWOJ_MAX_CG 64
#define TWOJMAX_CG_SHIFT 6

namespace AngularFuncs {
    using cg_index_type = unsigned int;
    double TriangleCoeffTrans(int j1, int j2, int j3);
    double ClebschGordanTrans(int j1, int j2, int j, int m1, int m2, int m);
    double ClebschGordanTransInter(int j1, int j2, int j, int m1, int m2, int m);
    double ClebschGordanTransCache(int j1, int j2, int j, int m1, int m2, int m);
    void InitCGTransCache(int maxj);
    double ClebschGordan(int twoj1, int twoj2, int twoj, int twom1, int twom2, int twom);
    double ClebschGordanInt(int l1, int l2, int l, int m1, int m2, int m);

    double Wigner3jInt(int l1, int l2, int l3, int m1, int m2, int m3);
    double Wigner3jIntInter(int l1, int l2, int l3, int m1, int m2, int m3);
    double Wigner3jIntCache(int l1, int l2, int l3, int m1, int m2, int m3);

    /**
     * We compute legendre polynomial firstly using following formula:
     *  P_{l+1, m}(x) = (2*l+1)/(l-m+1)*x*P_{l,m}(x) - (l+m)/(l-m+1)*P_{l-1,m}
     * where m = 0, 1, 2, ..., l, -1 <= x <= 1, the ALP is Hobson defined, with
     * Condon-Shortley phase factor. (actually this relation is invariant of phase factor)
     * 
     * To avoid ill-condition, we use a normalized ALP here:
     *  P'_{l, m}(x) = sqrt((2*l+1)*(l-m)!/(2*(l+m)!)) P_{l,m}(x)
     * Since we only care about sph, the form we considered in code is 
     *  P'_{l, m}(x) = sqrt((2*l+1)*(l-m)!/(4*pi*(l+m)!)) P_{l,m}(x)
     * Then we get
     *  P'_{l,m}(x) = a_{l,m} * (x * P'_{l-1,m} + b_{l,m} * P'_{l-2,m})
     * where
     *  a_{l,m} = sqrt((4*l^2 - 1)/(l^2-m^2))
     *  b_{l,m} = -sqrt(((l-1)^2-m^2) / (4*(l-1)^2-1))
     * 
     * For @f$Y_{l, m}(\theta, \phi)@f$ we have
     *  @f$Y_{l,m}(\theta, \phi) = \sqrt{(2l+1)(l-m)!/(4\pi(l+m)!)} \cdot P_{l,m}(\cos\theta) \cdot e^{im\phi}@f$
     *                        @f$= P'_{l,m}(\cos\theta) \cdot e^{im\phi}@f$
     * and if m < 0 we have
     *  @f$Y_{l,m} = (-1)^m \cdot Y^*_{l,|m|}@f$
     * The Condon-Shortley phase factor is contained in ALP.
     *
     * See https://arxiv.org/pdf/1410.1748.pdf
     * and https://docs.abinit.org/theory/spherical_harmonics/
     * and https://en.wikipedia.org/wiki/Spherical_harmonics
     *
     * Real spherical harmonics:
     * @f[
     * y_{l,m} = \begin{cases}
     *   \frac{1}{\sqrt{2}} (Y_{l,m} + Y^*_{l,m})                     & m > 0 \\
     *   Y_{l,0}                                                       & m = 0 \\
     *   \frac{1}{i\sqrt{2}} (Y_{l,|m|} - Y^*_{l,|m|})               & m < 0
     * \end{cases}
     * @f]
     * which in terms of P' is
     * @f[
     * y_{l,m} = \begin{cases}
     *   P'_{l,m}(\cos\theta) \cdot \cos(m\phi)                      & m > 0 \\
     *   P'_{l,0}(\cos\theta) / \sqrt{2}                              & m = 0 \\
     *   P'_{l,|m|}(\cos\theta) \cdot \sin(|m|\phi)                  & m < 0
     * \end{cases}
     * @f]
     * For cos(m*phi) and sin(m*phi), recurrence:
     *  @f$\cos(m\phi) = 2\cos\phi \cdot \cos((m-1)\phi) - \cos((m-2)\phi)@f$
     *  @f$\sin(m\phi) = 2\cos\phi \cdot \sin((m-1)\phi) - \sin((m-2)\phi)@f$
     * 
     * TODO: tabulate ones using Mathematica
     * 
     * **/
    class ComplexSphericalHarmoics {
    public:
        double *P = nullptr;
        double *dP = nullptr;
        double *Y = nullptr;
        double *dY = nullptr;//costheta
        double *dYs = nullptr;//sinphi
        double *dYc = nullptr;//cosphi
        double *Yi = nullptr;
        double *dYis = nullptr;
        double *dYic = nullptr;
        double *dYi = nullptr;
        double *A = nullptr;
        double *B = nullptr;
        double *dYdx = nullptr;
        double *dYdy = nullptr;
        double *dYdz = nullptr;
        double *dYidx = nullptr;
        double *dYidy = nullptr;
        double *dYidz = nullptr;
        int maxL = 6;
        int maxM = 6;

        ComplexSphericalHarmoics() {};

        ComplexSphericalHarmoics(int maxl, int maxm = -1);

        void init(int maxl, int maxm);

        ~ComplexSphericalHarmoics();

        void computeP(const double x);

        void computedP(const double x);

        void computeY(const double theta, const double phi);

        void computeY(const double costheta, const double sinphi, const double cosphi);

        void computedY(const double costheta, const double sinphi, const double cosphi);

        void computeYCar(const double &x, const double &y, const double &z, const double &r);
        void computeYCarSafe(const double &x, const double &y, const double &z, const double &r);

        void computedYCar(const double &x, const double &y, const double &z, const double &r);

        std::complex<double> get(const int &idl, const int &idm) const;

    };
    // Orthonormal tesseral harmonics, Y index l*l+l+m. For m>0 use
    // sqrt(2)*Re(Y_lm), for m<0 sqrt(2)*Im(Y_l|m|), with Condon--Shortley
    // phase. Cartesian calls reject the origin/inconsistent radii and are
    // regular on both polar axes. Value/first-gradient tests cover l<=32.
    class RealSphericalHarmoics {
    public:
        double *P = nullptr;
        double *dP = nullptr;
        double *Y = nullptr;
        double *dYdc = nullptr;
        double *dYds = nullptr;
        double *dYdt = nullptr;
        double *dYdx = nullptr;
        double *dYdy = nullptr;
        double *dYdz = nullptr;
        double *A = nullptr;
        double *B = nullptr;
        int maxL = 6;
        int maxM = 4;

        RealSphericalHarmoics() {};

        RealSphericalHarmoics(int maxl, int maxm = -1);

        void init(int maxl, int maxm = -1);

        ~RealSphericalHarmoics();

        void computeP(const double x);

        void computedP(const double x);

        double get(const int &l, const int &m) const;

        void computeY(const double phi, const double theta);

        void computeY(const double costheta, const double sinphi, const double cosphi);

        void computedY(const double costheta, const double sinphi, const double cosphi);

        void computeYCar(const double &x, const double &y, const double &z, const double &r);
        void computeYCarSafe(const double &x, const double &y, const double &z, const double &r);

        void computedYCar(const double &x, const double &y, const double &z, const double &r);
    };
}

#endif
