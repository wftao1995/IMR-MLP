// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#include "combinatorics.h"
#include "angular_functions.h"
#include "tools/tools.h"
#include <limits>
#include <cstdio>
#include <numeric>
#include <stdexcept>
#include <array>
#include "real_spherical_recurrence.h"

using namespace Combinatorics;
#define COMPLEX_ONE (1.0 + 0.0i)
#define COMPLEX_ZERO (0.0 + 0.0i)
#define COMPLEX_IMG (0.0 + 1.0i)
#define COMPLEX_DIA (1.0 + 1.0i)

using namespace std::literals;
using namespace std::complex_literals;

namespace AngularFuncs {
    std::unordered_map<cg_index_type, double> cgcoeff;
    std::unordered_map<cg_index_type, double> cgcoeffint;
    std::unordered_map<cg_index_type, double> wigner3jm;
    std::unordered_map<cg_index_type, double> wigner3jmint;

    double TriangleCoeffTrans(int j1, int j2, int j3) {
        return factorial((j1 + j2 - j3) / 2) *
               factorial((j1 - j2 + j3) / 2) *
               factorial((-j1 + j2 + j3) / 2) /
               factorial((j1 + j2 + j3) / 2 + 1);
    }

    /**
     * ClebschGordan
     * return 0 if unphysical
     * j1 j2 j
     * m1 m2 m
     * Use j1 <= j2 <= j and m >= 0 as standard form
     * If j1 == j2, then transform into m1 <= m2
     * **/
    double ClebschGordan(int twoj1, int twoj2, int twoj, int twom1, int twom2, int twom) {
        return ClebschGordanTrans(twoj1, twoj2, twoj, (twoj1 + twom1) / 2, (twoj2 + twom2) / 2, (twoj + twom) / 2);
    }

    cg_index_type
    compute_shiftcg_index(cg_index_type j1, cg_index_type j2, cg_index_type j, cg_index_type m1, cg_index_type m2,
                          cg_index_type m) {
        // one can use this compact form:
        // j1%2 j2%2 j1/2 j2/2 j/2 m1 m2 m to save 1 bit
        // however, still lots of vacancies
        return (((((((j1 << TWOJMAX_CG_SHIFT) + j2)
                << TWOJMAX_CG_SHIFT) +
                   j)
                << TWOJMAX_CG_SHIFT) +
                 m1)
                << TWOJMAX_CG_SHIFT) +
               m2;
    }

    double ClebschGordanTrans(int j1, int j2, int j, int m1, int m2, int m) {
        static bool inited = false;
        static int twojmax = 0;
        if (!inited) {
            char file[] = "cgcoeffstrans.dat";
            FILE *fp = fopen(file, "rb");
            int32_t ids[6];
            double res;
            cg_index_type ind;
            int32_t ntotal;
            cg_index_type *indarr;
            if (fp == NULL) {
                twojmax = -1;
            } else {
                fread(&twojmax, 4, 1, fp);
                if (twojmax > MAX_TWOJ_MAX_CG) {
                    // GPERROR("TROJAMX EXCEED!!")
                    ;
                }
                fread(&ntotal, 4, 1, fp);
                indarr = (cg_index_type *) malloc(ntotal * sizeof(cg_index_type));
                for (int i = 0; i < ntotal; ++i) {
                    fread(ids, 4, 6, fp);
                    ind = compute_shiftcg_index(ids[0], ids[1], ids[2], ids[3], ids[4], ids[5]);
                    indarr[i] = ind;
                }
                for (int i = 0; i < ntotal; ++i) {
                    fread(&res, 8, 1, fp);
                    cgcoeff[indarr[i]] = res;
                }
                free(indarr);
            }
            inited = true;
        }
        bool select(true);
        // may be duplate, todo: check this
        // return 0 if unphysical
        select = (2 * m - j == 2 * m1 - j1 + 2 * m2 - j2) &&
                 ((j1 + j2 + j) % 2 == 0) &&
                 (std::fabs(j1 - j2) <= j) && (j <= j1 + j2) &&
                 (m1 >= 0) && (m2 >= 0) && (m >= 0) &&
                 (m1 <= j1) && (m2 <= j2) && (m <= j);
        if (!select)
            return 0;
        int phase = 1;
        double sfac = 1.0;
#ifdef CG_USE_SYMM
        // sort j,
        int p = 1;
        int q = 1;
        while (true)
        {
            if (j1 <= j2 && j2 <= j)
            {
                break;
            }
            if (j1 > j2)
            {
                if ((j1 + j2 - j) % 4 != 0)
                    phase = -phase;
                std::swap(j1, j2);
                std::swap(m1, m2);
            }
            if (j2 > j)
            {
                if ((j1 - m1) % 2 != 0)
                    phase = -phase;
                p *= j + 1;
                q *= j2 + 1;
                std::swap(j2, j);
                std::swap(m2, m);
                m2 = j2 - m2;
            }
        }
        sfac = sqrt(static_cast<double>(p) / static_cast<double>(q));
#endif // CG_USE_SYMM \
    // makes m <= j/2
        if (2 * m > j) {
            if ((j1 + j2 - j) % 4 != 0)
                phase = -phase;
            m1 = j1 - m1;
            m2 = j2 - m2;
            m = j - m;
        }
        sfac *= phase;
// compute index
#ifndef CG_USE_SYMM
        if (j1 <= twojmax && j2 <= twojmax && j <= twojmax)
#else
            if (j <= twojmax)
#endif
        {
            cg_index_type index = compute_shiftcg_index(j1, j2, j, m1, m2, m);
            return sfac * cgcoeff[index];
        } else {
            return sfac * ClebschGordanTransInter(j1, j2, j, m1, m2, m);
        }
    }

    double ClebschGordanTransInter(int j1, int j2, int j, int m1, int m2, int m) {
        int aa2 = 2 * m1 - j1;
        int bb2 = 2 * m2 - j2;
        // m = (aa2 + bb2 + j) / 2
        double sum = 0.0;
        int phase;
        for (int z = __max(0, __max(-(j - j2 + aa2) / 2, -(j - j1 - bb2) / 2));
             z <= __min((j1 + j2 - j) / 2, __min((j1 - aa2) / 2, (j2 + bb2) / 2)); z++) {
            phase = z % 2 ? -1 : 1;
            sum += phase /
                   (factorial(z) *
                    factorial((j1 + j2 - j) / 2 - z) *
                    factorial((j1 - aa2) / 2 - z) *
                    factorial((j2 + bb2) / 2 - z) *
                    factorial((j - j2 + aa2) / 2 + z) *
                    factorial((j - j1 - bb2) / 2 + z));
        }
        int cc2 = 2 * m - j;
        double dcg = TriangleCoeffTrans(j1, j2, j);
        double sfaccg = sqrt(factorial((j1 + aa2) / 2) *
                             factorial((j1 - aa2) / 2) *
                             factorial((j2 + bb2) / 2) *
                             factorial((j2 - bb2) / 2) *
                             factorial((j + cc2) / 2) *
                             factorial((j - cc2) / 2) *
                             (j + 1));
        return sum * dcg * sfaccg;
    }

    // I WON'T CHECK BOUNDARY!!
    double ClebschGordanTransCache(int j1, int j2, int j, int m1, int m2, int m) {
        cg_index_type index = compute_shiftcg_index(j1, j2, j, m1, m2, m);
        if (cgcoeff.count(index) == 1)
            return cgcoeff[index];
        else {
            double cgval = ClebschGordanTrans(j1, j2, j, m1, m2, m);
            cgcoeff[index] = cgval;
            return cgval;
        }
    }

    void InitCGTransCache(int maxj) {
        for (int j1 = 0; j1 <= maxj; ++j1) {
            for (int j2 = 0; j2 <= maxj; ++j2) {
                for (int j3 = 0; j3 <= __min(j1 + j2, maxj); ++j3) {
                    if ((j1 + j2 + j3) % 2 == 1)
                        continue;
                    for (int m1 = 0; m1 <= j1; ++m1) {
                        for (int m2 = 0; m2 <= j2; ++m2) {
                            int m3 = m1 + m2 - (j1 + j2 + j3) / 2;
                            ClebschGordanTransCache(j1, j2, j3, m1, m2, m3);
                        }
                    }
                }
            }
        }
    }

    double ClebschGordanInt(int l1, int l2, int l, int m1, int m2, int m) {
        cg_index_type index = compute_shiftcg_index(l1, l2, l, m1, m2, m);
        if (cgcoeffint.count(index) == 1)
            return cgcoeffint[index];
        else {
            double cgval = ClebschGordan(2 * l1, 2 * l2, 2 * l, 2 * m1, 2 * m2, 2 * m);
            cgcoeffint[index] = cgval;
            return cgval;
        }
    }


    double Wigner3jIntInter(int l1, int l2, int l3, int m1, int m2, int m3) {
        double tfac = sqrt(Combinatorics::factorial(l1 + l2 - l3) *
                           Combinatorics::factorial(l1 - l2 + l3) *
                           Combinatorics::factorial(-l1 + l2 + l3) /
                           Combinatorics::factorial(l1 + l2 + l3 + 1));
        double mfac = sqrt(Combinatorics::factorial(l1 + m1) *
                           Combinatorics::factorial(l1 - m1) *
                           Combinatorics::factorial(l2 + m2) *
                           Combinatorics::factorial(l2 - m2) *
                           Combinatorics::factorial(l3 + m3) *
                           Combinatorics::factorial(l3 - m3));
        int phase;
        double sum = 0;
        for (int k = __max(0, __max(-l3 + l2 - m1, -l3 + l1 + m2));
             k <= __min(l1 + l2 - l3, __min(l1 - m1, l2 + m2)); ++k) {
            phase = k % 2 == 0 ? 1 : -1;
            sum += phase / Combinatorics::factorial(k) /
                   Combinatorics::factorial(l3 - l2 + k + m1) /
                   Combinatorics::factorial(l3 - l1 + k - m2) /
                   Combinatorics::factorial(l1 + l2 - l3 - k) /
                   Combinatorics::factorial(l1 - k - m1) /
                   Combinatorics::factorial(l2 - k + m2);
        }
        int osfre = (l1 - l2 - m3) % 2 == 0 ? 1 : -1;
        return osfre * tfac * mfac * sum;
    }

    double Wigner3jInt(int l1, int l2, int l3, int m1, int m2, int m3) {
        bool select = (abs(m1) <= l1) && (abs(m2) <= l2) && (abs(m3) <= l3) &&
                      (m1 + m2 == -m3) && (abs(l1 - l2) <= l3) && (l3 <= l1 + l2);
        if (!select)
            return 0;
        return Wigner3jIntInter(l1, l2, l3, m1, m2, m3);
    }

    double Wigner3jIntCache(int l1, int l2, int l3, int m1, int m2, int m3) {
        return Wigner3jInt(l1, l2, l3, m1, m2, m3);
    }

#define PVT(l, m) ((m) + ((l) * ((l) + 1)) / 2)
#define YVR(l, m) ((m) + (l) + ((l) * (l)))
#define YVRTRANS(l, m) ((m) + ((l) * (l)))

    ComplexSphericalHarmoics::ComplexSphericalHarmoics(int maxl, int maxm) {
        if (maxm < 0 || maxm > maxl) {
            maxm = maxl;
        }
        init(maxl, maxm);
    }

    void ComplexSphericalHarmoics::init(int maxl, int maxm) {
        maxL = maxl;
        maxM = maxm;
        int psize = (maxl + 1) * (maxl + 2) / 2;
        int ysize = (maxl + 1) * (maxl + 1);
        if (P)
            delete[] P;
        if (Y)
            delete[] Y;
        if (Yi)
            delete[] Yi;
        if (A)
            delete[] A;
        if (B)
            delete[] B;
        if (dP)
            delete[] dP;
        if (dY)
            delete[] dY;
        if (dYi)
            delete[] dYi;
        if (dYis)
            delete[] dYis;
        if (dYic)
            delete[] dYic;
        if (dYs)
            delete[] dYs;
        if (dYc)
            delete[] dYc;
        if (dYdx)
            delete[] dYdx;
        if (dYdy)
            delete[] dYdy;
        if (dYdz)
            delete[] dYdz;
        if (dYidx)
            delete[] dYidx;
        if (dYidy)
            delete[] dYidy;
        if (dYidz)
            delete[] dYidz;
        P = new double[psize];
        Y = new double[ysize];
        Yi = new double[ysize];
        A = new double[psize];
        B = new double[psize];
        dP = new double[psize];
        dY = new double[ysize];
        dYi = new double[ysize];
        dYs = new double[ysize];
        dYc = new double[ysize];
        dYis = new double[ysize];
        dYic = new double[ysize];
        dYdx = new double[ysize];
        dYdy = new double[ysize];
        dYdz = new double[ysize];
        dYidx = new double[ysize];
        dYidy = new double[ysize];
        dYidz = new double[ysize];

        for (int l = 2; l <= maxL; ++l) {
            double ls = l * l;
            double lm1s = (l - 1) * (l - 1);
            for (int m = 0; m < __min(l - 1, maxM); ++m) {
                double ms = m * m;
                A[PVT(l, m)] = sqrt((4 * ls - 1.) / (ls - ms));
                B[PVT(l, m)] = -sqrt((lm1s - ms) / (4. * lm1s - 1.));
            }
        }
    }

    // x = cos(theta)
    void ComplexSphericalHarmoics::computeP(const double x) {
        const double sintheta = sqrt(1. - x * x);
        constexpr double sq1 = 0.282094791773878143474039725780; // = sqrt(0.25/M_PI)
        double temp;
        P[PVT(0, 0)] = sq1;
        if (maxL == 0) return;

        constexpr double sq2 = 0.488602511902919921586384622838; // = sqrt(0.75/M_PI)
        P[PVT(1, 0)] = x * sq2;
        // use negative value to include phase factor
        constexpr double sp3 = -0.345494149471335479265244646032; // = sqrt(3/8/pi)
        // constexpr double sp3 = 0.345494149471335479265244646032;
        temp = sp3 * sintheta;
        P[PVT(1, 1)] = temp;

        for (int l = 2; l <= maxL; l++) {
            for (int m = 0; m < __min(l - 1, maxM); m++) {
                P[PVT(l, m)] = A[PVT(l, m)] * (x * P[PVT(l - 1, m)] + B[PVT(l, m)] * P[PVT(l - 2, m)]);
            }
            P[PVT(l, l - 1)] = x * sqrt(2 * (l - 1) + 3) * temp;
            temp = -sqrt(1.0 + 0.5 / l) * sintheta * temp;
            // temp = sqrt(1.0 + 0.5 / l) * sintheta * temp; // without phase factor
            P[PVT(l, l)] = temp;
        }
    }

    void ComplexSphericalHarmoics::computedP(const double x) {
        const double sintheta = sqrt(1. - x * x);
        constexpr double sq1 = 0.282094791773878143474039725780;
        double temp, dtemp;
        P[PVT(0, 0)] = sq1;
        dP[PVT(0, 0)] = 0.0;
        if (maxL == 0) return;

        constexpr double sq2 = 0.488602511902919921586384622838;
        P[PVT(1, 0)] = x * sq2;
        dP[PVT(1, 0)] = sq2;
        // use negative value to include phase factor
        constexpr double sp3 = -0.345494149471335479265244646032;
        // constexpr double sp3 = 0.345494149471335479265244646032;
        temp = sp3 * sintheta;
        dtemp = -sp3 * x / sintheta;
        P[PVT(1, 1)] = temp;
        dP[PVT(1, 1)] = dtemp;

        for (int l = 2; l <= maxL; l++) {
            for (int m = 0; m < __min(l - 1, maxM); m++) {
                P[PVT(l, m)] = A[PVT(l, m)] * (x * P[PVT(l - 1, m)] + B[PVT(l, m)] * P[PVT(l - 2, m)]);
                dP[PVT(l, m)] =
                        A[PVT(l, m)] * (P[PVT(l - 1, m)] + x * dP[PVT(l - 1, m)] + B[PVT(l, m)] * dP[PVT(l - 2, m)]);
            }
            P[PVT(l, l - 1)] = x * sqrt(2 * (l - 1) + 3) * temp;
            dP[PVT(l, l - 1)] = sqrt(2 * (l - 1) + 3) * (temp + x * dtemp);
            dtemp = -sqrt(1.0 + 0.5 / l) * (-x / sintheta * temp + sintheta * dtemp);
            temp = -sqrt(1.0 + 0.5 / l) * sintheta * temp;
            // temp = sqrt(1.0 + 0.5 / l) * sintheta * temp; // without phase factor
            P[PVT(l, l)] = temp;
            dP[PVT(l, l)] = dtemp;
        }
    }

    void ComplexSphericalHarmoics::computeY(const double theta, const double phi) {
        const double costheta = cos(theta);
        const double sinphi = sin(phi);
        const double cosphi = cos(phi);
        computeY(costheta, sinphi, cosphi);
    }

    void ComplexSphericalHarmoics::computeY(const double costheta, const double sinphi, const double cosphi) {
        computeP(costheta);
        for (int l = 0; l <= maxL; l++) {
            Y[YVR(l, 0)] = P[PVT(l, 0)];
            Yi[YVR(l, 0)] = 0;
        }

        double c1 = 1.0, c2 = cosphi;
        double s1 = 0.0, s2 = -sinphi;
        double tc = 2.0 * c2;
        int sign = 1;
        for (int m = 1; m <= maxM; m++) {
            double s = tc * s1 - s2;
            double c = tc * c1 - c2;
            s2 = s1;
            s1 = s;
            c2 = c1;
            c1 = c;
            sign *= -1;
            for (int l = m; l <= maxL; l++) {
                Y[YVR(l, m)] = P[PVT(l, m)] * c;
                Yi[YVR(l, m)] = P[PVT(l, m)] * s;
                Y[YVR(l, -m)] = sign * P[PVT(l, m)] * c;
                Yi[YVR(l, -m)] = -sign * P[PVT(l, m)] * s;
            }
        }

        // a little more accurate but cost more times
        // for (int l = 1; l <= ll; ++ll)
        // {
        //     for (int m = 1; m <=ll; ++m)
        //     {
        //         Y[YVR(l, m)] = P[PVT(l, m)] * cos(m*phi) ;
        //         Yi[YVR(l, m)] = P[PVT(l, m)] * sin(m*phi) ;
        //         Y[YVR(l, -m)] = sign * P[PVT(l, m)] * cos(m*phi) ;
        //         Yi[YVR(l, -m)] = -sign * P[PVT(l, m)] * sin(m*phi) ;
        //     }
        // }
    }

    void ComplexSphericalHarmoics::computedY(const double costheta, const double sinphi, const double cosphi) {
        computedP(costheta);
        for (int l = 0; l <= maxL; l++) {
            Y[YVR(l, 0)] = P[PVT(l, 0)];
            Yi[YVR(l, 0)] = 0;
            dY[YVR(l, 0)] = dP[PVT(l, 0)];
            dYs[YVR(l, 0)] = 0;
            dYc[YVR(l, 0)] = 0;
            dYi[YVR(l, 0)] = 0;
            dYis[YVR(l, 0)] = 0;
            dYic[YVR(l, 0)] = 0;
        }

        double c1 = 1.0, c2 = cosphi;
        double s1 = 0.0, s2 = -sinphi;
        double tc = 2.0 * c2;

        double dc1s = 0, dc2s = -sinphi / cosphi;
        double ds1s = 0, ds2s = -1;
        double dtcs = 2.0 * dc2s;

        double dc1c = 0, dc2c = 1;
        double ds1c = 0, ds2c = cosphi / sinphi;
        double dtcc = 2.0 * dc2c;

        int sign = 1;
        for (int m = 1; m <= maxM; m++) {
            double s = tc * s1 - s2;
            double c = tc * c1 - c2;
            double dss = dtcs * s1 + tc * ds1s - ds2s;
            double dsc = dtcc * s1 + tc * ds1c - ds2c;
            double dcs = dtcs * c1 + tc * dc1s - dc2s;
            double dcc = dtcc * c1 + tc * dc1c - dc2c;
            s2 = s1;
            ds2s = ds1s;
            ds2c = ds1c;
            s1 = s;
            ds1s = dss;
            ds1c = dsc;
            c2 = c1;
            dc2s = dc1s;
            dc2c = dc1c;
            c1 = c;
            dc1s = dcs;
            dc1c = dcc;
            sign *= -1;
            for (int l = m; l <= maxL; l++) {
                // TODO: put M_SQRT1_2 into P
                Y[YVR(l, m)] = P[PVT(l, m)] * c;
                Yi[YVR(l, m)] = P[PVT(l, m)] * s;
                Y[YVR(l, -m)] = sign * Y[YVR(l, m)];
                Yi[YVR(l, -m)] = -sign * Yi[YVR(l, m)];

                dY[YVR(l, m)] = dP[PVT(l, m)] * c;
                dYi[YVR(l, m)] = dP[PVT(l, m)] * s;
                dY[YVR(l, -m)] = sign * dY[YVR(l, m)];
                dYi[YVR(l, -m)] = -sign * dYi[YVR(l, m)];

                dYs[YVR(l, m)] = P[PVT(l, m)] * dcs;
                dYis[YVR(l, m)] = P[PVT(l, m)] * dss;
                dYs[YVR(l, -m)] = sign * dYs[YVR(l, m)];
                dYis[YVR(l, -m)] = -sign * dYis[YVR(l, m)];

                dYc[YVR(l, m)] = P[PVT(l, m)] * dcc;
                dYic[YVR(l, m)] = P[PVT(l, m)] * dsc;
                dYc[YVR(l, -m)] = sign * dYc[YVR(l, m)];
                dYic[YVR(l, -m)] = -sign * dYic[YVR(l, m)];
            }
        }
    }

    void ComplexSphericalHarmoics::computeYCar(const double &x, const double &y, const double &z, const double &r) {
        const double rxy = sqrt(x * x + y * y);
        const double costheta = z / r;
        const double sinphi = y / rxy;
        const double cosphi = x / rxy;
        computeY(costheta, sinphi, cosphi);
    }

    void ComplexSphericalHarmoics::computeYCarSafe(const double &x, const double &y, const double &z, const double &r) {
        const double rxy = sqrt(x * x + y * y);
        const double costheta = z / r;
        double sinphi, cosphi;
        if (rxy < std::numeric_limits<double>::epsilon()) {
            sinphi = 0;
            cosphi = 1;
        } else {
            sinphi = y / rxy;
            cosphi = x / rxy;
        }
        computeY(costheta, sinphi, cosphi);
    }

    void ComplexSphericalHarmoics::computedYCar(const double &x, const double &y, const double &z, const double &r) {
        const double rxy2 = x * x + y * y;
        const double costheta = z / r;
        const double invr = 1.0 / r;
        const double invr3 = invr / r / r;
        const double dcosthetadx = -x * z * invr3;
        const double dcosthetady = -y * z * invr3;
        const double dcosthetadz = -z * z * invr3 + invr;

        // Handle z-axis (rxy=0) separately: only |m|=1 has nonzero
        // transverse derivatives, with parity (-1)^(l+1) at the south pole.
        const bool on_z_axis = (rxy2 <= std::numeric_limits<double>::epsilon());

        if (on_z_axis) {
            computedP(costheta);
            // For |m|=1 on the z-axis, dY/dx and dY/dy have well-defined limits.
            // P(l,m) ∝ sin^m θ near θ=0, so for m=1: P(l,1) = F[l] * sinθ.
            // Y(l,1) = F[l] * x/r,  Yi(l,1) = F[l] * y/r  →  dYdx = dYidy = F[l]/r.
            // Y(l,-1) = -F[l] * x/r,  Yi(l,-1) = F[l] * y/r (sign convention).
            // For |m|>1, dY/dx = dY/dy = 0 since Y ∼ θ^m with m>1.
            double *coeff_m1 = new double[maxL + 1]();
            if (maxL >= 1) {
                constexpr double sp3 = -0.345494149471335479265244646032; // sqrt(3/8/pi)
                coeff_m1[0] = 0.0;
                coeff_m1[1] = sp3;
                for (int l = 2; l <= maxL; ++l) {
                    double A_l1 = sqrt((4.0 * l * l - 1.0) / (l * l - 1.0));
                    double B_l1 = -sqrt(((l - 1.0) * (l - 1.0) - 1.0) / (4.0 * (l - 1.0) * (l - 1.0) - 1.0));
                    coeff_m1[l] = A_l1 * (coeff_m1[l - 1] + B_l1 * coeff_m1[l - 2]);
                }
            }
            for (int l = 0; l <= maxL; l++) {
                int idx0 = YVR(l, 0);
                Y[idx0] = P[PVT(l, 0)];
                Yi[idx0] = 0;
                dYdx[idx0] = 0;
                dYdy[idx0] = 0;
                dYdz[idx0] = dP[PVT(l, 0)] * dcosthetadz;
                dYidx[idx0] = 0;
                dYidy[idx0] = 0;
                dYidz[idx0] = 0;
                if (l >= 1) {
                    double const invr_val = 1.0 / r;
                    double const Fl = (z < 0 && l % 2 == 0) ? -coeff_m1[l] : coeff_m1[l];
                    int p1 = YVR(l, 1);
                    int n1 = YVR(l, -1);
                    Y[p1] = Yi[p1] = Y[n1] = Yi[n1] = 0;
                    // m=1: Y = F[l] * x/r, Yi = F[l] * y/r
                    dYdx[p1] = Fl * invr_val;
                    dYdy[p1] = 0;
                    dYdz[p1] = 0;
                    dYidx[p1] = 0;
                    dYidy[p1] = Fl * invr_val;
                    dYidz[p1] = 0;
                    // m=-1: Y = -F[l] * x/r, Yi = F[l] * y/r
                    dYdx[n1] = -Fl * invr_val;
                    dYdy[n1] = 0;
                    dYdz[n1] = 0;
                    dYidx[n1] = 0;
                    dYidy[n1] = Fl * invr_val;
                    dYidz[n1] = 0;
                }
                for (int m = 2; m <= l; ++m) {
                    int idx_p = YVR(l, m);
                    int idx_n = YVR(l, -m);
                    Y[idx_p] = Yi[idx_p] = Y[idx_n] = Yi[idx_n] = 0;
                    dYdx[idx_p] = dYdy[idx_p] = dYdz[idx_p] = 0;
                    dYdx[idx_n] = dYdy[idx_n] = dYdz[idx_n] = 0;
                    dYidx[idx_p] = dYidy[idx_p] = dYidz[idx_p] = 0;
                    dYidx[idx_n] = dYidy[idx_n] = dYidz[idx_n] = 0;
                }
            }
            delete[] coeff_m1;
            return;
        }

        const double rxy = sqrt(rxy2);
        const double sinphi = y / rxy;
        const double cosphi = x / rxy;
        const double invrxy2 = 1.0 / rxy2;
        const double dphidx = -y * invrxy2;
        const double dphidy = x * invrxy2;

        computedY(costheta, sinphi, cosphi);
        int idx;
        for (int m = -maxM; m <= maxM; ++m) {
            for (int l = std::abs(m); l <= maxL; ++l) {
                idx = YVR(l, m);
                // ∂Re(Y)/∂φ = -m * Im(Y),  ∂Im(Y)/∂φ =  m * Re(Y)
                dYdx[idx] = dY[idx] * dcosthetadx + (-m * Yi[idx]) * dphidx;
                dYdy[idx] = dY[idx] * dcosthetady + (-m * Yi[idx]) * dphidy;
                dYdz[idx] = dY[idx] * dcosthetadz;
                dYidx[idx] = dYi[idx] * dcosthetadx + (m * Y[idx]) * dphidx;
                dYidy[idx] = dYi[idx] * dcosthetady + (m * Y[idx]) * dphidy;
                dYidz[idx] = dYi[idx] * dcosthetadz;
            }
        }
    }

    ComplexSphericalHarmoics::~ComplexSphericalHarmoics() {
        if (P)
            delete[] P;
        if (Y)
            delete[] Y;
        if (Yi)
            delete[] Yi;
        if (A)
            delete[] A;
        if (B)
            delete[] B;
        if (dP)
            delete[] dP;
        if (dY)
            delete[] dY;
        if (dYi)
            delete[] dYi;
        if (dYis)
            delete[] dYis;
        if (dYic)
            delete[] dYic;
        if (dYs)
            delete[] dYs;
        if (dYc)
            delete[] dYc;
        if (dYdx)
            delete[] dYdx;
        if (dYdy)
            delete[] dYdy;
        if (dYdz)
            delete[] dYdz;
        if (dYidx)
            delete[] dYidx;
        if (dYidy)
            delete[] dYidy;
        if (dYidz)
            delete[] dYidz;
    }

    std::complex<double> ComplexSphericalHarmoics::get(const int &l, const int &m) const {
        return std::complex<double>(Y[YVR(l, m)], Yi[YVR(l, m)]);
    }

    RealSphericalHarmoics::RealSphericalHarmoics(int maxl, int maxm) {
        init(maxl, maxm);
    }

    void RealSphericalHarmoics::init(int maxl, int maxm) {
        if (maxl < 0 || maxm < -1 || maxm > maxl ||
            (static_cast<long long>(maxl)+1)*(static_cast<long long>(maxl)+2) > std::numeric_limits<int>::max())
            throw std::invalid_argument("Invalid real spherical harmonic dimensions");
        maxL = maxl;
        maxM = maxm;
        if (maxm < 0)
            maxM = maxL;
        int psize = (maxl + 1) * (maxl + 2) / 2;
        int ysize = (maxl + 1) * (maxl + 1);
        if (P)
            delete[] P;
        if (Y)
            delete[] Y;
        if (A)
            delete[] A;
        if (B)
            delete[] B;
        if (dYdc)
            delete[] dYdc;
        if (dYds)
            delete[] dYds;
        if (dYdt)
            delete[] dYdt;
        if (dYdx)
            delete[] dYdx;
        if (dYdy)
            delete[] dYdy;
        if (dYdz)
            delete[] dYdz;
        if (dP)
            delete[] dP;
        P = new double[psize];
        Y = new double[ysize];
        A = new double[psize];
        B = new double[psize];
        dP = new double[psize];
        dYdc = new double[ysize];
        dYdt = new double[ysize];
        dYds = new double[ysize];
        dYdx = new double[ysize];
        dYdy = new double[ysize];
        dYdz = new double[ysize];

        for (int l = 2; l <= maxL; ++l) {
            double ls = l * l;
            double lm1s = (l - 1) * (l - 1);
            for (int m = 0; m < l - 1; ++m) {
                double ms = m * m;
                A[PVT(l, m)] = sqrt((4 * ls - 1.) / (ls - ms));
                B[PVT(l, m)] = -sqrt((lm1s - ms) / (4. * lm1s - 1.));
            }
        }
    }

    void RealSphericalHarmoics::computeP(const double x) {
        const double sintheta = sqrt(1. - x * x);
        constexpr double sq1 = 0.398942280401432677939946; // = sqrt(0.5/M_PI)
        double temp;
        P[PVT(0, 0)] = sq1;
        if (maxL == 0) return;

        constexpr double sq2 = 0.69098829894267095853048929206377936627875474; // = sqrt(1.5/M_PI)
        P[PVT(1, 0)] = x * sq2;
        // omit negative - here to remove phase factor
        constexpr double sp3 = -0.488602511902919921586384622838347; // = sqrt(3/4/pi)
        temp = sp3 * sintheta;
        P[PVT(1, 1)] = temp;

        for (int l = 2; l <= maxL; ++l) {
            for (int m = 0; m <= __min(l - 2, maxM); ++m) {
                P[PVT(l, m)] = A[PVT(l, m)] * (x * P[PVT(l - 1, m)] + B[PVT(l, m)] * P[PVT(l - 2, m)]);
            }
            P[PVT(l, l - 1)] = x * sqrt(2 * (l - 1) + 3) * temp;
            temp = -sqrt(1.0 + 0.5 / l) * sintheta * temp;
            P[PVT(l, l)] = temp;
        }
    }

    void RealSphericalHarmoics::computedP(const double x) {
        const double sintheta = sqrt(1. - x * x);
        constexpr double sq1 = 0.398942280401432677939946; // = sqrt(0.5/M_PI)
        double temp, dtemp;
        P[PVT(0, 0)] = sq1;
        dP[PVT(0, 0)] = 0.0;
        if (maxL == 0) return;

        constexpr double sq2 = 0.69098829894267095853048929206377936627875474; // = sqrt(1.5/M_PI)
        P[PVT(1, 0)] = x * sq2;
        dP[PVT(1, 0)] = sq2;
        // omit negative - here to remove phase factor
        constexpr double sp3 = -0.488602511902919921586384622838347; // = sqrt(3/4/pi)
        temp = sp3 * sintheta;
        dtemp = -sp3 * x / sintheta;
        P[PVT(1, 1)] = temp;
        dP[PVT(1, 1)] = dtemp;

        for (int l = 2; l <= maxL; ++l) {
            for (int m = 0; m <= __min(l - 2, maxM); ++m) {
                P[PVT(l, m)] = A[PVT(l, m)] * (x * P[PVT(l - 1, m)] + B[PVT(l, m)] * P[PVT(l - 2, m)]);
                dP[PVT(l, m)] =
                        A[PVT(l, m)] * (P[PVT(l - 1, m)] + x * dP[PVT(l - 1, m)] + B[PVT(l, m)] * dP[PVT(l - 2, m)]);
            }
            P[PVT(l, l - 1)] = x * sqrt(2 * (l - 1) + 3) * temp;
            dP[PVT(l, l - 1)] = sqrt(2 * (l - 1) + 3) * (temp + x * dtemp);
            dtemp = -sqrt(1.0 + 0.5 / l) * (-x / sintheta * temp + sintheta * dtemp);
            temp = -sqrt(1.0 + 0.5 / l) * sintheta * temp;
            P[PVT(l, l)] = temp;
            dP[PVT(l, l)] = dtemp;
        }
    }

    double RealSphericalHarmoics::get(const int &l, const int &m) const {
        return Y[YVR(l, m)];
    }

    // P should be computed first
    void RealSphericalHarmoics::computeY(const double theta, const double phi) {
        const double costheta = cos(theta);
        const double cosphi = cos(phi);
        const double sinphi = sin(phi);
        computeY(costheta, sinphi, cosphi);

        // a little more accurate but cost more times
        // for (int l = 1; l <= ll; ++ll)
        // {
        //     for (int m = 1; m <=ll; ++m)
        //     {
        //         Y[YVR(l, -m)] = P[PVT(l, m)] * sin(m*phi);
        //         Y[YVR(l, m)] = P[PVT(l, m)] * cos(m*phi);
        //     }
        // }
    }

    void RealSphericalHarmoics::computeY(const double costheta, const double sinphi, const double cosphi) {
        computeP(costheta);
        for (int l = 0; l <= maxL; l++)
            Y[YVR(l, 0)] = P[PVT(l, 0)] * M_SQRT1_2;

        double c1 = 1.0, c2 = cosphi;
        double s1 = 0.0, s2 = -sinphi;
        double tc = 2.0 * c2;
        for (int m = 1; m <= maxM; m++) {
            double s = tc * s1 - s2;
            double c = tc * c1 - c2;
            s2 = s1;
            s1 = s;
            c2 = c1;
            c1 = c;
            for (int l = m; l <= maxL; l++) {
                Y[YVR(l, -m)] = P[PVT(l, m)] * s;
                Y[YVR(l, m)] = P[PVT(l, m)] * c;
            }
        }
    }
    namespace {
    struct CartesianFirst {
        double value=0.;std::array<double,3> gradient{};
        CartesianFirst(double v=0.) : value(v) {}
    };
    CartesianFirst operator+(const CartesianFirst& a,const CartesianFirst& b) {
        CartesianFirst c(a.value+b.value);
        for(int i=0;i<3;++i)c.gradient[i]=a.gradient[i]+b.gradient[i];return c;
    }
    CartesianFirst operator-(const CartesianFirst& a,const CartesianFirst& b) {
        CartesianFirst c(a.value-b.value);
        for(int i=0;i<3;++i)c.gradient[i]=a.gradient[i]-b.gradient[i];return c;
    }
    CartesianFirst operator*(const CartesianFirst& a,const CartesianFirst& b) {
        CartesianFirst c(a.value*b.value);
        for(int i=0;i<3;++i)c.gradient[i]=a.gradient[i]*b.value+a.value*b.gradient[i];return c;
    }
    void real_cartesian(RealSphericalHarmoics& sh,double x,double y,double z,double r,bool derivatives) {
        const double norm=std::hypot(x,y,z);
        if (!std::isfinite(norm) || !std::isfinite(r) || norm<=0. || r<=0. ||
            std::abs(norm-r)>64*std::numeric_limits<double>::epsilon()*std::max(norm,r))
            throw std::invalid_argument("Real spherical harmonics require finite nonzero coordinates and their radius");
        if (!sh.Y) throw std::logic_error("Real spherical harmonics are not initialized");
        const std::array<double,3> u={x/norm,y/norm,z/norm};
        if(!derivatives) {
            const auto values=detail::real_solid_harmonics(sh.maxL,sh.maxM,u[0],u[1],u[2]);
            std::copy(values.begin(),values.end(),sh.Y);return;
        }
        std::array<CartesianFirst,3> direction;
        for(int a=0;a<3;++a) {
            direction[a].value=u[a];
            for(int b=0;b<3;++b)direction[a].gradient[b]=((a==b)-u[a]*u[b])/norm;
        }
        const auto values=detail::real_solid_harmonics(sh.maxL,sh.maxM,direction[0],direction[1],direction[2]);
        for(size_t i=0;i<values.size();++i) {
            sh.Y[i]=values[i].value;
            sh.dYdx[i]=values[i].gradient[0];sh.dYdy[i]=values[i].gradient[1];sh.dYdz[i]=values[i].gradient[2];
        }
    }
    }
    void RealSphericalHarmoics::computeYCar(const double& x,const double& y,const double& z,const double& r) {
        real_cartesian(*this,x,y,z,r,false);
    }
    void RealSphericalHarmoics::computeYCarSafe(const double& x,const double& y,const double& z,const double& r) {
        real_cartesian(*this,x,y,z,r,false);
    }

    void RealSphericalHarmoics::computedY(const double costheta, const double sinphi, const double cosphi) {
        computedP(costheta);
        for (int l = 0; l <= maxL; l++) {
            Y[YVR(l, 0)] = P[PVT(l, 0)] * M_SQRT1_2;
            dYdt[YVR(l, 0)] = dP[PVT(l, 0)] * M_SQRT1_2;
            dYds[YVR(l, 0)] = 0;
            dYdc[YVR(l, 0)] = 0;
        }

        double c1 = 1.0, c2 = cosphi;
        double s1 = 0.0, s2 = -sinphi;
        double tc = 2.0 * c2;

        double dc1s = 0, dc2s = -sinphi / cosphi;
        double ds1s = 0, ds2s = -1;
        double dtcs = 2.0 * dc2s;

        double dc1c = 0, dc2c = 1;
        double ds1c = 0, ds2c = cosphi / sinphi;
        double dtcc = 2.0 * dc2c;

        for (int m = 1; m <= maxM; m++) {
            double s = tc * s1 - s2;
            double c = tc * c1 - c2;
            double dss = dtcs * s1 + tc * ds1s - ds2s;
            double dsc = dtcc * s1 + tc * ds1c - ds2c;
            double dcs = dtcs * c1 + tc * dc1s - dc2s;
            double dcc = dtcc * c1 + tc * dc1c - dc2c;

            s2 = s1;
            ds2s = ds1s;
            ds2c = ds1c;
            s1 = s;
            ds1s = dss;
            ds1c = dsc;
            c2 = c1;
            dc2s = dc1s;
            dc2c = dc1c;
            c1 = c;
            dc1s = dcs;
            dc1c = dcc;

            for (int l = m; l <= maxL; l++) {
                Y[YVR(l, m)] = P[PVT(l, m)] * c;
                Y[YVR(l, -m)] = P[PVT(l, m)] * s;

                dYdt[YVR(l, m)] = dP[PVT(l, m)] * c;
                dYdt[YVR(l, -m)] = dP[PVT(l, m)] * s;

                dYds[YVR(l, m)] = P[PVT(l, m)] * dcs;
                dYds[YVR(l, -m)] = P[PVT(l, m)] * dss;

                dYdc[YVR(l, m)] = P[PVT(l, m)] * dcc;
                dYdc[YVR(l, -m)] = P[PVT(l, m)] * dsc;
            }
        }
    }

    void RealSphericalHarmoics::computedYCar(const double& x,const double& y,const double& z,const double& r) {
        real_cartesian(*this,x,y,z,r,true);
    }

    RealSphericalHarmoics::~RealSphericalHarmoics() {
        if (P)
            delete[] P;
        if (Y)
            delete[] Y;
        if (A)
            delete[] A;
        if (B)
            delete[] B;
        if (dYdc)
            delete[] dYdc;
        if (dYds)
            delete[] dYds;
        if (dYdt)
            delete[] dYdt;
        if (dYdx)
            delete[] dYdx;
        if (dYdy)
            delete[] dYdy;
        if (dYdz)
            delete[] dYdz;
        if (dP)
            delete[] dP;
    }
}
