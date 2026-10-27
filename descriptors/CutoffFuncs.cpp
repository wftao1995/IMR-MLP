// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#include "CutoffFuncs.h"
#include <stdexcept>
#include <cmath>

// #define TANH_PRE 10.13313248223860242047407046222649266841935669440587255062758086
#define TANH_PRE 2.26375379522535079537792974633292536376619020514776237861236

using namespace std;

CutoffFunctions::CutoffFunctions() : cutofftype_(CT_HARD),
                                     radius_(1.0),
                                     radinv_(1.0),
                                     radinner_(1.0),
                                     alpha_(0.0),
                                     iw(0.0),
                                     fPtr(&CutoffFunctions::fHARD),
                                     dfPtr(&CutoffFunctions::dfHARD),
                                     fdfPtr(&CutoffFunctions::fdfHARD)
{
    ;
}

void CutoffFunctions::setCutoffType(CutoffType const cutoffType)
{
    this->cutofftype_ = cutoffType;

    if (cutoffType == CT_HARD)
    {
        fPtr = &CutoffFunctions::fHARD;
        dfPtr = &CutoffFunctions::dfHARD;
        fdfPtr = &CutoffFunctions::fdfHARD;
    }
    else if (cutoffType == CT_COS)
    {
        fPtr = &CutoffFunctions::fCOS;
        dfPtr = &CutoffFunctions::dfCOS;
        fdfPtr = &CutoffFunctions::fdfCOS;
    }
    else if (cutoffType == CT_COS_BASIC)
    {
        fPtr = &CutoffFunctions::fCOSBasic;
        dfPtr = &CutoffFunctions::dfCOSBasic;
        fdfPtr = &CutoffFunctions::fdfCOSBasic;
    }
    else if (cutoffType == CT_TANHU)
    {
        fPtr = &CutoffFunctions::fTANHU;
        dfPtr = &CutoffFunctions::dfTANHU;
        fdfPtr = &CutoffFunctions::fdfTANHU;
    }
    else if (cutoffType == CT_TANH)
    {
        fPtr = &CutoffFunctions::fTANH;
        dfPtr = &CutoffFunctions::dfTANH;
        fdfPtr = &CutoffFunctions::fdfTANH;
    }
    else if (cutoffType == CT_EXP)
    {
        fPtr = &CutoffFunctions::fEXP;
        dfPtr = &CutoffFunctions::dfEXP;
        fdfPtr = &CutoffFunctions::fdfEXP;
    }
    else if (cutoffType == CT_POLY1)
    {
        fPtr = &CutoffFunctions::fPOLY1;
        dfPtr = &CutoffFunctions::dfPOLY1;
        fdfPtr = &CutoffFunctions::fdfPOLY1;
    }
    else if (cutoffType == CT_POLY2)
    {
        fPtr = &CutoffFunctions::fPOLY2;
        dfPtr = &CutoffFunctions::dfPOLY2;
        fdfPtr = &CutoffFunctions::fdfPOLY2;
    }
    else if (cutoffType == CT_POLY3)
    {
        fPtr = &CutoffFunctions::fPOLY3;
        dfPtr = &CutoffFunctions::dfPOLY3;
        fdfPtr = &CutoffFunctions::fdfPOLY3;
    }
    else if (cutoffType == CT_POLY4)
    {
        fPtr = &CutoffFunctions::fPOLY4;
        dfPtr = &CutoffFunctions::dfPOLY4;
        fdfPtr = &CutoffFunctions::fdfPOLY4;
    }
    else
    {
        throw invalid_argument("ERROR: Unknown cutoff type.\n");
    }

    return;
}

void CutoffFunctions::setCutoffRadius(double const cutoffRadius, double const cutoffInner)
{
    radius_ = cutoffRadius;
    radinv_ = 1.0 / cutoffRadius;
    iw = 1.0 / (radius_ - radinner_);
    return;
}

void CutoffFunctions::setCutoffParameter(double const alpha)
{
    if (alpha < 0.0 || alpha >= 1.0)
    {
        throw invalid_argument("ERROR: 0 <= alpha < 1.0 is required.\n");
    }
    alpha_ = alpha;
    radinner_ = radius_ * alpha;
    iw = 1.0 / (radius_ - radinner_);
    return;
}

double CutoffFunctions::fCOSBasic(double r) const
{
    return 0.5 * (cos(M_PI * r * radinv_) + 1.0);
}

double CutoffFunctions::dfCOSBasic(double r) const
{
    return -M_PI_2 * radinv_ * sin(M_PI * r * radinv_);
}

void CutoffFunctions::fdfCOSBasic(double r, double &fc, double &dfc) const
{
    double const t = M_PI * r * radinv_;
    double const c = cos(t);
    // double const s = sin(t);
    fc = 0.5 * (c + 1.0);
    dfc = - radinv_ * M_PI_2 * sqrt(1.0 - c * c);
    return;
}

double CutoffFunctions::fCOS(double r) const
{
    if (r < radinner_)
        return 1.0;
    double const x = (r - radinner_) * iw;
    return 0.5 * (cos(M_PI * x) + 1.0);
}

double CutoffFunctions::dfCOS(double r) const
{
    if (r < radinner_)
        return 0.0;
    double const x = (r - radinner_) * iw;
    return -M_PI_2 * iw * sin(M_PI * x);
}

void CutoffFunctions::fdfCOS(double r, double &fc, double &dfc) const
{
    if (r < radinner_)
    {
        fc = 1.0;
        dfc = 0.0;
        return;
    }
    double const x = (r - radinner_) * iw;
    double const temp = cos(M_PI * x);
    fc = 0.5 * (temp + 1.0);
    dfc = -0.5 * iw * M_PI * sqrt(1.0 - temp * temp);
    return;
}

double CutoffFunctions::fTANHU(double r) const
{
    double const temp = tanh(1.0 - r * radinv_);
    return temp * temp * temp;
}

double CutoffFunctions::dfTANHU(double r) const
{
    double temp = tanh(1.0 - r * radinv_);
    temp *= temp;
    return 3.0 * temp * (temp - 1.0) * radinv_;
}

void CutoffFunctions::fdfTANHU(double r, double &fc, double &dfc) const
{
    double const temp = tanh(1.0 - r * radinv_);
    double const temp2 = temp * temp;
    fc = temp * temp2;
    dfc = 3.0 * temp2 * (temp2 - 1.0) * radinv_;
    return;
}

double CutoffFunctions::fTANH(double r) const
{
    double const temp = tanh(1.0 - r * radinv_);
    return TANH_PRE * temp * temp * temp;
}

double CutoffFunctions::dfTANH(double r) const
{
    double temp = tanh(1.0 - r * radinv_);
    temp *= temp;
    return 3.0 * TANH_PRE * temp * (temp - 1.0) * radinv_;
}

void CutoffFunctions::fdfTANH(double r, double &fc, double &dfc) const
{
    double const temp = tanh(1.0 - r * radinv_);
    double const temp2 = temp * temp;
    fc = TANH_PRE * temp * temp2;
    dfc = 3.0 * TANH_PRE * temp2 * (temp2 - 1.0) * radinv_;
    return;
}

double CutoffFunctions::fEXP(double r) const
{
    if (r < radinner_)
        return 1.0;
    double const x = (r - radinner_) * iw;
    return M_E * exp(1.0 / (x * x - 1.0));
}

double CutoffFunctions::dfEXP(double r) const
{
    if (r < radinner_)
        return 0.0;
    double const x = (r - radinner_) * iw;
    double const temp = 1.0 / (x * x - 1.0);
    return -2.0 * iw * M_E * x * temp * temp * exp(temp);
}

void CutoffFunctions::fdfEXP(double r, double &fc, double &dfc) const
{
    if (r < radinner_)
    {
        fc = 1.0;
        dfc = 0.0;
        return;
    }
    double const x = (r - radinner_) * iw;
    double const temp = 1.0 / (x * x - 1.0);
    double const temp2 = exp(temp);
    fc = M_E * temp2;
    dfc = -2.0 * iw * M_E * x * temp * temp * temp2;
    return;
}

double CutoffFunctions::fPOLY1(double r) const
{
    if (r < radinner_)
        return 1.0;
    double const x = (r - radinner_) * iw;
    return (2.0 * x - 3.0) * x * x + 1.0;
}

double CutoffFunctions::dfPOLY1(double r) const
{
    if (r < radinner_)
        return 0.0;
    double const x = (r - radinner_) * iw;
    return iw * x * (6.0 * x - 6.0);
}

void CutoffFunctions::fdfPOLY1(double r, double &fc, double &dfc) const
{
    if (r < radinner_)
    {
        fc = 1.0;
        dfc = 0.0;
        return;
    }
    double const x = (r - radinner_) * iw;
    fc = (2.0 * x - 3.0) * x * x + 1.0;
    dfc = iw * x * (6.0 * x - 6.0);
    return;
}

double CutoffFunctions::fPOLY2(double r) const
{
    if (r < radinner_)
        return 1.0;
    double const x = (r - radinner_) * iw;
    return ((15.0 - 6.0 * x) * x - 10.0) * x * x * x + 1.0;
}

double CutoffFunctions::dfPOLY2(double r) const
{
    if (r < radinner_)
        return 0.0;
    double const x = (r - radinner_) * iw;
    return iw * x * x * ((60.0 - 30.0 * x) * x - 30.0);
}

void CutoffFunctions::fdfPOLY2(double r, double &fc, double &dfc) const
{
    if (r < radinner_)
    {
        fc = 1.0;
        dfc = 0.0;
        return;
    }
    double const x = (r - radinner_) * iw;
    double const x2 = x * x;
    fc = ((15.0 - 6.0 * x) * x - 10.0) * x * x2 + 1.0;
    dfc = iw * x2 * ((60.0 - 30.0 * x) * x - 30.0);
    return;
}

double CutoffFunctions::fPOLY3(double r) const
{
    if (r < radinner_)
        return 1.0;
    double const x = (r - radinner_) * iw;
    double const x2 = x * x;
    return (x * (x * (20.0 * x - 70.0) + 84.0) - 35.0) * x2 * x2 + 1.0;
}

double CutoffFunctions::dfPOLY3(double r) const
{
    if (r < radinner_)
        return 0.0;
    double const x = (r - radinner_) * iw;
    return iw * x * x * x * (x * (x * (140.0 * x - 420.0) + 420.0) - 140.0);
}

void CutoffFunctions::fdfPOLY3(double r, double &fc, double &dfc) const
{
    if (r < radinner_)
    {
        fc = 1.0;
        dfc = 0.0;
        return;
    }
    double const x = (r - radinner_) * iw;
    double const x2 = x * x;
    fc = (x * (x * (20.0 * x - 70.0) + 84.0) - 35.0) * x2 * x2 + 1.0;
    dfc = iw * x2 * x * (x * (x * (140.0 * x - 420.0) + 420.0) - 140.0);
    return;
}

double CutoffFunctions::fPOLY4(double r) const
{
    if (r < radinner_)
        return 1.0;
    double const x = (r - radinner_) * iw;
    double const x2 = x * x;
    return (x * (x * ((315.0 - 70.0 * x) * x - 540.0) + 420.0) - 126.0) *
               x2 * x2 * x +
           1.0;
}

double CutoffFunctions::dfPOLY4(double r) const
{
    if (r < radinner_)
        return 0.0;
    double const x = (r - radinner_) * iw;
    double const x2 = x * x;
    return iw * x2 * x2 *
           (x * (x * ((2520.0 - 630.0 * x) * x - 3780.0) + 2520.0) - 630.0);
}

void CutoffFunctions::fdfPOLY4(double r, double &fc, double &dfc) const
{
    if (r < radinner_)
    {
        fc = 1.0;
        dfc = 0.0;
        return;
    }
    double const x = (r - radinner_) * iw;
    double x4 = x * x;
    x4 *= x4;
    fc = (x * (x * ((315.0 - 70.0 * x) * x - 540.0) + 420.0) - 126.0) *
             x * x4 +
         1.0;
    dfc = iw * x4 *
          (x * (x * ((2520.0 - 630.0 * x) * x - 3780.0) + 2520.0) - 630.0);
    return;
}
