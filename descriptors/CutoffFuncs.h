// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

/**
 * @brief Cutoff functions for smoothly truncating pairwise interactions.
 * @ingroup descriptors
 *
 * Provides cosine, polynomial, and other functional forms with inner cutoff support.
 */
#ifndef IMRMLP_CUTOFFFUNCS_H
#define IMRMLP_CUTOFFFUNCS_H
class CutoffFunctions
{
public:
    /** List of available cutoff function types.
     *
     * Most cutoff types allow the definition of an inner cutoff
     * @f$ r_{ci} := \alpha \, r_c@f$. Then the cutoff is equal to @f$1@f$ up
     * to the inner cutoff:
     *
     * @f$ f_c(r) =
     * \begin{cases}
     * 1, & \text{for } 0 \le r < r_{ci} \\
     * f(x), & \text{for } r_{ci} \le r < r_c \text{ where }
     * x := \frac{r - r_{ci}}{r_c - r_{ci}} \\
     * 0 & \text{for } r \geq r_c
     * \end{cases} @f$
     */
    enum CutoffType
    {
        /** @f$f(x) = 1@f$ */
        CT_HARD,
        /** @f$f(x) = \frac{1}{2} \left[ \cos (\pi x) + 1\right] @f$ */
        CT_COS_BASIC,
        CT_COS,
        /** @f$f_c(r) = \tanh^3 \left(1 - \frac{r}{r_c} \right) @f$ */
        CT_TANHU,
        /** @f$f_c(r) = c \tanh^3 \left(1 - \frac{r}{r_c} \right),\,
         *  f(0) = 1 @f$ */
        CT_TANH,
        /** @f$f(x) = e^{1 - \frac{1}{1 - x^2}}@f$ */
        CT_EXP,
        /** @f$f(x) = (2x - 3)x^2 + 1@f$ */
        CT_POLY1,
        /** @f$f(x) = ((15 - 6x)x - 10) x^3 + 1@f$ */
        CT_POLY2,
        /** @f$f(x) = (x(x(20x - 70) + 84) - 35)x^4 + 1@f$ */
        CT_POLY3,
        /** @f$f(x) = (x(x((315 - 70x)x - 540) + 420) - 126)x^5 + 1@f$ */
        CT_POLY4
    };

private:
    CutoffType cutofftype_;
    double radius_ = 5.0;
    double radinv_ = 0.2;                                                           // inverse of radius
    double radinner_ = 1.0;                                                         // inner cutoff
    double iw = 4.0;                                                                // inverse width of cutoff function @f$\frac{1}{r_c - r_{ci}}@f$.
    double alpha_;                                                            // Cutoff function parameter for `CT_POLYn` and `CT_EXP` @f$\alpha@f$.
    double (CutoffFunctions::*fPtr)(double r) const;                          //Function pointer to fcut
    double (CutoffFunctions::*dfPtr)(double r) const;                         // Function pointer to df.
    void (CutoffFunctions::*fdfPtr)(double r, double &fc, double &dfc) const; // Function pointer to fdf.

    // Cutoff functions
    // double fHARD(double r) const;
    // double dfHARD(double r) const;
    // void fdfHARD(double r, double &fc, double &dfc) const;

    double fCOS(double r) const;
    double dfCOS(double r) const;
    void fdfCOS(double r, double &fc, double &dfc) const;

    double fCOSBasic(double r) const;
    double dfCOSBasic(double r) const;
    void fdfCOSBasic(double r, double &fc, double &dfc) const;

    double fTANHU(double r) const;
    double dfTANHU(double r) const;
    void fdfTANHU(double r, double &fc, double &dfc) const;

    double fTANH(double r) const;
    double dfTANH(double r) const;
    void fdfTANH(double r, double &fc, double &dfc) const;

    double fPOLY1(double r) const;
    double dfPOLY1(double r) const;
    void fdfPOLY1(double r, double &fc, double &dfc) const;

    double fPOLY2(double r) const;
    double dfPOLY2(double r) const;
    void fdfPOLY2(double r, double &fc, double &dfc) const;

    double fPOLY3(double r) const;
    double dfPOLY3(double r) const;
    void fdfPOLY3(double r, double &fc, double &dfc) const;

    double fPOLY4(double r) const;
    double dfPOLY4(double r) const;
    void fdfPOLY4(double r, double &fc, double &dfc) const;

    double fEXP(double r) const;
    double dfEXP(double r) const;
    void fdfEXP(double r, double &fc, double &dfc) const;

public:
    // Constructor, initializes to CT_HARD
    CutoffFunctions();
    ~CutoffFunctions(){};
    /** Set cutoff type.
     * @param[in] ctype  Type of cutoff function to use.
    */
    void setCutoffType(CutoffType const ctype);
    inline CutoffType getCutoffType(void) const { return cutofftype_; };
    void setCutoffRadius(double radius, double const cutoffInner = 1.0);
    inline double getCutoffRadius(void) const { return radius_; };
    void setCutoffParameter(double const alpha);
    inline double getCutoffParameter(void) const { return alpha_; };
    inline double fHARD(double) const { return 1; };
    inline double dfHARD(double) const { return 0; };
    inline void fdfHARD(double, double &fc, double &dfc) const
    {
        fc = 1.0;
        dfc = 0.0;
        return;
    };
    /** Cutoff function
     * parameter r
     * */
    inline double fcut(double r) const
    {
        if (r >= radius_)
            return 0.0;
        else
            return (this->*fPtr)(r);
    }
    // derivative of cutoff
    inline double dfcut(double r) const
    {
        if (r >= radius_)
            return 0.0;
        else
            return (this->*dfPtr)(r);
    }
    // cutoff and derivative calculated also
    inline void fdfcut(double r, double &fc, double &dfc) const
    {
        if (r >= radius_)
        {
            fc = 0.0;
            dfc = 0.0;
        }
        else
            (this->*fdfPtr)(r, fc, dfc);
        return;
    }
};

#endif // IMRMLP_CUTOFFFUNCS_H