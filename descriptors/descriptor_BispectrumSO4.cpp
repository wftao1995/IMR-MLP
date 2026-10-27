#include "descriptor_BispectrumSO4.h"
#include "tools/tools.h"
#include "so4_density_kernel.h"
#include "so4_wigner_spline.h"
#include <array>
#include <stdexcept>

#define COMPLEX_ONE (1.0 + 0.0i)
#define COMPLEX_ZERO (0.0 + 0.0i)
#define COMPLEX_IMG (0.0 + 1.0i)
#define COMPLEX_DIA (1.0 + 1.0i)

using namespace std::literals;
using namespace std::complex_literals;

Descriptor_BispectrumSO4::Descriptor_BispectrumSO4(double _cutoff, int _twojmax, bool _der, double _radius, CutoffFunctions::CutoffType ctype)
{
    cutoff_ = _cutoff;
    twojmax_ = _twojmax;
    compute_der = _der;
    radius_ = _radius;
    init_internals(ctype);
};

Descriptor_BispectrumSO4::Descriptor_BispectrumSO4(const GPSettings &_settings)
{
    // Set defaults, then let ApplySettings override
    cutoff_ = 5.0;
    twojmax_ = 9;
    radius_ = -1.0;
    compute_der = false;
    InitSettings();
    ApplySettings(_settings);
    init_internals(CutoffFunctions::CT_COS_BASIC);
};

void Descriptor_BispectrumSO4::init_internals(CutoffFunctions::CutoffType ctype)
{
    if (twojmax_ < 0)
        throw std::invalid_argument("BispectrumSO4: twojmax must be nonnegative");
    if (radius_ < 0.0)
        radius_ = cutoff_ * 4.0 / 3.0;
    if (!(radius_ > 0.) || !(cutoff_ > 0.) || !std::isfinite(radius_) || !std::isfinite(cutoff_))
        throw std::invalid_argument("BispectrumSO4: radius and cutoff must be positive");
    if (radial_angle_map_ != "linear" && radial_angle_map_ != "atan")
        throw std::invalid_argument("BispectrumSO4: radial_angle_map must be linear or atan");
    if (quaternion_convention_ == "legacy") {
#if defined(DES_SOF_Q1)
        convention_ = 1;
#elif defined(DES_SOF_Q2)
        convention_ = 2;
#elif defined(DES_SOF_Q3)
        convention_ = 3;
#else
        convention_ = 4;
#endif
    } else if (quaternion_convention_.size()==2 && quaternion_convention_[0]=='q' &&
               quaternion_convention_[1]>='1' && quaternion_convention_[1]<='4')
        convention_ = quaternion_convention_[1]-'0';
    else throw std::invalid_argument("BispectrumSO4: invalid quaternion_convention");
    cutfun_.setCutoffType(ctype);
    cutfun_.setCutoffRadius(cutoff_);
    invz0_ = M_PI / radius_;
    if (average_!="outer" && average_!="inner-density")
        throw std::invalid_argument("BispectrumSO4: average must be outer or inner-density");
    if (contraction_backend_!="reference" && contraction_backend_!="snap")
        throw std::invalid_argument("BispectrumSO4: contraction_backend must be reference or snap");
    if(wigner_backend_!="direct" && wigner_backend_!="spline")
        throw std::invalid_argument("BispectrumSO4: wigner_backend must be direct or spline");
    if(wigner_backend_=="spline")spline_=SO4Density::make_wigner_spline(twojmax_);
    coefficient_offsets_=SO4Density::offsets(twojmax_);
    bi_size = 0;
    for (int tj = 0; tj <= twojmax_; ++tj)
    {
        for (int tj2 = 0; tj2 <= tj; ++tj2)
        {
            for (int tj1 = tj - tj2; tj1 <= tj2; ++tj1)
            {
                if (tj2 - tj1 <= tj && tj2 + tj1 >= tj && (tj1 + tj2 - tj) % 2 == 0)
                    ++bi_size;
            }
        }
    }
    bindex.resize(bi_size, 3);
    zarray.resize(bi_size);
    int ct = 0;
    for (int tj = 0; tj <= twojmax_; ++tj)
    {
        for (int tj2 = 0; tj2 <= tj; ++tj2)
        {
            for (int tj1 = tj - tj2; tj1 <= tj2; ++tj1)
            {
                if (tj2 - tj1 <= tj && tj2 + tj1 >= tj && (tj1 + tj2 - tj) % 2 == 0)
                {
                    bindex(ct, 0) = tj1;
                    bindex(ct, 1) = tj2;
                    bindex(ct, 2) = tj;
                    zarray[ct].resize(tj + 1, tj + 1);
                    ++ct;
                }
            }
        }
    }
    uarray.resize(twojmax_ + 1);
    uarray_j.resize(twojmax_ + 1);
    for (int i = 0; i <= twojmax_; ++i)
    {
        uarray[i].resize(i + 1, i + 1);
        uarray_j[i].resize(i + 1, i + 1);
    }
    rootpqarr.resize(twojmax_ + 1, twojmax_ + 1);
    for (int i = 0; i <= twojmax_; ++i)
    {
        for (int j = 1; j <= twojmax_; ++j)
        {
            rootpqarr(i, j) = sqrt(static_cast<double>(i) / static_cast<double>(j));
        }
    }
    // Prime the CG cache once and keep immutable, contiguous nonzero terms.
    // Concurrent descriptor evaluation never inserts into the global cache.
    coupling_terms_.clear();coupling_offsets_.clear();reference_terms_=0;
    for (int feature=0;feature<bi_size;++feature) {
        coupling_offsets_.push_back(coupling_terms_.size());
        const int j1=bindex(feature,0),j2=bindex(feature,1),j=bindex(feature,2),shift=(j1+j2-j)/2;
        for(int m=0;m<=j;++m)for(int mp=0;mp<=j/2;++mp)
            for(int m2=0;m2<=j2;++m2) {
                const int m1=m-m2+shift;if(m1<0 || m1>j1)continue;
                const double cg=AngularFuncs::ClebschGordanTransCache(j1,j2,j,m1,m2,m);
                for(int m2p=0;m2p<=j2;++m2p) {
                    const int m1p=mp-m2p+shift;if(m1p<0 || m1p>j1)continue;
                    const double cgp=AngularFuncs::ClebschGordanTransCache(j1,j2,j,m1p,m2p,mp);
                    ++reference_terms_;
                    const double w=((j%2==0 && mp==j/2)?1.:2.)*cg*cgp;
                    if(w!=0.)coupling_terms_.push_back({SO4Density::index(coefficient_offsets_,j,m,mp),
                        SO4Density::index(coefficient_offsets_,j1,m1,m1p),
                        SO4Density::index(coefficient_offsets_,j2,m2,m2p),w});
                }
            }
    }
    coupling_offsets_.push_back(coupling_terms_.size());
    if (weights_.empty()) weights_.resize(10, 1.0);
    if (selfweights_.empty()) selfweights_.resize(10, 0.6);
}

Eigen::Vector4d Descriptor_BispectrumSO4::compute_coord(Eigen::RowVector3d &x, double r)
{
    Eigen::Vector4d coord;
    double phi, theta, theta0;
    phi = std::atan2(x(1), x(0));
    theta = std::acos(x(2) / r);
    theta0 = radial_angle_map_ == "atan" ? std::atan(r/radius_) : invz0_ * r;
    double stheta0 = radius_ * std::sin(theta0);
    double stheta1 = stheta0 * std::sin(theta);
    coord(0) = radius_ * std::cos(theta0);
    coord(1) = stheta0 * std::cos(theta);
    coord(2) = stheta1 * std::cos(phi);
    coord(3) = stheta1 * std::sin(phi);
    return coord;
}

void Descriptor_BispectrumSO4::add_uarray_self(int typei, uarrtype &uarr)
{
    double swig = selfweights_[typei];
    for (int i = 0; i < uarr.size(); ++i)
    {
        for (int m1 = 0; m1 < uarr[i].rows(); ++m1)
        {
            uarr[i](m1, m1) = swig*basis_scale(i);
        }
    }
}
// TODO: merge compute and add
/**
 * U(\theta0, \theta, \phi) = D(\alpha, \beta, \gamma)
 * a = D_{1/2,1/2}^{1/2}
 * b = D_{-1/2,1/2}^{1/2}
 * a* = D_{-1/2,-1/2}^{1/2}
 * b* = -D_{1/2,-1/2}^{1/2}
 *
 * D_{m1,m2,J} = sqrt((J-m1)/(J+m2)) * b * D_{m1+1/2, m2-1/2,J-1/2}
 *         + sqrt((J+m1)/(J+m2)) * a * D_{m1-1/2,m2-1/2,J-1/2}
 * D_{m1,m2,J} = sqrt((J-m1)/(J-m2)) * conj(a) * D_{m1+1/2,m2+1/2,J-1/2}
 *         + sqrt((J+m1)/(J-m2)) * conj(b) * D_{m1-1/2,m2+1/2,J-1/2}
 * Since m2 != -J, when m2 = J, we convert it using
 * D_{m1,m2,J} = (-1)^(m2-m1) * conj(D_{-m1,-m2,J})
 * Then we let m2 >= 0, the left case can be built from this
 * and from the iteration relation we have
 *  2*m2 >= (2*J) mod 2
 *  m2 <= J
 *  -J+1 <= m1 <= J - 1
 *
 * see VMK 4.6.3(13) or VMK 4.3 (2-5)
 *
 * If we use convension:
 *  j = 2*J
 *  ma = J + m1
 *  mb = J + m2
 * then we have
 *  0 <= j <= 2*J
 *  0 <= ma, mb <= j
 * and the symmetry of wigner U function becoms
 * W_{ma,mb,j} = (-1)^(ma - mb) * conj(W_{j-ma,j-mb,j})
 * and relation becoms
 *  W_{ma,mb,j} = sqrt((j - ma)/(2 * mb)) * b * W_{ma,mb-1,j-1}
 *        + sqrt(ma/mb) * a * W_{ma-1,mb-1,j-1}
 *  W_{ma,mb,j} = sqrt((j-ma)/(j-mb)) * conj(a) * W_{ma,mb,j-1}
 *        + sqrt((ma)/(j-mb)) * conj(b) * W_{ma-1,mb,j-1}
 * and
 *  W_{1,1,1} = a
 *  W_{0,1,1} = b
 *  W_{0,0,1} = a*
 *  -W_{1,0,1} = b*
 *
 * The final bispectrum compoments becoms:
 *  B_{j1,j2,j} = P_{J1,J2,J}
 *  J1 = 2*j1
 *  J2 = 2*j2
 *  J = 2*J
 * **/
void Descriptor_BispectrumSO4::compute_uarray(cmpld &acj, cmpld &bcj, uarrtype &uarr)
{
    // VMK 4.8.2 (14,15)
    uarr[0](0, 0) = 1;

    for (int j = 1; j <= twojmax_; ++j)
    {
        int offset = j / 2; // ma = offset .=. m1 = J%2
        // <= or < ??
        for (int mb = 0; mb <= offset; ++mb)
        {
            uarr[j](0, mb) = rootpqarr(j, j - mb) * acj * uarr[j - 1](0, mb);
            uarr[j](j, mb) = -rootpqarr(j, j - mb) * bcj * uarr[j - 1](j - 1, mb);
            for (int ma = 1; ma < j; ++ma)
            {
                uarr[j](ma, mb) = rootpqarr(j - ma, j - mb) * acj * uarr[j - 1](ma, mb) -
                                  rootpqarr(ma, j - mb) * bcj * uarr[j - 1](ma - 1, mb);
            }
        }
        // use symmetry to fill other blocks
        for (int mb = offset + 1; mb <= j; ++mb)
        {
            int sign = mb % 2 == 0 ? 1 : -1;
            for (int ma = 0; ma <= j; ++ma)
            {
                if (sign == 1)
                    uarr[j](ma, mb) = std::conj(uarr[j](j - ma, j - mb));
                else
                    uarr[j](ma, mb) = -std::conj(uarr[j](j - ma, j - mb));
                sign = -sign;
            }
        }
    }
}

void Descriptor_BispectrumSO4::compute_zarray(complexdsqmat &zarr, uarrtype &uarr, int j1, int j2, int j)
{
    zarr.setZero();
    int cj = (j1 + j2 - j) / 2;
    for (int m = 0; m <= j; ++m)
    {
        for (int m2 = 0; m2 <= j2; ++m2)
        {
            int m1 = m - m2 + cj;
            if (m1 < 0 || m1 > j1) continue;   // CG coeff would be zero
            for (int mp = 0; mp <= j / 2; ++mp)
            {
                for (int m2p = 0; m2p <= j2; ++m2p)
                {
                    int m1p = mp - m2p + cj;
                    if (m1p < 0 || m1p > j1) continue;  // CG coeff would be zero
                    zarr(m, mp) +=
                        AngularFuncs::ClebschGordanTransCache(j1, j2, j, m1, m2, m) *
                        AngularFuncs::ClebschGordanTransCache(j1, j2, j, m1p, m2p, mp) *
                        uarr[j1](m1, m1p) *
                        uarr[j2](m2, m2p);
                }
            }
        }
    }
}

// Differentiate the same recurrence as compute_uarray, then conjugate once
// all levels are complete. Each Cartesian component has independent storage.
void Descriptor_BispectrumSO4::compute_duarray_cj(cmpld &acj, cmpld &bcj, cmplarr3 &dacj, cmplarr3 &dbcj, uarrtype &uarr, duarrtype &duarr)
{
    duarr.resize(twojmax_ + 1);
    for (int j = 0; j <= twojmax_; ++j)
        for (auto &component : duarr[j])
            component.resize(j + 1, j + 1);
    for (int axis = 0; axis < 3; ++axis) {
        duarr[0][axis](0, 0) = COMPLEX_ZERO;
        for (int j = 1; j <= twojmax_; ++j) {
            auto &du = duarr[j][axis];
            const auto &prev = duarr[j - 1][axis];
            const int offset = j / 2;
            for (int mb = 0; mb <= offset; ++mb) {
                du(0, mb) = rootpqarr(j, j - mb) *
                    (uarr[j - 1](0, mb) * dacj[axis] + acj * prev(0, mb));
                du(j, mb) = -rootpqarr(j, j - mb) *
                    (uarr[j - 1](j - 1, mb) * dbcj[axis] + bcj * prev(j - 1, mb));
                for (int ma = 1; ma < j; ++ma) {
                    du(ma, mb) = rootpqarr(j - ma, j - mb) *
                        (uarr[j - 1](ma, mb) * dacj[axis] + acj * prev(ma, mb)) -
                        rootpqarr(ma, j - mb) *
                        (uarr[j - 1](ma - 1, mb) * dbcj[axis] + bcj * prev(ma - 1, mb));
                }
            }
            for (int mb = offset + 1; mb <= j; ++mb) {
                int sign = mb % 2 == 0 ? 1 : -1;
                for (int ma = 0; ma <= j; ++ma) {
                    du(ma, mb) = static_cast<double>(sign) * std::conj(du(j - ma, j - mb));
                    sign = -sign;
                }
            }
        }
    }
    for (auto &level : duarr)
        for (auto &component : level)
            component = component.conjugate().eval();
}

void Descriptor_BispectrumSO4::add_uarray(uarrtype &uarrj, uarrtype &uarr, double fcut)
{
    for (int j = 0; j <= twojmax_; ++j)
    {
        for (int ma = 0; ma <= j; ma++)
        {
            for (int mb = 0; mb <= j; mb++)
            {
                uarr[j](ma, mb) += basis_scale(j)*fcut * uarrj[j](ma, mb);
            }
        }
    }
}

void Descriptor_BispectrumSO4::compute_descriptors_derivative(Neighborhood &nbh, bi_type &ret, dbi_type &dret)
{
    using MatrixXcd = Eigen::Matrix<cmpld, Eigen::Dynamic, Eigen::Dynamic>;
    using UArray = std::vector<MatrixXcd>;
    using DUArray = std::array<UArray, 3>;

    auto make_uarray = [this]() {
        UArray arr;
        arr.reserve(twojmax_ + 1);
        for (int j = 0; j <= twojmax_; ++j) {
            arr.emplace_back(MatrixXcd::Zero(j + 1, j + 1));
        }
        return arr;
    };

    auto make_duarray = [&make_uarray]() {
        DUArray darr;
        for (int a = 0; a < 3; ++a) {
            darr[a] = make_uarray();
        }
        return darr;
    };

    auto compute_uarray_full = [this](const cmpld &acj, const cmpld &bcj, UArray &uarr) {
        uarr[0](0, 0) = COMPLEX_ONE;
        for (int j = 1; j <= twojmax_; ++j) {
            int offset = j / 2;
            for (int mb = 0; mb <= offset; ++mb) {
                uarr[j](0, mb) = rootpqarr(j, j - mb) * acj * uarr[j - 1](0, mb);
                uarr[j](j, mb) = -rootpqarr(j, j - mb) * bcj * uarr[j - 1](j - 1, mb);
                for (int ma = 1; ma < j; ++ma) {
                    uarr[j](ma, mb) =
                        rootpqarr(j - ma, j - mb) * acj * uarr[j - 1](ma, mb) -
                        rootpqarr(ma, j - mb) * bcj * uarr[j - 1](ma - 1, mb);
                }
            }
            for (int mb = offset + 1; mb <= j; ++mb) {
                int sign = mb % 2 == 0 ? 1 : -1;
                for (int ma = 0; ma <= j; ++ma) {
                    uarr[j](ma, mb) = sign == 1
                        ? std::conj(uarr[j](j - ma, j - mb))
                        : -std::conj(uarr[j](j - ma, j - mb));
                    sign = -sign;
                }
            }
        }
    };

    auto compute_duarray_full = [this](const cmpld &acj, const cmpld &bcj,
                                       const std::array<cmpld, 3> &dacj,
                                       const std::array<cmpld, 3> &dbcj,
                                       const UArray &uarr, DUArray &duarr) {
        for (int j = 1; j <= twojmax_; ++j) {
            int offset = j / 2;
            for (int mb = 0; mb <= offset; ++mb) {
                for (int a = 0; a < 3; ++a) {
                    duarr[a][j](0, mb) = rootpqarr(j, j - mb) *
                        (dacj[a] * uarr[j - 1](0, mb) + acj * duarr[a][j - 1](0, mb));
                    duarr[a][j](j, mb) = -rootpqarr(j, j - mb) *
                        (dbcj[a] * uarr[j - 1](j - 1, mb) + bcj * duarr[a][j - 1](j - 1, mb));
                    for (int ma = 1; ma < j; ++ma) {
                        duarr[a][j](ma, mb) =
                            rootpqarr(j - ma, j - mb) *
                                (dacj[a] * uarr[j - 1](ma, mb) + acj * duarr[a][j - 1](ma, mb)) -
                            rootpqarr(ma, j - mb) *
                                (dbcj[a] * uarr[j - 1](ma - 1, mb) + bcj * duarr[a][j - 1](ma - 1, mb));
                    }
                }
            }
            for (int mb = offset + 1; mb <= j; ++mb) {
                int sign = mb % 2 == 0 ? 1 : -1;
                for (int ma = 0; ma <= j; ++ma) {
                    for (int a = 0; a < 3; ++a) {
                        duarr[a][j](ma, mb) = sign == 1
                            ? std::conj(duarr[a][j](j - ma, j - mb))
                            : -std::conj(duarr[a][j](j - ma, j - mb));
                    }
                    sign = -sign;
                }
            }
        }
    };

    auto compute_zarray_full = [](const UArray &uarr, int j1, int j2, int j) {
        MatrixXcd zarr = MatrixXcd::Zero(j + 1, j + 1);
        const int cj = (j1 + j2 - j) / 2;
        for (int m = 0; m <= j; ++m) {
            for (int m2 = 0; m2 <= j2; ++m2) {
                const int m1 = m - m2 + cj;
                if (m1 < 0 || m1 > j1) continue;
                const double cg = AngularFuncs::ClebschGordanTransCache(j1, j2, j, m1, m2, m);
                for (int mp = 0; mp <= j; ++mp) {
                    for (int m2p = 0; m2p <= j2; ++m2p) {
                        const int m1p = mp - m2p + cj;
                        if (m1p < 0 || m1p > j1) continue;
                        const double cgp = AngularFuncs::ClebschGordanTransCache(j1, j2, j, m1p, m2p, mp);
                        zarr(m, mp) += cg * cgp * uarr[j1](m1, m1p) * uarr[j2](m2, m2p);
                    }
                }
            }
        }
        return zarr;
    };

    auto compute_dzarray_full = [](const UArray &uarr, const UArray &duarr, int j1, int j2, int j) {
        MatrixXcd dzarr = MatrixXcd::Zero(j + 1, j + 1);
        const int cj = (j1 + j2 - j) / 2;
        for (int m = 0; m <= j; ++m) {
            for (int m2 = 0; m2 <= j2; ++m2) {
                const int m1 = m - m2 + cj;
                if (m1 < 0 || m1 > j1) continue;
                const double cg = AngularFuncs::ClebschGordanTransCache(j1, j2, j, m1, m2, m);
                for (int mp = 0; mp <= j; ++mp) {
                    for (int m2p = 0; m2p <= j2; ++m2p) {
                        const int m1p = mp - m2p + cj;
                        if (m1p < 0 || m1p > j1) continue;
                        const double cgp = AngularFuncs::ClebschGordanTransCache(j1, j2, j, m1p, m2p, mp);
                        dzarr(m, mp) += cg * cgp *
                            (duarr[j1](m1, m1p) * uarr[j2](m2, m2p) +
                             uarr[j1](m1, m1p) * duarr[j2](m2, m2p));
                    }
                }
            }
        }
        return dzarr;
    };

    ret.assign(bi_size, 0.0);
    dret.resize(bi_size);
    for (auto &it : dret) {
        it.resize(nbh.nbhcount_, 3);
        it.setZero();
    }

    UArray total_u = make_uarray();
    const int typei = nbh.centertype;
    const double swig = selfweights_[typei];
    for (int j = 0; j <= twojmax_; ++j) {
        for (int ma = 0; ma <= j; ++ma) {
            total_u[j](ma, ma) = swig*basis_scale(j);
        }
    }

    std::vector<DUArray> total_du;
    total_du.reserve(nbh.nbhcount_);

    for (int i = 0; i < nbh.nbhcount_; ++i) {
        const int typej = nbh.chemmap_[i];
        const double r = nbh.distances_[i];
        const double x = nbh.pos_[i](0);
        const double y = nbh.pos_[i](1);
        const double z = nbh.pos_[i](2);
        if (!(r>0.)) throw std::invalid_argument("SO4 requires nonzero neighbor distance");
        using namespace SO4Density;
        const auto jx=FirstOrder::variable(x,0), jy=FirstOrder::variable(y,1), jz=FirstOrder::variable(z,2);
        const auto jr=SO4Density::sqrt(jx*jx+jy*jy+jz*jz);
        FirstOrder ar,ai,br,bi;
        SO4Density::parameters(jx,jy,jz,jr,radius_,convention_,radial_angle_map_=="atan",ar,ai,br,bi);
        const cmpld acj(ar.value,ai.value),bcj(br.value,bi.value);
        std::array<cmpld,3> dacj,dbcj;
        for (int a=0;a<3;++a) {
            dacj[a]=cmpld(ar.gradient[a],ai.gradient[a]);
            dbcj[a]=cmpld(br.gradient[a],bi.gradient[a]);
        }
        const double fcut = cutfun_.fcut(r)*weights_[typej];
        const double dfcutdr = cutfun_.dfcut(r)*weights_[typej];
        const std::array<double,3> dfcut={dfcutdr*x/r,dfcutdr*y/r,dfcutdr*z/r};

        UArray atom_u = make_uarray();
        compute_uarray_full(acj, bcj, atom_u);

        DUArray atom_du = make_duarray();
        compute_duarray_full(acj, bcj, dacj, dbcj, atom_u, atom_du);

        DUArray neighbor_du = make_duarray();
        for (int j = 0; j <= twojmax_; ++j) {
            total_u[j] += basis_scale(j)*fcut * atom_u[j];
            for (int a = 0; a < 3; ++a) {
                neighbor_du[a][j] = basis_scale(j)*(dfcut[a] * atom_u[j] + fcut * atom_du[a][j]);
            }
        }
        total_du.push_back(std::move(neighbor_du));
    }

    for (int cbi = 0; cbi < bi_size; ++cbi) {
        const int j1 = bindex(cbi, 0);
        const int j2 = bindex(cbi, 1);
        const int j = bindex(cbi, 2);
        const MatrixXcd zarr = compute_zarray_full(total_u, j1, j2, j);

        for (int m = 0; m <= j; ++m) {
            for (int mp = 0; mp <= j / 2; ++mp) {
                const double factor = (j % 2 == 0 && mp == j / 2) ? 1.0 : 2.0;
                ret[cbi] += factor * std::real(std::conj(total_u[j](m, mp)) * zarr(m, mp));
            }
        }

        for (int k = 0; k < nbh.nbhcount_; ++k) {
            for (int a = 0; a < 3; ++a) {
                const MatrixXcd dzarr = compute_dzarray_full(total_u, total_du[k][a], j1, j2, j);
                double db = 0.0;
                for (int m = 0; m <= j; ++m) {
                    for (int mp = 0; mp <= j / 2; ++mp) {
                        const double factor = (j % 2 == 0 && mp == j / 2) ? 1.0 : 2.0;
                        db += factor * std::real(std::conj(total_du[k][a][j](m, mp)) * zarr(m, mp) +
                                                 std::conj(total_u[j](m, mp)) * dzarr(m, mp));
                    }
                }
                dret[cbi](k, a) = db;
            }
        }
    }
}

void Descriptor_BispectrumSO4::compute_descriptors(Neighborhood &nbh, bi_type &ret, dbi_type *dret)
{
    if (uses_snap() || spline_) { compute_snap(nbh,ret,dret);return; }
    ret.clear();
    ret.resize(bi_size);
    if (dret != nullptr) {
        compute_descriptors_derivative(nbh, ret, *dret);
        return;
    }
    std::for_each(uarray.begin(), uarray.end(), [](complexdsqmat &a)
                  { a.setZero(); });
    const int &typei = nbh.centertype;
    add_uarray_self(typei, uarray);
    // do not parallize this loop using shared memory method
    for (int i = 0; i < nbh.nbhcount_; ++i)
    {
        int typej = nbh.chemmap_[i];
        double theta0 = invz0_ * nbh.distances_[i];
        double &r = nbh.distances_[i];
        const double &x = nbh.pos_[i](0);
        const double &y = nbh.pos_[i](1);
        const double &z = nbh.pos_[i](2);
        const double ux = x / r;
        const double uy = y / r;
        const double uz = z / r;
        double fcut = cutfun_.fcut(r) * weights_[typej];
        if (!(r>0.)) throw std::invalid_argument("SO4 requires nonzero neighbor distance");
        double ar,ai,br,bi;
        SO4Density::parameters(x,y,z,r,radius_,convention_,radial_angle_map_=="atan",ar,ai,br,bi);
        cmpld acj(ar,ai), bcj(br,bi);
        compute_uarray(acj, bcj, uarray_j);
        add_uarray(uarray_j, uarray, fcut);
    }
    // compute B{j1,j2,j}
    /**
     * Firstly we compute Z_{m1,m2,j,j1,j2}
     * See Tompson's paper, the right part of eq(5),
     *  since m1, m2 determines m, we can write the formula as
     * B_{j1,j2,j} = \sum_{m,mp=0}^{j} conj(u_{m,mp}^j) * Z_{m,mp,j,j1,j2}
     * Z_{m,mp,j,j1,j2} = \sum_{m1,m1p=0}^{j1} \sum_{m2,m2p=0}^{j2} C_{m1,m2,m}^{j1,j2,j} * C_{m1p,m2p,mp}^{j1,j2,j} * u_{m1,m1p}^j1 * u_{m2,m2p}^j2
     * All formula above adopt the current convension in code.
     *
     * We notice that:
     *  Z_{m,mp,j,j1,j2} = (-1)^(m-mp) * Z*_{j-m,j-mp,j,j1,j2}
     * thus we can only hold half of Z to save memory and computational cost
     * by tracking mp <= j/2, and we can simplely prove that Bi is real number
     *
     * we have
     *  dBidRj = \sum_{m,mp} conj(du_{m,mp,j}) * Z_{m,mp,j,j1,j2}
     *       + (j+1)/(j1+1) * \sum_{m1,m1p} conj(du_{m1,m1p,j}) * Z_{m1,m1p,j1,j,j2}
     *       + (j+1)/(j2+1) * \sum_{m2,m2p} conj(du_{m2,m2p,j}) * Z_{m2,m2p,j2,j1,j}
     * TODO: optimize dBidRj process
     * */
    for (int cbi = 0; cbi < bi_size; ++cbi)
    {
        int j1 = bindex(cbi, 0);
        int j2 = bindex(cbi, 1);
        int j = bindex(cbi, 2);
        compute_zarray(zarray[cbi], uarray, j1, j2, j);
    }
    // Next, we compute B_{j1,j2,j}
    // B_{j1,j2,j} = \sum_{m,mp=0}^j conj(u_{m,mp}^j) * Z_{m1,m2}^{j1,j2,j}
    for (int cbi = 0; cbi < bi_size; ++cbi)
    {
        int j = bindex(cbi, 2);
        // For even j the whole middle column is already present; count it once.
        for (int m1 = 0; m1 <= j; ++m1)
        {
            for (int m2 = 0; m2 <= j / 2; ++m2)
            {
                const double factor = (j % 2 == 0 && m2 == j / 2) ? 1.0 : 2.0;
                ret[cbi] += factor * (uarray[j](m1, m2).real() * zarray[cbi](m1, m2).real() +
                                   uarray[j](m1, m2).imag() * zarray[cbi](m1, m2).imag());
            }
        }
    }
}

std::vector<double> Descriptor_BispectrumSO4::compute_descriptors_powerspectrum(Neighborhood &nbh)
{
    std::vector<double> ret(twojmax_ + 1, 0);
    std::for_each(uarray.begin(), uarray.end(), [](complexdsqmat &a)
                  { a.setZero(); });
    const int &typei = nbh.centertype;
    add_uarray_self(typei, uarray);
    // do not parallize this loop using shared memory method
    for (int i = 0; i < nbh.nbhcount_; ++i)
    {
        int typej = nbh.chemmap_[i];
        double theta0 = invz0_ * nbh.distances_[i];
        double &r = nbh.distances_[i];
        const double &x = nbh.pos_[i](0);
        const double &y = nbh.pos_[i](1);
        const double &z = nbh.pos_[i](2);
        const double ux = x / r;
        const double uy = y / r;
        const double uz = z / r;
        double fcut = cutfun_.fcut(r) * weights_[typej];
        if (!(r>0.)) throw std::invalid_argument("SO4 requires nonzero neighbor distance");
        double ar,ai,br,bi;
        SO4Density::parameters(x,y,z,r,radius_,convention_,radial_angle_map_=="atan",ar,ai,br,bi);
        cmpld a(ar,ai), b(br,bi);
        compute_uarray(a, b, uarray_j);
        add_uarray(uarray_j, uarray, fcut);
    }
    for (int j = 0; j <= twojmax_; ++j)
        for (int m1 = 0; m1 <= j; ++m1)
            for (int m2 = 0; m2 <= j; ++m2)
                ret[j] += (uarray[j](m1, m2)).real() * (uarray[j](m1, m2)).real() +
                          (uarray[j](m1, m2)).imag() * (uarray[j](m1, m2)).imag();
    return ret;
}

std::vector<int> Descriptor_BispectrumSO4::get_bi_ind(int ind) const
{
    std::vector<int> ret(3);
    ret[0] = bindex(ind, 0);
    ret[1] = bindex(ind, 1);
    ret[2] = bindex(ind, 2);
    return ret;
}

void Descriptor_BispectrumSO4::compute(Neighborhood &nbh, Eigen::VectorXd &out)
{
    bi_type bi;
    compute_descriptors(nbh, bi);
    out.resize(bi_size);
    for (int i = 0; i < bi_size; ++i)
        out[i] = bi[i];
}

void Descriptor_BispectrumSO4::compute(Neighborhood &nbh, Eigen::VectorXd &out,
                                        const ComputeOptions &opts)
{
    bi_type bi;
    dbi_type dbi;
    bool need_der = opts.compute_derivative && opts.gradient;

    compute_descriptors(nbh, bi, need_der ? &dbi : nullptr);

    out.resize(bi_size);
    for (int i = 0; i < bi_size; ++i)
        out[i] = bi[i];

    if (need_der) {
        auto &grad = opts.gradient->grad;
        grad.resize(bi_size, nbh.nbhcount_, 3);
        for (int f = 0; f < bi_size; ++f) {
            for (int j = 0; j < nbh.nbhcount_; ++j) {
                for (int a = 0; a < 3; ++a) {
                    grad(f, j, a) = dbi[f](j, a);
                }
            }
        }
    }
}
