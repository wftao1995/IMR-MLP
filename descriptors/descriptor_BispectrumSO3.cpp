#include "descriptor_BispectrumSO3.h"
#include "tools/tools.h"
#include "solid_harmonic_curvature.h"
#include "mathextra/real_spherical_recurrence.h"
#include <tuple>
#include <array>
#include <algorithm>

#define CIDX(n, l, m) ((n)*ysize + (l) * (l + 1) + (m))
#define YIDX(l, m) ((m) + (l) + ((l) * (l)))

using namespace std::literals;
using namespace std::complex_literals;

namespace {
// The polar formula obtains sin(theta) from sqrt(1-(z/r)^2), which loses
// relative precision near the axis. Use the equivalent Cartesian recurrence
// there for both values and gradients so energy differences remain smooth.
bool cartesian_polar_harmonics(AngularFuncs::ComplexSphericalHarmoics &sph,
                              const Eigen::RowVector3d &position, int maximum_l,
                              bool derivatives) {
    const double radius_squared = position.squaredNorm();
    if (!(radius_squared > 0.) || position.head<2>().squaredNorm() >= 1e-6*radius_squared)
        return false;
    const auto values = DescriptorCurvature::solid_harmonics(position,maximum_l,true);
    for (int index = 0; index < static_cast<int>(values.size()); ++index) {
        const auto &value = values[index];
        sph.Y[index] = value.real.value;
        sph.Yi[index] = value.imag.value;
        if (derivatives) {
            sph.dYdx[index] = value.real.gradient[0];
            sph.dYdy[index] = value.real.gradient[1];
            sph.dYdz[index] = value.real.gradient[2];
            sph.dYidx[index] = value.imag.gradient[0];
            sph.dYidy[index] = value.imag.gradient[1];
            sph.dYidz[index] = value.imag.gradient[2];
        }
    }
    return true;
}
}

Descriptor_BispectrumSO3::Descriptor_BispectrumSO3(int _lmax, double rmin, double rmax, int rsize,bool real_harmonics)
{
    real_harmonics_=real_harmonics;
    l_max = _lmax;
    ysize = (l_max + 1) * (l_max + 1);
    r_min = rmin;
    r_max = rmax;
    r_size = rsize;
    cutoff_ = rmax;
    sph.init(l_max, l_max);
    radical_basis = new RBasis_PolyB(rmin, rmax, rsize);
    normalize_factor.resize(l_max + 1);
    for (int i = 0; i <= l_max; ++i)
    {
        normalize_factor[i] = 1.0 / sqrt(2.0 * i + 1.0); //TODO: or sqrt(1.0 / (2.0 * i + 1.0))
    }
    // bsize
    b_size = 0;
    for (int n = 0; n < r_size; ++n)
        for (int l = 0; l <= l_max; ++l)
        {
            for (int l2 = 0; l2 <= l; ++l2)
            {
                for (int l1 = 0; l1 <= l2; ++l1)
                {
                    b_size++;
                }
            }
        }
    bQsize = 0;
    for (int n1 = 0; n1 < r_size; ++n1)
    {
        for (int n2 = 0; n2 < r_size; ++n2)
        {
            for (int l = 0; l <= l_max; ++l)
            {
                for (int l1 = 0; l1 <= l_max; ++l1)
                {
                    for (int l2 = 0; l2 <= l_max; ++l2)
                    {
                        ++bQsize;
                    }
                }
            }
        }
    }
    bvector.resize(b_size);
    bQvector.resize(bQsize);
    b_index.resize(b_size, 4);
    zarray.resize(b_size);
    bQ_index.resize(bQsize, 5);
    int ct = 0;
    for (int n = 0; n < r_size; ++n)
        for (int l = 0; l <= l_max; ++l)
        {
            for (int l2 = 0; l2 <= l; ++l2)
            {
                for (int l1 = 0; l1 <= l2; ++l1)
                {
                    b_index(ct, 0) = n;
                    b_index(ct, 1) = l1;
                    b_index(ct, 2) = l2;
                    b_index(ct, 3) = l;
                    zarray[ct].resize(2 * l + 1);
                    ++ct;
                }
            }
        }
    ct = 0;
    for (int n1 = 0; n1 < r_size; ++n1)
    {
        for (int n2 = 0; n2 < r_size; ++n2)
        {
            for (int l = 0; l <= l_max; ++l)
            {
                for (int l1 = 0; l1 <= l_max; ++l1)
                {
                    for (int l2 = 0; l2 <= l_max; ++l2)
                    {
                        bQ_index(ct, 0) = n1;
                        bQ_index(ct, 1) = n2;
                        bQ_index(ct, 2) = l;
                        bQ_index(ct, 3) = l1;
                        bQ_index(ct, 4) = l2;
                        ++ct;
                    }
                }
            }
        }
    }
    carray.resize(r_size);
    for (int i = 0; i < r_size; ++i)
    {
        carray[i].resize(ysize); // using c_{n,l,-m} = (-1)^m * c*_{n,l,m} to reduce memory and computational cost
    }
    if(real_harmonics_) initialize_real_couplings();
};
// compute and add
void Descriptor_BispectrumSO3::compute_carray(carr_type &carr, Eigen::RowVector3d &coord)
{
    const double r = coord.norm();
    const double x = coord[0];
    const double y = coord[1];
    const double z = coord[2];
    // const double costheta = z / r; // since costheta can be both positive and negative, sintheta must be positive
    if (!cartesian_polar_harmonics(sph,coord,l_max,compute_der)) {
        if (compute_der)
            sph.computedYCar(x, y, z, r);
        else
            sph.computeYCarSafe(x, y, z, r);
    }
    if (compute_der)
        radical_basis->calc_der(r);
    else
        radical_basis->calc_val(r);
    for (int n = 0; n < r_size; ++n)
    {
        for (int l = 0; l <= l_max; ++l)
        {
            for (int m = -l; m <= l; ++m)
            {
                //carr[n][YIDX(l, m)] += radical_basis->vals[n] * sph.Y[YIDX(l, m)];
                carr[n][YIDX(l, m)] += radical_basis->vals[n] * sph.get(l, m);
            }
        }
    }
}

void Descriptor_BispectrumSO3::compute_zarray(std::vector<std::complex<double>> &zarr, std::vector<std::complex<double>> &carr, int l1, int l2, int l)
{
    for (int m = -l; m <= l; ++m)
    {
        zarr[l + m] = COMPLEX_ZERO;
        for (int m1 = -l1; m1 <= l1; ++m1)
        {
            int m2 = m - m1;
            if (abs(m2) <= l2)
                zarr[l + m] += AngularFuncs::ClebschGordanInt(l1, l2, l, m1, m2, m) * carr[YIDX(l1, m1)] * carr[YIDX(l2, m2)];
        }
    }
}

Descriptor_BispectrumSO3::~Descriptor_BispectrumSO3()
{
    delete radical_basis;
}

Descriptor_BispectrumSO3::b_type Descriptor_BispectrumSO3::compute(Neighborhood &nbh)
{
    if(real_harmonics_) {
        Eigen::VectorXd out;compute_real(nbh,out);b_type result(b_size);
        for(int i=0;i<b_size;++i) result[i]={out[2*i],out[2*i+1]};
        return result;
    }
    b_type bret(b_size, 0);
    for (int i = 0; i < carray.size(); ++i)
    {
        for (int j = 0; j < carray[i].size(); ++j)
        {
            carray[i][j] = 0;
        }
    }
    // compute carray
    for (int i = 0; i < nbh.nbhcount_; ++i)
    {
        compute_carray(carray, nbh.pos_[i]);
    }
    // compute zarray and b_{}
    for (int i = 0; i < b_index.rows(); i++)
    {
        const int n = b_index(i, 0);
        const int l1 = b_index(i, 1);
        const int l2 = b_index(i, 2);
        const int l = b_index(i, 3);
        compute_zarray(zarray[i], carray[n], l1, l2, l);
        // compute b
        for (int m = -l; m <= l; ++m)
        {
            bret[i] += std::conj(carray[n][YIDX(l, m)]) * zarray[i][l + m];
            // _DEBUG_OUTPUT_(std::conj(carray[n][YIDX(l, m)]));
            // _DEBUG_OUTPUT_(zarray[i][l+m]);
        }
    }
    return bret;
}

std::vector<int> Descriptor_BispectrumSO3::get_bi_ind(int ind) const
{
    std::vector<int> ret(4);
    ret[0] = b_index(ind, 0);
    ret[1] = b_index(ind, 1);
    ret[2] = b_index(ind, 2);
    ret[3] = b_index(ind, 3);
    return ret;
}

void Descriptor_BispectrumSO3::compute(Neighborhood &nbh, Eigen::VectorXd &out)
{
    if(real_harmonics_) {compute_real(nbh,out);return;}
    auto b = compute(nbh); // calls existing b_type compute(Neighborhood&)
    out.resize(b_size * 2);
    for (int i = 0; i < b_size; ++i) {
        out[2 * i]     = b[i].real();
        out[2 * i + 1] = b[i].imag();
    }
}

void Descriptor_BispectrumSO3::compute(Neighborhood &nbh, Eigen::VectorXd &out,
                                       const ComputeOptions &opts)
{
    if(real_harmonics_) {compute_real(nbh,out,opts.compute_derivative ? opts.gradient : nullptr);return;}
    // If gradient not requested, delegate to the 2-arg version
    if (!opts.compute_derivative || !opts.gradient) {
        compute(nbh, out);
        return;
    }

    // ---- Step 1: Normal bispectrum computation (with derivative flag on) ----
    compute_der = true;

    b_type bret(b_size, 0);
    for (int i = 0; i < static_cast<int>(carray.size()); ++i)
        for (int j = 0; j < static_cast<int>(carray[i].size()); ++j)
            carray[i][j] = 0;

    for (int i = 0; i < nbh.nbhcount_; ++i)
        compute_carray(carray, nbh.pos_[i]);

    for (int i = 0; i < b_index.rows(); i++)
    {
        const int n  = b_index(i, 0);
        const int l1 = b_index(i, 1);
        const int l2 = b_index(i, 2);
        const int l  = b_index(i, 3);
        compute_zarray(zarray[i], carray[n], l1, l2, l);
        for (int m = -l; m <= l; ++m)
            bret[i] += std::conj(carray[n][YIDX(l, m)]) * zarray[i][l + m];
    }

    out.resize(b_size * 2);
    for (int i = 0; i < b_size; ++i) {
        out[2 * i]     = bret[i].real();
        out[2 * i + 1] = bret[i].imag();
    }

    // ---- Step 2: Compute position gradient ----
    auto &grad = opts.gradient->grad;
    grad.resize(b_size * 2, nbh.nbhcount_, 3);
    grad.setZero();

    for (int i_nbh = 0; i_nbh < nbh.nbhcount_; ++i_nbh)
    {
        const auto &pos = nbh.pos_[i_nbh];
        const double r  = pos.norm();
        const double invr = 1.0 / r;
        const double drdx = pos.x() * invr;
        const double drdy = pos.y() * invr;
        const double drdz = pos.z() * invr;

        // Compute per-neighbor spherical harmonics (Y + derivatives) and
        // radial-basis values + derivatives.
        if (!cartesian_polar_harmonics(sph,pos,l_max,true))
            sph.computedYCar(pos.x(), pos.y(), pos.z(), r);
        radical_basis->calc_der(r);

        // ---------- Pre-compute dc/d{x,y,z} for every (n,l,m) ----------
        // dc/dp^{(j)}_{n,l,m}  =  f'_n(r) * (dr/dp) * Y_{l,m}
        //                         + f_n(r) * dY_{l,m}/dp
        std::vector<std::vector<std::complex<double>>> dcdx(r_size);
        std::vector<std::vector<std::complex<double>>> dcdy(r_size);
        std::vector<std::vector<std::complex<double>>> dcdz(r_size);
        for (int n = 0; n < r_size; ++n)
        {
            dcdx[n].resize(ysize);
            dcdy[n].resize(ysize);
            dcdz[n].resize(ysize);
            const double fn  = radical_basis->vals[n];
            const double dfn = radical_basis->ders[n];
            for (int l = 0; l <= l_max; ++l)
            {
                for (int m = -l; m <= l; ++m)
                {
                    const int yidx = YIDX(l, m);
                    std::complex<double> Ylm(sph.Y[yidx], sph.Yi[yidx]);
                    std::complex<double> dYdx(sph.dYdx[yidx], sph.dYidx[yidx]);
                    std::complex<double> dYdy(sph.dYdy[yidx], sph.dYidy[yidx]);
                    std::complex<double> dYdz(sph.dYdz[yidx], sph.dYidz[yidx]);

                    dcdx[n][yidx] = dfn * drdx * Ylm + fn * dYdx;
                    dcdy[n][yidx] = dfn * drdy * Ylm + fn * dYdy;
                    dcdz[n][yidx] = dfn * drdz * Ylm + fn * dYdz;
                }
            }
        }

        // ---------- Chain rule for every bispectrum component ----------
        for (int i_b = 0; i_b < b_size; ++i_b)
        {
            const int n  = b_index(i_b, 0);
            const int l1 = b_index(i_b, 1);
            const int l2 = b_index(i_b, 2);
            const int l  = b_index(i_b, 3);

            std::complex<double> db_dx(0, 0), db_dy(0, 0), db_dz(0, 0);

            // --- Part 1:  sum_m  conj(dc_{n,l,m}/dp) * z_i(m) ---
            for (int m = -l; m <= l; ++m)
            {
                const int yidx = YIDX(l, m);
                const int zidx = l + m;

                db_dx += std::conj(dcdx[n][yidx]) * zarray[i_b][zidx];
                db_dy += std::conj(dcdy[n][yidx]) * zarray[i_b][zidx];
                db_dz += std::conj(dcdz[n][yidx]) * zarray[i_b][zidx];
            }

            // --- Part 2:  sum_m  conj(c_{n,l,m}) * dz_i(m)/dp ---
            // dz_i(m)/dp  =  sum_{m1} CG * dc_{l1,m1}/dp * c_{l2,m-m1}
            //              + sum_{m1} CG * c_{l1,m1} * dc_{l2,m-m1}/dp
            for (int m = -l; m <= l; ++m)
            {
                const int zidx = l + m;
                std::complex<double> dz_dx(0, 0), dz_dy(0, 0), dz_dz(0, 0);

                for (int m1 = -l1; m1 <= l1; ++m1)
                {
                    const int m2 = m - m1;
                    if (std::abs(m2) > l2)
                        continue;

                    const double cg = AngularFuncs::ClebschGordanInt(l1, l2, l, m1, m2, m);
                    const int yidx1 = YIDX(l1, m1);
                    const int yidx2 = YIDX(l2, m2);

                    // term = CG * (dc_{l1,m1}/dp * c_{l2,m2} + c_{l1,m1} * dc_{l2,m2}/dp)
                    dz_dx += cg * (dcdx[n][yidx1] * carray[n][yidx2]
                                   + carray[n][yidx1] * dcdx[n][yidx2]);
                    dz_dy += cg * (dcdy[n][yidx1] * carray[n][yidx2]
                                   + carray[n][yidx1] * dcdy[n][yidx2]);
                    dz_dz += cg * (dcdz[n][yidx1] * carray[n][yidx2]
                                   + carray[n][yidx1] * dcdz[n][yidx2]);
                }

                const int yidx_lm = YIDX(l, m);
                db_dx += std::conj(carray[n][yidx_lm]) * dz_dx;
                db_dy += std::conj(carray[n][yidx_lm]) * dz_dy;
                db_dz += std::conj(carray[n][yidx_lm]) * dz_dz;
            }

            // Store in gradient tensor  (feature, neighbor, xyz)
            grad(2 * i_b,     i_nbh, 0) = db_dx.real();
            grad(2 * i_b + 1, i_nbh, 0) = db_dx.imag();
            grad(2 * i_b,     i_nbh, 1) = db_dy.real();
            grad(2 * i_b + 1, i_nbh, 1) = db_dy.imag();
            grad(2 * i_b,     i_nbh, 2) = db_dz.real();
            grad(2 * i_b + 1, i_nbh, 2) = db_dz.imag();
        }
    }

    compute_der = false;
}


// T maps real tesseral density coefficients to the public complex convention.
// Complex numbers occur only once while transforming Clebsch--Gordan tables;
// every subsequent density, contraction and derivative operation is real.
void Descriptor_BispectrumSO3::initialize_real_couplings() {
    using Complex=std::complex<double>;
    const auto transform=[](int m) {
        std::vector<std::pair<int,Complex>> terms;
        if(m==0) terms.push_back({0,1.});
        else {
            const int a=std::abs(m);const double factor=(m<0 && a%2 ? -1. : 1.)/std::sqrt(2.);
            terms.push_back({a,factor});terms.push_back({-a,Complex(0.,m>0 ? factor : -factor)});
        }
        return terms;
    };
    for(int feature=0;feature<b_index.rows();++feature) {
        const int n=b_index(feature,0),l1=b_index(feature,1),l2=b_index(feature,2),l=b_index(feature,3);
        std::map<std::tuple<int,int,int>,Complex> transformed;
        for(int m=-l;m<=l;++m) for(int m1=-l1;m1<=l1;++m1) {
            const int m2=m-m1;if(std::abs(m2)>l2) continue;
            const double cg=AngularFuncs::ClebschGordanInt(l1,l2,l,m1,m2,m);
            if(cg==0.) continue;
            ++complex_coupling_count_;
            for(const auto& [a,ta]:transform(m)) for(const auto& [b,tb]:transform(m1)) for(const auto& [c,tc]:transform(m2)) {
                std::array<int,3> indices{n*ysize+l*(l+1)+a,n*ysize+l1*(l1+1)+b,n*ysize+l2*(l2+1)+c};
                std::sort(indices.begin(),indices.end());
                transformed[{indices[0],indices[1],indices[2]}]+=cg*std::conj(ta)*tb*tc;
            }
        }
        for(const auto& [indices,weight]:transformed) if(weight!=Complex{}) {
            const auto [a,b,c]=indices;
            real_couplings_.push_back({feature,a,b,c,weight.real(),weight.imag()});
        }
    }
}

void Descriptor_BispectrumSO3::real_density(Neighborhood& nbh,Eigen::VectorXd& density,
    std::vector<std::vector<DescriptorCurvature::Jet6>>* derivatives) {
    using DescriptorCurvature::Jet6;
    density.setZero(r_size*ysize);
    if(derivatives) derivatives->assign(nbh.nbhcount_,std::vector<Jet6>(r_size*ysize));
    std::vector<double> radial_value(r_size),radial_first(r_size),radial_second(r_size);
    for(int neighbor=0;neighbor<nbh.nbhcount_;++neighbor) {
        const auto& position=nbh.pos_[neighbor];const double radius=position.norm();
        if(!(radius>0.)) GPERROR("SO3 real spherical harmonics require nonzero neighbor distances");
        if(derivatives) {
            const auto x=Jet6::variable(position[0],0),y=Jet6::variable(position[1],1),z=Jet6::variable(position[2],2);
            const auto r=DescriptorCurvature::sqrt(x*x+y*y+z*z);
            const auto harmonics=AngularFuncs::detail::real_solid_harmonics(l_max,l_max,x/r,y/r,z/r);
            radical_basis->calc_derder(radius,radial_value,radial_first,radial_second);
            for(int n=0;n<r_size;++n) {
                const auto radial=DescriptorCurvature::compose(r,radial_value[n],radial_first[n],radial_second[n]);
                for(int lm=0;lm<ysize;++lm) {
                    const auto coefficient=radial*harmonics[lm];
                    density[n*ysize+lm]+=coefficient.value;(*derivatives)[neighbor][n*ysize+lm]=coefficient;
                }
            }
        } else {
            const auto harmonics=AngularFuncs::detail::real_solid_harmonics(l_max,l_max,
                position[0]/radius,position[1]/radius,position[2]/radius);
            radical_basis->calc_val(radius,radial_value);
            for(int n=0;n<r_size;++n) for(int lm=0;lm<ysize;++lm) density[n*ysize+lm]+=radial_value[n]*harmonics[lm];
        }
    }
}

void Descriptor_BispectrumSO3::compute_real(Neighborhood& nbh,Eigen::VectorXd& out,DescriptorGradient* gradient) {
    Eigen::VectorXd density;std::vector<std::vector<DescriptorCurvature::Jet6>> local;
    real_density(nbh,density,gradient ? &local : nullptr);
    out.setZero(feature_size());
    if(gradient) {gradient->grad.resize(feature_size(),nbh.nbhcount_,3);gradient->grad.setZero();}
    for(const auto& term:real_couplings_) {
        const double a=density[term.a],b=density[term.b],c=density[term.c];
        out[2*term.feature]+=term.real*a*b*c;out[2*term.feature+1]+=term.imag*a*b*c;
        if(gradient) for(int neighbor=0;neighbor<nbh.nbhcount_;++neighbor) for(int axis=0;axis<3;++axis) {
            const double derivative=local[neighbor][term.a].gradient[axis]*b*c
                +a*local[neighbor][term.b].gradient[axis]*c+a*b*local[neighbor][term.c].gradient[axis];
            gradient->grad(2*term.feature,neighbor,axis)+=term.real*derivative;
            gradient->grad(2*term.feature+1,neighbor,axis)+=term.imag*derivative;
        }
    }
}
