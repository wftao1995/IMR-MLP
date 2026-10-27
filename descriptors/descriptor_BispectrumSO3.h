#ifndef IMRMLP_DESCRIPTOR_BISPECTRUM_SO3_H
#define IMRMLP_DESCRIPTOR_BISPECTRUM_SO3_H

#include "base/structures.h"
#include "descriptors.h"
#include "CutoffFuncs.h"
#include "mathextra/mathextra.h"
#include <Eigen/Core>
#include <complex>
#include <unordered_map>
#include <map>
#include "base/anyneighborhoods.h"
#include "basis/basis.h"
#include "local_curvature_jet.h"

/**
 * This is bispectrum SO(3) impl of descriptor based on bartok's paper.
 * SOAP is not included here.
 *
 * The bispectrum in SO(3) is
 * @f$b_{n,l,l1,l2} = \sum_{m=-l}^{l}\sum_{m_1=-l_1}^{l_1}\sum_{m_2=-l_2}^{l_2}
 *                  c^*_{nlm} \cdot C_{m,m1,m2}^{l,l1,l2} \cdot c_{n,l_1,m_1} \cdot c_{n,l_2,m_2}@f$
 * And a symmetry is present:
 * @f$b_{n,l,l1,l2} / \sqrt{2l+1} = (-1)^{l2} \cdot b_{n,l1,l,l2} / \sqrt{2l_1+1}@f$
 *                               @f$= (-1)^{l1} \cdot b_{n,l2,l1,l} / \sqrt{2l_2+1}@f$
 * hence we only need to track l1 <= l2 <= l
 *
 * If we define
 * @f$z_{m1,m2,l1,l2}(m, l) = \sum_{m1=-l1}^{l1}\sum_{m2=-l2}^{l2} C_{m,m1,m2}^{l,l1,l2} \cdot c_{n,l_1,m_1} \cdot c_{n,l_2,m_2}@f$
 * then
 * @f$b_{n,l,l1,l2} = \sum_{m=-l}^{l} c^*_{n,l,m} \cdot z_{m1,m2,l1,l2}(m, l)@f$
 *
 *
 * There is also a more complex version, which impl in Q-suffix version.
 * @f$b_{n1,n2,l,l1,l2} = \sum_{m=-l}^{l}\sum_{m_1=-l_1}^{l_1}\sum_{m_2=-l_2}^{l_2}@f$
 *                          c*_{n_1,l,m} * C_{m,m1,m2}^{l,l1,l2} * c_{n_2,l_1,m_1} * c_{n_2,l_2,m_2}
 * and no symmetry found currently.
 *
 * radical function used here already statisify cutoff condition, in rcut function value
 *  and its deritive goes to zero. One can choose a external orth-basis with a cutoff function.
 *
 * The spherical convention we adopt here is
 * @f$x = r \sin\theta \cos\phi@f$, @f$y = r \sin\theta \sin\phi@f$, @f$z = r \cos\theta@f$
 * @f$\theta \in [0,\pi]@f$, @f$\phi \in [0,2\pi]@f$
 *
 * We choose convension
 *  L = l
 *  M = l + m
 * such that
 *  0 <= L <= l_max
 *  0 <= M <= 2*l
 *
 *
 * We can use real spherical harmonics here for speed, however some symmetry may change their forms, i.e
 *  Y_{l,-m} = (-1)^m Y*_{l,m}
 * becomes
 *  y_{l,-m}^2 + y_{l,m}^2 = P'_{l,|m|}^2
 * which makes symmetry of indices hard to handle
 *
 * The optional real backend transforms the complete coupling tensor; both
 * public real/imaginary slots retain the complex convention.
 * **/

class Descriptor_BispectrumSO3 : public Descriptors
{
private:
    //types
    typedef std::vector<std::vector<std::complex<double>>> carr_type;

    AngularFuncs::ComplexSphericalHarmoics sph;
    AnyBasis *radical_basis;
    Descriptor_BispectrumSO3(const Descriptor_BispectrumSO3 &) = delete;
    Descriptor_BispectrumSO3 &operator=(const Descriptor_BispectrumSO3 &) = delete;

    int l_max;
    double r_min;
    double r_max;
    int r_size;

    int b_size;
    int bQsize;

    int maxnum;                   // size of b_{}, equal to radical_size * ysize
    int ysize;                    // size of Y_{l,m}, equal to (lmax + 1)^2
    std::vector<double> cgvector; // C_{m,m1,m2}^{l,l1,l2}
    std::vector<double> normalize_factor;

    Eigen::Matrix<int, Eigen::Dynamic, 4, Eigen::RowMajor> b_index;
    Eigen::Matrix<int, Eigen::Dynamic, 5, Eigen::RowMajor> bQ_index;

    bool compute_der = false;
    bool real_harmonics_ = false;
    struct RealCoupling {
        int feature,a,b,c;
        double real,imag;
    };
    std::vector<RealCoupling> real_couplings_;
    size_t complex_coupling_count_=0;
    void initialize_real_couplings();
    void real_density(Neighborhood&,Eigen::VectorXd&,
        std::vector<std::vector<DescriptorCurvature::Jet6>>* = nullptr);
    void compute_real(Neighborhood&,Eigen::VectorXd&,DescriptorGradient* = nullptr);


    //template
    std::vector<double> bvector; // result b_{n,l,l1,l2}
    std::vector<double> bQvector; // result b_{n1,n2,l,l1,l2}
    std::vector<double> cvector; // c_nlm1
    carr_type carray;
    carr_type zarray;

public:
    typedef std::vector<std::complex<double>> b_type;
    Descriptor_BispectrumSO3(int _lmax, double rmin, double rmax, int rsize, bool real_harmonics=false);
    ~Descriptor_BispectrumSO3();

    std::unique_ptr<Descriptors> clone() const override {
        return std::make_unique<Descriptor_BispectrumSO3>(l_max, r_min, r_max, r_size,real_harmonics_);
    }

    void compute_carray(carr_type &carr, Eigen::RowVector3d &coord);
    void compute_zarray(std::vector<std::complex<double>> &zarr, std::vector<std::complex<double>> &carr, int l1, int l2, int l);

    b_type compute(Neighborhood &nbhs);
    // compute radical coupled version of bispectrum
    b_type computeQ(Neighborhood &nbhs);

    // --- Descriptors interface ---
    void compute(Neighborhood &nbh, Eigen::VectorXd &out) override;
    void compute(Neighborhood &nbh, Eigen::VectorXd &out,
                 const ComputeOptions &opts) override;
    int feature_size() const override { return b_size * 2; }
    bool uses_real_harmonics() const {return real_harmonics_;}
    size_t real_coupling_count() const {return real_couplings_.size();}
    size_t complex_coupling_count() const {return complex_coupling_count_;}
    bool supports_weighted_site_hvp() const override;
    bool supports_weighted_site_hessian() const override;
    void compute_weighted_site_hvp(Neighborhood &, const Eigen::VectorXd &,
        const DescriptorSiteMatrix &, double &, DescriptorSiteMatrix &, DescriptorSiteMatrix &) override;
    void compute_weighted_site_hessian(Neighborhood &, const Eigen::VectorXd &,
        double &, DescriptorSiteMatrix &, Eigen::MatrixXd &) override;

    std::vector<int> get_bi_ind(int ind) const;
};


#endif
