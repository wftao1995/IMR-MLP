#ifndef IMRMLP_DESCRIPTOR_BISPECTRUMSO4_H
#define IMRMLP_DESCRIPTOR_BISPECTRUMSO4_H

#include "base/structures.h"
#include "descriptors.h"
#include "CutoffFuncs.h"
#include "mathextra/mathextra.h"
#include <Eigen/Core>
#include <array>
#include <vector>
#include <complex>
#include <unordered_map>
#include <map>
#include <span>
#include "base/anyneighborhoods.h"
#include "utils/gputils.h"
#include "solid_harmonic_curvature.h"
#include "so4_first_order.h"

// SO(4) bispectrum, see Bartok 2013
// No SOAP support
/***
 * Note:
 *  This is only a reference implementation of very low efficiency
 *  The algorithms are straight forward
 * **/

/**
 * @brief SO(4) bispectrum descriptor (Bartok 2013, reference implementation).
 * @ingroup descriptors
 *
 * B_{j1,j2,j}, in which j1 <= j2 <= j, |j1 - j2| <= j <= j1 + j2.
 * @note This is a reference implementation with low efficiency.
 *
 * In the code we use theta_0 = invz0 * r, where invz0 = pi / r_m.
 * Different theta_0 conventions exist in Bartok's GAP and Wood's SNAP codes.
 *
 * Runtime radial_angle_map selects linear or atan embedding.
 *
 * Quaternion representations (r0 = sqrt(r^2 + z0^2), w/2 = theta0):
 * - DES_SOF_Q1 (Bartok GAP): a = z0/r0 - z/r0 * I,  b = x/r0 - y/r0 * I
 * - DES_SOF_Q2 (Wood SNAP): a = z0/r0 - z/r0 * I,  b = y/r0 - x/r0 * I
 * - DES_SOF_Q3 (ZhuQiang PyxtalFF): a = cos(w/2) - sin(w/2)*z/r*I, b = sin(w/2)*(y-x*I)/r
 * - DES_SOF_Q4 (Jiantao Wang): a = cos(w/2) - sin(w/2)*x/r*I, b = sin(w/2)*(z-y*I)/r
 * wigner_backend=spline enables the experimental C2 unit-quaternion table (J<=12).
 * contraction_backend=snap selects the sparse Z/adjoint contraction;
 * average=inner-density forms the invariant after averaging local densities.
 * Cutoff, angular order, S3 radius, and species/self weights are configured through GPSettings.
 */
namespace SO4Density { class WignerSpline; }
//#define DES_SOF_Q1
class Descriptor_BispectrumSO4 : public Descriptors, public ConnectSettings
{
public:
    typedef std::vector<double> weight_type; // TODO: test map and unordered_map
    typedef std::complex<double> cmpld;
    using complexdsqmat = Eigen::Matrix<cmpld, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
    typedef std::vector<complexdsqmat> uarrtype;
    using duarrtype = std::vector<std::array<complexdsqmat, 3>>; // [2j][xyz](m1, m2)
    using cmplarr3 = std::array<cmpld, 3>;
    using dbi_type = std::vector<Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor>>;

private:
    double radius_ = 6.8; // r0 of a S3 4d sphere, r0 > rcut, usually r0 = 4/3 rcut is resonable.
    // cutoff_ is inherited from Descriptors base class
    int twojmax_ = 9; // max value of 2*j, since j is half-integer
    bool compute_der = false;

    CutoffFunctions cutfun_;
    // list of CG coeffs, we do not use functions in combinatircs.h on the fly for efficient evaluation.
    std::vector<double> cglist;
    // theta0 = invz0 * r
    double invz0_;
    bool normalize_U = false;
    std::string quaternion_convention_ = "legacy";
    std::string radial_angle_map_ = "linear";
    int convention_ = 4;
    std::string average_ = "outer";
    std::vector<int> coefficient_offsets_;
    std::string contraction_backend_ = "reference";
    std::string wigner_backend_ = "direct";
    std::shared_ptr<const SO4Density::WignerSpline> spline_;
public:
    struct CouplingTerm { int a,b,c; double weight; };
private:
    std::vector<CouplingTerm> coupling_terms_;
    std::vector<std::size_t> coupling_offsets_;
    std::size_t reference_terms_ = 0;
    std::vector<double> wdnorm;
    int ucount_; // count of wigner U
    int ncoeff_;
    int bi_size; // size of each Bispectrum
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> rootpqarr;
    Eigen::Matrix<int, Eigen::Dynamic, 3, Eigen::RowMajor> bindex;

    /*Templates*/
    /**
     * Alloc and deloc??
     * TODO: use 3d-array of complex
     * TODO: make a buffer struct to accoperate openmp
     * */
    // u_{m1,m2}^j = wself_i * U_{m1,m2}^j(0,0,0) + \sum_{rij < rcut}(fc * w_j * U_{m1,m2}^j(theht0, theta, phi))
    uarrtype uarray; // a suqre matrix for each i
    uarrtype uarray_j; // terms in sum
    uarrtype zarray;

public:
    // int nspecies_ = 1;
    // std::vector<int> species_; // species indices for each element, i.e. atomic numbers
    weight_type weights_;       // spice value pair
    weight_type selfweights_;

public:
    Descriptor_BispectrumSO4(double _cutoff = 4, int _twojmax = 9, bool _c_der = false, double _radius = -1, CutoffFunctions::CutoffType ctype = CutoffFunctions::CT_COS_BASIC);
    ~Descriptor_BispectrumSO4(){};

    std::unique_ptr<Descriptors> clone() const override {
        auto copy = std::make_unique<Descriptor_BispectrumSO4>(*this);
        copy->InitSettings();
        return copy;
    }

    // ── GPSettings constructor (MSD pattern) ──
    void InitSettings()
    {
        Connect(cutoff_, "cutoff", "cutoff radius");
        Connect(twojmax_, "twojmax", "2*jmax (max angular order, odd, default 9)");
        Connect(compute_der, "compute_der", "compute derivatives");
        Connect(radius_, "radius", "S3 sphere radius, default 4/3*cutoff");
        Connect(weights_, "weights", "species weights (vector)", -1);
        Connect(selfweights_, "selfweights", "self-term weights (vector)", -1);
        Connect(normalize_U, "normalize_U", "normalize Wigner U matrices");
        Connect(quaternion_convention_, "quaternion_convention", "legacy|q1|q2|q3|q4 embedding");
        Connect(radial_angle_map_, "radial_angle_map", "linear|atan radial angle");
        Connect(average_, "average", "outer|inner-density structure fingerprint");
        Connect(contraction_backend_, "contraction_backend", "reference|snap exact contraction");
        Connect(wigner_backend_, "wigner_backend", "direct|spline (experimental, J<=12, no FC3)");
    }
    Descriptor_BispectrumSO4(const GPSettings &_settings);

    typedef std::vector<double> bi_type;

private:
    double basis_scale(int j) const { return normalize_U ? 1./std::sqrt(j+1.) : 1.; }
    std::vector<DescriptorCurvature::ComplexJet> neighbor_coefficients(
        const Eigen::RowVector3d &, int type,bool second_order=true) const;
    void neighbor_coefficient_jacobian(const Eigen::RowVector3d &,int type,
        std::span<SO4Density::ComplexFirstOrder>) const;
    std::vector<cmpld> neighbor_coefficient_values(const Eigen::RowVector3d &,int type) const;
    void compute_snap(Neighborhood &,std::vector<double> &,dbi_type *) const;
    void init_internals(CutoffFunctions::CutoffType ctype);
    void compute_descriptors_derivative(Neighborhood &nbh, bi_type &ret, dbi_type &dret);

public:

    Eigen::Vector4d compute_coord(Eigen::RowVector3d &x, double r); // x,y,z,r

    void compute_descriptors(Neighborhood &nbh, bi_type &ret, dbi_type *dret = nullptr);
    std::vector<double> compute_descriptors_powerspectrum(Neighborhood &nbh);

    // --- Descriptors interface ---
    void compute(Neighborhood &nbh, Eigen::VectorXd &out) override;
    void compute(Neighborhood &nbh, Eigen::VectorXd &out,
                 const ComputeOptions &opts) override;
    int feature_size() const override { return bi_size; }
    // Native weighted curvature currently covers the default quaternion
    // convention with cosine/basic or hard cutoff; other branches opt out.
    bool supports_weighted_site_hvp() const override;
    bool supports_weighted_site_hessian() const override;
    void compute_weighted_site_hvp(Neighborhood &, const Eigen::VectorXd &,
        const DescriptorSiteMatrix &, double &, DescriptorSiteMatrix &, DescriptorSiteMatrix &) override;
    void compute_weighted_site_hessian(Neighborhood &, const Eigen::VectorXd &,
        double &, DescriptorSiteMatrix &, Eigen::MatrixXd &) override;

    int size() const { return bi_size; };
    bool uses_inner_density() const { return average_ == "inner-density"; }
    bool uses_snap() const { return contraction_backend_ == "snap"; }
    std::span<const CouplingTerm> contraction_terms(int feature) const {
        return std::span<const CouplingTerm>(coupling_terms_).subspan(coupling_offsets_[feature],
            coupling_offsets_[feature+1]-coupling_offsets_[feature]);
    }
    std::size_t reference_contraction_count() const { return reference_terms_; }
    std::size_t sparse_contraction_count() const { return coupling_terms_.size(); }
    // Global density is averaged before forming invariants. Local compute()
    // always retains its local meaning, even with average=inner-density.
    void compute_global_density(const AnyNeighborhoods &, Eigen::VectorXd &,
        Eigen::Tensor<double,3,Eigen::RowMajor> *atom_jacobian=nullptr,
        Eigen::Tensor<double,3,Eigen::RowMajor> *strain_jacobian=nullptr) const;
    void compute_global_density_cell(const Atoms &,Eigen::VectorXd &,
        Eigen::Tensor<double,3,Eigen::RowMajor> &cell_jacobian) const;
    // q=(cell[9], Cartesian residual[3N]) with r=f*C+u when include_cell;
    // otherwise q is Cartesian positions at a fixed cell.
    void compute_weighted_global_hvp(const Atoms &,const Eigen::VectorXd &weights,
        const Eigen::VectorXd &direction,bool include_cell,double &value,
        Eigen::VectorXd &gradient,Eigen::VectorXd &product) const;
    void compute_weighted_global_hessian(const Atoms &,const Eigen::VectorXd &weights,
        bool include_cell,double &value,Eigen::VectorXd &gradient,Eigen::MatrixXd &) const;

    // compute U_{m1,m2}^j
    void compute_uarray(cmpld &acj, cmpld &bcj, uarrtype &uarr);
    void compute_zarray(complexdsqmat &zarr, uarrtype &uarr, int j1, int j2, int j);
    // Derivatives of conj(U): duarr[2j][axis](m1, m2). Resizes the output.
    void compute_duarray_cj(cmpld &acj, cmpld &bcj, cmplarr3 &dacj, cmplarr3 &dbcj, uarrtype &uarr, duarrtype &duarr);
    void add_uarray_self(int typei, uarrtype &uarr);
    // NOTE: in EME version separate each species
    void add_uarray(uarrtype &uarrj, uarrtype& uarr, double fcut);
    std::vector<int> get_bi_ind(int ind) const;
};

#endif
