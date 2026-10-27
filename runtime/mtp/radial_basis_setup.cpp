#include "radial_basis.h"
#include <Eigen/Eigenvalues>
#include <limits>

namespace MLIPCore::Radial {
void initialize_polynomial(mlip::radial_basis_t &b) {
    const auto n=std::size_t(b.size);
    if(n==0 || n>std::size_t(100000000)/n) radial_parse_error("PolyB matrix exceeds allocation limit");
    Eigen::MatrixXd overlap(n,n);
    b.polynomial_normalizations.resize(n);
    for(std::size_t i=0;i<n;++i) {
        const double ipower=i+1;
        b.polynomial_normalizations[i]=std::sqrt((2*ipower+5)/std::pow(b.max_dist-b.min_dist,2*ipower+5));
        for(std::size_t j=i;j<n;++j) {
            const double jpower=j+1;
            overlap(i,j)=overlap(j,i)=std::sqrt((2*ipower+5)*(2*jpower+5))/(ipower+jpower+5);
        }
    }
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(overlap,Eigen::ComputeEigenvectors);
    if(solver.info()!=Eigen::Success || !solver.eigenvalues().allFinite() || solver.eigenvalues().minCoeff()<=0)
        radial_parse_error("PolyB overlap matrix is not positive definite");
    // Same expression and storage order as basis/basis.cpp.
    Eigen::MatrixXd weights=Eigen::MatrixXd::Identity(n,n);
    for(std::size_t i=0;i<n;++i) weights(i,i)=1/std::sqrt(solver.eigenvalues()(i));
    const Eigen::MatrixXd matrix=solver.eigenvectors()*weights*solver.eigenvectors().transpose();
    if(!matrix.allFinite()) radial_parse_error("non-finite PolyB orthogonalization matrix");
    for(double x:b.polynomial_normalizations) if(!std::isfinite(x) || !(x>0)) radial_parse_error("invalid PolyB normalization");
    b.polynomial_matrix.assign(matrix.data(),matrix.data()+n*n);
}
}
