#pragma once
#include "hvp_io.h"
#include <limits>

// Dense labels use the existing EXTXYZ order: Cartesian atom-major xyz rows,
// then columns, in eV/Angstrom^2. Feature strings remain a legacy metadata API.
namespace HessianIO {
inline void validate(const Atoms &atom) {
    if (!atom.have_hessian) return;
    if (atom.size()<=0 || atom.size()>std::numeric_limits<int>::max()/3 || !atom.has_hessian() || !atom.hessian_.allFinite())
        throw std::invalid_argument("Full Hessian labels require a finite Cartesian 3N by 3N matrix");
}
inline bool read_feature(Atoms &atom,std::string_view text) {
    const size_t n=atom.size();
    if (!n || n>size_t(std::numeric_limits<int>::max())/3) return false;
    const size_t ndof=3*n;
    if (ndof>std::numeric_limits<size_t>::max()/ndof) return false;
    const size_t expected=ndof*ndof;
    // Even one-character numbers need separating whitespace. Avoid reserving a
    // huge matrix for a short arbitrary legacy Feature value.
    if (expected>size_t(std::numeric_limits<Eigen::Index>::max()) || expected>text.size()/2+1) return false;
    std::vector<double> values; values.reserve(expected);
    const char *p=text.data(),*end=p+text.size();
    while (p<end) {
        while (p<end && std::isspace(static_cast<unsigned char>(*p))) ++p;
        if (p==end) break;
        if (values.size()==expected) return false;
        double value; const auto result=fast_float::from_chars(p,end,value);
        if (result.ec!=std::errc() || !std::isfinite(value) ||
            (result.ptr<end && !std::isspace(static_cast<unsigned char>(*result.ptr)))) return false;
        values.push_back(value); p=result.ptr;
    }
    if (values.size()!=expected) return false;
    if (atom.have_hessian) throw std::invalid_argument("Duplicate full Hessian Feature");
    atom.hessian_.resize(ndof,ndof);
    for (size_t i=0;i<ndof;++i) for (size_t j=0;j<ndof;++j)
        atom.hessian_(i,j)=values[i*ndof+j];
    atom.have_hessian=true;
    return true;
}
inline std::string write(const Atoms &atom) {
    validate(atom);
    // HVP's serializer is column-major; transpose explicitly for dense rows.
    return HVPIO::write(atom.hessian_.transpose());
}
}
