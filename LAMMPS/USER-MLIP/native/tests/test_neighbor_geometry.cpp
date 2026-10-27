#include "../native_neighbor_geometry.h"

#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>

namespace {
std::size_t checks = 0, bitwise_checks = 0, nan_checks = 0;
#if defined(__GNUC__) && !defined(__clang__)
#define NATIVE_GEOMETRY_NOINLINE __attribute__((noinline,noipa))
#elif defined(__clang__)
#define NATIVE_GEOMETRY_NOINLINE __attribute__((noinline))
#else
#define NATIVE_GEOMETRY_NOINLINE
#endif
NATIVE_GEOMETRY_NOINLINE double reference_distance(const Eigen::RowVector3d &center,
                                                  const Eigen::RowVector3d &neighbor)
{
  // Deliberately retain the original materialized expression as the reference.
  const Eigen::RowVector3d displacement = neighbor-center;
  return displacement.norm();
}
NATIVE_GEOMETRY_NOINLINE double candidate_distance(const Eigen::RowVector3d &center,
                                                  const Eigen::RowVector3d &neighbor)
{
  return LAMMPS_NS::NativeNeighborGeometry::distance(center,neighbor);
}
void check(const Eigen::RowVector3d &center,const Eigen::RowVector3d &neighbor)
{
  // Distinct non-inlined calls prevent common-subexpression elimination from
  // turning the equivalence check into a comparison with itself.
  const double reference = reference_distance(center,neighbor);
  const double actual = candidate_distance(center,neighbor);
  if (std::isnan(reference) || std::isnan(actual)) {
    // NaN signs/payloads can depend on the reduction code generated at -Og.
    // Their physical contract is identical rejection below, not bit identity.
    if (!(std::isnan(reference) && std::isnan(actual)))
      throw std::runtime_error("native packed NaN classification changed");
    ++nan_checks;
  } else {
    // Keep exact equality for every finite value and infinity: no tolerance.
    if (std::bit_cast<std::uint64_t>(reference) != std::bit_cast<std::uint64_t>(actual))
      throw std::runtime_error("native packed distance differs from the materialized reference");
    ++bitwise_checks;
  }
  for (double cutoff : {std::nextafter(reference,0.),reference,
                        std::nextafter(reference,std::numeric_limits<double>::infinity())}) {
    const auto outcome = [cutoff](double r) {
      if (!(r < cutoff)) return 0; // Outside the original physical neighbor list.
      return r > 0. && std::isfinite(r) ? 1 : -1;
    };
    if (outcome(reference) != outcome(actual))
      throw std::runtime_error("native packed cutoff/error decision changed");
  }
  ++checks;
}
}

int main()
{
  const Eigen::RowVector3d origin(0.,0.,0.);
  // This diagonal makes norm(delta)==cutoff while delta.squaredNorm()<cutoff^2.
  const Eigen::RowVector3d boundary(2.4515835088113644,3.7189776864989215,0.);
  check(origin,boundary);
  const double cutoff = 4.454330110505138;
  if (LAMMPS_NS::NativeNeighborGeometry::distance(origin,boundary) < cutoff)
    throw std::runtime_error("the historical sqrt cutoff boundary changed");
  for (double value : {0.,-0.,std::numeric_limits<double>::denorm_min(),1e-300,
      1e-160,1e-100,1.,1e100,1e160,1e300,std::numeric_limits<double>::max(),
      std::numeric_limits<double>::infinity(),-std::numeric_limits<double>::infinity(),
      std::numeric_limits<double>::quiet_NaN()}) {
    check(origin,Eigen::RowVector3d(value,0.,0.));
    check(origin,Eigen::RowVector3d(value,value,value));
    check(Eigen::RowVector3d(-value,0.,value),Eigen::RowVector3d(value,value,-value));
  }
  // Wide exponents cover cancellation, subnormal squares, overflow, and
  // ordinary coordinates. Every finite distance also tests its adjacent ULPs.
  std::mt19937_64 random(20261024);
  for (int i = 0; i < 1000000; ++i) {
    Eigen::RowVector3d a,b;
    for (int k = 0; k < 3; ++k) {
      const double fraction = static_cast<double>(random() >> 11)*0x1p-53-.5;
      const int exponent = static_cast<int>(random()%2098)-1074;
      a[k] = std::ldexp(fraction,exponent);
      b[k] = i%2 ? std::nextafter(a[k],0.) :
        std::ldexp(static_cast<double>(random() >> 11)*0x1p-53-.5,exponent);
    }
    check(a,b);
  }
  std::cout << "Native packed geometry: " << checks << " comparisons ("
            << bitwise_checks << " non-NaN bitwise, " << nan_checks
            << " NaN classification) and adjacent-cutoff decisions passed\n";
}
