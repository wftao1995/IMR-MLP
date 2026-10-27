#ifndef LMP_NATIVE_NEIGHBOR_GEOMETRY_H
#define LMP_NATIVE_NEIGHBOR_GEOMETRY_H

#include <Eigen/Core>

namespace LAMMPS_NS::NativeNeighborGeometry {
// Keep Eigen's norm and its floating-point cutoff boundary. The GPU packing
// loop only needs the distance; materializing a three-component displacement
// can force masked tail stores to the stack before the norm reduction.
inline double distance(const Eigen::RowVector3d &center,
                       const Eigen::RowVector3d &neighbor)
{
  return (neighbor-center).norm();
}
}
#endif
