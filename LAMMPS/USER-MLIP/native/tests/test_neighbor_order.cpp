#include "../native_neighbor_order.h"

#include <array>
#include <iostream>
#include <stdexcept>

int main()
{
  using namespace LAMMPS_NS::NativeNeighborOrder;
  const std::array<int,4> tags{7,7,9,10};
  std::array<std::array<double,3>,4> positions{};
  auto move = [&](double x) {
    // Two periodic forwarding paths for the same Cartesian component.
    // Other components distinguish the images and make the tie observable.
    positions[0] = {x+10.,2.,0.};
    positions[1] = {(x+5.)+5.,1.,0.};
    positions[2] = {0.,0.,0.};
    positions[3] = {-1.,0.,0.};
  };
  move(.001);
  if (!(positions[0][0] < positions[1][0])) throw std::runtime_error("missing first rounding direction");
  std::vector<int> indices{0,1,2,3};
  std::vector<ImageGroup> groups;
  append_image_groups(indices,0,4,tags,groups);
  if (groups.size()!=1 || groups[0].begin!=0 || groups[0].end!=2)
    throw std::runtime_error("unexpected same-tag group boundaries");
  refresh_images(indices,groups,positions);
  if (indices!=std::vector<int>{0,1,2,3}) throw std::runtime_error("stable ordering changed");
  move(.002);
  if (!(positions[0][0] > positions[1][0])) throw std::runtime_error("missing reversed rounding direction");
  refresh_images(indices,groups,positions);
  if (indices!=std::vector<int>{1,0,2,3}) throw std::runtime_error("moving images retained stale ordering");
  move(.001);
  refresh_images(indices,groups,positions);
  if (indices!=std::vector<int>{0,1,2,3}) throw std::runtime_error("returning images retained stale ordering");
  groups.clear(); append_image_groups(indices,2,4,tags,groups);
  if (!groups.empty()) throw std::runtime_error("distinct tags created an image group");
  std::cout << "Native neighbor image ordering: 7 checks passed\n";
}
