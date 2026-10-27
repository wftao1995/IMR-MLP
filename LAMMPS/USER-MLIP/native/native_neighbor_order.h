#pragma once

#include <algorithm>
#include <vector>

namespace LAMMPS_NS::NativeNeighborOrder {

struct ImageGroup { int begin, end; };

template<class Positions>
bool image_less(int a, int b, const Positions &positions)
{
  for (int k = 0; k < 3; ++k)
    if (positions[a][k] != positions[b][k]) return positions[a][k] < positions[b][k];
  return a < b;
}

template<class Tags, class Positions>
bool less(int a, int b, const Tags &tags, const Positions &positions)
{
  return tags[a] != tags[b] ? tags[a] < tags[b] : image_less(a,b,positions);
}

template<class Tags>
void append_image_groups(const std::vector<int> &indices, int begin, int end,
                         const Tags &tags, std::vector<ImageGroup> &groups)
{
  for (int first = begin; first < end;) {
    int last = first+1;
    while (last < end && tags[indices[last]] == tags[indices[first]]) ++last;
    if (last-first > 1) groups.push_back({first,last});
    first = last;
  }
}

template<class Positions>
void refresh_images(std::vector<int> &indices, const std::vector<ImageGroup> &groups,
                    const Positions &positions)
{
  // Periodic forwarding can reach equal Cartesian image components through
  // different floating-point addition paths. Their last-bit ordering can
  // change as an atom moves, even at a fixed cell. Preserve the original
  // coordinate comparator rather than assuming that order is invariant.
  const auto compare = [&](int a,int b) { return image_less(a,b,positions); };
  for (const auto &group : groups) {
    const auto begin = indices.begin()+group.begin, end = indices.begin()+group.end;
    if (!std::is_sorted(begin,end,compare)) std::sort(begin,end,compare);
  }
}

} // namespace LAMMPS_NS::NativeNeighborOrder
