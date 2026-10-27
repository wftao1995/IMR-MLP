#include "potentials/potential_eamalloy_spline.h"
#include "eam_spline_curvature_fixture.h"
using namespace Potentials;
TEST_CASE("EAMAlloySpline native curvature follows radial tail and embedding branches","[eam][curvature][batch39]"){const int mode=GENERATE(0,1,2);const bool partial=GENERATE(false,true);EAMSplineTest::native<EAMAlloySpline>(false,mode,partial);}
TEST_CASE("EAMAlloySpline analytical cell consumer maps fixed positions strain and cutoff","[eam][curvature][batch39]"){const bool strain=GENERATE(false,true);const double cutoff=GENERATE(0.,3.1,4.7);EAMSplineTest::consumer<EAMAlloySpline>(false,strain,cutoff);}
TEST_CASE("EAMAlloySpline cell keeps periodic self background and element remapping","[eam][curvature][batch39]"){for(int mode:{0,1,2})EAMSplineTest::self<EAMAlloySpline>(false,mode);EAMSplineTest::mapping<EAMAlloySpline>(false);}
