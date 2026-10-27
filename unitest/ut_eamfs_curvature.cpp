#include "potentials/potential_eamfs_spline.h"
#include "eam_spline_curvature_fixture.h"
using namespace Potentials;
TEST_CASE("EAMFsSpline native curvature follows radial tail and embedding branches","[eam][curvature][batch40]"){const int mode=GENERATE(0,1,2);const bool partial=GENERATE(false,true);EAMSplineTest::native<EAMFsSpline>(true,mode,partial);}
TEST_CASE("EAMFsSpline analytical cell consumer maps fixed positions strain and cutoff","[eam][curvature][batch40]"){const bool strain=GENERATE(false,true);const double cutoff=GENERATE(0.,3.1,4.7);EAMSplineTest::consumer<EAMFsSpline>(true,strain,cutoff);}
TEST_CASE("EAMFsSpline cell keeps periodic self background and element remapping","[eam][curvature][batch40]"){for(int mode:{0,1,2})EAMSplineTest::self<EAMFsSpline>(true,mode);EAMSplineTest::mapping<EAMFsSpline>(true);}
