#include <catch2/catch_all.hpp>
#include "mathextra/combinatorics.h"
#include "mathextra/spline.h"
#include <cstdint>
#include <limits>
#include <numbers>
using Catch::Approx;
TEST_CASE("Binomial exact typed recurrence handles factorial overflow and domain boundaries", "[combinatorics][contract]") {
    using Combinatorics::binomial;
    REQUIRE(binomial<int>(0,0)==1);
    for(int n:{169,170,171}) {
        REQUIRE(binomial<int>(n,0)==1);
        REQUIRE(binomial<int>(n,n)==1);
        REQUIRE(binomial<int>(n,1)==n);
        REQUIRE(binomial<int>(n,2)==n*(n-1)/2);
    }
    REQUIRE(binomial<int>(1000000,1)==1000000);
    REQUIRE(binomial<std::uint64_t>(67,33)==14226520737620288370ULL);
    REQUIRE(binomial<std::uint64_t>(std::numeric_limits<std::uint64_t>::max(),1)==std::numeric_limits<std::uint64_t>::max());
    REQUIRE_THROWS_AS(binomial<int>(34,17),std::overflow_error);
    REQUIRE_THROWS_AS(binomial<std::uint64_t>(68,34),std::overflow_error);
    REQUIRE_THROWS_AS(binomial<unsigned>(-1,0),std::invalid_argument);
    REQUIRE_THROWS_AS(binomial<int>(2,3),std::invalid_argument);
    REQUIRE_THROWS_AS(binomial<double>(5.5,2),std::invalid_argument);
    REQUIRE_THROWS_AS(binomial<double>(std::numeric_limits<double>::infinity(),1),std::invalid_argument);
    REQUIRE_THROWS_AS(binomial<double>(std::ldexp(1.,64),1),std::overflow_error);
    REQUIRE(binomial<int>(5.,2.)==10);
    for(int n=1;n<=30;++n)for(int m=1;m<n;++m) {
        REQUIRE(binomial<std::uint64_t>(n,m)==binomial<std::uint64_t>(n,n-m));
        REQUIRE(binomial<std::uint64_t>(n,m)==binomial<std::uint64_t>(n-1,m-1)+binomial<std::uint64_t>(n-1,m));
    }
}
TEST_CASE("Binomial floating results remain finite whenever the chosen type can represent them", "[combinatorics][contract]") {
    using Combinatorics::binomial;
    REQUIRE(binomial<double>(200,2)==Approx(19900.).margin(1e-12));
    REQUIRE(binomial<double>(1000,500)==Approx(2.7028824094543655e299).epsilon(2e-14));
    REQUIRE(binomial<long double>(2000,1000)==Approx(std::exp(std::lgamma(2001.L)-2*std::lgamma(1001.L))).epsilon(2e-14));
    REQUIRE_THROWS_AS(binomial<double>(2000,1000),std::overflow_error);
    REQUIRE_THROWS_AS(binomial<float>(200,100),std::overflow_error);
}
TEST_CASE("Generic linear tables include endpoints own copies and survive rejected reinitialization", "[interpolation][contract]") {
    Spline::LinearInterpolationUnivar<double> table(-2,3,6,[](double x){return 2*x+1;});
    for(double x:{-2.,-1.5,0.,2.7,3.})REQUIRE(table.eval(x)==Approx(2*x+1).margin(1e-14));
    auto copy=table;
    table.init(0.,1.,3,[](double x){return -x;});
    REQUIRE(copy.eval(3)==Approx(7));
    double x[]={0,1,2},y[]={1,3,5},k[]={2,2};
    table.init(x,y,3,k);x[0]=-100;y[0]=-200;
    REQUIRE(table.eval(0)==Approx(1));
    REQUIRE(table.eval(2)==Approx(5));
    REQUIRE_THROWS_AS(table.eval(-1),std::out_of_range);
    REQUIRE_THROWS_AS(table.eval(std::numeric_limits<double>::quiet_NaN()),std::out_of_range);
    REQUIRE_THROWS_AS(table.init(0.,1.,1,[](double){return 1.;}),std::invalid_argument);
    REQUIRE_THROWS_AS(table.init(0.,1.,3,[](double){return std::numeric_limits<double>::infinity();}),std::invalid_argument);
    REQUIRE(table.eval(1)==Approx(3));
    double unsorted[]={0,1,.5};
    REQUIRE_THROWS_AS(table.init(unsorted,y,3),std::invalid_argument);
    Spline::LinearInterpolationUnivar<double> empty;
    REQUIRE_THROWS_AS(empty.eval(0),std::logic_error);
}
TEST_CASE("Adaptive flat tables refine nonuniform intervals and preserve valid state on all failures", "[interpolation][contract]") {
    Spline::LinearInterpolationAdp<double> table;
    const auto smooth=[](double x){return std::exp(x);};
    REQUIRE(table.init(0.,2.,1e-5,.4,smooth,1e-4));
    REQUIRE(table.size()>6);
    for(int i=0;i<=1000;++i) {
        const double x=2.*i/1000;
        REQUIRE(table.eval(x)==Approx(smooth(x)).margin(1.01e-4));
    }
    const auto size=table.size();const double before=table.eval(.37);
    REQUIRE_FALSE(table.init(0.,2.,.5,1.,smooth,1e-12));
    REQUIRE(table.size()==size);REQUIRE(table.eval(.37)==before);
    REQUIRE_THROWS_AS(table.init(0.,2.,0.,1.,smooth),std::invalid_argument);
    REQUIRE_THROWS_AS(table.init(0.,2.,.01,.5,[](double){return std::numeric_limits<double>::quiet_NaN();}),std::invalid_argument);
    REQUIRE(table.eval(.37)==before);
    auto copy=table;table.init(-1.,1.,.01,.4,[](double x){return x;});
    REQUIRE(copy.eval(.37)==before);
    double x[]={0,.001,.2,1.,100.},y[]={1,1.002,1.4,3.,201.};
    table.init(x,y,5);
    for(double query:{0.,.0003,.05,.7,40.,100.})REQUIRE(table.eval(query)==Approx(1+2*query).margin(1e-12));
    REQUIRE(table.findindex(100)==3);
    REQUIRE_THROWS_AS(table.eval(101),std::out_of_range);
}
TEST_CASE("Generic bilinear grids reproduce affine cross terms on all closed boundaries", "[interpolation][contract]") {
    const auto f=[](double x,double y){return 1+2*x-3*y+4*x*y;};
    Spline::BiLinearInterpolationUnivar<double> table(3,4,-1.,-2.,2.,3.,f);
    for(double x:{-1.,-.3,1.,2.})for(double y:{-2.,-.5,1.,3.})
        REQUIRE(table.eval(x,y)==Approx(f(x,y)).margin(2e-14));
    auto copy=table;
    double x[]={0,.2,2},y[]={-3,-1,4},z[9];
    for(int i=0;i<3;++i)for(int j=0;j<3;++j)z[3*i+j]=f(x[i],y[j]);
    table.init(x,y,z,3,3);
    REQUIRE(table.eval(.7,2)==Approx(f(.7,2)).margin(1e-14));
    REQUIRE(copy.eval(-1,-2)==Approx(f(-1,-2)));
    REQUIRE_THROWS_AS(table.eval(-.1,0),std::out_of_range);
    REQUIRE_THROWS_AS(table.init(x,y,z,1,3),std::invalid_argument);
    REQUIRE(table.eval(.7,2)==Approx(f(.7,2)).margin(1e-14));
}
