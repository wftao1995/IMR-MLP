#include <catch2/catch_all.hpp>
#include "descriptors/descriptor_Behler.h"
#include "potentials/potential_descriptor_linear.h"
#include <cmath>
#include <filesystem>
#include <fstream>
#include <unistd.h>

namespace {
GPSettings settings(bool weighted = true) {
    GPSettings s;
    s.Modify("dispcount", "5"); s.Modify("dispselect", "1 2 3 4 5");
    s.Modify("cutoff", "5"); s.Modify("eta", ".2"); s.Modify("rshift", ".3");
    s.Modify("kppa", ".7"); s.Modify("zeta", "2"); s.Modify("lambda", "1");
    s.Modify("weighted", weighted ? "true" : "false");
    if (weighted) s.Modify("weights", "1 8");
    return s;
}
Neighborhood environment() {
    Neighborhood n;
    n.nbhcount_ = 3; n.centertype = 0; n.chemmap_ = {0, 1, 0}; n.indices_ = {1, 2, 3};
    n.pos_ = {{1.2, .3, .1}, {-.4, 1.6, .2}, {.2, -.3, 2.3}};
    for (const auto &p : n.pos_) n.distances_.push_back(p.norm());
    return n;
}
double fc(double r, double rc) { return r >= rc ? 0. : .5 * (std::cos(M_PI*r/rc) + 1.); }
Eigen::VectorXd independent_values(DescriptorBehler &descriptor, Neighborhood &n,
    bool weighted, const std::vector<double> &chemical) {
    const auto &p = descriptor.feature_parameters();
    Eigen::VectorXd out = Eigen::VectorXd::Zero(5);
    for (int f = 0; f < 5; ++f) {
        const double rc=p(f,0), eta=p(f,1), rs=p(f,2), k=p(f,3), z=p(f,4), l=p(f,5);
        for (int i = 0; i < n.nbhcount_; ++i) {
            const double r=n.distances_[i], w=weighted ? chemical[n.chemmap_[i]] : 1.;
            if (f == 0) out[f] += w*fc(r,rc);
            if (f == 1) out[f] += w*std::exp(-eta*(r-rs)*(r-rs))*fc(r,rc);
            if (f == 2) out[f] += w*std::cos(k*r)*fc(r,rc);
            // Deliberately enumerate ordered pairs for weighted angular features.
            if (f >= 3) for (int j = 0; j < n.nbhcount_; ++j) {
                if (i == j || (!weighted && j <= i)) continue;
                const double t=n.distances_[j], d=(n.pos_[i]-n.pos_[j]).norm();
                const double shift=weighted ? rs : 0.;
                double q=(r-shift)*(r-shift)+(t-shift)*(t-shift), cut=fc(r,rc)*fc(t,rc);
                if (f == 3) { q+=(d-shift)*(d-shift); cut*=fc(d,rc); }
                out[f] += std::pow(2.,1.-z)*w*(weighted ? chemical[n.chemmap_[j]] : 1.)*
                    std::pow(1.+l*n.pos_[i].dot(n.pos_[j])/(r*t),z)*std::exp(-eta*q)*cut;
            }
        }
    }
    return out;
}
DescriptorSiteMatrix contracted(DescriptorBehler &d, Neighborhood &n, const Eigen::VectorXd &weights) {
    Eigen::VectorXd value; DescriptorGradient g; ComputeOptions o;
    o.compute_derivative=true; o.gradient=&g; d.compute(n,value,o);
    DescriptorSiteMatrix result=DescriptorSiteMatrix::Zero(n.nbhcount_,3);
    for(int f=0;f<weights.size();++f) for(int i=0;i<n.nbhcount_;++i) for(int a=0;a<3;++a)
        result(i,a)+=weights[f]*g.grad(f,i,a);
    return result;
}
}

TEST_CASE("Behler explicit chemical weights implement ordered shifted wACSF", "[descriptors][behler][wacsf]") {
    const bool weighted=GENERATE(false,true);
    auto s=settings(weighted); DescriptorBehler d(s); auto n=environment();
    Eigen::VectorXd value; d.compute(n,value);
    REQUIRE((value-independent_values(d,n,weighted,{1,8})).norm()<1e-12);
    n.centertype=1; Eigen::VectorXd changed; d.compute(n,changed);
    REQUIRE(value==changed);
    if(weighted) {
        s.Modify("weights","8 1"); DescriptorBehler remapped(s);
        for(auto &t:n.chemmap_) t=1-t;
        remapped.compute(n,changed); REQUIRE((value-changed).norm()<1e-12);
        s.Modify("weights","1 1"); DescriptorBehler ones(s);
        auto plain=settings(false); plain.Modify("rshift","0");
        auto zero_shift=s; zero_shift.Modify("rshift","0");
        DescriptorBehler old(plain), current(zero_shift);
        Eigen::VectorXd a,b; old.compute(n,a); current.compute(n,b);
        REQUIRE((a.head(3)-b.head(3)).norm()<1e-12);
        REQUIRE((2.*a.tail(2)-b.tail(2)).norm()<1e-12);
    }
}
TEST_CASE("Behler resolves feature rows and preserves effective settings", "[descriptors][behler][wacsf]") {
    auto s=settings(); s.Modify("elements","H O"); s.Modify("weights","2 3");
    s.Modify("rcs","2 3 4 5 6"); s.Modify("etas",".1 .2 .3 .4 .5");
    s.Modify("rshifts","0 .1 .2 .3 .4"); s.Modify("zetas","0 1 2 3 4");
    DescriptorBehler d(s); REQUIRE(d.cutoff_==6.); REQUIRE(d.species_weights()==std::vector<double>{2,3});
    REQUIRE(d.feature_parameters()(3,2)==.3);
    DescriptorBehler restored(d.CurrentSettings());
    REQUIRE(restored.feature_parameters()==d.feature_parameters());
    REQUIRE(restored.species_weights()==d.species_weights());
    auto n=environment(); Eigen::VectorXd value; d.compute(n,value);
    REQUIRE((value-independent_values(d,n,true,{2,3})).norm()<1e-12);
    auto snapshot=d.feature_parameters(); d.compute(n,value); REQUIRE(snapshot==d.feature_parameters());
    s.Modify("weights",""); DescriptorBehler atomic(s);
    REQUIRE(atomic.species_weights()==std::vector<double>{1,8});
    int selection[]={2,2}; double row0[]={2,.1,.2,.3,1,-1}, row1[]={4,.4,.5,.6,2,1};
    double *rows[]={row0,row1}; DescriptorBehler manual;
    manual.set_d_parameters(2,selection,rows);
    REQUIRE(manual.feature_parameters()(0,2)==.2); REQUIRE(manual.feature_parameters()(1,1)==.4);
    REQUIRE(manual.cutoff_==4.); manual.compute(n,value); REQUIRE(value.size()==2);
    auto clone=manual.clone(); Eigen::VectorXd cloned; clone->compute(n,cloned); REQUIRE(value==cloned);
}
TEST_CASE("Behler shifted chemical values gradients and curvature agree", "[descriptors][behler][wacsf][curvature]") {
    const bool weighted=GENERATE(false,true);
    auto s=settings(weighted); s.Modify("rcs","3 4 5 4.5 5.5");
    s.Modify("etas","0 .1 .2 .3 .4"); s.Modify("rshifts","0 .2 .4 .6 .8");
    s.Modify("zetas","0 0 0 1 2");
    DescriptorBehler d(s); auto n=environment(); Eigen::VectorXd value;
    DescriptorGradient jac; ComputeOptions o; o.compute_derivative=true; o.gradient=&jac;
    d.compute(n,value,o); Eigen::VectorXd weights=Eigen::VectorXd::LinSpaced(5,-.3,.7);
    double contracted_value; DescriptorSiteMatrix gradient,product,direction(3,3); Eigen::MatrixXd hessian;
    direction<<.2,-.1,.3,-.4,.5,.2,.1,.2,-.3;
    d.compute_weighted_site_hessian(n,weights,contracted_value,gradient,hessian);
    REQUIRE(contracted_value==Catch::Approx(weights.dot(value)).margin(1e-12));
    REQUIRE((gradient-contracted(d,n,weights)).norm()<1e-10);
    REQUIRE((hessian-hessian.transpose()).norm()<1e-10);
    double hv; DescriptorSiteMatrix hg;
    d.compute_weighted_site_hvp(n,weights,direction,hv,hg,product);
    REQUIRE((Eigen::Map<Eigen::VectorXd>(product.data(),9)-hessian*Eigen::Map<Eigen::VectorXd>(direction.data(),9)).norm()<1e-10);
    for(double step:{1e-4,1e-5}) for(int i=0;i<3;++i) for(int a=0;a<3;++a) {
        auto plus=n,minus=n; plus.pos_[i][a]+=step; minus.pos_[i][a]-=step;
        plus.distances_[i]=plus.pos_[i].norm(); minus.distances_[i]=minus.pos_[i].norm();
        Eigen::VectorXd vp,vm; d.compute(plus,vp); d.compute(minus,vm);
        for(int f=0;f<5;++f) REQUIRE(jac.grad(f,i,a)==Catch::Approx((vp[f]-vm[f])/(2.*step)).margin(2e-6));
        const DescriptorSiteMatrix fd=(contracted(d,plus,weights)-contracted(d,minus,weights))/(2.*step);
        REQUIRE((hessian.col(3*i+a)-Eigen::Map<const Eigen::VectorXd>(fd.data(),9)).norm()<3e-5);
    }
}
TEST_CASE("Behler rejects incomplete weights parameters and undefined geometry", "[descriptors][behler][wacsf]") {
    for(const auto &[key,val]:std::vector<std::pair<std::string,std::string>>{
        {"weights","nan 1"},{"elements","H Unknown"},{"rcs","2 3"},
        {"etas","-1 0 0 0 0"},{"zetas","-1 1 1 1 1"},{"lambdas","0 0 0 0 2"},
        {"cutoff","0"},{"rshift","nan"}}) {
        auto s=settings(); s.Modify(key,val); REQUIRE_THROWS(DescriptorBehler{s});
    }
    auto s=settings(); s.Modify("elements","H"); REQUIRE_THROWS(DescriptorBehler{s});
    DescriptorBehler d(settings()); auto n=environment(); Eigen::VectorXd value;
    n.chemmap_[0]=2; REQUIRE_THROWS(d.compute(n,value));
    n=environment(); n.distances_[0]=0.; REQUIRE_THROWS(d.compute(n,value));
    n=environment(); n.pos_[1]=n.pos_[0]; n.distances_[1]=n.distances_[0]; REQUIRE_THROWS(d.compute(n,value));
    auto smooth=settings(); smooth.Modify("zeta","1.5"); smooth.Modify("lambda","-1");
    DescriptorBehler limits(smooth); n=environment(); n.nbhcount_=2;
    n.pos_={{1,0,0},{2,0,0}}; n.distances_={1,2}; n.chemmap_={0,1};
    double v; DescriptorSiteMatrix g; Eigen::MatrixXd h;
    limits.compute_weighted_site_hessian(n,Eigen::VectorXd::Ones(5),v,g,h); REQUIRE(h.allFinite());
    smooth.Modify("zeta",".5"); DescriptorBehler singular(smooth);
    REQUIRE_THROWS(singular.compute(n,value));
    REQUIRE_THROWS(singular.compute_weighted_site_hessian(n,Eigen::VectorXd::Zero(5),v,g,h));
}
TEST_CASE("Weighted Behler linear model saves effective chemistry and feature rows", "[potential][descriptor_linear][wacsf]") {
    GPSettings s; s.Modify("descriptor:dtype","behler"); s.Modify("species_count","2");
    auto local=settings(); local.Modify("elements","H O"); local.Modify("weights","");
    local.Modify("etas",".1 .2 .3 .4 .5"); local.Modify("rcs","3 4 5 4 5");
    for(const auto &[key,val]:local.settings) s.Modify("descriptor:behler:"+key,val);
    Potentials::PotentialDescriptorLinear original(s);
    Eigen::VectorXd coeff=Eigen::VectorXd::LinSpaced(5,-.2,.3); original.nl_set_coeffs(coeff.data());
    char name[]="/tmp/imrmlp_wacsf_XXXXXX"; const int fd=mkstemp(name); REQUIRE(fd>=0); close(fd);
    struct Cleanup { const char *p; ~Cleanup(){std::filesystem::remove(p);} } cleanup{name};
    original.Save(std::string(name)); Potentials::PotentialDescriptorLinear restored{std::string(name)};
    GPSettings saved{std::string(name)}; REQUIRE(!saved["descriptor:behler:weights"].empty());
    Atoms a; a.cell_=20.*Eigen::Matrix3d::Identity(); a.pos_={{0,0,0},{1.2,.3,.1},{-.4,1.6,.2}}; a.chemmap_={0,1,0};
    Atoms b(a); static_cast<Potentials::AnyPotential&>(original).getEFS(a);
    static_cast<Potentials::AnyPotential&>(restored).getEFS(b);
    REQUIRE(a.energy_==Catch::Approx(b.energy_).margin(1e-12));
    REQUIRE((a.stress_-b.stress_).norm()<1e-11);
    for(int i=0;i<a.size();++i) REQUIRE((a.forces_[i]-b.forces_[i]).norm()<1e-11);
}

TEST_CASE("Behler reinitialization preserves inherited and explicit parameter provenance", "[descriptors][behler][wacsf][settings]") {
    GPSettings s;s.Modify("dispcount","1");s.Modify("dispselect","2");
    s.Modify("cutoff","5");s.Modify("eta",".2");
    DescriptorBehler inherited(s);GPSettings changed;changed.Modify("cutoff","3");changed.Modify("eta",".8");
    inherited.ApplySettings(changed);inherited.init();
    REQUIRE(inherited.cutoff_==3.);REQUIRE(inherited.feature_parameters()(0,0)==3.);
    REQUIRE(inherited.feature_parameters()(0,1)==.8);
    auto n=environment();Eigen::VectorXd value;inherited.compute(n,value);
    double expected=0.;for(double r:n.distances_)expected+=std::exp(-.8*r*r)*fc(r,3.);
    REQUIRE(value[0]==Catch::Approx(expected).margin(1e-13));
    DescriptorBehler snapshot(inherited.CurrentSettings());
    REQUIRE(snapshot.feature_parameters()==inherited.feature_parameters());
    auto clone=inherited.clone();auto& clone_typed=dynamic_cast<DescriptorBehler&>(*clone);
    changed.Modify("dispcount","2");changed.Modify("dispselect","2 2");
    changed.Modify("cutoff","2.5");changed.Modify("eta","1.1");
    clone_typed.ApplySettings(changed);clone_typed.init();
    REQUIRE(clone_typed.feature_size()==2);REQUIRE(clone_typed.cutoff_==2.5);
    REQUIRE(clone_typed.feature_parameters()(0,1)==1.1);
    REQUIRE(clone_typed.feature_parameters()(1,1)==1.1);
    REQUIRE(inherited.feature_size()==1);REQUIRE(inherited.cutoff_==3.);
    REQUIRE(inherited.feature_parameters()(0,1)==.8);
    s.Modify("rcs","4");s.Modify("etas",".4");DescriptorBehler explicit_rows(s);
    GPSettings scalar;scalar.Modify("cutoff","2");scalar.Modify("eta",".9");
    explicit_rows.ApplySettings(scalar);explicit_rows.init();
    REQUIRE(explicit_rows.feature_parameters()(0,0)==4.);REQUIRE(explicit_rows.feature_parameters()(0,1)==.4);
}

TEST_CASE("Behler elements-derived weights follow settings while explicit weights remain authoritative", "[descriptors][behler][wacsf][settings]") {
    auto s=settings();s.Modify("weights","");s.Modify("elements","H O");
    DescriptorBehler derived(s);REQUIRE(derived.species_weights()==std::vector<double>{1,8});
    GPSettings changed;changed.Modify("elements","C O");derived.ApplySettings(changed);derived.init();
    REQUIRE(derived.species_weights()==std::vector<double>{6,8});
    DescriptorBehler snapshot(derived.CurrentSettings());REQUIRE(snapshot.species_weights()==derived.species_weights());
    auto clone=derived.clone();auto& typed=dynamic_cast<DescriptorBehler&>(*clone);
    changed.Modify("elements","N O");typed.ApplySettings(changed);typed.init();
    REQUIRE(typed.species_weights()==std::vector<double>{7,8});
    REQUIRE(derived.species_weights()==std::vector<double>{6,8});
    s.Modify("weights","2 3");DescriptorBehler explicit_weights(s);
    explicit_weights.ApplySettings(changed);explicit_weights.init();
    REQUIRE(explicit_weights.species_weights()==std::vector<double>{2,3});
}

TEST_CASE("Behler G4 has the smooth coincident-neighbor Cartesian limit when eta times shift is zero", "[descriptors][behler][wacsf][curvature]") {
    const bool weighted=GENERATE(false,true);
    const bool zero_eta=GENERATE(false,true);
    GPSettings s;s.Modify("dispcount","1");s.Modify("dispselect","4");s.Modify("cutoff","5");
    s.Modify("eta",zero_eta ? "0" : ".2");s.Modify("rshift",zero_eta ? ".4" : "0");
    s.Modify("zeta","2");s.Modify("lambda","1");s.Modify("weighted",weighted ? "true" : "false");
    if(weighted)s.Modify("weights","1 8");DescriptorBehler d(s);
    Neighborhood n;n.nbhcount_=2;n.centertype=0;n.chemmap_={0,1};n.pos_={{1,0,0},{1,0,0}};n.distances_={1,1};
    Eigen::VectorXd value;DescriptorGradient jac;ComputeOptions opts;opts.compute_derivative=true;opts.gradient=&jac;
    d.compute(n,value,opts);
    const double eta=zero_eta ? 0. : .2;
    const double scale=weighted ? 16. : 1.;
    REQUIRE(value[0]==Catch::Approx(scale*2.*std::exp(-2.*eta)*fc(1.,5.)*fc(1.,5.)).margin(1e-12));
    const double dfc=-.5*M_PI/5.*std::sin(M_PI/5.);
    const double expected_gradient=scale*2.*std::exp(-2.*eta)*(dfc*fc(1.,5.)-2.*eta*fc(1.,5.)*fc(1.,5.));
    REQUIRE(jac.grad(0,0,0)==Catch::Approx(expected_gradient).margin(1e-12));
    REQUIRE(jac.grad(0,1,0)==Catch::Approx(expected_gradient).margin(1e-12));
    REQUIRE(jac.grad(0,0,1)==0.);REQUIRE(jac.grad(0,0,2)==0.);
    Eigen::VectorXd coefficient=Eigen::VectorXd::Ones(1);double v;DescriptorSiteMatrix g,hv,dir(2,3);Eigen::MatrixXd h;
    dir<<.2,-.3,.4,-.1,.6,-.2;d.compute_weighted_site_hessian(n,coefficient,v,g,h);
    REQUIRE(h.allFinite());REQUIRE((h-h.transpose()).norm()<1e-12);
    REQUIRE((g-contracted(d,n,coefficient)).norm()<1e-12);
    d.compute_weighted_site_hvp(n,coefficient,dir,v,g,hv);
    REQUIRE((Eigen::Map<const Eigen::VectorXd>(hv.data(),6)-h*Eigen::Map<const Eigen::VectorXd>(dir.data(),6)).norm()<1e-11);
    for(double step:{1e-4,1e-5})for(int i=0;i<2;++i)for(int a=0;a<3;++a) {
        auto plus=n,minus=n;plus.pos_[i][a]+=step;minus.pos_[i][a]-=step;
        plus.distances_[i]=plus.pos_[i].norm();minus.distances_[i]=minus.pos_[i].norm();
        Eigen::VectorXd vp,vm;d.compute(plus,vp);d.compute(minus,vm);
        REQUIRE(jac.grad(0,i,a)==Catch::Approx((vp[0]-vm[0])/(2.*step)).margin(3e-6));
        const DescriptorSiteMatrix fd=(contracted(d,plus,coefficient)-contracted(d,minus,coefficient))/(2.*step);
        REQUIRE((h.col(3*i+a)-Eigen::Map<const Eigen::VectorXd>(fd.data(),6)).norm()<3e-5);
        // Value-only second differences independently check the coincidence Hessian diagonal.
        REQUIRE(h(3*i+a,3*i+a)==Catch::Approx((vp[0]-2.*value[0]+vm[0])/(step*step)).margin(step<5e-5 ? 3e-3 : 3e-5));
    }
    if(weighted) {
        s.Modify("eta",".2");s.Modify("rshift",".4");DescriptorBehler cusp(s);
        REQUIRE_THROWS_WITH(cusp.compute(n,value),Catch::Matchers::ContainsSubstring("cusp"));
        REQUIRE_THROWS(cusp.compute_weighted_site_hessian(n,coefficient,v,g,h));
    }
}
