#include <catch2/catch_all.hpp>
#include "train/varpro.h"
#include "train/train_nonlinear_potential.h"
#include "potentials/potential_mtp_LES.h"
#include <sstream>
#include <filesystem>
#include <fstream>
#include <regex>
#include "apps/apps.h"
#include "io/gpio.h"
#include <unistd.h>

using Catch::Approx;
namespace {
using namespace Potentials;
Atoms geometry(int k) {
    Atoms a;a.id=k;a.cell_=12.*Eigen::Matrix3d::Identity();a.pbc_={false,false,false};
    a.pos_={{2.,2.,2.},{4.1+.13*std::sin(k),2.2,2.},{2.3,4.2+.12*std::cos(k),2.4}};
    a.chemmap_={0,1,0};a.fitness=.5+.2*k;a.atomic_fitness={.3,1.,1.7};return a;
}
Eigen::VectorXd flatten_forces(const Atoms &a) {
    Eigen::VectorXd out(3*a.size());
    for(int i=0;i<a.size();++i) out.segment<3>(3*i)=a.forces_[i].transpose();
    return out;
}
// Independent force oracle for the labels (no production observation helper).
Eigen::VectorXd force_difference(AnyPotential &p,const Atoms &a,const Eigen::VectorXd &direction,double step=1e-4) {
    double maximum=0.;for(int i=0;i<a.size();++i) maximum=std::max(maximum,direction.segment<3>(3*i).norm());
    const double epsilon=step/maximum;auto plus=a,minus=a;
    for(int i=0;i<a.size();++i) { plus.pos_[i]+=epsilon*direction.segment<3>(3*i).transpose();minus.pos_[i]-=epsilon*direction.segment<3>(3*i).transpose(); }
    p.getEFS(plus);p.getEFS(minus);return -(flatten_forces(plus)-flatten_forces(minus))/(2.*epsilon);
}
void label(AnyPotential &p,std::vector<Atoms> &data,std::vector<Neighborhoods> &neighbors,const std::string &channel,int count=5) {
    for(int k=0;k<count;++k) {
        auto a=geometry(k);
        if(dynamic_cast<PotMTP_LES *>(&p)) a.pbc_={true,true,true};
        p.getEFS(a);const int ndof=3*a.size();
        if(channel=="hvp" || channel=="both") {
            a.hvp_directions_.resize(ndof,2);a.hvp_targets_.resize(ndof,2);
            for(int j=0;j<2;++j) {
                for(int i=0;i<ndof;++i) a.hvp_directions_(i,j)=std::sin(.3*(i+1)*(j+1));
                a.hvp_targets_.col(j)=force_difference(p,a,a.hvp_directions_.col(j));
            }
        }
        if(channel=="fullh" || channel=="both") {
            a.hessian_.resize(ndof,ndof);a.have_hessian=true;
            for(int j=0;j<ndof;++j) a.hessian_.col(j)=force_difference(p,a,Eigen::VectorXd::Unit(ndof,j));
        }
        data.push_back(a);neighbors.emplace_back(a,p.cutoff);
    }
}
EFSFitWeights weights() { EFSFitWeights w;w.energy=.7;w.force=.05;w.stress=.002;w.norm_f=.4;return w; }
LinearRefitSettings refit_settings(const std::string &channel) {
    LinearRefitSettings s;s.mode="sr";s.report="";
    s.hvp={channel=="hvp" || channel=="both" ? .4 : 0.,.6,1e-4};
    s.hessian={channel=="fullh" || channel=="both" ? .3 : 0.,.8,1e-4};return s;
}
GPSettings trainer_settings(const LinearRefitSettings &r) {
    GPSettings s;s.Modify("energy_weight",".7");s.Modify("force_weight",".05");s.Modify("stress_weight",".002");
    s.Modify("force_wgt_f",".4");s.Modify("weight_L2","0");s.Modify("penalty_weight","0");
    s.Modify("hvp_weight",std::to_string(r.hvp.weight));s.Modify("hvp_pow",".6");s.Modify("hvp_fd_step",".0001");
    s.Modify("hessian_weight",std::to_string(r.hessian.weight));s.Modify("hessian_pow",".8");s.Modify("hessian_fd_step",".0001");return s;
}
Eigen::VectorXd coefficients(AnyPotential &p) { Eigen::VectorXd x(p.get_NL_coeffs_count());p.nl_get_coeffs(x.data());return x; }
void configure_sog(PotMTP_LES &p) {
    p.les_kernel="sog";p.sog_terms=2;p.sog_channels=1;p.train_init();
    p.les_species_coeffs[0]=.23;p.les_species_coeffs[1]=-.17;
    std::fill_n(p.les_moment_coeffs,p.alpha_scalar_moments,0.);
    p.les_moment_coeffs[0]=.35;p.les_moment_coeffs[1]=-.21;
    p.model_les.sog_weights[0]=1.2;p.model_les.sog_weights[1]=-.6;
    p.model_les.sog_log_bandwidths[0]=std::log(.7);p.model_les.sog_log_bandwidths[1]=std::log(1.2);
    p.train_init();p.set_inited();
}
}

TEST_CASE("Narrow refit uses consistent HVP and full-Frobenius force-difference observations", "[refit][refit-curvature]") {
    const auto channel=GENERATE(std::string("hvp"),std::string("fullh"),std::string("both"));
    const bool sog=GENERATE(false,true);
    std::unique_ptr<PotMTP> p;
    if(sog) { auto model=std::make_unique<PotMTP_LES>("unitest/test-data/portable-mtp/les.mtp");configure_sog(*model);p=std::move(model); }
    else { p=std::make_unique<PotMTP>("unitest/test-data/portable-mtp/les.mtp");p->train_init();p->set_inited(); }
    std::vector<Atoms> data;std::vector<Neighborhoods> neighbors;label(*p,data,neighbors,channel);
    auto options=refit_settings(channel);if(sog) options.mode="sr_sog";
    TrainNonlinear trainer(trainer_settings(options));trainer.init(data,neighbors,p.get());
    p->linear_coeffs[0]+=.03;p->linear_coeffs[1]-=.02;
    for(int i=0;i<3;++i) p->moment_coeffs[i]+=.02*std::cos(i+1.);
    if(sog) { auto &model=static_cast<PotMTP_LES &>(*p);model.model_les.sog_weights[0]+=.2;model.model_les.sog_weights[1]-=.1; }
    const auto before=coefficients(*p);
    const auto result=linear_refit(*p,data,neighbors,weights(),0.,options);
    const auto after=coefficients(*p);
    REQUIRE(result.after<result.before*1e-5);
    REQUIRE(result.after==Approx(result.predicted).margin(1e-10));
    REQUIRE((after.head(p->radial_coeff_size).array()==before.head(p->radial_coeff_size).array()).all());
    if(sog) {
        auto &model=static_cast<PotMTP_LES &>(*p);
        const int offset=p->reg_coeff_size;
        const int count=model.latent_species_count()+model.latent_moment_count();
        REQUIRE((after.segment(offset,count).array()==before.segment(offset,count).array()).all());
    }
    double value;Eigen::VectorXd gradient(after.size());trainer.AccumulateGard(value,gradient);
    REQUIRE(value==Approx(result.after).margin(2e-10));
    REQUIRE(gradient.segment(p->radial_coeff_size,p->lin_coeff_size).lpNorm<Eigen::Infinity>()<1e-6);
    if(sog) {
        auto &model=static_cast<PotMTP_LES &>(*p);
        const int offset=p->reg_coeff_size+model.latent_species_count()+model.latent_moment_count();
        REQUIRE(gradient.segment(offset,model.kernel_count()).lpNorm<Eigen::Infinity>()<1e-6);
        REQUIRE((after.tail(model.kernel_count()).array()==before.tail(model.kernel_count()).array()).all());
    }
}

TEST_CASE("Profiled curvature objective has a stationary inner solve and an envelope gradient", "[varpro][refit-curvature]") {
    const auto channel=GENERATE(std::string("hvp"),std::string("fullh"),std::string("both"));
    PotMTP p("unitest/test-data/portable-mtp/les.mtp");p.train_init();p.set_inited();
    std::vector<Atoms> data;std::vector<Neighborhoods> neighbors;label(p,data,neighbors,channel,7);
    auto options=refit_settings(channel);auto s=trainer_settings(options);s.Modify("weight_L2",".0001");
    TrainNonlinear trainer(s);trainer.init(data,neighbors,&p);
    const auto original=coefficients(p);
    MTPProfiledObjective objective(p,data,neighbors,weights(),0.,options,
        [&](double &value,Eigen::VectorXd &gradient) { trainer.AccumulateGard(value,gradient); },2e-6,3);
    Eigen::VectorXd radial=original.head(p.radial_coeff_size);radial[0]+=.004;
    const auto profiled=objective.evaluate(radial);
    REQUIRE(profiled.linear_gradient_inf<2e-6);
    REQUIRE(profiled.objective==Approx(profiled.linear.after).margin(2e-10));
    REQUIRE((coefficients(p).array()==original.array()).all());
    for(const double step : {2e-4,8e-5}) {
        auto plus=radial,minus=radial;plus[0]+=step;minus[0]-=step;
        const auto ep=objective.evaluate(plus),em=objective.evaluate(minus);
        REQUIRE(profiled.gradient[0]==Approx((ep.objective-em.objective)/(2.*step)).margin(2e-5).epsilon(3e-3));
        REQUIRE((coefficients(p).array()==original.array()).all());
    }
}

namespace {
struct CurvatureCommandDirectory {
    std::filesystem::path original=std::filesystem::current_path(),path;
    CurvatureCommandDirectory() {
        char pattern[]="/tmp/imrmlp_refit_curvature_cli_XXXXXX";
        const auto *directory=mkdtemp(pattern);
        if(!directory) throw std::runtime_error("Cannot create curvature command directory");
        path=directory;std::filesystem::current_path(path);
    }
    ~CurvatureCommandDirectory() {
        CMDOption::global_cmdopts.reset();std::filesystem::current_path(original);std::filesystem::remove_all(path);
    }
};
int curvature_command(std::vector<std::string> args) {
    CMDOption::global_cmdopts.reset();std::vector<char *> argv;
    for(auto &arg:args) argv.push_back(arg.data());return trainnlpot(argv.size(),argv.data());
}
std::string text_file(const std::string &path) { std::ifstream file(path);return {std::istreambuf_iterator<char>(file),{}}; }
double number_field(const std::string &text,const std::string &name) {
    std::smatch match;
    const bool found=std::regex_search(text,match,std::regex("\\\""+name+"\\\":([-+0-9.eE]+)"));
    if(!found) throw std::runtime_error("Missing report field: "+name);
    return std::stod(match[1]);
}
}

TEST_CASE("trainnlpot curvature VP and conditional refit commands publish complete objective reports", "[apps][refit-curvature]") {
    const auto channel=GENERATE(std::string("hvp"),std::string("fullh"));
    const bool profiled=GENERATE(false,true);
    PotMTP p("unitest/test-data/portable-mtp/les.mtp");p.train_init();p.set_inited();
    std::vector<Atoms> data;std::vector<Neighborhoods> neighbors;label(p,data,neighbors,channel,7);
    p.moment_coeffs[0]+=.02;p.moment_coeffs[1]-=.01;if(profiled) p.radial_coeffs[0]+=.004;
    CurvatureCommandDirectory directory;
    p.Save("input.mtp");GPIO io;REQUIRE(io.write(data,"train.bin","bin"));
    std::ofstream("index.map")<<"Al\nNi\n";
    const auto options=refit_settings(channel);
    auto fit=trainer_settings(options);fit.Modify("save_to","fitted.mtp");fit.Modify("bfgs_maxiteration","1");
    fit.Modify("bfgs_grad_tol","0");fit.Modify("bfgs_conv_tol","0");fit.Modify("bfgs_step_tol","0");
    fit.Modify("linear_refit_report","refit.json");fit.Modify("varpro_report","profile.jsonl");
    if(profiled) fit.Modify("weight_L2",".0001");else fit.Modify("linear_refit","sr");
    {
        std::ofstream settings_file("fit.ini");settings_file<<"pot:pottype mtp\npot:potfile input.mtp\n";
        for(const auto &[key,value]:fit.settings) settings_file<<"fit:"<<key<<' '<<value<<'\n';
    }
    const std::vector<std::string> args={"trainnlpot",profiled ? "--do-varpro" : "--linear-refit-only",
        "--settings=fit.ini","--trainset=train.bin","--trainset-format=bin","--index=index.map","--output=published.mtp"};
    REQUIRE(curvature_command(args)==0);
    REQUIRE(std::filesystem::exists("published.mtp"));REQUIRE(std::filesystem::exists("fitted.mtp"));
    PotMTP published("published.mtp"),fitted("fitted.mtp");
    REQUIRE((coefficients(published).array()==coefficients(fitted).array()).all());
    if(profiled) {
        std::istringstream report(text_file("profile.jsonl"));std::string line,final;
        while(std::getline(report,line)) if(line.find("\"event\":\"final\"")!=std::string::npos) final=line;
        REQUIRE_FALSE(final.empty());
        const double data_loss=number_field(final,"energy")+number_field(final,"force")+number_field(final,"virial")
            +number_field(final,"hvp")+number_field(final,"hessian");
        REQUIRE(data_loss==Approx(number_field(final,"data")).margin(1e-12));
        REQUIRE(data_loss+number_field(final,"radial_penalty")+number_field(final,"linear_penalty")
            ==Approx(number_field(final,"total")).margin(1e-12));
        REQUIRE(number_field(final,"linear_gradient_inf")<1e-7);
    } else {
        const auto report=text_file("refit.json");
        REQUIRE(number_field(report,"after")==Approx(number_field(report,"predicted")).margin(1e-10));
        REQUIRE(number_field(report,channel=="hvp" ? "hvp_weight" : "hessian_weight")>0.);
        REQUIRE(number_field(report,"after")<number_field(report,"before"));
    }
    // A failed second CLI invocation must retain already published results.
    const auto saved=text_file("published.mtp");
    if(!profiled) {
        auto settings_text=text_file("fit.ini");
        const auto position=settings_text.find("refit.json");REQUIRE(position!=std::string::npos);
        settings_text.replace(position,std::string("refit.json").size(),"missing/report.json");
        std::ofstream("fit.ini")<<settings_text;
    }
    REQUIRE_THROWS(curvature_command(args));
    REQUIRE(text_file("published.mtp")==saved);
}

TEST_CASE("Curvature profile failures restore complete radial and linear coefficients", "[varpro][refit-curvature]") {
    PotMTP p("unitest/test-data/portable-mtp/les.mtp");p.train_init();p.set_inited();
    std::vector<Atoms> data;std::vector<Neighborhoods> neighbors;label(p,data,neighbors,"both",5);
    const auto original=coefficients(p);auto options=refit_settings("both");
    MTPProfiledObjective objective(p,data,neighbors,weights(),0.,options,
        [](double &,Eigen::VectorXd &) { throw std::runtime_error("injected complete-curvature gradient failure"); });
    auto radial=original.head(p.radial_coeff_size).eval();radial[0]+=.004;
    REQUIRE_THROWS_WITH(objective.evaluate(radial),Catch::Matchers::ContainsSubstring("complete-curvature"));
    REQUIRE((coefficients(p).array()==original.array()).all());
}
