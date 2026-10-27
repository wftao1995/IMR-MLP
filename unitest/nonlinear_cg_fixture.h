#pragma once
#include "train/train_nonlinear_potential.h"
#include "utils/comm.h"
#include <filesystem>
#include <fstream>
#include <limits>
#include <unistd.h>

namespace CGFixture {
namespace fs = std::filesystem;
class Potential : public Potentials::AnyPotential {
public:
    enum class Fault { none, energy, gradient, nan_energy, nan_gradient, penalty, nan_penalty,
                       validation, nan_validation, validation_getter, alglib };
    Eigen::Vector2d coefficients{3., .5};
    Fault fault = Fault::none;
    int fail_after = 0, fail_rank = 0, evaluations = 0, validations = 0, saves = 0;
    int coefficient_count = 2;
    bool fail_save = false, empty_save = false, sidecar_save = false;
    mutable bool pending_getter_failure = false;
    Potential() { set_inited(); }
    bool failing() const { return mpi.rank == fail_rank && evaluations > fail_after; }
    int get_NL_coeffs_count() const override { return coefficient_count; }
    void nl_get_coeffs(double *out) const override {
        if (pending_getter_failure) {
            pending_getter_failure = false;
            throw std::runtime_error("injected CG validation getter failure");
        }
        Eigen::Map<Eigen::Vector2d> mapped(out); mapped = coefficients;
    }
    void nl_set_coeffs(const double *in) override { coefficients = Eigen::Map<const Eigen::Vector2d>(in); }
    std::vector<Potentials::NLParameterGroup> nl_parameter_groups() const override {
        return {{"radial",0,1},{"linear",1,2}};
    }
    void getEFS(Atoms &a, AnyNeighborhoods &) override {
        ++evaluations;
        if (a.id == 42) {
            ++validations;
            if (mpi.rank == fail_rank && fault == Fault::validation)
                throw std::runtime_error("injected CG validation failure");
            if (mpi.rank == fail_rank && fault == Fault::validation_getter)
                pending_getter_failure = true;
        }
        if (failing() && fault == Fault::energy) throw std::runtime_error("injected CG energy failure");
        if (failing() && fault == Fault::alglib) {
            alglib::real_1d_array point; point.setlength(1); point[0] = 1.;
            alglib::mincgstate invalid;
            alglib::mincgcreate(0, point, invalid);
        }
        const double c2 = coefficients[0]*coefficients[0];
        const double r = a.pos_[0][0];
        a.energy_ = c2*r+coefficients[1]; a.have_energy = true;
        a.forces_.assign(a.size(), Eigen::Vector3d::Zero()); a.forces_[0][0] = -c2;
        a.stress_.setZero(); a.stress_(0,0) = -c2*r; a.have_stress = true;
        if ((failing() && fault == Fault::nan_energy)
            || (a.id == 42 && mpi.rank == fail_rank && fault == Fault::nan_validation))
            a.energy_ = std::numeric_limits<double>::quiet_NaN();
    }
    void AccumulateEFSCombinationGard(Atoms &a, AnyNeighborhoods &, double dE, double *g) override {
        if (failing() && fault == Fault::gradient) throw std::runtime_error("injected CG gradient failure");
        g[0] += failing() && fault == Fault::nan_gradient ? std::numeric_limits<double>::infinity()
            : dE*2.*coefficients[0]*a.pos_[0][0];
        g[1] += dE;
    }
    void AccumulateEFSCombinationGard(Atoms &a, AnyNeighborhoods &n, double dE,
        Eigen::Matrix<double,-1,3> &dF, Eigen::Matrix3d &dS,
        Eigen::Matrix<double,-1,3,Eigen::RowMajor> &, double *g) override {
        AccumulateEFSCombinationGard(a,n,dE,g);
        g[0] -= 2.*coefficients[0]*(dF(0,0)+dS(0,0)*a.pos_[0][0]);
    }
    void AddPenaltyGard(double, double &loss, double *) override {
        if (mpi.rank == fail_rank && fault == Fault::penalty) throw std::runtime_error("injected CG penalty failure");
        if (mpi.rank == fail_rank && fault == Fault::nan_penalty) loss = std::numeric_limits<double>::infinity();
    }
    void Save(const std::string &path) override {
        ++saves;
        if (empty_save) return;
        std::ofstream out(path);
        out << std::setprecision(17) << coefficients[0] << ' ' << coefficients[1] << '\n';
        out.close();
        if (!out) throw std::runtime_error("CG fixture failed to write");
        if (sidecar_save) std::ofstream(path+".sidecar") << "unsupported sidecar\n";
        if (fail_save) throw std::runtime_error("injected CG partial save failure");
    }
};

struct Case {
    Potential potential;
    std::vector<Atoms> data;
    std::vector<Neighborhoods> neighbors;
    fs::path directory;
    Case() {
        static int sequence = 0;
        int owner = getpid(); MPI_Bcast(&owner, 1, MPI_INT, 0, mpi.comm);
        directory = fs::path("build/todo-audit-20261018/batch7/cg-isolated") /
            ("case-"+std::to_string(owner)+"-"+std::to_string(++sequence));
        if (mpi.rank == 0) fs::create_directories(directory);
        MPI_Barrier(mpi.comm);
        for (double r : {1., 2., 3.}) {
            Atoms a; a.cell_ = Eigen::Matrix3d::Identity()*10.; a.pos_ = {{r,1.,1.}}; a.chemmap_ = {0};
            a.energy_ = r+1.; a.have_energy = true; a.fitness = 1.;
            a.forces_ = {{-1.,0.,0.}};
            a.stress_.setZero(); a.stress_(0,0) = -r; a.have_stress = true;
            data.push_back(a); neighbors.emplace_back(data.back(), potential.cutoff);
        }
    }
    ~Case() {
        MPI_Barrier(mpi.comm);
        if (mpi.rank == 0) { std::error_code ec; fs::remove_all(directory,ec); }
    }
    GPSettings settings() const {
        GPSettings s;
        for (const auto &[k,v] : std::vector<std::pair<std::string,std::string>>{
            {"energy_weight","1"},{"force_weight","0"},{"stress_weight","0"},{"penalty_weight","0"},
            {"weight_L1","0"},{"weight_L2","0"},{"bfgs_m","2"},{"bfgs_maxiteration","1"},
            {"bfgs_grad_tol","0"},{"bfgs_conv_tol","0"},{"bfgs_step_tol","0"},{"save_every","1"}}) s.Modify(k,v);
        s.Modify("save_to",(directory/"fitted.mtp").string()); return s;
    }
    void preserve() const {
        if (mpi.rank == 0) std::ofstream(directory/"fitted.mtp") << "previous successful model\n";
        MPI_Barrier(mpi.comm);
    }
    std::string saved() const {
        std::ifstream in(directory/"fitted.mtp"); return std::string(std::istreambuf_iterator<char>(in),{});
    }
    double energy_loss() {
        double loss = 0.;
        for (std::size_t i=0; i<data.size(); ++i) {
            Atoms out=data[i]; potential.getEFS(out,neighbors[i]);
            loss += std::pow(out.energy_-data[i].energy_,2);
        }
        return loss / data.size();
    }
};
}
