// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#include <catch2/catch_all.hpp>
#include "train/train_mtp.h"
#include <Eigen/Geometry>
#include <filesystem>
#include <fstream>
#include <sstream>

namespace {
struct TrainingData {
    Potentials::PotMTP potential{"unitest/test-data/portable-mtp/les.mtp"};
    std::vector<Atoms> atoms;
    std::vector<Neighborhoods> neighborhoods;

    TrainingData() {
        potential.train_init();
        potential.set_inited();
        for (int k = 0; k < 16; ++k) {
            Atoms a;
            a.id = k;
            a.cell_ = Eigen::Matrix3d::Identity() * 12.;
            a.pos_ = {{2., 2., 2.}, {4.1 + .2 * std::sin(k), 2.2, 2.},
                      {2.3, 4.2 + .2 * std::cos(k), 2.4},
                      {5., 4.8, 3.1 + .15 * std::sin(2 * k)}};
            a.chemmap_ = {0, 1, 0, 1};
            potential.getEFS(a);
            atoms.push_back(a);
            neighborhoods.emplace_back(a, potential.cutoff);
        }
        for (int i = 0; i < potential.radial_coeff_size; ++i)
            potential.radial_coeffs[i] += .01 * std::cos(i);
        for (int i = 0; i < potential.lin_coeff_size; ++i)
            potential.linear_coeffs[i] += .03 * std::sin(i + 1);
    }
};

GPSettings settings(int limit, bool refit = true) {
    GPSettings s;
    s.Modify("skip_preinit", "true");
    s.Modify("do_orthogonalization", "false");
    s.Modify("linear_fit_bfgs", refit ? "true" : "false");
    s.Modify("bfgs_maxiteration", std::to_string(limit));
    s.Modify("bfgs_grad_tol", "0");
    s.Modify("bfgs_conv_tol", "0");
    s.Modify("bfgs_step_tol", "0");
    s.Modify("save_every", "100000");
    s.Modify("save_to", "mtp-refit-test.mtp");
    return s;
}

int occurrences(const std::string &text, const std::string &needle) {
    int count = 0;
    for (size_t pos = 0; (pos = text.find(needle, pos)) != std::string::npos;
         pos += needle.size())
        ++count;
    return count;
}
}

TEST_CASE("MTP training exposes EFS and Hessian target weights", "[training][mtp]") {
    Trainer_MTP trainer;
    std::ostringstream help;
    trainer.ListSettings(help, "fit:");
    const auto output = help.str();
    for (const auto *key : {"energy_weight", "force_weight", "stress_weight"})
        REQUIRE(output.find(key) != std::string::npos);
    REQUIRE(output.find("hessian_weight") != std::string::npos);
    REQUIRE(output.find("hessian_pow") != std::string::npos);
}

TEST_CASE("MTP EFS training ignores attached Hessian data", "[training][mtp]") {
    const bool fast = GENERATE(false, true);
    auto *log = new std::ostringstream;
    ScopedStreamRedirect redirect(global_ostream, log);
    TrainingData plain, annotated;
    for (auto &atom : annotated.atoms) {
        atom.hessian_ = 1e6 * Eigen::MatrixXd::Identity(3 * atom.size(), 3 * atom.size());
        atom.have_hessian = true;
        REQUIRE(atom.has_hessian());
    }
    auto s = settings(1, false);
    s.Modify("lfit_fastalgo", fast ? "true" : "false");
    Trainer_MTP plain_trainer(s), annotated_trainer(s);
    plain_trainer.init(plain.atoms, plain.neighborhoods, &plain.potential);
    annotated_trainer.init(annotated.atoms, annotated.neighborhoods, &annotated.potential);

    double plain_loss, annotated_loss;
    alglib::real_1d_array plain_gradient, annotated_gradient;
    plain_gradient.setlength(plain.potential.reg_coeff_size);
    annotated_gradient.setlength(annotated.potential.reg_coeff_size);
    plain_trainer.AccumulateGard(plain_loss, plain_gradient);
    annotated_trainer.AccumulateGard(annotated_loss, annotated_gradient);
    REQUIRE(std::isfinite(plain_loss));
    REQUIRE(annotated_loss == Catch::Approx(plain_loss));
    for (int i = 0; i < plain_gradient.length(); ++i)
        REQUIRE(annotated_gradient[i] == Catch::Approx(plain_gradient[i]).margin(1e-12));

    plain_trainer.TrainLinear();
    annotated_trainer.TrainLinear();
    for (int i = 0; i < plain.potential.lin_coeff_size; ++i)
        REQUIRE(annotated.potential.linear_coeffs[i]
            == Catch::Approx(plain.potential.linear_coeffs[i]).margin(1e-12));
    plain_trainer.AccumulateLoss(plain_loss);
    annotated_trainer.AccumulateLoss(annotated_loss);
    REQUIRE(std::isfinite(plain_loss));
    REQUIRE(annotated_loss == Catch::Approx(plain_loss));
}

TEST_CASE("MTP six-row stress export reproduces the full tensor least squares system", "[training][mtp][stress]") {
    TrainingData data;
    Atoms original=data.atoms.front();
    original.stress_(0,0)+=.7;
    original.stress_(0,1)+=.2;original.stress_(1,0)+=.2;
    auto config=settings(0);
    config.Modify("energy_weight","0");config.Modify("force_weight","0");
    config.Modify("stress_weight","1");
    const int n=data.potential.lin_coeff_size;
    const std::string filename="MatrixA_Stress_CMPT_ON_CORE"+std::to_string(mpi.rank);
    Eigen::MatrixXd baseline_gram;
    Eigen::VectorXd baseline_rhs;
    double baseline_target_norm=0;
    for(const double angle:{0.,M_PI/4.}) {
        const Eigen::Matrix3d rotation=Eigen::AngleAxisd(angle,Eigen::Vector3d::UnitZ()).toRotationMatrix();
        Atoms ref=original;
        ref.cell_=(ref.cell_*rotation.transpose()).eval();
        for(auto& r:ref.pos_) r=(r*rotation.transpose()).eval();
        for(auto& f:ref.forces_) f=(rotation*f).eval();
        ref.stress_=(rotation*ref.stress_*rotation.transpose()).eval();
        std::vector<Atoms> frames{ref};
        std::vector<Neighborhoods> nbhs;
        nbhs.emplace_back(frames[0],data.potential.cutoff);
        Trainer_MTP trainer(config);trainer.init(frames,nbhs,&data.potential);
        trainer.get_A();

        std::ifstream input(filename,std::ios::binary);
        REQUIRE(input.is_open());
        Eigen::MatrixXd A(6,n);
        Eigen::VectorXd target(6);
        for(int row=0;row<6;++row) {
            int id=-1;input.read(reinterpret_cast<char*>(&id),sizeof(id));
            REQUIRE(id==ref.id);
            for(int k=0;k<n;++k) input.read(reinterpret_cast<char*>(&A(row,k)),sizeof(double));
            input.read(reinterpret_cast<char*>(&target[row]),sizeof(double));
            REQUIRE(input.good());
        }
        REQUIRE(input.peek()==std::char_traits<char>::eof());
        input.close();std::filesystem::remove(filename);

        std::vector<double> energy;
        Eigen::Tensor<double,3> force,stress;
        data.potential.CalcEFSCmpt(ref,nbhs[0],energy,force,stress);
        Eigen::MatrixXd full(9,n);Eigen::VectorXd full_target(9);
        int row=0;
        for(int i=0;i<3;++i) for(int j=0;j<3;++j) {
            for(int k=0;k<n;++k) full(3*i+j,k)=stress(k,i,j);
            full_target[3*i+j]=ref.stress_(i,j);
            if(j>=i) {
                const double factor=i==j?1.:std::sqrt(2.);
                REQUIRE((A.row(row)-factor*full.row(3*i+j)).norm()<1e-12);
                REQUIRE(target[row]==Catch::Approx(factor*full_target[3*i+j]).margin(1e-12));
                ++row;
            }
        }
        const Eigen::MatrixXd gram=A.transpose()*A;
        const Eigen::VectorXd rhs=A.transpose()*target;
        REQUIRE((gram-full.transpose()*full).norm()<1e-10);
        REQUIRE((rhs-full.transpose()*full_target).norm()<1e-10);
        REQUIRE(target.squaredNorm()==Catch::Approx(full_target.squaredNorm()).margin(1e-10));
        if(angle==0) {baseline_gram=gram;baseline_rhs=rhs;baseline_target_norm=target.squaredNorm();}
        else {
            REQUIRE((gram-baseline_gram).norm()<1e-10);
            REQUIRE((rhs-baseline_rhs).norm()<1e-10);
            REQUIRE(target.squaredNorm()==Catch::Approx(baseline_target_norm).margin(1e-10));
        }
    }
}


TEST_CASE("MTP ALGLIB refits at accepted iterations across restarts", "[training][mtp][bfgs]") {
    auto *log = new std::ostringstream;
    ScopedStreamRedirect redirect(global_ostream, log);
    TrainingData data;
    Trainer_MTP trainer(settings(201));
    trainer.init(data.atoms, data.neighborhoods, &data.potential);
    double before, after;
    trainer.AccumulateLoss(before);
    trainer.TrainBFGS();
    trainer.AccumulateLoss(after);
    REQUIRE(trainer.bfgs_iter == 201);
    REQUIRE(std::isfinite(after));
    REQUIRE(after < before);
    const std::string output = log->str();
    REQUIRE(output.find("f_hess") == std::string::npos);
    REQUIRE(output.find("HRMS") == std::string::npos);
    REQUIRE(occurrences(output, "BFGS linear fit at iteration ") == 4);
    for (int step : {25, 100, 200}) {
        REQUIRE(occurrences(output, "BFGS linear fit at iteration "
            + std::to_string(step) + " (scheduled)") == 1);
    }
    REQUIRE(occurrences(output, "BFGS linear fit at iteration 201 (end)") == 1);
    REQUIRE(occurrences(output, "BFGS restart after iteration ") == 3);
    REQUIRE(occurrences(output, "Iteration ") == 201);
    alglib::real_1d_array x;
    alglib::minlbfgsreport report;
    alglib::minlbfgsresults(trainer.bfgsstate, x, report);
    REQUIRE(report.iterationscount == 1); // Only the last segment is in ALGLIB's report.
}

TEST_CASE("MTP training rejects fixing every coefficient", "[training][mtp][errors]") {
    TrainingData data;
    auto config = settings(1);
    config.Modify("fix_linear_coeffs", "true");
    config.Modify("fix_radial_coeffs", "true");
    Trainer_MTP trainer(config);
    REQUIRE_THROWS_WITH(trainer.init(data.atoms, data.neighborhoods, &data.potential),
                        Catch::Matchers::ContainsSubstring("Nothing to be optimized"));
}

TEST_CASE("MTP loss and gradient paths apply the same atomic force weights", "[training][mtp][weights]") {
    const double weight = GENERATE(0., .25, 2.);
    TrainingData data;
    data.atoms.resize(1);
    data.neighborhoods.resize(1);
    auto &atom = data.atoms.front();
    data.potential.getEFS(atom);
    atom.forces_[0][0] += 1.;
    atom.atomic_fitness = {weight, 1., 1., 1.};
    auto config = settings(1);
    config.Modify("energy_weight", "0");
    config.Modify("force_weight", "1");
    config.Modify("stress_weight", "0");
    config.Modify("penalty_weight", "0");
    config.Modify("weight_L2", "0");
    Trainer_MTP trainer(config);
    trainer.init(data.atoms, data.neighborhoods, &data.potential);
    alglib::real_1d_array gradient;
    gradient.setlength(trainer.get_relaxed_coeffs_count());
    double loss, gradient_loss;
    trainer.AccumulateGard(gradient_loss, gradient);
    trainer.AccumulateLoss(loss);
    REQUIRE(gradient_loss == Catch::Approx(weight).margin(1e-14));
    REQUIRE(loss == Catch::Approx(weight).margin(1e-14));
}

TEST_CASE("MTP SGD updates every trainable coordinate with the averaged penalized gradient", "[training][mtp][sgd]") {
    const std::string coordinates = GENERATE("all", "radial", "linear");
    const bool clipped = GENERATE(false, true);
    CAPTURE(coordinates, clipped);
    TrainingData data;
    auto &model = data.potential;
    model.set_scale(.2);
    Atoms atom = data.atoms.front();
    model.getEFS(atom);
    atom.energy_ += 1.;
    data.atoms.assign(2, atom);
    data.neighborhoods.assign(2, Neighborhoods(atom, model.cutoff));
    auto config = settings(1);
    config.Modify("fix_species_coeffs", "true");
    config.Modify("fix_radial_coeffs", coordinates == "linear" ? "true" : "false");
    config.Modify("fix_linear_coeffs", coordinates == "radial" ? "true" : "false");
    config.Modify("force_weight", "0");
    config.Modify("stress_weight", "0");
    config.Modify("penalty_weight", "0.01");
    config.Modify("weight_L2", "0.02");
    config.Modify("epoch_limit", "1");
    config.Modify("batch_size", "2");
    config.Modify("learning_rate", "0.0001");
    config.Modify("momentum", "0");
    config.Modify("clip_gradient", clipped ? "0.001" : "0");
    Trainer_MTP trainer(config);
    trainer.init(data.atoms, data.neighborhoods, &model);
    alglib::real_1d_array before, gradient, after;
    before.setlength(trainer.get_relaxed_coeffs_count());
    gradient.setlength(before.length());
    after.setlength(before.length());
    model.ae_get_coeffs(before, trainer.fix_radial_coeffs, trainer.fix_linear_coeffs);
    const Eigen::VectorXd species = Eigen::Map<Eigen::VectorXd>(model.species_coeffs, model.species_count);
    double loss;
    trainer.AccumulateGard(loss, gradient);
    trainer.TrainSGD();
    model.ae_get_coeffs(after, trainer.fix_radial_coeffs, trainer.fix_linear_coeffs);
    for (int i = 0; i < before.length(); ++i) {
        CAPTURE(i);
        const double g = clipped ? std::clamp(gradient[i], -.001, .001) : gradient[i];
        REQUIRE(after[i] == Catch::Approx(before[i] - .0001 * g).epsilon(1e-12).margin(1e-14));
    }
    for (int i = 0; i < model.species_count; ++i)
        REQUIRE(model.species_coeffs[i] == species[i]);
}

TEST_CASE("MTP BFGS preconditioning supports zero and tiny coefficients", "[training][mtp][bfgs]") {
    const double initial = GENERATE(0., 1e-14);
    TrainingData data;
    auto &model = data.potential;
    model.set_scale(.2);
    model.species_coeffs[0] = initial;
    auto config = settings(1, false);
    config.Modify("fix_radial_coeffs", "true");
    config.Modify("precondition", "true");
    config.Modify("weight_L2", "0.01");
    Trainer_MTP trainer(config);
    trainer.init(data.atoms, data.neighborhoods, &model);
    double before, after;
    trainer.AccumulateLoss(before);
    REQUIRE_NOTHROW(trainer.TrainNonLinearBFGS(1));
    trainer.AccumulateLoss(after);
    REQUIRE(std::isfinite(after));
    REQUIRE(after < before);
    // Each four-atom structure has two atoms of species 0: d2L/dc0^2 = 2 * 2^2 / 4.
    REQUIRE(trainer.hess_diag[0] == Catch::Approx(2.).epsilon(1e-5));
    for (int i = 0; i < trainer.hess_diag.length(); ++i) {
        REQUIRE(std::isfinite(trainer.hess_diag[i]));
        REQUIRE(trainer.hess_diag[i] > 0.);
    }
}

TEST_CASE("MTP curvature checks restore the potential coefficients", "[training][mtp][diagnostics]") {
    TrainingData data;
    auto &model = data.potential;
    Trainer_MTP trainer(settings(1));
    trainer.init(data.atoms, data.neighborhoods, &model);
    const Eigen::VectorXd original = Eigen::Map<Eigen::VectorXd>(model.regression_coeffs, model.reg_coeff_size);
    trainer.test_curv(0, 1e-5);
    for (int i = 0; i < model.reg_coeff_size; ++i)
        REQUIRE(model.regression_coeffs[i] == original[i]);
}

TEST_CASE("MTP fixed species retain penalties for every moment coefficient", "[training][mtp][regularization]") {
    TrainingData data;
    auto config = settings(1);
    config.Modify("fix_species_coeffs", "true");
    config.Modify("fix_radial_coeffs", "true");
    config.Modify("weight_L1", "0.02");
    config.Modify("weight_L2", "0.03");
    Trainer_MTP trainer(config);
    auto &model = data.potential;
    trainer.init(data.atoms, data.neighborhoods, &model);
    REQUIRE(model.reg_L1_vector.size() == model.lin_coeff_size);
    REQUIRE(model.reg_L2_vector.size() == model.lin_coeff_size);

    const int first = model.species_count;
    const int last = model.lin_coeff_size - 1;
    std::fill(model.linear_coeffs + first, model.linear_coeffs + last + 1, 0.);
    model.linear_coeffs[first] = .4;
    model.linear_coeffs[last] = -.7;
    double loss = 0.;
    std::vector<double> gradient(model.lin_coeff_size, 0.);
    model.AddPenaltyGard(0., loss, gradient.data(), true, false, true);
    REQUIRE(loss == Catch::Approx(.02 * (.4 + .7) + .03 * (.4 * .4 + .7 * .7)));
    REQUIRE(gradient[first] == Catch::Approx(2 * .03 * .4 + .02));
    REQUIRE(gradient[last] == Catch::Approx(-2 * .03 * .7 - .02));
    for (int i = 0; i < first; ++i)
        REQUIRE(gradient[i] == 0.);
}

TEST_CASE("MTP fixed species linear fits match an augmented ridge solve", "[training][mtp][regularization]") {
    const bool fast = GENERATE(false, true);
    const bool energy_only = GENERATE(false, true);
    const bool adapt_regularization = GENERATE(false, true);
    CAPTURE(fast, energy_only, adapt_regularization);
    TrainingData data;
    auto &model = data.potential;
    model.set_scale(.2);
    auto config = settings(1);
    config.Modify("fix_species_coeffs", "true");
    config.Modify("lfit_fastalgo", fast ? "true" : "false");
    config.Modify("lfit_eng_only", energy_only ? "true" : "false");
    config.Modify("force_weight", "0");
    config.Modify("stress_weight", "0");
    config.Modify("weight_L2", "0.01");
    Trainer_MTP trainer(config);
    trainer.init(data.atoms, data.neighborhoods, &model);
    REQUIRE(model.reg_L2_vector.size() == model.lin_coeff_size);
    const int n = model.lin_coeff_size - model.species_count;
    const int rows = data.atoms.size();
    const Eigen::VectorXd species = Eigen::Map<Eigen::VectorXd>(model.species_coeffs, model.species_count);
    Eigen::MatrixXd augmented = Eigen::MatrixXd::Zero(rows + n, n);
    Eigen::VectorXd target = Eigen::VectorXd::Zero(rows + n);
    for (int k = 0; k < rows; ++k) {
        std::vector<double> basis;
        model.CalcECmpt(data.atoms[k], data.neighborhoods[k], basis);
        const double weight = 1. / std::sqrt(data.atoms[k].size());
        double energy = data.atoms[k].energy_;
        for (int i = 0; i < model.species_count; ++i) energy -= basis[i] * species[i];
        target[k] = weight * energy;
        for (int i = 0; i < n; ++i)
            augmented(k, i) = weight * basis[model.species_count + i];
    }
    Eigen::VectorXd penalties(n);
    for (int i = 0; i < n; ++i) {
        const double diagonal = augmented.col(i).squaredNorm();
        const double initial = .01 * std::max(1., diagonal / rows) * (1. + .01 * i);
        // Zero weights trigger the trainer's automatic regularization update.
        model.reg_L2_vector[model.species_count + i] = adapt_regularization ? 0. : initial;
        penalties[i] = adapt_regularization ? .01 * std::max(1., diagonal) / rows : initial;
        augmented(rows + i, i) = std::sqrt(rows * penalties[i]);
    }
    const Eigen::VectorXd reference = augmented.colPivHouseholderQr().solve(target);
    trainer.TrainLinear();
    const Eigen::VectorXd fitted = Eigen::Map<Eigen::VectorXd>(model.moment_coeffs, n);
    REQUIRE((augmented * (fitted - reference)).norm() < 1e-8 * (1. + target.norm()));
    for (int i = 0; i < n; ++i)
        REQUIRE(model.reg_L2_vector[model.species_count + i] == Catch::Approx(penalties[i]));
    for (int i = 0; i < model.species_count; ++i)
        REQUIRE(model.species_coeffs[i] == species[i]);
}

TEST_CASE("MTP ALGLIB respects the passed limit and saves accepted parameters", "[training][mtp][bfgs]") {
    auto *log = new std::ostringstream;
    ScopedStreamRedirect redirect(global_ostream, log);
    TrainingData data;
    auto s = settings(1000, false);
    s.Modify("save_every", "1");
    Trainer_MTP trainer(s);
    trainer.init(data.atoms, data.neighborhoods, &data.potential);
    trainer.TrainNonLinearBFGS(1);
    REQUIRE(trainer.bfgs_iter == 1);
    alglib::real_1d_array x;
    alglib::minlbfgsreport report;
    alglib::minlbfgsresults(trainer.bfgsstate, x, report);
    REQUIRE(report.iterationscount == 1);
    REQUIRE(report.nfev > report.iterationscount);
    Potentials::PotMTP saved("_Current_potential_save_1.mtp");
    REQUIRE(saved.reg_coeff_size == data.potential.reg_coeff_size);
    for (int i = 0; i < saved.reg_coeff_size; ++i) {
        REQUIRE(saved.regression_coeffs[i] == x[i]);
        REQUIRE(data.potential.regression_coeffs[i] == x[i]);
    }
    REQUIRE(occurrences(log->str(), "Iteration ") == 1);
}

TEST_CASE("MTP ALGLIB disabled refits leave an uninterrupted run", "[training][mtp][bfgs]") {
    auto *log = new std::ostringstream;
    ScopedStreamRedirect redirect(global_ostream, log);
    TrainingData data;
    Trainer_MTP trainer(settings(26, false));
    trainer.init(data.atoms, data.neighborhoods, &data.potential);
    trainer.TrainBFGS();
    REQUIRE(trainer.bfgs_iter == 26);
    REQUIRE(log->str().find("BFGS linear fit") == std::string::npos);
    alglib::real_1d_array x;
    alglib::minlbfgsreport report;
    alglib::minlbfgsresults(trainer.bfgsstate, x, report);
    REQUIRE(report.iterationscount == 26);
}

TEST_CASE("MTP ALGLIB merges a scheduled and final refit", "[training][mtp][bfgs]") {
    const bool fast = GENERATE(false, true);
    const bool energy_only = GENERATE(false, true);
    auto *log = new std::ostringstream;
    ScopedStreamRedirect redirect(global_ostream, log);
    TrainingData data;
    auto s = settings(25);
    s.Modify("lfit_fastalgo", fast ? "true" : "false");
    s.Modify("lfit_eng_only", energy_only ? "true" : "false");
    Trainer_MTP trainer(s);
    trainer.init(data.atoms, data.neighborhoods, &data.potential);
    trainer.TrainBFGS();
    REQUIRE(trainer.bfgs_iter == 25);
    REQUIRE(occurrences(log->str(), "BFGS linear fit at iteration ") == 1);
    REQUIRE(occurrences(log->str(), "BFGS linear fit at iteration 25 (end)") == 1);
    REQUIRE(log->str().find("BFGS restart") == std::string::npos);
}

TEST_CASE("MTP ALGLIB initial convergence still performs the final refit", "[training][mtp][bfgs]") {
    auto *log = new std::ostringstream;
    ScopedStreamRedirect redirect(global_ostream, log);
    TrainingData data;
    auto s = settings(201);
    s.Modify("bfgs_grad_tol", "1e30");
    Trainer_MTP trainer(s);
    trainer.init(data.atoms, data.neighborhoods, &data.potential);
    trainer.TrainBFGS();
    REQUIRE(trainer.bfgs_iter == 0);
    REQUIRE(occurrences(log->str(), "BFGS linear fit at iteration ") == 1);
    REQUIRE(occurrences(log->str(), "BFGS linear fit at iteration 0 (end)") == 1);
}

TEST_CASE("MTP ALGLIB pretraining has its own iteration budget", "[training][mtp][bfgs]") {
    auto *log = new std::ostringstream;
    ScopedStreamRedirect redirect(global_ostream, log);
    TrainingData data;
    auto s = settings(3);
    s.Modify("skip_preinit", "false");
    s.Modify("pre_init_iter", "2");
    Trainer_MTP trainer(s);
    trainer.init(data.atoms, data.neighborhoods, &data.potential);
    trainer.TrainBFGS();
    REQUIRE(trainer.bfgs_iter == 3);
    REQUIRE(occurrences(log->str(), "Iteration ") == 5);
    REQUIRE(occurrences(log->str(), "Iteration 1 :") == 2);
    REQUIRE(occurrences(log->str(), "BFGS linear fit at iteration ") == 1);
}

TEST_CASE("MTP ALGLIB reloads radial-only optimizer parameters after refitting", "[training][mtp][bfgs]") {
    auto *log = new std::ostringstream;
    ScopedStreamRedirect redirect(global_ostream, log);
    TrainingData data;
    auto s = settings(26);
    s.Modify("fix_linear_coeffs", "true");
    Trainer_MTP trainer(s);
    trainer.init(data.atoms, data.neighborhoods, &data.potential);
    trainer.TrainNonLinearBFGS(26);
    REQUIRE(trainer.bfgs_iter == 26);
    REQUIRE(occurrences(log->str(), "BFGS linear fit at iteration 25 (scheduled)") == 1);
    alglib::real_1d_array x;
    alglib::minlbfgsreport report;
    alglib::minlbfgsresults(trainer.bfgsstate, x, report);
    REQUIRE(x.length() == data.potential.radial_coeff_size);
    for (int i = 0; i < x.length(); ++i)
        REQUIRE(data.potential.radial_coeffs[i] == x[i]);
}
