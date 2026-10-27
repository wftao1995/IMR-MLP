// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#include "io/gpio.h"
#include "external/spglib_driver.h"
#include "base/neighborhoodKD.h"
#include "symmetry/symmetry.h"
#include <fmt/format.h>
#include <numeric>

namespace {
Eigen::Matrix3i directed_orthogonal_cell(const Eigen::Matrix3d &cell,
    const Eigen::RowVector3i &direction, int target, int budget, double tolerance, bool first) {
    // Keep the requested primitive vector fixed; enumerate the other integer
    // rows within the same coefficient/expansion budget as the full search.
    const Eigen::Vector3d a = (direction.cast<double>()*cell).transpose();
    const Eigen::Vector3d ua = a/a.stableNorm();
    const Eigen::Matrix3d inverse = cell.inverse();
    Eigen::Matrix3i best = Eigen::Matrix3i::Zero();
    long long best_copies = static_cast<long long>(budget)+1;
    double best_metric = std::numeric_limits<double>::infinity();
    for (int p = -budget; p <= budget; ++p)
    for (int q = -budget; q <= budget; ++q)
    for (int r = -budget; r <= budget; ++r) {
        const Eigen::RowVector3i second(p, q, r);
        if (std::gcd(std::gcd(p, q), r) != 1) continue;
        const int leading = p != 0 ? p : q != 0 ? q : r;
        if (leading < 0) continue; // opposite rows generate the same candidates
        const Eigen::Vector3d b = (second.cast<double>()*cell).transpose();
        const Eigen::Vector3d ub = b/b.stableNorm();
        if (std::abs(ua.dot(ub)) > tolerance) continue;
        const Eigen::RowVector3d normal = a.cross(b).transpose()*inverse;
        Eigen::Index pivot;
        normal.cwiseAbs().maxCoeff(&pivot);
        if (!normal.allFinite() || normal[pivot] == 0) continue;
        const Eigen::RowVector3d ratio = normal/std::abs(normal[pivot]);
        for (int scale = 1; scale <= budget; ++scale) {
            Eigen::RowVector3i third = (scale*ratio).array().round().matrix().cast<int>();
            if (std::gcd(std::gcd(third[0], third[1]), third[2]) != 1) continue;
            // Use a wider accumulator before accepting an int-sized index.
            const long double determinant =
                static_cast<long double>(direction[0])*(static_cast<long double>(second[1])*third[2]-static_cast<long double>(second[2])*third[1])
                -static_cast<long double>(direction[1])*(static_cast<long double>(second[0])*third[2]-static_cast<long double>(second[2])*third[0])
                +static_cast<long double>(direction[2])*(static_cast<long double>(second[0])*third[1]-static_cast<long double>(second[1])*third[0]);
            if (determinant == 0 || std::abs(determinant) > budget) continue;
            if (determinant*cell.determinant() < 0) third = -third;
            const Eigen::Vector3d c = (third.cast<double>()*cell).transpose();
            const Eigen::Vector3d uc = c/c.stableNorm();
            const double metric = std::max({std::abs(ua.dot(ub)), std::abs(ua.dot(uc)), std::abs(ub.dot(uc))});
            if (!std::isfinite(metric) || metric > tolerance) continue;
            const int copies = static_cast<int>(std::abs(determinant));
            if (copies < best_copies || (copies == best_copies && metric < best_metric)) {
                best.row(target) = direction;
                best.row((target+1)%3) = second;
                best.row((target+2)%3) = third;
                best_copies = copies;
                best_metric = metric;
                if (first) return best;
            }
        }
    }
    if (best_copies > budget)
        GPERROR("No integer orthogonal completion found within --max-expand coefficient/expansion budget; no strain was applied");
    return best;
}

void align_directed_cell(Atoms &atom, int target) {
    const Eigen::Vector3d a = atom.cell_.row(target).normalized().transpose();
    Eigen::Vector3d b = atom.cell_.row((target+1)%3).transpose();
    b = (b-a*a.dot(b)).normalized();
    Eigen::Matrix3d frame, axes = Eigen::Matrix3d::Zero();
    frame.col(0) = a; frame.col(1) = b; frame.col(2) = a.cross(b);
    axes(target, 0) = axes((target+1)%3, 1) = axes((target+2)%3, 2) = 1;
    const Eigen::Matrix3d rotation = axes*frame.transpose();
    const Eigen::AngleAxisd angle(rotation);
    atom.rotate(angle.axis(), angle.angle());
    // Atoms::rotate handles geometry. Rotate the same spatial properties as
    // structure standardization, retaining their existing supercell semantics.
    for (auto &f : atom.forces_) f = rotation*f;
    for (auto &v : atom.velocities_) v = rotation*v;
    for (auto &m : atom.magmoms_) m = rotation*m;
    auto initial = atom.local_attribs_v_.find("initial_magmoms");
    if (initial != atom.local_attribs_v_.end() && initial->second.rows() == 3 && initial->second.cols() == atom.size())
        initial->second = (rotation*initial->second).eval();
    for (auto &z : atom.born_eff_chgs_) z = (rotation*z*rotation.transpose()).eval();
    if (atom.have_stress) atom.stress_ = (rotation*atom.stress_*rotation.transpose()).eval();
    if (atom.has_hessian())
        for (int i = 0; i < atom.size(); ++i)
            for (int j = 0; j < atom.size(); ++j) {
                auto block = atom.hessian_.block<3,3>(3*i, 3*j);
                block = (rotation*block*rotation.transpose()).eval();
            }
    for (auto *values : {&atom.hvp_directions_, &atom.hvp_targets_})
        if (values->rows() == 3*atom.size())
            for (int i = 0; i < atom.size(); ++i)
                values->middleRows(3*i, 3) = (rotation*values->middleRows(3*i, 3)).eval();
}
}

// Quick orthogonal search, optionally fixing a specified integer direction.
int mk_orth(int argc, char *argv[]) {
    std::vector<int> axis_choice;
    std::string axis_to = "x";
    auto &cmdopts = CMDOption::global_cmdopts;
    GPIO iobj;
    double tol = 1e-2;
    int max_expand = 20;
    std::string strfile = "POSCAR";
    std::string outfile = "POSCAR_ORTH";
    std::string informat = "";
    std::string outformat = "";
    std::string index = "index.map";
    std::vector<Atoms> atoms;
    bool find_one = false;
    bool search_only = false;
    bool overwrite = false;
    cmdopts.addop(tol, "tol", "tol", "tolerance")
            .addop(strfile, "input", "in", "input structure file", true)
            .addop(outfile, "output", "out", "output structure file", true)
            .addop(informat, "input-format", "if")
            .addop(outformat, "output-format", "of")
            .addop(max_expand, "max-expand", "mx")
            .addop(axis_choice, "axis-choice", "", "Three integer lattice-direction coefficients (reduced to primitive). Directed search bounds all integer coefficients and the volume expansion by max-expand; requires full PBC", true, -1)
            .addop(axis_to, "axis-to", "", "Align the specified direction to positive x, y or z by a rigid rotation (default x)")
            .addop(index, "index", "id")
            .addop(find_one, "one", "one", "break search when find one result", false, 0)
            .addop(search_only, "search-only", "s", "only search cell, do not generate supercell", false, 0)
            .addop(overwrite, "overwrite", "ov", "overwrite output file", false, 0)
            .add_request("input")
            .phrase_options(argc, argv);
    if (max_expand < 2)
        GPERROR("--max-expand must be at least 2");
    int target = 0;
    Eigen::RowVector3i direction = Eigen::RowVector3i::Zero();
    if (!axis_choice.empty()) {
        if (axis_choice.size() != 3)
            GPERROR("--axis-choice requires exactly three integer coefficients");
        if (axis_to == "y") target = 1;
        else if (axis_to == "z") target = 2;
        else if (axis_to != "x") GPERROR("--axis-to must be x, y or z");
        const long long divisor = std::gcd(std::gcd(static_cast<long long>(axis_choice[0]), static_cast<long long>(axis_choice[1])), static_cast<long long>(axis_choice[2]));
        if (divisor == 0) GPERROR("--axis-choice must be a nonzero integer direction");
        for (int i = 0; i < 3; ++i) {
            const long long value = axis_choice[i]/divisor;
            if (std::abs(value) > max_expand)
                GPERROR("Primitive --axis-choice exceeds the --max-expand coefficient budget");
            direction[i] = static_cast<int>(value);
        }
        if (!std::isfinite(tol) || tol <= 0 || tol >= 1)
            GPERROR("Directed orthogonal search requires a finite --tol between 0 and 1 (maximum absolute cosine)");
    } else if (cmdopts.was_provided("axis-choice") || cmdopts.was_provided("axis-to")) {
        GPERROR("--axis-to requires a nonempty --axis-choice direction");
    }
    if (GPIO::file_exist(index))
        iobj.readmap(index);
    else 
        iobj.setmap();
    if (GPIO::file_exist(outfile) && !overwrite) {
        std::cerr << "Error: output file " << outfile << " already exists; use --overwrite" << std::endl;
        return 1;
    }
    if (!iobj.read(atoms, strfile, informat) || atoms.empty())
        GPERROR("No structures loaded from " + strfile);
    for (auto &atom : atoms) {
        if (!atom.cell_.allFinite() || std::abs(atom.cell_.determinant()) <= 1e-14)
            GPERROR("Orthogonal cell search requires a finite, non-singular cell");
        if (!axis_choice.empty()) {
            if (!atom.is_fully_periodic() || !std::isfinite(atom.cell_.determinant()))
                GPERROR("Directed integer orthogonal completion requires a finite full-PBC lattice");
            const auto supercell = directed_orthogonal_cell(atom.cell_, direction, target, max_expand, tol, find_one);
            std::cout << "Directed supercell matrix (copies = " << std::abs(supercell.determinant()) << "):\n" << supercell << '\n';
            if (!search_only) {
                atom.mksuperL(supercell);
                align_directed_cell(atom, target);
                atom.wrap();
            }
            continue;
        }
        // if ((abs(atom.cell_.row(0).dot(atom.cell_.row(1))) < 0.1*tol &&
        //     abs(atom.cell_.row(0).dot(atom.cell_.row(2))) < 0.1*tol &&
        //     abs(atom.cell_.row(2).dot(atom.cell_.row(1))) < 0.1*tol)) continue;
        // 0. change direction

        // 1. mklower
        atom.to_lower_standard();
        atom.wrap();
        // 2. search m3, m4, m6
        // a' = m1 a ; b' = m2 a + m3 b ; c' = m4 a + m5 b + m6 c
        // m1 = 1
        // m2 a^2 + m3 ab == 0
        // m4 a^2 + m5 ab + m6 ac == 0
        // m2m4 a^2 + m3m5 b^2 + (m2m5 + m3m4) ab + m2m6 ac + m3m6 bc == 0
        // m3 > 0, m6 > 0, m3m6 < max_expand
        double a2 = atom.cell_.row(0).squaredNorm();
        double b2 = atom.cell_.row(1).squaredNorm();
        double ab = atom.cell_.row(0).dot(atom.cell_.row(1));
        double ac = atom.cell_.row(0).dot(atom.cell_.row(2));
        double bc = atom.cell_.row(1).dot(atom.cell_.row(2));
        std::array<int, 6> best{};
        bool found_best = false;
        double cmin = std::numeric_limits<double>::max();
        int ct = 1;
        for (int m3 = 1; m3 < max_expand; ++m3) {
            int m2 = round(-m3 * ab / a2);
            for (int m6 = 1; m6 < max_expand / m3; ++m6) {
                int m4 = round((ac*b2-ab*bc)*m6/(ab*ab-a2*b2));
                int m5 = round((ab*ac-a2*bc)*m6/(a2*b2-ab*ab));
                double c1 = m2 * a2 + m3 * ab;
                double c2 = m4 * a2 + m5 * ab + m6 * ac;
                double c3 = m2 * m4 * a2 + m3 * m5 * b2 + (m2 * m5 + m3 * m4) * ab + m2 * m6 * ac + m3 * m6 * bc;
                double c = fabs(c1) + fabs(c2) + fabs(c3);
                if (c < cmin) {
                    found_best = true;
                    cmin = c;
                    best[0] = 1;
                    best[1] = m2;
                    best[2] = m3;
                    best[3] = m4;
                    best[4] = m5;
                    best[5] = m6;
                }
                if (c < tol) {
                    std::cout << fmt::format("Solution #{} for structure #{} c = {} n = {}: {} {} {}  {} {} {}\n", ct, atom.id, c, m3 * m6, 1, m2, m3, m4, m5, m6);
                    ++ct;
                    if (find_one) goto lab_out;
                }
            }
        }
        lab_out:
        // use best
        if (!found_best)
            GPERROR("Orthogonal cell search found no valid candidate");
        if (cmin > tol) {
            std::cerr << "Warning: c_min = " << cmin << " is larger than tol" << std::endl;
        } else {
            std::cout << "c_min = " << cmin << std::endl;
        }
        Eigen::Matrix3i sup = Eigen::Matrix3i::Identity();
        sup(1, 0) = best[1];
        sup(1, 1) = best[2];
        sup(2, 0) = best[3];
        sup(2, 1) = best[4];
        sup(2, 2) = best[5];
        std::cout << sup << std::endl;
        if (!search_only) {
            atom.mksuperL(sup);
            Symmetry3D::trans_reduce(atom, 1e-5);
        }
    }
    if (!search_only && !iobj.write(atoms, outfile, outformat))
        GPERROR("Failed to write structures: " + outfile);
    return 0;
}
// full search
int mk_orth_search(int argc, char *argv[]) {
    auto &cmdopts = CMDOption::global_cmdopts;
    GPIO iobj;
    double tol = 1e-2;
    int max_expand = 20;
    std::string strfile = "POSCAR";
    std::string outfile = "POSCAR_ORTH";
    std::string informat = "";
    std::string outformat = "";
    std::string index = "index.map";
    std::vector<Atoms> atoms;
    bool find_one = false;
    bool overwrite = false;
    cmdopts.addop(tol, "tol", "tol", "tolerance")
            .addop(strfile, "input", "in", "input structure file", true)
            .addop(outfile, "output", "out", "output structure file", true)
            .addop(informat, "input-format", "if")
            .addop(outformat, "output-format", "of")
            .addop(max_expand, "max-expand", "mx")
            .addop(index, "index", "id")
            .addop(find_one, "one", "one", "break search when find one result")
            .addop(overwrite, "overwrite", "ov", "overwrite output file", false, 0)
            .add_request("input")
            .phrase_options(argc, argv);
    if (GPIO::file_exist(index))
        iobj.readmap(index);
    else 
        iobj.setmap();
    if (GPIO::file_exist(outfile) && !overwrite) {
        std::cerr << "Error: output file " << outfile << " already exists; use --overwrite" << std::endl;
        return 1;
    }
    if (!iobj.read(atoms, strfile, informat) || atoms.empty())
        GPERROR("No structures loaded from " + strfile);
    for (auto &atom : atoms) {
        Symmetry3D::correct_cell(atom);
        if ((abs(atom.cell_.row(0).dot(atom.cell_.row(1))) < 0.1*tol &&
            abs(atom.cell_.row(0).dot(atom.cell_.row(2))) < 0.1*tol &&
            abs(atom.cell_.row(2).dot(atom.cell_.row(1))) < 0.1*tol)) continue;
        double cmin = std::numeric_limits<double>::max();
        std::vector<Eigen::Matrix3i> sups_valid;
        Eigen::Matrix3i sup_best = Eigen::Matrix3i::Identity();

        double local_cmin = std::numeric_limits<double>::max();
        Eigen::Matrix3i local_sup_best = Eigen::Matrix3i::Identity();
        std::vector<Eigen::Matrix3i> local_sups_valid;
        bool found_one = false;

        #pragma omp parallel firstprivate(local_cmin, local_sup_best) \
                             shared(cmin, sup_best, sups_valid, found_one, atom, tol, max_expand, find_one)
        {
            std::vector<Eigen::Matrix3i> thread_sups_valid;
            double thread_cmin = std::numeric_limits<double>::max();
            Eigen::Matrix3i thread_sup_best = Eigen::Matrix3i::Identity();
            bool thread_found_one = false;
            Eigen::Vector3d veca = atom.cell_.row(0).transpose();
            Eigen::Vector3d vecb = atom.cell_.row(1).transpose();
            Eigen::Vector3d vecc = atom.cell_.row(2).transpose();
            #pragma omp for collapse(2) schedule(dynamic) nowait
            for (int a11 = 1; a11 <= max_expand; ++a11) {
            for (int a12 = -max_expand; a12 <= max_expand; ++a12) {
            if (find_one) {
                bool stop_search;
                #pragma omp atomic read
                stop_search = found_one;
                if (stop_search) continue;
            }
            for (int a13 = -max_expand; a13 <= max_expand; ++a13) {
                int gca = std::gcd(std::gcd(a11, a12), a13);
                if (gca > 1) continue;
                // build veca
                Eigen::Vector3d nva = a11 * veca + a12 * vecb + a13 * vecc;
                double ana = nva.dot(veca);
                double anb = nva.dot(vecb);
                double anc = nva.dot(vecc);
            for (int a21 = -max_expand; a21 <= max_expand; ++a21) {
            for (int a22 = 1; a22 <= max_expand; ++a22) {
                // here a23 can be determined
                int a23 = 0;
                if (abs(anc) >= tol) a23 = round(-(a21 * ana + a22 * anb)/anc);
                if (abs(a23) > max_expand) continue;
            // for (int a23 = -max_expand; a23 <= max_expand; ++a23) {
                int gcb = std::gcd(std::gcd(a21, a22), a23);
                if (gcb > 1) continue;
                // build vecb and check if ab is valid
                Eigen::Vector3d nvb = a21 * veca + a22 * vecb + a23 * vecc;
                if (abs(nva.dot(nvb)) > tol) continue;
                double bna = nvb.dot(veca);
                double bnb = nvb.dot(vecb);
                double bnc = nvb.dot(vecc);
            // for (int a31 = -max_expand; a31 <= max_expand; ++a31) {
            for (int a33 = 1; a33 <= max_expand; ++a33) {
                // solve m33
                double s1 = (anb * bna - ana * bnb);
                double s2 = (ana * bnb - anb * bna);
                int a31, a32;
                if (abs(s1) < tol) a31 = 0;
                else a31 = a33 * round((anc * bnb - anb * bnc)/s1);
                if (abs(s2) < tol) a32 = 0;
                else a32 = a33 * round((anc * bna - ana * bnc)/s2);
                if (abs(a31) > max_expand || abs(a32) > max_expand) continue;
            // for (int a32 = -max_expand; a32 <= max_expand; ++a32) {
                int gcc = std::gcd(std::gcd(a31, a32), a33);
                if (gcc > 1) continue;
                Eigen::Matrix3i sup;
                sup << a11, a12, a13,
                        a21, a22, a23,
                        a31, a32, a33;
                int det = sup.determinant();
                if (det <= 0 || det > max_expand) continue;
                Eigen::Matrix3d new_cell = sup.cast<double>() * atom.cell_;
                double lna = new_cell.row(0).norm();
                double lnb = new_cell.row(1).norm();
                double lnc = new_cell.row(2).norm();
                double orth_metric = std::max({
                    abs(new_cell.row(0).dot(new_cell.row(1))) / lna / lnb,
                    abs(new_cell.row(0).dot(new_cell.row(2))) / lna / lnc,
                    abs(new_cell.row(1).dot(new_cell.row(2))) / lnb / lnc
                });
                if (orth_metric < tol) {
                    thread_sups_valid.push_back(sup);
                    
                    if (find_one) {
                        thread_sup_best = sup;
                        thread_cmin = orth_metric;
                        thread_found_one = true;
                        #pragma omp atomic write
                        found_one = true;
                    }
                }
                if (orth_metric < thread_cmin) {
                    thread_cmin = orth_metric;
                    thread_sup_best = sup;
                }
            }}}}}}
            #pragma omp critical
            {
                sups_valid.insert(sups_valid.end(), 
                                 thread_sups_valid.begin(), 
                                 thread_sups_valid.end());
                
                if (thread_cmin < cmin) {
                    cmin = thread_cmin;
                    sup_best = thread_sup_best;
                }
                
                if (thread_found_one && find_one) {
                    sup_best = thread_sup_best;
                    cmin = thread_cmin;
                }
            }
        }
        if (find_one && !sups_valid.empty()) {
            sups_valid.clear();
            sups_valid.push_back(sup_best);
        }
        lab_out:
        // use best
        if (cmin > tol) {
            std::cerr << "Warning: c_min = " << cmin << " is larger than tol" << std::endl;
        } else {
            std::cout << "c_min = " << cmin << std::endl;
        }
        // reduce sup_best by gca to avoid non-primitive cell
        int gca = 0;
        gca = std::gcd(std::gcd(sup_best(0,0), sup_best(0,1)), sup_best(0, 2));
        sup_best.row(0) /= gca;
        gca = std::gcd(std::gcd(sup_best(1,0), sup_best(1,1)), sup_best(1, 2));
        sup_best.row(1) /= gca;
        gca = std::gcd(std::gcd(sup_best(2,0), sup_best(2,1)), sup_best(2, 2));
        sup_best.row(2) /= gca;
        std::cout << "Best supercell matrix (det = " << sup_best.determinant() << "):\n" 
                  << sup_best << std::endl;
        atom.mksuperL(sup_best);
        if (!find_one && !sups_valid.empty()) {
            std::cout << "Found " << sups_valid.size() << " valid supercells" << std::endl;
            // if (sups_valid.size() <= 10) {
                std::cout << "Valid supercells:" << std::endl;
                for (const auto& sup : sups_valid) {
                    std::cout << "det = " << sup.determinant() << ":\n" << sup << std::endl;
                }
            // }
        }
    }
    if (!iobj.write(atoms, outfile, outformat))
        GPERROR("Failed to write structures: " + outfile);
    return 0;
}
// reduce to quasi-primitive cell (only try to reduce cell length)
// TODO: move to Symmetry3D
// TODO: test this 
int trans_reduce(int argc, char *argv[]) {
    auto &cmdopts = CMDOption::global_cmdopts;
    GPIO iobj;
    double tol = 1e-3; // symmetry tolerance
    std::string strfile = "POSCAR";
    std::string outfile = "POSCAR_TRED";
    std::string informat = "";
    std::string outformat = "";
    std::string index = "index.map";
    std::vector<Atoms> atoms;
    bool overwrite = false;
    bool optimize_search = false;
    cmdopts.addop(tol, "tol", "tol", "tolerance")
            .addop(strfile, "input", "in", "input structure file", true)
            .addop(outfile, "output", "out", "output structure file", true)
            .addop(optimize_search,"optimize-search","","Opt-in exact candidate pruning/cache for 3D/1D searches; 2D primitive path unchanged",false,0)
            .addop(informat, "input-format", "if")
            .addop(outformat, "output-format", "of")
            .addop(index, "index", "id")
            .addop(overwrite, "overwrite", "ov", "overwrite output file", false, 0)
            .add_request("input")
            .phrase_options(argc, argv);
    if (GPIO::file_exist(index))
        iobj.readmap(index);
    else 
        iobj.setmap();
    if (GPIO::file_exist(outfile) && !overwrite) {
        std::cerr << "Error: output file " << outfile << " already exists; use --overwrite" << std::endl;
        return 1;
    }
    if (!iobj.read(atoms, strfile, informat) || atoms.empty())
        GPERROR("No structures loaded from " + strfile);
    for (auto &atom : atoms) {
        Symmetry3D::ReductionSearchOptions options;options.enabled=optimize_search;
        Symmetry3D::ReductionSearchStats stats;
        int n = Symmetry3D::trans_reduce(atom,tol,options,optimize_search?&stats:nullptr);
        if(optimize_search)std::cout<<fmt::format("Search candidates: {}, full checks: {}, cache hits: {}\n",stats.candidates,stats.full_checks,stats.cache_hits);
        std::cout << fmt::format("Cell size reduced by {} times\n", n);
    }
    if (!iobj.write(atoms, outfile, outformat))
        GPERROR("Failed to write structures: " + outfile);
    return 0;
}
// find primitive cell
int primi_reduce(int argc, char *argv[]) {
    auto &cmdopts = CMDOption::global_cmdopts;
    GPIO iobj;
    double tol = 1e-5; // symmetry tolerance
    std::string strfile = "POSCAR";
    std::string outfile = "POSCAR_PRIM";
    std::string informat = "";
    std::string outformat = "";
    std::string index = "index.map";
    bool trans_red_first = true;
    bool correct_cell = false;
    std::vector<Atoms> atoms;
    bool overwrite = false;
    bool optimize_search = false;
    cmdopts.addop(tol, "tol", "tol", "tolerance")
            .addop(strfile, "input", "in", "input structure file", true)
            .addop(outfile, "output", "out", "output structure file", true)
            .addop(optimize_search,"optimize-search","","Opt-in exact candidate pruning/cache for 3D/1D searches; 2D primitive path unchanged",false,0)
            .addop(informat, "input-format", "if")
            .addop(outformat, "output-format", "of")
            .addop(overwrite, "overwrite", "ov", "overwrite output file", false, 0)
            .addop(trans_red_first, "two-stage", "t", "do trans_reduce first")
            .addop(correct_cell, "correct-cell", "c", "try to find make cell as orthgonal as possible", false, 0)
            .addop(index, "index", "id")
            .add_request("input")
            .phrase_options(argc, argv);
    if (GPIO::file_exist(index))
        iobj.readmap(index);
    else 
        iobj.setmap();
    if (GPIO::file_exist(outfile) && !overwrite) {
        std::cerr << "Error: output file " << outfile << " already exists; use --overwrite" << std::endl;
        return 1;
    }
    if (!iobj.read(atoms, strfile, informat) || atoms.empty())
        GPERROR("No structures loaded from " + strfile);
    int n = 0;
    for (auto &atom : atoms) {
        std::vector<int> periodic_axes;
        for (int axis = 0; axis < 3; ++axis)
            if (atom.pbc_[axis]) periodic_axes.push_back(axis);
        Symmetry3D::ReductionSearchOptions options;options.enabled=optimize_search;
        Symmetry3D::ReductionSearchStats stats;
        if (periodic_axes.size() == 2) {
            // The 2D routine searches the complete periodic translation group.
            // Neither 3D pre-reduction nor cell correction may change its open row.
            n = Symmetry3D::find_primitive_2d(atom, periodic_axes[0], periodic_axes[1], tol);
        } else if (trans_red_first) {
            n = Symmetry3D::all_reduce(atom,tol,options,optimize_search?&stats:nullptr);
        } else {
            n = Symmetry3D::find_primitive(atom,tol,options,optimize_search?&stats:nullptr);
        }
        if(optimize_search) {
            if(periodic_axes.size()==2)std::cout<<"Search optimization applies to 3D/1D paths; 2D primitive path unchanged\n";
            else std::cout<<fmt::format("Search candidates: {}, full checks: {}, cache hits: {}\n",stats.candidates,stats.full_checks,stats.cache_hits);
        }
        if (correct_cell && periodic_axes.size() != 2) {
            Symmetry3D::correct_cell(atom, tol);
            atom.to_righthand();
        }
        std::cout << fmt::format("Cell size reduced by {} times\n", n);
    }
    if (!iobj.write(atoms, outfile, outformat))
        GPERROR("Failed to write structures: " + outfile);
    return 0;
}
// make cell lower trangle
// a axis along x
// TODO: give a new axis vector, rotate this axis alone x/y/z and make orth cell
int mk_lower(int argc, char *argv[]) {
    auto &cmdopts = CMDOption::global_cmdopts;
    GPIO iobj;
    // double tol = 1e-8;
    std::string strfile = "POSCAR";
    std::string outfile = "POSCAR_REV";
    std::string informat = "";
    std::string outformat = "";
    std::string index = "index.map";
    std::string vec2x = ""; // a or b or c
    std::string order = ""; // xyz -> xyz xzy yxz yzx zxy zyx
    bool overwrite = false;
    int mode = 123;
    std::vector<Atoms> atoms;
    cmdopts.addop(strfile, "input", "in", "input structure file", true)
            .addop(outfile, "output", "out", "output structure file", true)
            .addop(informat, "input-format", "if")
            .addop(outformat, "output-format", "of")
            .addop(vec2x, "ax", "ax", "a or b or c, which vector align to x")
            .addop(index, "index", "index")
            .addop(order, "order", "order", "axis order: xyz -> (xyz|xzy|yxz|yzx|zxy|zyx)")
            .addop(overwrite, "overwrite", "ov", "overwrite output file", false, 0)
            .add_request("input")
            .phrase_options(argc, argv);
    if (GPIO::file_exist(index))
        iobj.readmap(index);
    else 
        iobj.setmap();
    if (GPIO::file_exist(outfile) && !overwrite) {
        std::cerr << "Error: output file " << outfile << " already exists; use --overwrite" << std::endl;
        return 1;
    }
    if (!iobj.read(atoms, strfile, informat) || atoms.empty())
        GPERROR("No structures loaded from " + strfile);
    if (vec2x == "a") mode = 123;
    else if (vec2x == "b") mode = 231;
    else if (vec2x == "c") mode = 312;
    if (order == "xyz" || order == "abc") mode = 123;
    else if (order == "xzy" || order == "acb") mode = 132;
    else if (order == "yxz" || order == "bac") mode = 213;
    else if (order == "yzx" || order == "bca") mode = 231;
    else if (order == "zxy" || order == "cab") mode = 312;
    else if (order == "zyx" || order == "cba") mode = 321;
    for (auto &atom : atoms) {
        atom.swap_axis(mode);
        atom.to_lower_standard();
    }
    if (!iobj.write(atoms, outfile, outformat))
        GPERROR("Failed to write structures: " + outfile);
    return 0;
}
// make super cell
int mk_super(int argc, char *argv[]) {
    auto &cmdopts = CMDOption::global_cmdopts;
    GPIO iobj;
    std::string strfile = "POSCAR";
    std::string outfile = "POSCAR_REV";
    std::string informat = "";
    std::string outformat = "";
    std::string index = "index.map";
    std::vector<Atoms> atoms;
    std::vector<int> suplat;
    bool check_orient = true;
    bool overwrite = false;
    cmdopts.addop(strfile, "input", "in", "input structure file", true)
            .addop(informat, "input-format", "if")
            .addop(outformat, "output-format", "of")
            .addop(index, "index", "index")
            .addop(check_orient, "check", "c", "check if right hand", false, 1)
            .addop(suplat, "supercell", "s", "supercell array (3 elem) or matrix (9 elem), left, input row by row", true, -1)
            .addop(outfile, "output", "out", "output structure file", true)
            .addop(overwrite, "overwrite", "ov", "overwrite output file", false, 0)
            .add_request("input")
            .add_request("supercell")
            .phrase_options(argc, argv);
    Eigen::Matrix3i sup = Eigen::Matrix3i::Identity();
    if (suplat.size() == 3) {
        sup(0, 0) = suplat[0];
        sup(1, 1) = suplat[1];
        sup(2, 2) = suplat[2];
    } else if (suplat.size() == 9) {
        for (int i = 0; i < 9; ++i) {
            sup(i/3, i%3) = suplat[i];
        }
    } else {
        std::cerr << "ERROR: supercell must be length of 3 or 9, but received " << suplat.size() << std::endl;
        return 128;
    }
    if (GPIO::file_exist(index))
        iobj.readmap(index);
    else 
        iobj.setmap();
    if (GPIO::file_exist(outfile) && !overwrite) {
        std::cerr << "Error: output file " << outfile << " already exists; use --overwrite" << std::endl;
        return 1;
    }
    if (!iobj.read(atoms, strfile, informat) || atoms.empty())
        GPERROR("No structures loaded from " + strfile);
    for (auto &atom : atoms) {
        atom.mksuperL(sup);
        if (check_orient) {
            if (atom.cell_.determinant() < 0) {
                atom.cell_.row(0) *= -1.0;
                atom.wrap();
            }
        }
    }
    if (!iobj.write(atoms, outfile, outformat))
        GPERROR("Failed to write structures: " + outfile);
    return 0;
}
// swap axis
int swap_axis(int argc, char *argv[]) {
    auto &cmdopts = CMDOption::global_cmdopts;
    GPIO iobj;
    double tol = 1e-6;
    std::string strfile = "POSCAR";
    std::string outfile = "POSCAR_REV";
    std::string informat = "";
    std::string outformat = "";
    std::string index = "index.map";
    std::vector<Atoms> atoms;
    bool keephand = true;
    bool tolower = false;
    bool overwrite = false;
    std::string order = "xyz"; // xyz -> xyz xzy yxz yzx zxy zyx
    cmdopts.addop(tol, "tol", "tol", "tolerance")
            .addop(order, "order", "order", "axis order: xyz -> (xyz|xzy|yxz|yzx|zxy|zyx)", true)
            .addop(strfile, "input", "in", "", true)
            .addop(outfile, "output", "out", "", true)
            .addop(informat, "input-format", "if")
            .addop(outformat, "output-format", "of")
            .addop(index, "index", "index")
            .addop(keephand, "keephand", "k")
            .addop(tolower, "tolower", "l", "to lower standard form", false, 0)
            .addop(overwrite, "overwrite", "ov", "overwrite output file", false, 0)
            .add_request("order")
            .add_request("input")
            .phrase_options(argc, argv);
    if (GPIO::file_exist(index))
        iobj.readmap(index);
    else 
        iobj.setmap();
    if (GPIO::file_exist(outfile) && !overwrite) {
        std::cerr << "Error: output file " << outfile << " already exists; use --overwrite" << std::endl;
        return 1;
    }
    if (!iobj.read(atoms, strfile, informat) || atoms.empty())
        GPERROR("No structures loaded from " + strfile);
    int mode = 123;
    if (order == "xyz" || order == "abc") mode = 123;
    else if (order == "xzy" || order == "acb") mode = 132;
    else if (order == "yxz" || order == "bac") mode = 213;
    else if (order == "yzx" || order == "bca") mode = 231;
    else if (order == "zxy" || order == "cab") mode = 312;
    else if (order == "zyx" || order == "cba") mode = 321;
    for (auto &atom : atoms) {
        atom.swap_axis(mode, keephand);
        if (tolower) atom.to_lower_standard();
    }
    if (!iobj.write(atoms, outfile, outformat))
        GPERROR("Failed to write structures: " + outfile);
    return 0;
}
// mindist
int find_mindist(int argc, char *argv[]) {
    auto &cmdopts = CMDOption::global_cmdopts;
    GPIO iobj;
    std::string strfile = "POSCAR";
    std::string outfile = "";
    std::string informat = "";
    std::string outformat = "";
    std::string index = "index.map";
    std::vector<Atoms> atoms;
    bool overwrite = false;
    cmdopts.addop(strfile, "input", "in", "", true)
            .addop(outfile, "output", "out", "", true)
            .addop(informat, "input-format", "if")
            .addop(outformat, "output-format", "of")
            .addop(index, "index", "index")
            .addop(overwrite, "overwrite", "ov", "overwrite output file", false, 0)
            .add_request("input")
            .phrase_options(argc, argv);
    if (GPIO::file_exist(index))
        iobj.readmap(index);
    else 
        iobj.setmap();
    if (GPIO::file_exist(outfile) && !overwrite) {
        std::cerr << "Error: output file " << outfile << " already exists; use --overwrite" << std::endl;
        return 1;
    }
    if (!iobj.read(atoms, strfile, informat) || atoms.empty())
        GPERROR("No structures loaded from " + strfile);
    std::vector<double> mindists;
    double global_mindist = -1;
    for (auto &atom : atoms) {
        double mdist = -1;
        if (atom.size() > 0 && !atom.is_nonperiodic()) {
            mdist = atom.report_mindist();
        } else if (atom.size() > 1) {
            Atoms cartesian_atom = atom;
            if (!cartesian_atom.cartesian) cartesian_atom.to_cartesian();
            NeighborhoodKD nbhskd(&cartesian_atom);
            for (int i = 0; i < cartesian_atom.size(); ++i) {
                auto neighbors = nbhskd.query(cartesian_atom.pos_[i].data(), 2,
                                              cartesian_atom.chemmap_[i], true);
                for (int j = 0; j < neighbors.nbhcount_; ++j) {
                    if (neighbors.indices_[j] == i) continue;
                    if (mdist < 0 || neighbors.distances_[j] < mdist)
                        mdist = neighbors.distances_[j];
                }
            }
        }
        if (mdist >= 0 && (global_mindist < 0 || mdist < global_mindist))
            global_mindist = mdist;
        mindists.push_back(mdist);
    }
    // output mindists
    if (!outfile.empty()) {
        std::ofstream ofs(outfile);
        if (!ofs) GPERROR("Cannot open output file: " + outfile);
        for (size_t i = 0; i < mindists.size(); ++i) {
            ofs << fmt::format("{}\n", mindists[i]);
        }
        ofs << fmt::format("{}\n", global_mindist);
        ofs.close();
        if (!ofs) GPERROR("Failed to write distances: " + outfile);
    } else {
        for (size_t i = 0; i < mindists.size(); ++i) {
            std::cout << fmt::format("Structure #{}: min distance = {}\n", i, mindists[i]);
        }
        std::cout << fmt::format("Global minimum distance = {}\n", global_mindist);
    }
    return 0;
}
