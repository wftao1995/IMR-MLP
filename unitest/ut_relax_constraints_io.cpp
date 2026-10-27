#include <catch2/catch_all.hpp>
#include "mindriver/relax_constraints_io.h"

#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <limits>
#include <sstream>

using Catch::Approx;
using namespace Constraints;

namespace {

LatticeConstraintDeclarations parse(const std::string &text) {
    std::istringstream input(text);
    return parse_lattice_constraints(input);
}

std::string document(const std::string &items) {
    return "{\"version\":1,\"constraints\":[" + items + "]}";
}

const std::string volume = R"({"kind":"volume","residual_tolerance":1e-5})";
const std::string all_kinds =
    R"({"kind":"volume","target":74.5,"residual_tolerance":1e-5},)"
    R"({"kind":"lattice_length","axis":"c","target":5.2,"residual_tolerance":2e-5},)"
    R"({"kind":"lattice_angle","axes":["c","a"],"target":83.2,"residual_tolerance":3e-5},)"
    R"({"kind":"lattice_ratio","axes":["b","a"],"target":1.13,"residual_tolerance":4e-5})";

Eigen::Matrix3d skew_cell() {
    Eigen::Matrix3d cell;
    cell << 4.1, 0.3, 0.2,
            0.2, 4.7, 0.5,
            0.4, 0.6, 5.3;
    return cell;
}

std::pair<double, Eigen::Matrix<double, 9, 1>> evaluate(
    regulator &reg, const Eigen::Matrix3d &cell) {
    Atoms atom;
    atom.cell_ = cell;
    for (std::size_t i = 0; i < reg.inputs.size(); ++i)
        reg.x[i] = cell(reg.inputs[i] / 3, reg.inputs[i] % 3);
    reg.compute(reg.x.data(), reg.x.size(), reg.g.data(), reg.g.size(), &reg.f,
                reg.g0, &atom, reg.integer_specs.data(), reg.data);
    Eigen::Matrix<double, 9, 1> gradient = Eigen::Matrix<double, 9, 1>::Zero();
    for (std::size_t i = 0; i < reg.outputs.size(); ++i)
        gradient[reg.outputs[i]] = reg.g[i];
    return {reg.f, gradient};
}

struct TemporaryDirectory {
    std::filesystem::path path = std::filesystem::temp_directory_path() /
        ("imrmlp-constraint-json-" + std::to_string(
            std::chrono::steady_clock::now().time_since_epoch().count()));
    TemporaryDirectory() { std::filesystem::create_directory(path); }
    ~TemporaryDirectory() { std::error_code ec; std::filesystem::remove_all(path, ec); }
};

} // namespace

TEST_CASE("Lattice JSON parses four physical targets and binds AUL equality state", "[relax_constraints_io]") {
    const auto declarations = parse(document(all_kinds));
    REQUIRE(declarations.size() == 4);
    auto bound = bind_lattice_constraints(declarations, skew_cell());
    REQUIRE(bound.size() == 4);
    REQUIRE(bound[0].g0 == Approx(74.5));
    REQUIRE(bound[1].g0 == Approx(5.2 * 5.2));
    REQUIRE(bound[2].g0 == Approx(std::cos(83.2 * std::acos(-1.0) / 180.0)));
    REQUIRE(bound[3].g0 == Approx(1.13 * 1.13));
    REQUIRE(bound[1].inputs == std::vector<int>{6, 7, 8});
    REQUIRE(bound[2].inputs == std::vector<int>{6, 7, 8, 0, 1, 2});
    REQUIRE(bound[3].inputs == std::vector<int>{3, 4, 5, 0, 1, 2});
    for (std::size_t i = 0; i < bound.size(); ++i) {
        const auto &reg = bound[i];
        REQUIRE(reg.type == 10);
        REQUIRE(reg.to_target_g0);
        REQUIRE(reg.compute != nullptr);
        REQUIRE(reg.weight == 1.0);
        REQUIRE(reg.weight_init == 1.0);
        REQUIRE(reg.lambda == 0.0);
        REQUIRE(reg.lambda_init == 0.0);
        REQUIRE(reg.f == 0.0);
        REQUIRE(reg.mos == 0.0);
        REQUIRE(reg.mos2 == 0.0);
        REQUIRE(reg.curr_constraint_iter == 0);
        REQUIRE(reg.tol == Approx((i + 1) * 1e-5));
        REQUIRE(reg.x.size() == reg.inputs.size());
        REQUIRE(reg.g.size() == reg.outputs.size());
    }
    // Driver setup can run again; explicit physical targets remain bound.
    Atoms different;
    different.cell_ = 2.0 * skew_cell();
    constant_volume_setup(bound[0], different);
    constant_vector_setup(bound[1], different, 2);
    constant_angle_setup(bound[2], different, 2, 0);
    couple_vector_setup(bound[3], different, 1, 0);
    REQUIRE(bound[0].g0 == Approx(74.5));
    REQUIRE(bound[1].g0 == Approx(27.04));
    REQUIRE(bound[2].g0 == Approx(std::cos(83.2 * std::acos(-1.0) / 180.0)));
    REQUIRE(bound[3].g0 == Approx(1.2769));
}

TEST_CASE("Lattice JSON default targets and multipliers are independent per frame", "[relax_constraints_io]") {
    const auto declarations = parse(document(
        volume + R"(,{"kind":"lattice_length","axis":"a","residual_tolerance":1e-5},)"
        R"({"kind":"lattice_angle","axes":["a","b"],"residual_tolerance":1e-6},)"
        R"({"kind":"lattice_ratio","axes":["a","c"],"residual_tolerance":1e-6})"));
    auto first = bind_lattice_constraints(declarations, skew_cell());
    Eigen::Matrix3d next = skew_cell();
    next(0, 1) += 0.8;
    next.row(2) *= 1.6;
    first[0].lambda = 123;
    first[0].weight = 456;
    auto second = bind_lattice_constraints(declarations, next);
    for (std::size_t i = 0; i < first.size(); ++i) {
        REQUIRE(!declarations[i].target.has_value());
        REQUIRE(first[i].g0 != Approx(second[i].g0));
        REQUIRE(second[i].lambda == 0);
        REQUIRE(second[i].weight == 1);
        const auto [residual, gradient] = evaluate(second[i], next);
        REQUIRE(residual == Approx(0).margin(1e-12));
        REQUIRE(gradient.allFinite());
    }
}

TEST_CASE("Lattice JSON accepts reordered and escaped ASCII fields", "[relax_constraints_io]") {
    const auto declarations = parse(
        " \t\r\n" R"({"constraints":[{"residual_tolerance":2E-5,"targ\u0065t":5.0,)"
        R"("axis":"\u0063","kind":"lattice_leng\u0074h"}],"\u0076ersion":1})" " \n");
    REQUIRE(declarations.size() == 1);
    REQUIRE(declarations[0].axis == 2);
    REQUIRE(*declarations[0].target == 5.0);
    REQUIRE(declarations[0].residual_tolerance == Approx(2e-5));
    // Axis order is meaningful for ratios; reverse ratios are distinct.
    REQUIRE(parse(document(
        R"({"kind":"lattice_ratio","axes":["a","b"],"residual_tolerance":1e-5},)"
        R"({"kind":"lattice_ratio","axes":["b","a"],"residual_tolerance":1e-5})")).size() == 2);
}

TEST_CASE("Lattice JSON rejects malformed schema and duplicate identities", "[relax_constraints_io][errors]") {
    const std::vector<std::string> invalid_documents = {
        "", "{}", "[]", "null", "true", "/* comment */" + document(volume),
        R"({"constraints":[]})", R"({"version":1})", document(""),
        R"({"version":2,"constraints":[]})", R"({"version":1.0,"constraints":[]})",
        R"({"version":1e0,"constraints":[]})", R"({"version":true,"constraints":[]})",
        R"({"version":"1","constraints":[]})", R"({"version":1,"constraints":null})",
        R"({"version":1,"constraints":{}})", document(volume) + "{}",
        document(volume) + "garbage", document(volume) + std::string(1, '\0'),
        "{\"version\":1,\"version\":1,\"constraints\":[" + volume + "]}",
        "{\"version\":1,\"\\u0076ersion\":1,\"constraints\":[" + volume + "]}",
        "{\"version\":1,\"other\":0,\"constraints\":[" + volume + "]}",
        document(volume + ","), document(volume + "," + volume),
        document(R"({"kind":"volume","kind":"volume","residual_tolerance":1e-4})"),
        document(R"({"kind":"volume","k\u0069nd":"volume","residual_tolerance":1e-4})"),
        document(R"({"kind":"volume","residual_tolerance":1e-4,"weight":1})"),
        document(R"({"kind":"volume","residual_tolerance":1e-4,})"),
        document(R"({"kind":"volume"})"), document(R"({"residual_tolerance":1e-4})"),
        document(R"({"kind":1,"residual_tolerance":1e-4})"),
        document(R"({"kind":"unknown","residual_tolerance":1e-4})"),
        document(R"({"kind":"volume","axis":"a","residual_tolerance":1e-4})"),
        document(R"({"kind":"volume","axes":["a","b"],"residual_tolerance":1e-4})"),
        document(R"({"kind":"lattice_length","residual_tolerance":1e-4})"),
        document(R"({"kind":"lattice_length","axis":"A","residual_tolerance":1e-4})"),
        document(R"({"kind":"lattice_length","axis":0,"residual_tolerance":1e-4})"),
        document(R"({"kind":"lattice_length","axis":"a","axes":["b","c"],"residual_tolerance":1e-4})"),
        document(R"({"kind":"lattice_angle","axes":["a"],"residual_tolerance":1e-4})"),
        document(R"({"kind":"lattice_angle","axes":["a","b","c"],"residual_tolerance":1e-4})"),
        document(R"({"kind":"lattice_angle","axes":["a","a"],"residual_tolerance":1e-4})"),
        document(R"({"kind":"lattice_angle","axes":["a","b",],"residual_tolerance":1e-4})"),
        document(R"({"kind":"lattice_angle","axes":"ab","residual_tolerance":1e-4})"),
        document(R"({"kind":"lattice_angle","axis":"a","residual_tolerance":1e-4})"),
        document(R"({"kind":"lattice_angle","axes":["a","b"],"residual_tolerance":1e-4},)"
                 R"({"kind":"lattice_angle","axes":["b","a"],"residual_tolerance":1e-4})"),
        document(R"({"kind":"vol\ume","residual_tolerance":1e-4})"),
        document(R"({"kind":"vol\qume","residual_tolerance":1e-4})"),
        document(R"({"kind":"vol\uZZZZume","residual_tolerance":1e-4})"),
        document(R"({"kind":"vol\uD800ume","residual_tolerance":1e-4})"),
        document(R"({"kind":"vol\u00e9ume","residual_tolerance":1e-4})"),
        document(R"({"kind":"vol\u0000ume","residual_tolerance":1e-4})"),
        document("{\"kind\":\"vol\nume\",\"residual_tolerance\":1e-4}"),
        document("{\"kind\":\"vol\xc3\xa9ume\",\"residual_tolerance\":1e-4}"),
        document("{\"kind\":\"" + std::string(65, 'a') + "\",\"residual_tolerance\":1e-4}"),
        document("[" + volume + "]"),
    };
    for (const auto &text : invalid_documents) {
        INFO(text);
        REQUIRE_THROWS_AS(parse(text), std::invalid_argument);
    }
    const auto complete = document(all_kinds);
    for (std::size_t length = 0; length < complete.size(); ++length) {
        INFO("truncated at " << length);
        REQUIRE_THROWS_AS(parse(complete.substr(0, length)), std::invalid_argument);
    }
    REQUIRE_NOTHROW(parse(document(volume)));
}

TEST_CASE("Lattice JSON validates numeric syntax, domains, and conversions", "[relax_constraints_io][errors]") {
    for (const std::string token : {"true", "false", "null", "\"2\"", "[]", "{}", "NaN", "Infinity",
                                   "-Infinity", "+1", "01", "-01", ".5", "1.", "1e", "1e+", "1e-", "0x1", "1_0",
                                   "1e309", "1e-9999", "0", "-0", "-1"}) {
        INFO(token);
        REQUIRE_THROWS_AS(parse(document("{\"kind\":\"volume\",\"residual_tolerance\":" + token + "}")), std::invalid_argument);
        REQUIRE_THROWS_AS(parse(document("{\"kind\":\"volume\",\"residual_tolerance\":1e-5,\"target\":" + token + "}")), std::invalid_argument);
    }
    for (const std::string kind : {"lattice_length", "lattice_ratio"}) {
        const std::string axes = kind == "lattice_length" ? R"("axis":"a")" : R"("axes":["a","b"])";
        for (const std::string target : {"1e200", "1e-200"}) {
            REQUIRE_THROWS_AS(parse(document("{\"kind\":\"" + kind + "\"," + axes + ",\"target\":" + target +
                                             ",\"residual_tolerance\":1e-5}")), std::invalid_argument);
        }
    }
    for (const std::string target : {"180", "190", "1e-12", "179.99999999999999"}) {
        REQUIRE_THROWS_AS(parse(document(R"({"kind":"lattice_angle","axes":["a","b"],"target":)" + target +
                                        R"(,"residual_tolerance":1e-5})")), std::invalid_argument);
    }
    REQUIRE_NOTHROW(parse(document(R"({"kind":"volume","target":1.25e+2,"residual_tolerance":0.00001})")));
}

TEST_CASE("Lattice JSON input is bounded and IO failures retain context", "[relax_constraints_io][errors]") {
    const std::string valid = document(volume);
    REQUIRE_NOTHROW(parse(valid + std::string(lattice_constraints_max_bytes - valid.size(), ' ')));
    REQUIRE_THROWS_AS(parse(valid + std::string(lattice_constraints_max_bytes - valid.size() + 1, ' ')), std::invalid_argument);
    std::istringstream failed(valid);
    failed.setstate(std::ios::failbit);
    REQUIRE_THROWS_AS(parse_lattice_constraints(failed), std::runtime_error);
    std::istringstream bad(valid);
    bad.setstate(std::ios::badbit);
    REQUIRE_THROWS_AS(parse_lattice_constraints(bad), std::runtime_error);
    std::istringstream exhausted(valid);
    exhausted.setstate(std::ios::failbit | std::ios::eofbit);
    REQUIRE_THROWS_AS(parse_lattice_constraints(exhausted), std::runtime_error);
    std::istringstream exceptional(valid);
    exceptional.exceptions(std::ios::failbit | std::ios::badbit);
    REQUIRE_NOTHROW(parse_lattice_constraints(exceptional));

    TemporaryDirectory temporary;
    const auto file = temporary.path / "constraints.json";
    { std::ofstream output(file); output << valid; }
    REQUIRE(read_lattice_constraints(file.string()).size() == 1);
    { std::ofstream output(file); output << document(R"({"kind":"wrong","residual_tolerance":1e-4})"); }
    REQUIRE_THROWS_WITH(read_lattice_constraints(file.string()),
        Catch::Matchers::ContainsSubstring(file.string()) && Catch::Matchers::ContainsSubstring("constraint[0]"));
    REQUIRE_THROWS_WITH(read_lattice_constraints((temporary.path / "missing.json").string()),
        Catch::Matchers::ContainsSubstring("missing.json"));
    REQUIRE_THROWS_AS(read_lattice_constraints(temporary.path.string()), std::runtime_error);
}

TEST_CASE("Lattice constraint binding validates caller declarations and initial cell", "[relax_constraints_io][errors]") {
    const auto good = parse(document(all_kinds));
    auto declarations = good;
    declarations[0].target = std::numeric_limits<double>::quiet_NaN();
    REQUIRE_THROWS_AS(bind_lattice_constraints(declarations, skew_cell()), std::invalid_argument);
    declarations = good;
    declarations[1].axis = 3;
    REQUIRE_THROWS_AS(bind_lattice_constraints(declarations, skew_cell()), std::invalid_argument);
    declarations = good;
    declarations[0].kind = static_cast<LatticeConstraintKind>(100);
    REQUIRE_THROWS_AS(bind_lattice_constraints(declarations, skew_cell()), std::invalid_argument);
    declarations = good;
    declarations[0].residual_tolerance = std::numeric_limits<double>::infinity();
    REQUIRE_THROWS_AS(bind_lattice_constraints(declarations, skew_cell()), std::invalid_argument);
    declarations = good;
    declarations.push_back(declarations[0]);
    REQUIRE_THROWS_AS(bind_lattice_constraints(declarations, skew_cell()), std::invalid_argument);
    REQUIRE_THROWS_AS(bind_lattice_constraints({}, skew_cell()), std::invalid_argument);
    declarations.assign(lattice_constraints_max_count + 1, good[0]);
    REQUIRE_THROWS_AS(bind_lattice_constraints(declarations, skew_cell()), std::invalid_argument);

    auto cell = skew_cell();
    cell.row(1) = cell.row(0);
    REQUIRE_THROWS_AS(bind_lattice_constraints(good, cell), std::invalid_argument);
    cell = skew_cell(); cell.row(1) *= -1;
    REQUIRE_THROWS_AS(bind_lattice_constraints(good, cell), std::invalid_argument);
    cell = skew_cell(); cell(0, 0) = std::numeric_limits<double>::infinity();
    REQUIRE_THROWS_AS(bind_lattice_constraints(good, cell), std::invalid_argument);
    cell = skew_cell(); cell(0, 0) = std::numeric_limits<double>::quiet_NaN();
    REQUIRE_THROWS_AS(bind_lattice_constraints(good, cell), std::invalid_argument);
    REQUIRE_THROWS_AS(bind_lattice_constraints(good, Eigen::Matrix3d::Zero()), std::invalid_argument);
    REQUIRE_THROWS_AS(bind_lattice_constraints(good, Eigen::Matrix3d::Identity() * 1e200), std::invalid_argument);
    REQUIRE_THROWS_AS(bind_lattice_constraints(good, Eigen::Matrix3d::Identity() * 1e-200), std::invalid_argument);
    REQUIRE_NOTHROW(bind_lattice_constraints(good, skew_cell()));
}

TEST_CASE("Bound raw9 residual and AUL gradients agree with finite differences", "[relax_constraints_io][gradient]") {
    auto bound = bind_lattice_constraints(parse(document(all_kinds)), skew_cell());
    const auto cell = skew_cell();
    for (std::size_t i = 0; i < bound.size(); ++i) {
        auto &reg = bound[i];
        reg.weight = 0.7 + i;
        reg.lambda = -0.3 + 0.2 * i;
        const auto [residual, gradient] = evaluate(reg, cell);
        REQUIRE(std::abs(residual) > 1e-6);
        for (int k = 0; k < 9; ++k) {
            INFO("constraint=" << i << " component=" << k);
            Eigen::Matrix3d plus = cell, minus = cell;
            plus(k / 3, k % 3) += 1e-6;
            minus(k / 3, k % 3) -= 1e-6;
            const double rp = evaluate(reg, plus).first;
            const double rm = evaluate(reg, minus).first;
            REQUIRE(gradient[k] == Approx((rp - rm) / 2e-6).margin(2e-8));
            const auto objective = [&](double r) { return 0.5 * reg.weight * r * r + reg.lambda * r; };
            const double analytic = (reg.weight * residual + reg.lambda) * gradient[k];
            REQUIRE(analytic == Approx((objective(rp) - objective(rm)) / 2e-6).margin(3e-6));
        }
    }
}

TEST_CASE("Generic constraint JSON binds atomic physical targets and fresh frame defaults", "[relax_constraints_io][atomic]") {
    std::istringstream text(document(
        R"({"kind":"distance","atoms":[0,1],"target":1.5,"residual_tolerance":1e-6},)"
        R"({"kind":"angle","atoms":[0,1,2],"target":60,"residual_tolerance":2e-6})"));
    const ConstraintDeclarations declarations = parse_constraints(text);
    REQUIRE(declarations.size() == 2);
    Atoms frame;
    frame.pbc_ = {false, false, false};
    frame.pos_ = {{1, 0, 0}, {0, 0, 0}, {0, 2, 0}};
    frame.chemmap_ = {0, 0, 0};
    auto bound = bind_constraints(declarations, frame);
    REQUIRE(bound[0].regulator_t == regulator::DISTANCE);
    REQUIRE(bound[1].regulator_t == regulator::ANGLE);
    REQUIRE(bound[0].g0 == Approx(2.25));
    REQUIRE(bound[1].g0 == Approx(.5));
    REQUIRE(bound[0].integer_specs == std::vector<int>{0, 1});
    REQUIRE(bound[1].integer_specs == std::vector<int>{0, 1, 2});
    for (const auto &reg : bound) {
        REQUIRE(reg.type == 10);
        REQUIRE(reg.weight == 1);
        REQUIRE(reg.lambda == 0);
    }
    std::istringstream defaults(document(
        R"({"kind":"distance","atoms":[0,1],"residual_tolerance":1e-6},)"
        R"({"kind":"angle","atoms":[0,1,2],"residual_tolerance":1e-6})"));
    const auto implicit = parse_constraints(defaults);
    auto first = bind_constraints(implicit, frame);
    frame.pos_[0] = {2, 0, 0};
    frame.pos_[2] = {1, 1, 0};
    first[0].lambda = 12;
    auto second = bind_constraints(implicit, frame);
    REQUIRE(first[0].double_target == Approx(1));
    REQUIRE(first[1].double_target == Approx(90));
    REQUIRE(second[0].double_target == Approx(2));
    REQUIRE(second[1].double_target == Approx(45));
    REQUIRE(second[0].lambda == 0);
    REQUIRE(second[0].weight == 1);
    atomic_geometry_setup(first[0], frame);
    REQUIRE(first[0].double_target == Approx(1)); // Setup must not bind again after displacement.
    REQUIRE(evaluate_atomic_geometry(second[0], frame).residual == Approx(0).margin(1e-12));
    REQUIRE(evaluate_atomic_geometry(second[1], frame).residual == Approx(0).margin(1e-12));
}

TEST_CASE("Atomic JSON rejects ambiguous identities invalid targets and per-frame indices", "[relax_constraints_io][atomic][errors]") {
    for (const auto &entry : {
        R"({"kind":"distance","atoms":[0],"residual_tolerance":1e-6})",
        R"({"kind":"distance","atoms":[0,0],"residual_tolerance":1e-6})",
        R"({"kind":"distance","atoms":[0,1.0],"residual_tolerance":1e-6})",
        R"({"kind":"distance","atoms":[0,1],"target":0,"residual_tolerance":1e-6})",
        R"({"kind":"angle","atoms":[0,1,2],"target":180,"residual_tolerance":1e-6})",
        R"({"kind":"angle","atoms":[0,1,1],"residual_tolerance":1e-6})",
        R"({"kind":"volume","atoms":[0,1],"residual_tolerance":1e-6})",
        R"({"kind":"volume","atoms":[],"residual_tolerance":1e-6})",
        R"({"kind":"distance","atoms":[0,1],"axis":"a","residual_tolerance":1e-6})",
        R"({"kind":"distance","atoms":[0,1],"residual_tolerance":1e-6},{"kind":"distance","atoms":[1,0],"residual_tolerance":1e-6})",
        R"({"kind":"angle","atoms":[0,1,2],"residual_tolerance":1e-6},{"kind":"angle","atoms":[2,1,0],"residual_tolerance":1e-6})"}) {
        CAPTURE(entry);
        std::istringstream input(document(entry));
        REQUIRE_THROWS_AS(parse_constraints(input), std::invalid_argument);
    }
    std::istringstream input(document(R"({"kind":"distance","atoms":[0,2],"residual_tolerance":1e-6})"));
    const auto declarations = parse_constraints(input);
    Atoms frame; frame.pbc_ = {false, false, false};
    frame.pos_ = {{0,0,0}, {1,0,0}}; frame.chemmap_ = {0,0};
    REQUIRE_THROWS_AS(bind_constraints(declarations, frame), std::invalid_argument);
}
