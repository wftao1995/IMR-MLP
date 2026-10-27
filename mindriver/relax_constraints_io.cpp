#include "mindriver/relax_constraints_io.h"
#include "mindriver/constraint_expression.h"
#include "mindriver/relax_new.h"

#include <algorithm>
#include <array>
#include <charconv>
#include <cmath>
#include <fstream>
#include <numbers>
#include <limits>
#include <set>
#include <stdexcept>
#include <tuple>

namespace Constraints {
namespace {

[[noreturn]] void invalid(const std::string &message) {
    throw std::invalid_argument("Lattice constraints: " + message);
}

using Identity = std::tuple<int, int, int, std::vector<int>, std::string>;

Identity validate_declaration(const ConstraintDeclaration &item) {
    const bool symmetry=item.kind==ConstraintKind::Symmetry;
    if (!symmetry && (!std::isfinite(item.residual_tolerance) || item.residual_tolerance <= 0))
        invalid("residual_tolerance must be finite and positive");
    if (item.target && !std::isfinite(*item.target)) invalid("target must be finite");
    if (item.target && (item.lower || item.upper)) invalid("target and lower/upper are mutually exclusive");
    if ((item.lower && !std::isfinite(*item.lower)) || (item.upper && !std::isfinite(*item.upper)) ||
        (item.lower && item.upper && *item.lower>*item.upper)) invalid("invalid lower/upper interval");
    int first=item.axis, second=item.second_axis; auto atoms=item.atoms;
    const bool positive=item.kind==ConstraintKind::Volume || item.kind==ConstraintKind::Length ||
        item.kind==ConstraintKind::Ratio || item.kind==ConstraintKind::Distance;
    if (positive) for (auto v:{item.target,item.lower,item.upper})
        if(v && *v<=0) invalid("geometry target/bounds must be positive");
    const bool angle=item.kind==ConstraintKind::Angle || item.kind==ConstraintKind::AtomicAngle;
    if (angle) for (auto v:{item.target,item.lower,item.upper})
        if(v && !(*v>0 && *v<180)) invalid("angle target/bounds must lie strictly between 0 and 180 degrees");
    switch(item.kind) {
    case ConstraintKind::Volume: if(first!=-1 || second!=-1) invalid("volume does not accept axis or axes"); break;
    case ConstraintKind::Length:
        if(first<0 || first>2 || second!=-1) invalid("lattice_length requires one axis (a, b, or c)"); break;
    case ConstraintKind::Angle: case ConstraintKind::Ratio:
        if(first<0 || first>2 || second<0 || second>2 || first==second) invalid("lattice_angle/lattice_ratio requires two distinct axes");
        if(item.kind==ConstraintKind::Angle && first>second) std::swap(first,second); break;
    case ConstraintKind::Distance: case ConstraintKind::AtomicAngle: case ConstraintKind::Dihedral:
    case ConstraintKind::Position: case ConstraintKind::Wall: {
        const size_t count=item.kind==ConstraintKind::Distance?2:item.kind==ConstraintKind::AtomicAngle?3:item.kind==ConstraintKind::Dihedral?4:1;
        if(first!=-1 || second!=-1 || atoms.size()!=count) invalid("wrong geometry atom indices or axes");
        std::set<int> ids(atoms.begin(),atoms.end());
        if(*ids.begin()<0 || ids.size()!=atoms.size()) invalid("atom indices must be nonnegative and distinct");
        if(item.kind==ConstraintKind::Position && (item.component<0 || item.component>2)) invalid("position requires component x/y/z");
        if(item.kind==ConstraintKind::Wall && (!item.normal.allFinite() || item.normal.norm()==0 || (!item.lower && !item.upper))) invalid("wall requires a nonzero normal and lower or upper distance");
        if((item.kind==ConstraintKind::Distance || item.kind==ConstraintKind::AtomicAngle) && atoms.front()>atoms.back()) std::swap(atoms.front(),atoms.back());
        break;
    }
    case ConstraintKind::Formula: if(item.expression.empty()) invalid("formula requires expression"); break;
    case ConstraintKind::Symmetry:
        if(item.target || item.lower || item.upper || !atoms.empty()) invalid("symmetry accepts operations and optional sites");
        for(int op:item.operations) if(op<0) invalid("operation indices must be nonnegative");
        for(int site:item.sites) if(site<0) invalid("site indices must be nonnegative");
        break;
    }
    if ((item.kind==ConstraintKind::Length || item.kind==ConstraintKind::Ratio || item.kind==ConstraintKind::Distance) && item.target &&
        (!std::isfinite(*item.target * *item.target) || *item.target * *item.target==0)) invalid("squared target is not representable");
    if(item.kind==ConstraintKind::Angle && item.target) {
        const double cosine=std::cos(*item.target*std::numbers::pi/180.);
        if(!std::isfinite(cosine) || std::abs(cosine)>=1.)invalid("angle target is too close to 0 or 180 degrees to represent");
    }
    std::string identity=item.expression;
    if(item.kind==ConstraintKind::Position) identity=std::to_string(item.component);
    if(item.kind==ConstraintKind::Wall) {std::ostringstream o;o<<item.normal;identity=o.str();}
    return {static_cast<int>(item.kind),first,second,atoms,identity};
}

// This parser follows the fixed schema directly. There is no recursive generic
// JSON tree, so nesting and memory use are bounded independently of input data.
class Parser {
    const std::string &text;
    std::size_t pos = 0;
    int version = 1;

    [[noreturn]] void fail(const std::string &message) const {
        invalid("byte " + std::to_string(pos) + ": " + message);
    }
    void space() {
        while (pos < text.size() && (text[pos] == ' ' || text[pos] == '\t' ||
               text[pos] == '\n' || text[pos] == '\r')) ++pos;
    }
    bool take(char c) {
        space();
        if (pos < text.size() && text[pos] == c) { ++pos; return true; }
        return false;
    }
    void expect(char c) {
        if (!take(c)) fail(std::string("expected '") + c + "'");
    }
    static bool digit(char c) { return c >= '0' && c <= '9'; }
    std::string string() {
        expect('"');
        std::string result;
        while (pos < text.size()) {
            unsigned char c = text[pos++];
            if (c == '"') return result;
            if (c < 0x20 || c >= 0x80) fail("schema strings must contain ASCII JSON text");
            if (c == '\\') {
                if (pos == text.size()) fail("unfinished string escape");
                c = text[pos++];
                switch (c) {
                case '"': case '\\': case '/': break;
                case 'b': c = '\b'; break;
                case 'f': c = '\f'; break;
                case 'n': c = '\n'; break;
                case 'r': c = '\r'; break;
                case 't': c = '\t'; break;
                case 'u': {
                    unsigned value = 0;
                    for (int i = 0; i < 4; ++i) {
                        if (pos == text.size()) fail("unfinished unicode escape");
                        const char h = text[pos++];
                        unsigned hex;
                        if (h >= '0' && h <= '9') hex = h - '0';
                        else if (h >= 'a' && h <= 'f') hex = h - 'a' + 10;
                        else if (h >= 'A' && h <= 'F') hex = h - 'A' + 10;
                        else fail("invalid unicode escape");
                        value = 16 * value + hex;
                    }
                    if (value > 0x7f) fail("schema strings must be ASCII");
                    c = static_cast<unsigned char>(value);
                    break;
                }
                default: fail("invalid string escape");
                }
            }
            result.push_back(static_cast<char>(c));
            if (result.size() > 4096) fail("schema string exceeds 4096 bytes");
        }
        fail("unterminated string");
    }
    double number(bool integer_only = false) {
        space();
        const std::size_t start = pos;
        if (pos < text.size() && text[pos] == '-') ++pos;
        if (pos == text.size() || !digit(text[pos])) fail("expected JSON number");
        if (text[pos] == '0') {
            ++pos;
            if (pos < text.size() && digit(text[pos])) fail("leading zero in JSON number");
        } else {
            while (pos < text.size() && digit(text[pos])) ++pos;
        }
        if (pos < text.size() && text[pos] == '.') {
            if (integer_only) fail("expected an integer token");
            ++pos;
            if (pos == text.size() || !digit(text[pos])) fail("missing fractional digits");
            while (pos < text.size() && digit(text[pos])) ++pos;
        }
        if (pos < text.size() && (text[pos] == 'e' || text[pos] == 'E')) {
            if (integer_only) fail("expected an integer token");
            ++pos;
            if (pos < text.size() && (text[pos] == '+' || text[pos] == '-')) ++pos;
            if (pos == text.size() || !digit(text[pos])) fail("missing exponent digits");
            while (pos < text.size() && digit(text[pos])) ++pos;
        }
        double result = 0;
        const auto parsed = std::from_chars(text.data() + start, text.data() + pos, result);
        if (parsed.ec != std::errc{} || parsed.ptr != text.data() + pos || !std::isfinite(result))
            fail("number is outside the finite representable range");
        return result;
    }
    int axis() {
        const auto name = string();
        if (name.size() != 1 || name[0] < 'a' || name[0] > 'c')
            fail("axis must be a, b, or c");
        return name[0] - 'a';
    }
    LatticeConstraintDeclaration declaration() {
        LatticeConstraintDeclaration result;
        std::set<std::string> seen;
        bool has_axis = false, has_axes = false, has_atoms = false;
        expect('{');
        if (!take('}')) {
            do {
                const auto key = string();
                if (!seen.insert(key).second) fail("duplicate key '" + key + "'");
                expect(':');
                if (key == "kind") {
                    const auto kind = string();
                    if (kind == "volume") result.kind = LatticeConstraintKind::Volume;
                    else if (kind == "lattice_length") result.kind = LatticeConstraintKind::Length;
                    else if (kind == "lattice_angle") result.kind = LatticeConstraintKind::Angle;
                    else if (kind == "lattice_ratio") result.kind = LatticeConstraintKind::Ratio;
                    else if (kind == "distance") result.kind = ConstraintKind::Distance;
                    else if (kind == "angle") result.kind = ConstraintKind::AtomicAngle;
                    else if (kind == "dihedral") result.kind = ConstraintKind::Dihedral;
                    else if (kind == "position") result.kind = ConstraintKind::Position;
                    else if (kind == "wall") result.kind = ConstraintKind::Wall;
                    else if (kind == "formula") result.kind = ConstraintKind::Formula;
                    else if (kind == "symmetry") result.kind = ConstraintKind::Symmetry;
                    else fail("unknown constraint kind '" + kind + "'");
                } else if (key == "axis") {
                    has_axis = true;
                    result.axis = axis();
                } else if (key == "axes") {
                    has_axes = true;
                    expect('['); result.axis = axis(); expect(',');
                    result.second_axis = axis(); expect(']');
                } else if (key == "atoms") {
                    has_atoms = true;
                    expect('[');
                    if (!take(']')) {
                        do {
                            if (result.atoms.size() == 4) fail("at most four atom indices are allowed");
                            const double index = number(true);
                            if (index < 0 || index > std::numeric_limits<int>::max())
                                fail("atom index is outside the nonnegative integer range");
                            result.atoms.push_back(static_cast<int>(index));
                        } while (take(','));
                        expect(']');
                    }
                } else if (key == "component") {
                    const auto c=string(); if(c.size()!=1 || c[0]<'x' || c[0]>'z') fail("component must be x/y/z"); result.component=c[0]-'x';
                } else if (key == "normal") {
                    expect('['); for(int i=0;i<3;++i){if(i)expect(',');result.normal[i]=number();} expect(']');
                } else if (key == "operations" || key == "sites") {
                    auto &ids=key=="operations"?result.operations:result.sites;
                    expect('['); if(!take(']')) {do {const auto id=number(true);if(id<0 || id>std::numeric_limits<int>::max())fail("invalid index");ids.push_back(static_cast<int>(id));}while(take(','));expect(']');}
                } else if (key == "expression") result.expression=string();
                else if (key == "lower") result.lower=number();
                else if (key == "upper") result.upper=number();
                else if (key == "target") result.target = number();
                else if (key == "residual_tolerance") result.residual_tolerance = number();
                else fail("unknown constraint key '" + key + "'");
            } while (take(','));
            expect('}');
        }
        if (!seen.contains("kind") || (result.kind!=ConstraintKind::Symmetry && !seen.contains("residual_tolerance")))
            fail("kind and residual_tolerance are required");
        if ((has_axis && result.kind != LatticeConstraintKind::Length) ||
            (has_axes && result.kind != LatticeConstraintKind::Angle &&
                         result.kind != LatticeConstraintKind::Ratio))
            fail("axis/axes is not allowed for this kind");
        if (has_atoms && result.kind != ConstraintKind::Distance && result.kind != ConstraintKind::AtomicAngle &&
            result.kind != ConstraintKind::Dihedral && result.kind != ConstraintKind::Position && result.kind != ConstraintKind::Wall)
            fail("atoms is only allowed for distance or angle");
        if ((seen.contains("component") && result.kind!=ConstraintKind::Position) ||
            (seen.contains("normal") && result.kind!=ConstraintKind::Wall) ||
            (seen.contains("expression") && result.kind!=ConstraintKind::Formula) ||
            ((seen.contains("operations") || seen.contains("sites")) && result.kind!=ConstraintKind::Symmetry))
            fail("constraint key is not applicable to this kind");
        validate_declaration(result);
        return result;
    }
    LatticeConstraintDeclarations declarations() {
        LatticeConstraintDeclarations result;
        std::set<Identity> identities;
        expect('[');
        if (take(']')) fail("constraints must not be empty");
        do {
            const std::size_t index = result.size();
            if (index == lattice_constraints_max_count) fail("too many constraints");
            try {
                auto item = declaration();
                if (!identities.insert(validate_declaration(item)).second)
                    fail("duplicate constraint identity");
                result.push_back(std::move(item));
            } catch (const std::invalid_argument &error) {
                invalid("constraint[" + std::to_string(index) + "]: " + error.what());
            }
        } while (take(','));
        expect(']');
        return result;
    }
public:
    explicit Parser(const std::string &source) : text(source) {}
    LatticeConstraintDeclarations parse() {
        LatticeConstraintDeclarations result;
        std::set<std::string> seen;
        expect('{');
        if (!take('}')) {
            do {
                const auto key = string();
                if (!seen.insert(key).second) fail("duplicate key '" + key + "'");
                expect(':');
                if (key == "version") {
                    const auto v=number(true); if(v!=1 && v!=2) fail("only version 1 or 2 is supported"); version=static_cast<int>(v);
                } else if (key == "constraints") result = declarations();
                else fail("unknown top-level key '" + key + "'");
            } while (take(','));
            expect('}');
        }
        if (!seen.contains("version") || !seen.contains("constraints"))
            fail("version and constraints are required");
        for(auto &item:result) {
            item.version=version;
            if(version==1 && (item.lower || item.upper || static_cast<int>(item.kind)>static_cast<int>(ConstraintKind::AtomicAngle) ||
                !item.expression.empty() || item.component!=-1 || !item.normal.isZero() || !item.operations.empty() || !item.sites.empty()))
                fail("extended constraint fields require version 2");
        }
        space();
        if (pos != text.size()) fail("trailing content");
        return result;
    }
};

std::string read_bounded(std::istream &input) {
    if (!input.good())
        throw std::runtime_error("Lattice constraints: input stream is not readable");
    std::string text;
    std::array<char, 4096> buffer{};
    for (;;) {
        try {
            input.read(buffer.data(), buffer.size());
        } catch (const std::ios_base::failure &) {
            if (input.bad() || !input.eof())
                throw std::runtime_error("Lattice constraints: input stream read failed");
        }
        const auto count = static_cast<std::size_t>(input.gcount());
        if (count > lattice_constraints_max_bytes - text.size())
            invalid("input exceeds " + std::to_string(lattice_constraints_max_bytes) + " bytes");
        text.append(buffer.data(), count);
        if (input.bad() || (input.fail() && !input.eof()))
            throw std::runtime_error("Lattice constraints: input stream read failed");
        if (input.eof()) break;
    }
    return text;
}

} // namespace

LatticeConstraintDeclarations parse_lattice_constraints(std::istream &input) {
    const std::string text = read_bounded(input);
    return Parser(text).parse();
}

LatticeConstraintDeclarations read_lattice_constraints(const std::string &path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) throw std::runtime_error("Lattice constraints: cannot open '" + path + "'");
    try {
        return parse_lattice_constraints(input);
    } catch (const std::invalid_argument &error) {
        throw std::invalid_argument("Lattice constraints file '" + path + "': " + error.what());
    } catch (const std::runtime_error &error) {
        throw std::runtime_error("Lattice constraints file '" + path + "': " + error.what());
    }
}

void configure_symmetry(const ConstraintDeclarations &items, RelaxationDrvV2 &driver) {
    for(const auto &item:items) if(item.kind==ConstraintKind::Symmetry) {
        const auto join=[](const std::vector<int> &ids){std::string s;for(int i:ids){if(!s.empty())s+=",";s+=std::to_string(i);}return s;};
        driver.flags.keep_symm=true;
        driver.control.symmetry_operations=join(item.operations);driver.control.symmetry_sites=join(item.sites);
    }
}

ConstraintDeclarations parse_constraints(std::istream &input) {
    return parse_lattice_constraints(input);
}

ConstraintDeclarations read_constraints(const std::string &path) {
    return read_lattice_constraints(path);
}

std::vector<regulator> bind_constraints(const ConstraintDeclarations &declarations,
                                      const Atoms &initial_frame) {
    if (declarations.empty() || declarations.size() > lattice_constraints_max_count)
        invalid("constraints count must be between 1 and " + std::to_string(lattice_constraints_max_count));
    std::vector<regulator> result;
    std::set<Identity> identities;
    for (const auto &item : declarations) {
        try {
            if (!identities.insert(validate_declaration(item)).second)
                invalid("duplicate constraint identity");
            if(item.kind==ConstraintKind::Symmetry) continue;
            if(item.version==2) {
                Atoms initial=initial_frame; if(!initial.cartesian) initial.to_cartesian();
                const auto arg=[](int i){return std::to_string(i);};
                std::string expr=item.expression;
                switch(item.kind) {
                case ConstraintKind::Volume: expr="volume()";break;
                case ConstraintKind::Length: expr="lattice_length("+arg(item.axis)+")";break;
                case ConstraintKind::Angle: expr="lattice_angle("+arg(item.axis)+","+arg(item.second_axis)+")";break;
                case ConstraintKind::Ratio: expr="lattice_ratio("+arg(item.axis)+","+arg(item.second_axis)+")";break;
                case ConstraintKind::Distance: case ConstraintKind::AtomicAngle: case ConstraintKind::Dihedral:
                    expr=item.kind==ConstraintKind::Distance?"distance(":item.kind==ConstraintKind::AtomicAngle?"angle(":"dihedral(";
                    for(size_t i=0;i<item.atoms.size();++i){if(i)expr+=",";expr+=arg(item.atoms[i]);}expr+=")";break;
                case ConstraintKind::Position: expr=std::string(1,"xyz"[item.component])+"("+arg(item.atoms[0])+")";break;
                case ConstraintKind::Wall: {
                    const auto normal=item.normal.normalized(); std::ostringstream out;out.precision(17);
                    for(int i=0;i<3;++i){if(i)out<<"+";out<<"("<<normal[i]<<")*"<<"xyz"[i]<<"("<<item.atoms[0]<<")";}expr=out.str();break;
                }
                default:break;
                }
                regulator reg;reg.regulator_t=regulator::FORMULA;reg.type=10;reg.tol=item.residual_tolerance;
                reg.expression=std::make_shared<Expression>(expr,initial);
                reg.integer_specs.resize(initial.size());std::iota(reg.integer_specs.begin(),reg.integer_specs.end(),0);
                reg.bounded=item.lower.has_value() || item.upper.has_value();
                const double initial_value=reg.expression->evaluate(initial).value;
                reg.g0=reg.bounded?0:item.target.value_or(initial_value);
                reg.lower=item.lower.value_or(-std::numeric_limits<double>::infinity());
                reg.upper=item.upper.value_or(std::numeric_limits<double>::infinity());
                if(item.kind==ConstraintKind::Dihedral) {
                    if(reg.bounded) {
                        const double center=item.lower && item.upper?.5*(*item.lower+*item.upper):item.lower?*item.lower:*item.upper;
                        const double shift=360*std::round((initial_value-center)/360);
                        reg.lower+=shift;reg.upper+=shift;
                    } else reg.g0+=360*std::round((initial_value-reg.g0)/360);
                }
                result.push_back(std::move(reg));continue;
            }
            if (item.kind != ConstraintKind::Distance && item.kind != ConstraintKind::AtomicAngle) {
                auto bound = bind_lattice_constraints({item}, initial_frame.cell_);
                result.push_back(std::move(bound.front()));
                continue;
            }
            regulator reg{};
            reg.regulator_t = item.kind == ConstraintKind::Distance ? regulator::DISTANCE : regulator::ANGLE;
            reg.type = 10;
            reg.integer_specs = item.atoms;
            reg.to_target_g0 = item.target.has_value();
            reg.double_target = item.target.value_or(0.0);
            reg.tol = item.residual_tolerance;
            reg.weight_init = reg.weight = 1.0;
            reg.weight_step = reg.weight_multply = 1.0;
            reg.lambda_init = reg.lambda = 0.0;
            atomic_geometry_setup(reg, initial_frame);
            result.push_back(std::move(reg));
        } catch (const std::invalid_argument &error) {
            invalid("constraint[" + std::to_string(result.size()) + "]: " + error.what());
        }
    }
    return result;
}

std::vector<regulator> bind_lattice_constraints(
    const LatticeConstraintDeclarations &declarations,
    const Eigen::Matrix3d &initial_cell) {
    if (declarations.empty() || declarations.size() > lattice_constraints_max_count)
        invalid("constraints count must be between 1 and " + std::to_string(lattice_constraints_max_count));
    const double volume = initial_cell.determinant();
    if (!initial_cell.allFinite() || !std::isfinite(volume) || volume <= 0 ||
        !initial_cell.inverse().allFinite())
        invalid("initial cell must be finite, nondegenerate, and right-handed");
    std::array<double, 3> squared_lengths{};
    for (int axis = 0; axis < 3; ++axis) {
        squared_lengths[axis] = initial_cell.row(axis).squaredNorm();
        if (!std::isfinite(squared_lengths[axis]) || squared_lengths[axis] <= 0)
            invalid("initial lattice lengths must be finite and nonzero");
    }
    Atoms frame;
    frame.cell_ = initial_cell;
    std::vector<regulator> result;
    std::set<Identity> identities;
    for (const auto &item : declarations) {
        try {
            if (!identities.insert(validate_declaration(item)).second)
                invalid("duplicate constraint identity");
            regulator reg{};
            reg.type = 10; // AUL equality: lambda*f + weight*f*f/2.
            reg.to_target_g0 = true;
            reg.tol = item.residual_tolerance;
            reg.weight_init = reg.weight = 1.0;
            reg.weight_step = 1.0;
            reg.weight_multply = 1.0;
            reg.lambda_init = reg.lambda = 0.0;
            switch (item.kind) {
            case LatticeConstraintKind::Volume:
                reg.regulator_t = regulator::CONSTANT_VOLUME;
                reg.double_target = item.target.value_or(volume);
                reg.g0 = reg.double_target;
                constant_volume_setup(reg, frame);
                break;
            case LatticeConstraintKind::Length:
                reg.regulator_t = regulator::CONSTANT_VECTOR;
                reg.integer_specs = {item.axis};
                reg.double_target = item.target.value_or(std::sqrt(squared_lengths[item.axis]));
                reg.g0 = item.target ? *item.target * *item.target : squared_lengths[item.axis];
                constant_vector_setup(reg, frame, item.axis);
                break;
            case LatticeConstraintKind::Angle: {
                reg.regulator_t = regulator::CONSTANT_ANGLE;
                reg.integer_specs = {item.axis, item.second_axis};
                const double a = std::sqrt(squared_lengths[item.axis]);
                const double b = std::sqrt(squared_lengths[item.second_axis]);
                const double cosine = initial_cell.row(item.axis).dot(initial_cell.row(item.second_axis)) / a / b;
                if (!std::isfinite(cosine) || cosine <= -1 || cosine >= 1 ||
                    !std::isfinite(a*a*a*b) || !std::isfinite(a*b*b*b) ||
                    a*a*a*b <= 0 || a*b*b*b <= 0)
                    invalid("initial angle/gradient is not representable");
                reg.double_target = item.target.value_or(std::acos(cosine) * (180.0 / std::numbers::pi));
                reg.g0 = item.target ? std::cos(*item.target * (std::numbers::pi / 180.0)) : cosine;
                constant_angle_setup(reg, frame, item.axis, item.second_axis);
                break;
            }
            case LatticeConstraintKind::Ratio: {
                reg.regulator_t = regulator::COUPLE_VECTOR;
                reg.integer_specs = {item.axis, item.second_axis};
                const double a2 = squared_lengths[item.axis], b2 = squared_lengths[item.second_axis];
                if (!std::isfinite(a2 / b2) || a2 / b2 <= 0 ||
                    !std::isfinite(b2*b2) || b2*b2 <= 0 ||
                    !std::isfinite(1.0 / b2) || !std::isfinite(a2 / (b2*b2)))
                    invalid("initial ratio/gradient is not representable");
                reg.double_target = item.target.value_or(std::sqrt(a2 / b2));
                reg.g0 = item.target ? *item.target * *item.target : a2 / b2;
                couple_vector_setup(reg, frame, item.axis, item.second_axis);
                break;
            }
            default: invalid("unknown lattice constraint kind");
            }
            if (!std::isfinite(reg.g0)) invalid("converted target is not finite");
            result.push_back(std::move(reg));
        } catch (const std::invalid_argument &error) {
            invalid("constraint[" + std::to_string(result.size()) + "]: " + error.what());
        }
    }
    return result;
}

} // namespace Constraints
