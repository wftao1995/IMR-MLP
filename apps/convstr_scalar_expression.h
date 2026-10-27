#pragma once

// Scalar metadata expressions for convstr only; no geometry or derivative binding.
#include "base/structures.h"
#include <charconv>
#include <cctype>
#include <cmath>
#include <limits>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace ConvstrLabels {

class Expression {
    enum class Op { Number, Variable, Add, Subtract, Multiply, Divide, Power,
                    Negate, Sqrt, Exp, Log, Sin, Cos };
    struct Instruction { Op op; double number = 0; std::string variable; };
    std::string source_, target_;
    std::vector<Instruction> code_;
    std::size_t position_ = 0;
    int depth_ = 0;

    [[noreturn]] void invalid(const std::string &message) const {
        throw std::invalid_argument("convstr label expression '" + source_ + "': " + message);
    }
    void space() {
        while (position_ < source_.size() &&
               std::isspace(static_cast<unsigned char>(source_[position_]))) ++position_;
    }
    bool take(char character) {
        space();
        if (position_ == source_.size() || source_[position_] != character) return false;
        ++position_;
        return true;
    }
    std::string identifier() {
        space();
        const auto begin = position_;
        if (position_ < source_.size() &&
            (std::isalpha(static_cast<unsigned char>(source_[position_])) || source_[position_] == '_')) {
            ++position_;
            while (position_ < source_.size() &&
                   (std::isalnum(static_cast<unsigned char>(source_[position_])) || source_[position_] == '_'))
                ++position_;
        }
        return source_.substr(begin, position_ - begin);
    }
    void emit(Op op) { code_.push_back({op, 0, {}}); }
    // The bool records whether a subtree contains only constants.
    bool sum() {
        bool constant = product();
        while (true) {
            if (take('+')) { bool rhs = product(); constant = rhs && constant; emit(Op::Add); }
            else if (take('-')) { bool rhs = product(); constant = rhs && constant; emit(Op::Subtract); }
            else return constant;
        }
    }
    bool product() {
        bool constant = unary();
        while (true) {
            if (take('*')) { bool rhs = unary(); constant = rhs && constant; emit(Op::Multiply); }
            else if (take('/')) { bool rhs = unary(); constant = rhs && constant; emit(Op::Divide); }
            else return constant;
        }
    }
    bool unary() {
        if (++depth_ > 128) invalid("expression nesting exceeds 128");
        bool constant;
        if (take('+')) constant = unary();
        else if (take('-')) { constant = unary(); emit(Op::Negate); }
        else {
            constant = primary();
            if (take('^')) {
                if (!unary()) invalid("power exponent must be constant");
                emit(Op::Power);
            }
        }
        --depth_;
        return constant;
    }
    bool primary() {
        if (take('(')) {
            const bool constant = sum();
            if (!take(')')) invalid("expected ')'");
            return constant;
        }
        space();
        if (position_ < source_.size() &&
            (std::isdigit(static_cast<unsigned char>(source_[position_])) || source_[position_] == '.')) {
            double number;
            const char *begin = source_.data() + position_;
            const auto parsed = std::from_chars(begin, source_.data() + source_.size(), number);
            if (parsed.ec != std::errc{} || !std::isfinite(number)) invalid("invalid finite number");
            position_ = parsed.ptr - source_.data();
            code_.push_back({Op::Number, number, {}});
            return true;
        }
        const auto name = identifier();
        if (name.empty()) invalid("expected number, scalar variable or '('");
        if (!take('(')) {
            code_.push_back({Op::Variable, 0, name});
            return false;
        }
        Op op;
        if (name == "sqrt") op = Op::Sqrt;
        else if (name == "exp") op = Op::Exp;
        else if (name == "log") op = Op::Log;
        else if (name == "sin") op = Op::Sin;
        else if (name == "cos") op = Op::Cos;
        else invalid("unknown function '" + name + "'");
        const bool constant = sum();
        if (!take(')')) invalid("expected ')' after function argument");
        emit(op);
        return constant;
    }
    static double scalar(const Atoms &frame, const std::string &name) {
        if (name == "energy" && frame.has_energy()) return frame.energy_;
        if (name == "fenergy" && frame.has_fenergy()) return frame.fenergy_;
        if (name == "fitness") return frame.fitness;
        if (name == "volume") return frame.get_volume();
        if (name == "size") return static_cast<double>(frame.size());
        if (name == "id") return static_cast<double>(frame.id);
        // Missing physical fields must not fall through to stale metadata aliases.
        if (name != "energy" && name != "fenergy") {
            auto real = frame.global_attribs_d_.find(name);
            if (real != frame.global_attribs_d_.end()) return real->second;
            auto integer = frame.global_attribs_i_.find(name);
            if (integer != frame.global_attribs_i_.end()) return integer->second;
        }
        throw std::invalid_argument("missing or nonnumeric scalar '" + name + "'");
    }

public:
    static bool reserved_target(const std::string &name) {
        static const std::set<std::string> reserved = {
            "energy", "fenergy", "free_energy", "fitness", "Fitness", "volume", "size", "id",
            "Lattice", "lattice", "cell", "pbc", "Properties", "positions", "pos",
            "forces", "velocities", "virial", "stress", "charges", "eff_charges",
            "site_en", "magmoms", "total_charge", "hessian", "hvp_directions", "hvp_targets"
        };
        return reserved.count(name) != 0;
    }
    explicit Expression(std::string source) : source_(std::move(source)) {
        target_ = identifier();
        if (target_.empty() || !take('=')) invalid("expected metadata_name=expression");
        if (reserved_target(target_)) invalid("target '" + target_ + "' is reserved for a physical or built-in field");
        sum();
        space();
        if (position_ != source_.size()) invalid("unexpected token at character " + std::to_string(position_ + 1));
    }
    const std::string &target() const { return target_; }
    const std::string &source() const { return source_; }
    double evaluate(const Atoms &frame) const {
        try {
            std::vector<double> stack;
            stack.reserve(code_.size());
            for (const auto &instruction : code_) {
                const auto op = instruction.op;
                if (op == Op::Number) stack.push_back(instruction.number);
                else if (op == Op::Variable) stack.push_back(scalar(frame, instruction.variable));
                else if (op == Op::Negate) stack.back() = -stack.back();
                else if (op == Op::Sqrt || op == Op::Exp || op == Op::Log || op == Op::Sin || op == Op::Cos) {
                    double &value = stack.back();
                    if (op == Op::Sqrt) {
                        if (value < 0) throw std::domain_error("sqrt domain requires a nonnegative argument");
                        value = std::sqrt(value);
                    } else if (op == Op::Log) {
                        if (value <= 0) throw std::domain_error("log domain requires a positive argument");
                        value = std::log(value);
                    } else if (op == Op::Exp) value = std::exp(value);
                    else if (op == Op::Sin) value = std::sin(value);
                    else value = std::cos(value);
                } else {
                    double rhs = stack.back(); stack.pop_back();
                    double &lhs = stack.back();
                    if (op == Op::Add) lhs += rhs;
                    else if (op == Op::Subtract) lhs -= rhs;
                    else if (op == Op::Multiply) lhs *= rhs;
                    else if (op == Op::Divide) {
                        if (rhs == 0) throw std::domain_error("division by zero");
                        lhs /= rhs;
                    } else lhs = std::pow(lhs, rhs);
                }
                if (!std::isfinite(stack.back())) throw std::domain_error("nonfinite scalar or arithmetic/function domain/overflow error");
            }
            return stack.back();
        } catch (const std::exception &error) {
            throw std::invalid_argument("convstr frame " + std::to_string(frame.id) +
                " label expression '" + source_ + "': " + error.what());
        }
    }
};

inline std::vector<Expression> compile(const std::vector<std::string> &sources,
                                       const std::vector<std::string> &text_labels = {}) {
    std::vector<Expression> expressions;
    std::set<std::string> targets;
    for (const auto &source : sources) {
        Expression expression(source);
        if (!targets.insert(expression.target()).second)
            throw std::invalid_argument("convstr duplicate label expression target '" + expression.target() + "'");
        for (const auto &label : text_labels)
            if (label == expression.target())
                throw std::invalid_argument("convstr label expression target '" + label + "' also has a text --label");
        expressions.push_back(std::move(expression));
    }
    return expressions;
}

inline void assign(Atoms &frame, const std::vector<Expression> &expressions, bool cfg_text_bridge = false) {
    std::vector<double> values;
    values.reserve(expressions.size());
    // Do not mutate the frame until every expression has read the original snapshot.
    for (const auto &expression : expressions) {
        const auto &name = expression.target();
        if (frame.global_attribs_v_.count(name) || frame.global_attribs_vi_.count(name) ||
            frame.local_attribs_d_.count(name) || frame.local_attribs_i_.count(name) ||
            frame.local_attribs_v_.count(name) || frame.local_attribs_vi_.count(name))
            throw std::invalid_argument("convstr frame " + std::to_string(frame.id) +
                " label expression '" + expression.source() + "': target is an existing nonscalar attribute");
        values.push_back(expression.evaluate(frame));
    }
    for (std::size_t i = 0; i < expressions.size(); ++i) {
        const auto &name = expressions[i].target();
        frame.global_attribs_i_.erase(name);
        frame.global_attribs_d_[name] = values[i];
        // CFG's legacy Feature writer is textual. Other codecs write the double
        // map directly; retaining a Feature there would duplicate the header key.
        frame.features_.erase(name);
        if (cfg_text_bridge) {
            std::ostringstream text;
            text.precision(std::numeric_limits<double>::max_digits10);
            text << values[i];
            frame.features_[name] = text.str();
        }
    }
}

} // namespace ConvstrLabels
