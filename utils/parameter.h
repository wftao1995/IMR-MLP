// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#pragma once

#include <algorithm>
#include <cctype>
#include <complex>
#include <functional>
#include <limits>
#include <memory>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace Parameters {

class Error : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

enum class Source { Default, Settings, CommandLine };
using Values = std::vector<std::string>;
using Occurrences = std::vector<Values>;
using Commit = std::function<void()>;

template<class T> struct Container : std::false_type {};
template<class T, class A> struct Container<std::vector<T, A>> : std::true_type {
    using Element = T;
};
template<class T, class C, class A> struct Container<std::set<T, C, A>> : std::true_type {
    using Element = T;
};

template<class T> std::string type_name() {
    if constexpr (Container<T>::value)
        return type_name<typename Container<T>::Element>() + " array";
    else if constexpr (std::is_same_v<T, bool>) return "bool";
    else if constexpr (std::is_integral_v<T>) return "integer";
    else if constexpr (std::is_floating_point_v<T>) return "float";
    else if constexpr (std::is_same_v<T, std::string>) return "string";
    else if constexpr (std::is_same_v<T, std::complex<double>>) return "complex";
    else return "unsupported";
}

// One scalar grammar for CLI values, settings, and container elements.
// Keep the existing decimal integer, hexadecimal float, and float-suffix syntax.
template<class T> T parse_value(const std::string &text) {
    if constexpr (std::is_same_v<T, std::string>) {
        return text;
    } else if constexpr (std::is_same_v<T, bool>) {
        std::string value = text;
        std::transform(value.begin(), value.end(), value.begin(),
                       [](unsigned char c) { return std::tolower(c); });
        if (value == "true" || value == "yes" || value == "t" || value == "y" || value == "1") return true;
        if (value == "false" || value == "no" || value == "f" || value == "n" || value == "0") return false;
        throw Error("invalid boolean value '" + text + "'");
    } else {
        T result{};
        std::size_t parsed = 0;
        try {
            if constexpr (std::is_integral_v<T> && std::is_signed_v<T>) {
                auto value = std::stoll(text, &parsed);
                if (value < std::numeric_limits<T>::min() || value > std::numeric_limits<T>::max())
                    throw std::out_of_range("integer range");
                result = static_cast<T>(value);
            } else if constexpr (std::is_integral_v<T>) {
                const auto first = text.find_first_not_of(" \t\r\n\f\v");
                if (first != std::string::npos && text[first] == '-')
                    throw std::out_of_range("negative unsigned value");
                auto value = std::stoull(text, &parsed);
                if (value > std::numeric_limits<T>::max()) throw std::out_of_range("integer range");
                result = static_cast<T>(value);
            } else if constexpr (std::is_same_v<T, float>) {
                result = std::stof(text, &parsed);
                if (parsed + 1 == text.size() && (text.back() == 'f' || text.back() == 'F')) ++parsed;
            } else if constexpr (std::is_same_v<T, double>) {
                result = std::stod(text, &parsed);
            } else if constexpr (std::is_same_v<T, long double>) {
                result = std::stold(text, &parsed);
            } else if constexpr (std::is_same_v<T, std::complex<double>>) {
                std::istringstream stream(text);
                if (!(stream >> result) || stream.peek() != std::char_traits<char>::eof())
                    throw std::invalid_argument("complex value");
                parsed = text.size();
            } else {
                throw Error("unsupported parameter type");
            }
        } catch (const std::invalid_argument &) {
            throw Error("invalid " + type_name<T>() + " value '" + text + "'");
        } catch (const std::out_of_range &) {
            throw Error("out-of-range " + type_name<T>() + " value '" + text + "'");
        }
        if (parsed != text.size()) throw Error("invalid " + type_name<T>() + " value '" + text + "'");
        return result;
    }
}

template<class T> std::string format_value(const T &value) {
    std::ostringstream stream;
    stream.precision(std::numeric_limits<double>::max_digits10);
    stream << std::boolalpha;
    if constexpr (Container<T>::value) {
        bool first = true;
        for (const auto &element : value) {
            if (!first) stream << ' ';
            stream << element;
            first = false;
        }
    } else stream << value;
    return stream.str();
}

// A binding borrows its target; the target must outlive every view of it.
// CLI and module settings share this record, including source precedence.
struct Binding {
    std::string name, help, type;
    int nargs = 1;
    int level = 0;
    bool container = false;
    bool boolean = false;
    Source source = Source::Default;
    std::function<Commit(const Occurrences &, Source)> convert;
    std::function<std::string()> format;

    Commit prepare(const Occurrences &values, Source from, const std::string &location) {
        if (from < source) return {};
        try {
            auto commit = convert(values, from);
            return [this, from, commit = std::move(commit)]() {
                commit();
                source = from;
            };
        } catch (const Error &error) {
            throw Error(location + ": parameter " + name + " (" + type + "): " + error.what());
        }
    }

    void apply_setting(const std::string &text, const std::string &location, Source from = Source::Settings) {
        if (from < source) return;
        Values values;
        if (container) {
            std::istringstream stream(text);
            for (std::string value; stream >> value;) values.push_back(std::move(value));
        } else values.push_back(text);
        if (auto commit = prepare({values}, from, location)) commit();
    }
};

template<class T>
std::shared_ptr<Binding> bind(T &target, std::string name, std::string help = "",
                              int nargs = 1, int level = 0) {
    auto binding = std::make_shared<Binding>();
    binding->name = std::move(name);
    binding->help = std::move(help);
    binding->type = type_name<T>();
    binding->nargs = nargs;
    binding->level = level;
    binding->container = Container<T>::value;
    binding->boolean = std::is_same_v<T, bool>;
    binding->format = [&target] { return format_value(target); };
    binding->convert = [&target, nargs](const Occurrences &occurrences, Source source) -> Commit {
        T value{};
        if constexpr (!Container<T>::value) value = target;
        for (const auto &tokens : occurrences) {
            if constexpr (Container<T>::value) {
                if ((nargs == -1 && tokens.empty()) || nargs == 0 || nargs < -1 ||
                    (nargs > 0 && tokens.size() != static_cast<std::size_t>(nargs)))
                    throw Error("expected " + (nargs == -1 ? std::string("at least one value") : std::to_string(nargs) + " values") +
                                ", received " + std::to_string(tokens.size()));
                for (const auto &token : tokens) {
                    auto element = parse_value<typename Container<T>::Element>(token);
                    value.insert(value.end(), std::move(element));
                }
            } else {
                if constexpr (std::is_same_v<T, bool>) {
                    if (source == Source::CommandLine && nargs == 0 && tokens.empty()) {
                        value = !value;
                        continue;
                    }
                }
                if (tokens.size() != 1 || (nargs != 1 && !(std::is_same_v<T, bool> && nargs == 0)))
                    throw Error("expected one value, received " + std::to_string(tokens.size()));
                value = parse_value<T>(tokens.front());
            }
        }
        return [&target, value = std::move(value)]() mutable { target = std::move(value); };
    };
    return binding;
}

} // namespace Parameters
