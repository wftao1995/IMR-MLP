// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#pragma once

#include "base/structures.h"
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <string>
#include <type_traits>
#include <vector>

namespace gpio_binary {

inline constexpr char MAGIC[] = {'G', 'P', 'M', 'L', 'B', 'I', 'N', '2'};
inline constexpr std::uint32_t VERSION = 3;
inline constexpr std::uint32_t MIN_READABLE_VERSION = 2;
inline constexpr std::int32_t MAX_STRING_SIZE = 1 << 20;
inline constexpr std::int32_t MAX_VECTOR_SIZE = 1 << 28;
inline constexpr std::int32_t MAX_ATOMS = MAX_VECTOR_SIZE / 3;

enum class Tag : char {
    Cell = 'l',
    Positions = 'p',
    Types = 't',
    Forces = 'f',
    FreeEnergy = 'h',
    SiteEnergies = 'v',
    MagmomVector = 'm',
    MagmomScalar = 'M',
    Charges = 'c',
    Energy = 'e',
    Stress = 's',
    Velocities = 'u',
    Ids = 'i',
    Feature = 'F',
    LocalDouble = 'A',
    LocalDoubleVector = 'B',
    GlobalDouble = 'C',
    GlobalDoubleVector = 'D',
    GlobalInt = 'E',
    GlobalIntVector = 'G',
    PBC = 'b',
    LocalInt = 'H',
    LocalIntVector = 'I',
    End = '#',
};

template <typename T>
bool write_value(std::ostream &os, const T &value)
{
    static_assert(std::is_trivially_copyable_v<T>);
    os.write(reinterpret_cast<const char *>(&value), sizeof(T));
    return static_cast<bool>(os);
}

template <typename T>
bool read_value(std::istream &is, T &value)
{
    static_assert(std::is_trivially_copyable_v<T>);
    is.read(reinterpret_cast<char *>(&value), sizeof(T));
    return static_cast<bool>(is);
}

inline bool write_tag(std::ostream &os, Tag tag)
{
    return write_value(os, static_cast<char>(tag));
}

inline bool read_tag(std::istream &is, Tag &tag)
{
    char raw = 0;
    if (!read_value(is, raw)) return false;
    tag = static_cast<Tag>(raw);
    return true;
}

template <typename T>
bool write_span(std::ostream &os, const T *data, std::size_t count)
{
    if (count == 0) return true;
    if (count > static_cast<std::size_t>(std::numeric_limits<std::streamsize>::max()) / sizeof(T)) {
        os.setstate(std::ios::failbit);
        return false;
    }
    os.write(reinterpret_cast<const char *>(data), static_cast<std::streamsize>(sizeof(T) * count));
    return static_cast<bool>(os);
}

template <typename T>
bool read_span(std::istream &is, T *data, std::size_t count)
{
    if (count == 0) return true;
    if (count > static_cast<std::size_t>(std::numeric_limits<std::streamsize>::max()) / sizeof(T)) {
        is.setstate(std::ios::failbit);
        return false;
    }
    is.read(reinterpret_cast<char *>(data), static_cast<std::streamsize>(sizeof(T) * count));
    return static_cast<bool>(is);
}

inline bool write_string(std::ostream &os, const std::string &value)
{
    if (value.size() > static_cast<std::size_t>(MAX_STRING_SIZE)) {
        os.setstate(std::ios::failbit);
        return false;
    }
    auto size = static_cast<std::int32_t>(value.size());
    if (!write_value(os, size)) return false;
    return write_span(os, value.data(), value.size());
}

inline bool read_string(std::istream &is, std::string &value)
{
    std::int32_t size = 0;
    if (!read_value(is, size)) return false;
    if (size < 0 || size > MAX_STRING_SIZE) return false;
    value.resize(static_cast<std::size_t>(size));
    return read_span(is, value.data(), value.size());
}

inline bool write_header(std::ostream &os, int natoms)
{
    if (natoms < 0 || natoms > MAX_ATOMS) {
        os.setstate(std::ios::failbit);
        return false;
    }
    auto count = static_cast<std::int32_t>(natoms);
    return write_span(os, MAGIC, sizeof(MAGIC)) &&
           write_value(os, VERSION) &&
           write_value(os, count);
}

inline bool read_header(std::istream &is, int &natoms,
                        std::uint32_t *version_out = nullptr)
{
    char magic[sizeof(MAGIC)]{};
    if (!read_span(is, magic, sizeof(magic))) return false;
    if (std::memcmp(magic, MAGIC, sizeof(MAGIC)) != 0) return false;

    std::uint32_t version = 0;
    std::int32_t count = 0;
    if (!read_value(is, version) || !read_value(is, count)) return false;
    if (version < MIN_READABLE_VERSION || version > VERSION || count < 0 || count > MAX_ATOMS) return false;
    if (version_out) *version_out = version;
    natoms = static_cast<int>(count);
    return true;
}

template <typename T>
bool write_vector(std::ostream &os, const std::vector<T> &values)
{
    if (values.size() > static_cast<std::size_t>(MAX_VECTOR_SIZE)) {
        os.setstate(std::ios::failbit);
        return false;
    }
    auto size = static_cast<std::int32_t>(values.size());
    return write_value(os, size) && write_span(os, values.data(), values.size());
}

template <typename T>
bool read_vector(std::istream &is, std::vector<T> &values)
{
    std::int32_t size = 0;
    if (!read_value(is, size)) return false;
    if (size < 0 || size > MAX_VECTOR_SIZE) return false;
    values.resize(static_cast<std::size_t>(size));
    return read_span(is, values.data(), values.size());
}

inline bool write_vector(std::ostream &os, const Eigen::VectorXd &values)
{
    if (values.size() < 0 || values.size() > MAX_VECTOR_SIZE) {
        os.setstate(std::ios::failbit);
        return false;
    }
    auto size = static_cast<std::int32_t>(values.size());
    return write_value(os, size) && write_span(os, values.data(), static_cast<std::size_t>(values.size()));
}

inline bool read_vector(std::istream &is, Eigen::VectorXd &values)
{
    std::int32_t size = 0;
    if (!read_value(is, size)) return false;
    if (size < 0 || size > MAX_VECTOR_SIZE) return false;
    values.resize(size);
    return read_span(is, values.data(), static_cast<std::size_t>(values.size()));
}

inline bool write_vector(std::ostream &os, const Eigen::VectorXi &values)
{
    if (values.size() < 0 || values.size() > MAX_VECTOR_SIZE) {
        os.setstate(std::ios::failbit);
        return false;
    }
    auto size = static_cast<std::int32_t>(values.size());
    return write_value(os, size) && write_span(os, values.data(), static_cast<std::size_t>(values.size()));
}

inline bool read_vector(std::istream &is, Eigen::VectorXi &values)
{
    std::int32_t size = 0;
    if (!read_value(is, size)) return false;
    if (size < 0 || size > MAX_VECTOR_SIZE) return false;
    values.resize(size);
    return read_span(is, values.data(), static_cast<std::size_t>(values.size()));
}

inline bool write_named_string(std::ostream &os, const std::string &name, const std::string &value)
{
    return write_string(os, name) && write_string(os, value);
}

inline bool read_named_string(std::istream &is, std::string &name, std::string &value)
{
    return read_string(is, name) && read_string(is, value);
}

} // namespace gpio_binary
