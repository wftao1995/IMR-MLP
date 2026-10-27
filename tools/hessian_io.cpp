// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#include "tools/hessian_io.h"
#include "utils/compressed_stream.h"

#include <algorithm>
#include <cctype>
#include <fstream>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace {

std::string lower_copy(std::string value)
{
    std::transform(value.begin(), value.end(), value.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return value;
}

std::string read_text_file(const std::string &filename)
{
    auto input = open_ifile(filename);
    if (!input)
        throw std::runtime_error("Cannot open file: " + filename);
    std::ostringstream buffer;
    buffer << input->rdbuf();
    return buffer.str();
}

bool getline_nonempty(std::istream &input, std::string &line)
{
    while (std::getline(input, line)) {
        if (line.find_first_not_of(" \t\r\n") != std::string::npos)
            return true;
    }
    return false;
}

std::vector<double> parse_doubles(const std::string &line)
{
    std::istringstream iss(line);
    std::vector<double> values;
    double value = 0.0;
    while (iss >> value)
        values.push_back(value);
    return values;
}

int normalize_atom_index(int raw, int natoms)
{
    if (raw >= 1 && raw <= natoms)
        return raw - 1;
    if (raw >= 0 && raw < natoms)
        return raw;
    throw std::runtime_error("FORCE_CONSTANTS atom index out of range: " + std::to_string(raw));
}

Eigen::MatrixXd rows_to_hessian(const std::vector<std::vector<double>> &rows,
                                int natoms, const std::string &label)
{
    const int dim = 3 * natoms;
    Eigen::MatrixXd hessian(dim, dim);

    if (static_cast<int>(rows.size()) == dim) {
        bool rectangular = true;
        for (const auto &row : rows)
            rectangular = rectangular && static_cast<int>(row.size()) == dim;
        if (rectangular) {
            for (int i = 0; i < dim; ++i)
                for (int j = 0; j < dim; ++j)
                    hessian(i, j) = rows[i][j];
            return hessian;
        }
    }

    if (static_cast<int>(rows.size()) == natoms * natoms * 3) {
        bool block_rows = true;
        for (const auto &row : rows)
            block_rows = block_rows && row.size() == 3;
        if (block_rows) {
            int row_id = 0;
            for (int i = 0; i < natoms; ++i) {
                for (int j = 0; j < natoms; ++j) {
                    for (int a = 0; a < 3; ++a, ++row_id) {
                        for (int b = 0; b < 3; ++b)
                            hessian(3 * i + a, 3 * j + b) = rows[row_id][b];
                    }
                }
            }
            return hessian;
        }
    }

    if (static_cast<int>(rows.size()) == natoms * natoms) {
        bool flat_blocks = true;
        for (const auto &row : rows)
            flat_blocks = flat_blocks && row.size() == 9;
        if (flat_blocks) {
            int row_id = 0;
            for (int i = 0; i < natoms; ++i) {
                for (int j = 0; j < natoms; ++j, ++row_id) {
                    for (int a = 0; a < 3; ++a)
                        for (int b = 0; b < 3; ++b)
                            hessian(3 * i + a, 3 * j + b) = rows[row_id][3 * a + b];
                }
            }
            return hessian;
        }
    }

    std::vector<double> flat;
    for (const auto &row : rows)
        flat.insert(flat.end(), row.begin(), row.end());
    if (static_cast<int>(flat.size()) == dim * dim) {
        for (int i = 0; i < dim; ++i)
            for (int j = 0; j < dim; ++j)
                hessian(i, j) = flat[i * dim + j];
        return hessian;
    }

    throw std::runtime_error(label + " does not contain a Hessian shaped as 3N x 3N for N="
                             + std::to_string(natoms));
}

bool name_matches(const std::string &name, const std::string &key)
{
    const std::string lname = lower_copy(name);
    const std::string lkey = lower_copy(key);
    if (lkey != "auto")
        return lname == lkey;

    return lname == "hessian" || lname == "force_constants" || lname == "forceconstants"
           || lname == "force_constant" || lname == "forceconstant";
}

}

namespace HessianIO {

Eigen::MatrixXd read_force_constants(const std::string &filename, int natoms)
{
    auto input = open_ifile(filename);
    if (!input)
        throw std::runtime_error("Cannot open FORCE_CONSTANTS file: " + filename);

    std::string line;
    if (!getline_nonempty(*input, line))
        throw std::runtime_error("Empty FORCE_CONSTANTS file: " + filename);

    std::istringstream header(line);
    std::vector<int> dims;
    int value = 0;
    while (header >> value)
        dims.push_back(value);

    if (dims.size() != 1 && dims.size() != 2)
        throw std::runtime_error("Unsupported FORCE_CONSTANTS header: " + line);

    const int nprim = dims[0];
    const int nsupe = dims.size() == 1 ? dims[0] : dims[1];
    if (nprim != natoms || nsupe != natoms) {
        throw std::runtime_error("FORCE_CONSTANTS dimensions (" + std::to_string(nprim) + ", "
                                 + std::to_string(nsupe) + ") do not match structure atom count "
                                 + std::to_string(natoms));
    }

    Eigen::MatrixXd hessian = Eigen::MatrixXd::Zero(3 * natoms, 3 * natoms);
    std::vector<bool> seen(static_cast<size_t>(natoms * natoms), false);
    for (int block = 0; block < natoms * natoms; ++block) {
        if (!getline_nonempty(*input, line))
            throw std::runtime_error("Unexpected end of FORCE_CONSTANTS before block header");

        std::istringstream pair_line(line);
        int raw_i = 0;
        int raw_j = 0;
        if (!(pair_line >> raw_i >> raw_j))
            throw std::runtime_error("Invalid FORCE_CONSTANTS block header: " + line);

        const int i = normalize_atom_index(raw_i, natoms);
        const int j = normalize_atom_index(raw_j, natoms);
        const size_t pair_index = static_cast<size_t>(i * natoms + j);
        if (seen[pair_index])
            throw std::runtime_error("Duplicate FORCE_CONSTANTS block for atoms "
                                     + std::to_string(raw_i) + ", " + std::to_string(raw_j));
        seen[pair_index] = true;
        for (int a = 0; a < 3; ++a) {
            if (!getline_nonempty(*input, line))
                throw std::runtime_error("Unexpected end of FORCE_CONSTANTS inside block");
            std::istringstream row(line);
            for (int b = 0; b < 3; ++b) {
                if (!(row >> hessian(3 * i + a, 3 * j + b)))
                    throw std::runtime_error("Invalid FORCE_CONSTANTS matrix row: " + line);
            }
        }
    }
    if (getline_nonempty(*input, line))
        throw std::runtime_error("Unexpected trailing data in FORCE_CONSTANTS: " + line);
    return hessian;
}

Eigen::MatrixXd read_vasprun_hessian(const std::string &filename, int natoms,
                                     const std::string &key)
{
    const std::string text = read_text_file(filename);
    const std::regex varray_re("<varray\\b[^>]*\\bname\\s*=\\s*[\"']([^\"']+)[\"'][^>]*>([\\s\\S]*?)</varray>",
                               std::regex::icase);
    const std::regex row_re("<v\\b[^>]*>([^<]+)</v>", std::regex::icase);

    std::smatch match;
    auto begin = text.cbegin();
    while (std::regex_search(begin, text.cend(), match, varray_re)) {
        const std::string name = match[1].str();
        const std::string body = match[2].str();

        if (name_matches(name, key)) {
            std::vector<std::vector<double>> rows;
            auto row_begin = body.cbegin();
            std::smatch row_match;
            while (std::regex_search(row_begin, body.cend(), row_match, row_re)) {
                auto values = parse_doubles(row_match[1].str());
                if (!values.empty())
                    rows.push_back(std::move(values));
                row_begin = row_match.suffix().first;
            }
            if (rows.empty())
                rows.push_back(parse_doubles(body));
            return rows_to_hessian(rows, natoms, "vasprun.xml varray '" + name + "'");
        }

        begin = match.suffix().first;
    }

    if (lower_copy(key) == "auto" && lower_copy(text).find("name=\"dynmat\"") != std::string::npos) {
        throw std::runtime_error("vasprun.xml contains dynmat but no hessian/force_constants varray. "
                                 "dynmat may be mass-weighted; specify --vasprun-key only if it is a "
                                 "3N x 3N Hessian in eV/Angstrom^2.");
    }
    throw std::runtime_error("Cannot find requested Hessian varray in vasprun.xml: " + key);
}

}
