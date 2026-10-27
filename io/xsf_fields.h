#pragma once

#include <array>
#include <cstddef>
#include <iosfwd>
#include <string>
#include <vector>

// Independent static XSF DATAGRID objects. No geometry or Atoms state is stored.
// Values use logical C order (x,y[,z]); XSF files use x-fast order.
// Coordinates are Angstroms; scalar value units are caller defined.
struct XsfGridLayout {
    std::vector<std::size_t> shape;
    std::array<double, 3> origin{0, 0, 0};
    std::vector<std::array<double, 3>> spanning_vectors;

    std::size_t dimension() const { return shape.size(); }
    std::size_t size() const;
    void validate() const;
    // General grids retain both endpoints. A singleton axis contributes zero.
    std::array<double, 3> coordinate(const std::vector<std::size_t>& indices) const;
    bool operator==(const XsfGridLayout&) const = default;
};

struct XsfScalarField {
    std::string identifier;
    std::vector<double> values;
};

struct XsfDataGridBlock {
    std::string identifier;
    XsfGridLayout layout;
    std::vector<XsfScalarField> fields;
    void validate() const;
};

// Input can contain a static geometry prefix, which is discarded. AXSF and
// BXSF are unsupported. Streams need not support seeking. Errors throw.
std::vector<XsfDataGridBlock> read_xsf_fields(std::istream& input);
std::vector<XsfDataGridBlock> read_xsf_fields(const std::string& path);
// Validates all objects before emitting bytes. Geometry may be written to the
// same stream by GPIO first. Does not infer periodicity or scalar value units.
void write_xsf_fields(std::ostream& output, const std::vector<XsfDataGridBlock>& blocks);
void write_xsf_fields(const std::string& path, const std::vector<XsfDataGridBlock>& blocks);
