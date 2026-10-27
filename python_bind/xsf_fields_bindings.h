#pragma once

#include "io/xsf_fields.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <sstream>

inline void bind_xsf_fields(pybind11::module_& m) {
    namespace py = pybind11;
    py::class_<XsfGridLayout>(m, "_XsfGridLayout")
        .def(py::init<>())
        .def_readwrite("shape", &XsfGridLayout::shape)
        .def_readwrite("origin", &XsfGridLayout::origin)
        .def_readwrite("spanning_vectors", &XsfGridLayout::spanning_vectors)
        .def_property_readonly("dimension", &XsfGridLayout::dimension)
        .def("validate", &XsfGridLayout::validate)
        .def("coordinate", &XsfGridLayout::coordinate);
    py::class_<XsfScalarField>(m, "_XsfScalarField")
        .def(py::init<>())
        .def_readwrite("identifier", &XsfScalarField::identifier)
        .def_readwrite("values", &XsfScalarField::values);
    py::class_<XsfDataGridBlock>(m, "_XsfDataGridBlock")
        .def(py::init<>())
        .def_readwrite("identifier", &XsfDataGridBlock::identifier)
        .def_readwrite("layout", &XsfDataGridBlock::layout)
        .def_readwrite("fields", &XsfDataGridBlock::fields)
        .def("validate", &XsfDataGridBlock::validate);
    m.def("_read_xsf_fields", [](const std::string& path) {
        py::gil_scoped_release release;
        return read_xsf_fields(path);
    });
    m.def("_write_xsf_fields", [](const std::string& path, const std::vector<XsfDataGridBlock>& blocks) {
        py::gil_scoped_release release;
        write_xsf_fields(path, blocks);
    });
    m.def("_parse_xsf_fields", [](const std::string& text) {
        std::istringstream input(text);
        py::gil_scoped_release release;
        return read_xsf_fields(input);
    });
    m.def("_format_xsf_fields", [](const std::vector<XsfDataGridBlock>& blocks) {
        std::ostringstream output;
        {
            py::gil_scoped_release release;
            write_xsf_fields(output, blocks);
        }
        return output.str();
    });
}
