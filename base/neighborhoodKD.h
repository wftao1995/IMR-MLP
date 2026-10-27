// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#pragma once

#include "libalglib/alglibmisc.h"
#include "libalglib/ap.h"
#include "structures.h"
#include "anyneighborhoods.h"

class NeighborhoodKD
{
private:
    Atoms *ref;
    alglib::kdtree kdt;
    alglib::integer_1d_array tags_query_buffer;
    alglib::real_1d_array r_query_buffer;
    alglib::real_2d_array x_query_buffer;
    double r_extent;

public:
    NeighborhoodKD(){};
    NeighborhoodKD(Atoms *atom, double r);
    NeighborhoodKD(Atoms *atom);
    void init(Atoms *atom, double r);
    void init();
    void init_npbc();
    void reinit(Atoms *atom, double r);

    // For threadsafty
    struct tslocalbuffer {
        alglib::kdtreerequestbuffer buf;
        alglib::integer_1d_array tags_query_buffer;
        alglib::real_1d_array r_query_buffer;
        alglib::real_2d_array x_query_buffer;
    };
    void create_buffer(tslocalbuffer &buf) {
        alglib::kdtreecreaterequestbuffer(kdt, buf.buf);
    }
    Neighborhood queryts(double *r, const double &rcut, const int &type, tslocalbuffer &buf, bool self = false);
    Neighborhood queryuts(double *r, const double &rcut, const int &type, tslocalbuffer &buf, bool self = false);

    // u version are unsorted, but much faster version  

    Neighborhood query(int index, const double &rcut);
    Neighborhood queryu(int index, const double &rcut);

    Neighborhood query(double *r, const double &rcut, const int &type, bool self = false);
    Neighborhood queryu(double *r, const double &rcut, const int &type, bool self = false);

    Neighborhood query(int index, const double &rcut_inn, const double &rcut);
    Neighborhood queryu(int index, const double &rcut_inn, const double &rcut);
    
    Neighborhood query(double *r, const double &rcut_inn, const double &rcut, const int &type, bool self = false);
    Neighborhood queryu(double *r, const double &rcut_inn, const double &rcut, const int &type, bool self = false);

    Neighborhood query(int index, int knn); // do a knn query
    Neighborhood query(int index, int knn, const double &rcut); // do a knn query within specific cutoff
    Neighborhood query_type(int index, const double &rcut, int jtype);
    Neighborhood query(int index, const double &rcut, int knn); // do a distance query within specific knn
    Neighborhood query(double *r, const int &knn, const int &type, bool self = false);

    Neighborhood querybox(double *rlo, double *rhi);
    // check if there exists any atom at position r, return atom id, -1 means none
    int find_atom(double *r, const double tol);
    // find nearest atom at position r with radius tol, return atom id, -1 means find none, 
    // int find_nearest(double *r, const double tol); // same as find_atom

    inline int size() const { return ref->size(); };
};
