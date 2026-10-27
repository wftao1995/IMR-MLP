// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#include "neighborhoodKD.h"
#include "libalglib/alglibmisc.h"
#include "tools/tools.h"

NeighborhoodKD::NeighborhoodKD(Atoms *atom, double r)
{
    r_extent = r;
    ref = atom;
    init();
}

NeighborhoodKD::NeighborhoodKD(Atoms *atom)
{
    ref = atom;
    init_npbc();
}

void NeighborhoodKD::init(Atoms *atom, double r)
{
    r_extent = r;
    ref = atom;
    init();
}

void NeighborhoodKD::init_npbc()
{
    auto data = ref->pos_.data()->data();
    auto rows = ref->pos_.size();
    alglib::real_2d_array array2d;
    alglib::integer_1d_array tags;
    array2d.setcontent(rows, 3, data);
    tags.setlength(rows);
    for (int i = 0; i < rows; ++i)
        tags[i] = i;
    alglib::kdtreebuildtagged(array2d, tags, rows, 3, 0, 2, kdt);
}

void NeighborhoodKD::init()
{
    if (ref->is_nonperiodic()) {
        init_npbc();
        return;
    }
    auto pos = ref->pos_;
    const int size = pos.size();
    std::vector<int> indices_orig;
    Eigen::RowVector3d v2, v3, v4, v5, v6, v7, v8;
    v5 << 1, 1, 0;
    v6 << 1, 0, 1;
    v7 << 0, 1, 1;
    v8 << 1, 1, 1;
    v2 = ref->cell_.row(0);
    v3 = ref->cell_.row(1);
    v4 = ref->cell_.row(2);
    v5 = v2 + v3;
    v6 = v2 + v4;
    v7 = v3 + v4;
    v8 = v2 + v3 + v4;
    Eigen::RowVector3d min_pos;
    min_pos(0) = __min(__min(__min(__min(__min(__min(__min(0, v2(0)), v3(0)), v4(0)), v5(0)), v6(0)), v7(0)), v8(0));
    min_pos(1) = __min(__min(__min(__min(__min(__min(__min(0, v2(1)), v3(1)), v4(1)), v5(1)), v6(1)), v7(1)), v8(1));
    min_pos(2) = __min(__min(__min(__min(__min(__min(__min(0, v2(2)), v3(2)), v4(2)), v5(2)), v6(2)), v7(2)), v8(2));
    Eigen::RowVector3d max_pos;
    max_pos(0) = __max(__max(__max(__max(__max(__max(__max(0, v2(0)), v3(0)), v4(0)), v5(0)), v6(0)), v7(0)), v8(0));
    max_pos(1) = __max(__max(__max(__max(__max(__max(__max(0, v2(1)), v3(1)), v4(1)), v5(1)), v6(1)), v7(1)), v8(1));
    max_pos(2) = __max(__max(__max(__max(__max(__max(__max(0, v2(2)), v3(2)), v4(2)), v5(2)), v6(2)), v7(2)), v8(2));
    for (const auto &p : ref->pos_) {
        min_pos = min_pos.cwiseMin(p);
        max_pos = max_pos.cwiseMax(p);
    }

    bool add_new = true;
    for (int l = 1; add_new; ++l)
    {
        add_new = false;

        for (int i = -l; i <= l; ++i)
        {
            for (int j = -l; j <= l; ++j)
            {
                for (int k = -l; k <= l; ++k)
                {
                    if ((!ref->pbc_[0] && i != 0) ||
                        (!ref->pbc_[1] && j != 0) ||
                        (!ref->pbc_[2] && k != 0)) continue;
                    if (__max(__max(abs(i), abs(j)), abs(k)) == l)
                    {
                        for (int ind = 0; ind < size; ind++)
                        {
                            Eigen::RowVector3d newvec = pos[ind];
                            newvec += i * ref->cell_.row(0) +
                                      j * ref->cell_.row(1) +
                                      k * ref->cell_.row(2);

                            if ((newvec[0] < min_pos[0] - r_extent) ||
                                (newvec[1] < min_pos[1] - r_extent) ||
                                (newvec[2] < min_pos[2] - r_extent) ||
                                (newvec[0] > max_pos[0] + r_extent) ||
                                (newvec[1] > max_pos[1] + r_extent) ||
                                (newvec[2] > max_pos[2] + r_extent))
                                continue;
                            //@note:by wfao:确保newvec有近邻，个人认为没有必要计算这一步
                            // int jnd;
                            // for (jnd = 0;
                            //      (jnd < size) && (pos[jnd] - newvec).norm() > r_extent;
                            //      jnd++)
                            //     ;
                            // if (jnd < size)
                            // {
                            pos.emplace_back(newvec);
                            indices_orig.emplace_back(ind);
                            add_new = true;
                            //}
                        }
                    }
                }
            }
        }
    }
    //auto data = ref->pos_.data()->data();
    auto data = pos.data()->data();
    auto rows = pos.size();
    alglib::real_2d_array array2d;
    alglib::integer_1d_array tags;
    array2d.setcontent(rows, 3, data);
    tags.setlength(rows);
    for (int i = 0; i < size; ++i)
        tags[i] = i;
    for (int i = size; i < pos.size(); ++i)
        tags[i] = indices_orig[i - size];
    alglib::kdtreebuildtagged(array2d, tags, rows, 3, 0, 2, kdt);
}

void NeighborhoodKD::reinit(Atoms *atom, double r)
{
    ref = atom;
    r_extent = r;
    init();
}

Neighborhood NeighborhoodKD::queryts(double *r, const double &rcut, const int &type, tslocalbuffer &buf, bool self) {
    alglib::real_1d_array pos;
    pos.setcontent(3, r);
    int nfind = alglib::kdtreetsqueryrnn(kdt, buf.buf, pos, rcut, self);
    Neighborhood nbh;
    nbh.centertype = type;
    nbh.nbhcount_ = nfind;
    if (nfind == 0) {
        return nbh;
    }
    alglib::kdtreetsqueryresultstags(kdt, buf.buf, buf.tags_query_buffer);
    alglib::kdtreetsqueryresultsdistances(kdt, buf.buf, buf.r_query_buffer);
    alglib::kdtreetsqueryresultsx(kdt, buf.buf, buf.x_query_buffer);
    nbh.indices_.resize(nfind);
    nbh.chemmap_.resize(nfind);
    nbh.distances_.resize(nfind);
    nbh.pos_.resize(nfind);

    for (int i = 0; i < nfind; ++i)
    {
        nbh.indices_[i] = buf.tags_query_buffer[i];
        nbh.chemmap_[i] = ref->chemmap_[buf.tags_query_buffer[i]];
        nbh.pos_[i][0] = buf.x_query_buffer[i][0] - *r;
        nbh.pos_[i][1] = buf.x_query_buffer[i][1] - *(r+1);
        nbh.pos_[i][2] = buf.x_query_buffer[i][2] - *(r+2);
        nbh.distances_[i] = buf.r_query_buffer[i];
    }
    return nbh;
}

Neighborhood NeighborhoodKD::queryuts(double *r, const double &rcut, const int &type, tslocalbuffer &buf, bool self) {
    alglib::real_1d_array pos;
    pos.setcontent(3, r);
    int nfind = alglib::kdtreetsqueryrnnu(kdt, buf.buf, pos, rcut, self);
    Neighborhood nbh;
    nbh.centertype = type;
    nbh.nbhcount_ = nfind;
    if (nfind == 0) {
        return nbh;
    }
    alglib::kdtreetsqueryresultstags(kdt, buf.buf, buf.tags_query_buffer);
    alglib::kdtreetsqueryresultsdistances(kdt, buf.buf, buf.r_query_buffer);
    alglib::kdtreetsqueryresultsx(kdt, buf.buf, buf.x_query_buffer);
    nbh.indices_.resize(nfind);
    nbh.chemmap_.resize(nfind);
    nbh.distances_.resize(nfind);
    nbh.pos_.resize(nfind);

    for (int i = 0; i < nfind; ++i)
    {
        nbh.indices_[i] = buf.tags_query_buffer[i];
        nbh.chemmap_[i] = ref->chemmap_[buf.tags_query_buffer[i]];
        nbh.pos_[i][0] = buf.x_query_buffer[i][0] - *r;
        nbh.pos_[i][1] = buf.x_query_buffer[i][1] - *(r+1);
        nbh.pos_[i][2] = buf.x_query_buffer[i][2] - *(r+2);
        nbh.distances_[i] = buf.r_query_buffer[i];
    }
    return nbh;
}

Neighborhood NeighborhoodKD::query(int index, const double &rcut)
{
    return query(ref->pos_[index].data(), rcut, ref->chemmap_[index]);
}

Neighborhood NeighborhoodKD::queryu(int index, const double &rcut)
{
    return queryu(ref->pos_[index].data(), rcut, ref->chemmap_[index]);
}

Neighborhood NeighborhoodKD::query(int index, const double &rcut, int knn) {
    alglib::real_1d_array pos;
    pos.setcontent(3, ref->pos_[index].data());
    int nfind = alglib::kdtreequeryrnn(kdt, pos, rcut, false);
    int nres = nfind;
    if (nfind > knn) {
        nres = knn;
    }
    Neighborhood nbh;
    nbh.centertype = ref->chemmap_[index];
    nbh.nbhcount_ = nres;
    if (nfind == 0) {
        return nbh;
    }
    alglib::kdtreequeryresultstags(kdt, tags_query_buffer);
    alglib::kdtreequeryresultsdistances(kdt, r_query_buffer);
    alglib::kdtreequeryresultsx(kdt, x_query_buffer);
    nbh.indices_.resize(nres);
    nbh.chemmap_.resize(nres);
    nbh.distances_.resize(nres);
    nbh.pos_.resize(nres);

    for (int i = 0; i < nres; ++i)
    {
        nbh.indices_[i] = tags_query_buffer[i];
        nbh.chemmap_[i] = ref->chemmap_[tags_query_buffer[i]];
        nbh.pos_[i][0] = x_query_buffer[i][0] - pos[0];
        nbh.pos_[i][1] = x_query_buffer[i][1] - pos[1];
        nbh.pos_[i][2] = x_query_buffer[i][2] - pos[2];
        nbh.distances_[i] = r_query_buffer[i];
    }
    return nbh;
}

Neighborhood NeighborhoodKD::query(double *r, const double &rcut, const int &type, bool self) {
    alglib::real_1d_array pos;
    pos.setcontent(3, r);
    int nfind = alglib::kdtreequeryrnn(kdt, pos, rcut, self);
    Neighborhood nbh;
    nbh.centertype = type;
    nbh.nbhcount_ = nfind;
    if (nfind == 0) {
        return nbh;
    }
    alglib::kdtreequeryresultstags(kdt, tags_query_buffer);
    alglib::kdtreequeryresultsdistances(kdt, r_query_buffer);
    alglib::kdtreequeryresultsx(kdt, x_query_buffer);
    nbh.indices_.resize(nfind);
    nbh.chemmap_.resize(nfind);
    nbh.distances_.resize(nfind);
    nbh.pos_.resize(nfind);

    for (int i = 0; i < nfind; ++i)
    {
        nbh.indices_[i] = tags_query_buffer[i];
        nbh.chemmap_[i] = ref->chemmap_[tags_query_buffer[i]];
        nbh.pos_[i][0] = x_query_buffer[i][0] - *r;
        nbh.pos_[i][1] = x_query_buffer[i][1] - *(r+1);
        nbh.pos_[i][2] = x_query_buffer[i][2] - *(r+2);
        nbh.distances_[i] = r_query_buffer[i];
    }
    return nbh;
}

Neighborhood NeighborhoodKD::queryu(double *r, const double &rcut, const int &type, bool self) {
    alglib::real_1d_array pos;
    pos.setcontent(3, r);
    int nfind = alglib::kdtreequeryrnnu(kdt, pos, rcut, self);
    Neighborhood nbh;
    nbh.centertype = type;
    nbh.nbhcount_ = nfind;
    if (nfind == 0) {
        return nbh;
    }
    alglib::kdtreequeryresultstags(kdt, tags_query_buffer);
    alglib::kdtreequeryresultsdistances(kdt, r_query_buffer);
    alglib::kdtreequeryresultsx(kdt, x_query_buffer);
    nbh.indices_.resize(nfind);
    nbh.chemmap_.resize(nfind);
    nbh.distances_.resize(nfind);
    nbh.pos_.resize(nfind);

    for (int i = 0; i < nfind; ++i)
    {
        nbh.indices_[i] = tags_query_buffer[i];
        nbh.chemmap_[i] = ref->chemmap_[tags_query_buffer[i]];
        nbh.pos_[i][0] = x_query_buffer[i][0] - *r;
        nbh.pos_[i][1] = x_query_buffer[i][1] - *(r+1);
        nbh.pos_[i][2] = x_query_buffer[i][2] - *(r+2);
        nbh.distances_[i] = r_query_buffer[i];
    }
    return nbh;
}

Neighborhood NeighborhoodKD::query(int index, int knn) {
    return query(ref->pos_[index].data(), knn, ref->chemmap_[index]);
}

Neighborhood NeighborhoodKD::query(int index, int knn, const double &rcut) {
    alglib::real_1d_array pos;
    pos.setcontent(3, ref->pos_[index].data());
    int nfind = alglib::kdtreequeryknn(kdt, pos, knn, false);
    Neighborhood nbh;
    nbh.centertype = ref->chemmap_[index];
    nbh.nbhcount_ = 0;
    if (nfind == 0) {
        return nbh;
    }
    alglib::kdtreequeryresultstags(kdt, tags_query_buffer);
    alglib::kdtreequeryresultsdistances(kdt, r_query_buffer);
    alglib::kdtreequeryresultsx(kdt, x_query_buffer);
    nbh.indices_.reserve(nfind);
    nbh.chemmap_.reserve(nfind);
    nbh.distances_.reserve(nfind);
    nbh.pos_.reserve(nfind);
    Eigen::RowVector3d tmprow;
    for (int i = 0; i < nfind; ++i) {
        double dist = r_query_buffer[i];
        if (dist < rcut) {
            tmprow[0] = x_query_buffer[i][0] - pos[0];
            tmprow[1] = x_query_buffer[i][1] - pos[1];
            tmprow[2] = x_query_buffer[i][2] - pos[2];
            nbh.indices_.push_back(tags_query_buffer[i]);
            nbh.pos_.push_back(tmprow);
            nbh.chemmap_.push_back(ref->chemmap_[tags_query_buffer[i]]);
            nbh.distances_.push_back(dist);
            nbh.nbhcount_++;
        }
    }
    return nbh;
}

Neighborhood NeighborhoodKD::query_type(int index, const double &rcut, int jtype) {
    alglib::real_1d_array pos;
    pos.setcontent(3, ref->pos_[index].data());
    int nfind = alglib::kdtreequeryrnn(kdt, pos, rcut, false);
    Neighborhood nbh;
    nbh.centertype = ref->chemmap_[index];
    nbh.nbhcount_ = 0;
    if (nfind == 0) {
        return nbh;
    }
    alglib::kdtreequeryresultstags(kdt, tags_query_buffer);
    alglib::kdtreequeryresultsdistances(kdt, r_query_buffer);
    alglib::kdtreequeryresultsx(kdt, x_query_buffer);
    nbh.indices_.reserve(nfind);
    nbh.chemmap_.reserve(nfind);
    nbh.distances_.reserve(nfind);
    nbh.pos_.reserve(nfind);
    Eigen::RowVector3d tmprow;
    for (int i = 0; i < nfind; ++i) {
        if (ref->chemmap_[tags_query_buffer[i]] == jtype) {
            double dist = r_query_buffer[i];
            tmprow[0] = x_query_buffer[i][0] - pos[0];
            tmprow[1] = x_query_buffer[i][1] - pos[1];
            tmprow[2] = x_query_buffer[i][2] - pos[2];
            nbh.indices_.push_back(tags_query_buffer[i]);
            nbh.pos_.push_back(tmprow);
            nbh.chemmap_.push_back(ref->chemmap_[tags_query_buffer[i]]);
            nbh.distances_.push_back(dist);
            nbh.nbhcount_++;
        }
    }
    return nbh;
}
// 
Neighborhood NeighborhoodKD::query(double *r, const int &knn, const int &type, bool self)
{
    alglib::real_1d_array pos;
    pos.setcontent(3, r);
    int nfind = alglib::kdtreequeryknn(kdt, pos, knn, self);
    Neighborhood nbh;
    nbh.centertype = type;
    nbh.nbhcount_ = nfind;
    if (nfind == 0)
    {
        return nbh;
    }
    alglib::kdtreequeryresultstags(kdt, tags_query_buffer);
    alglib::kdtreequeryresultsdistances(kdt, r_query_buffer);
    alglib::kdtreequeryresultsx(kdt, x_query_buffer);
    nbh.indices_.resize(nfind);
    nbh.chemmap_.resize(nfind);
    nbh.distances_.resize(nfind);
    nbh.pos_.resize(nfind);

    for (int i = 0; i < nfind; ++i)
    {
        nbh.indices_[i] = tags_query_buffer[i];
        nbh.chemmap_[i] = ref->chemmap_[tags_query_buffer[i]];
        nbh.pos_[i][0] = x_query_buffer[i][0] - *r;
        nbh.pos_[i][1] = x_query_buffer[i][1] - *(r+1);
        nbh.pos_[i][2] = x_query_buffer[i][2] - *(r+2);
        nbh.distances_[i] = r_query_buffer[i];
    }
    return nbh;
}

Neighborhood NeighborhoodKD::query(int index, const double &rcut_inn, const double &rcut)
{
    return query(ref->pos_[index].data(), rcut_inn, rcut, ref->chemmap_[index]);
}

Neighborhood NeighborhoodKD::queryu(int index, const double &rcut_inn, const double &rcut)
{
    return queryu(ref->pos_[index].data(), rcut_inn, rcut, ref->chemmap_[index]);
}

Neighborhood NeighborhoodKD::query(double *r, const double &rcut_inn, const double &rcut, const int &type, bool self)
{
    Neighborhood nbh;
    alglib::real_1d_array pos;
    pos.setcontent(3, r);
    int nfind = alglib::kdtreequeryrnn(kdt, pos, rcut, self);
    nbh.centertype = type;
    nbh.nbhcount_ = 0;
    if (nfind == 0)
    {
        return nbh;
    }
    alglib::kdtreequeryresultstags(kdt, tags_query_buffer);
    alglib::kdtreequeryresultsdistances(kdt, r_query_buffer);
    alglib::kdtreequeryresultsx(kdt, x_query_buffer);
    nbh.indices_.reserve(nfind / 3);
    nbh.chemmap_.reserve(nfind / 3);
    nbh.distances_.reserve(nfind / 3);
    nbh.pos_.reserve(nfind / 3);

    for (int i = 0; i < nfind; ++i)
    {
        double dist = r_query_buffer[i];
        if (dist > rcut_inn)
        {
            nbh.nbhcount_++;
            nbh.indices_.push_back(tags_query_buffer[i]);
            nbh.chemmap_.push_back(ref->chemmap_[tags_query_buffer[i]]);
            nbh.pos_.emplace_back(x_query_buffer[i][0] - *r,
                                  x_query_buffer[i][1] - *(r+1),
                                  x_query_buffer[i][2] - *(r+2));
            nbh.distances_.push_back(r_query_buffer[i]);
        }
    }
    return nbh;
}

Neighborhood NeighborhoodKD::queryu(double *r, const double &rcut_inn, const double &rcut, const int &type, bool self)
{
    Neighborhood nbh;
    alglib::real_1d_array pos;
    pos.setcontent(3, r);
    int nfind = alglib::kdtreequeryrnnu(kdt, pos, rcut, self);
    nbh.centertype = type;
    nbh.nbhcount_ = 0;
    if (nfind == 0)
    {
        return nbh;
    }
    alglib::kdtreequeryresultstags(kdt, tags_query_buffer);
    alglib::kdtreequeryresultsdistances(kdt, r_query_buffer);
    alglib::kdtreequeryresultsx(kdt, x_query_buffer);
    nbh.indices_.reserve(nfind / 3);
    nbh.chemmap_.reserve(nfind / 3);
    nbh.distances_.reserve(nfind / 3);
    nbh.pos_.reserve(nfind / 3);

    for (int i = 0; i < nfind; ++i)
    {
        double dist = r_query_buffer[i];
        if (dist > rcut_inn)
        {
            nbh.nbhcount_++;
            nbh.indices_.push_back(tags_query_buffer[i]);
            nbh.chemmap_.push_back(ref->chemmap_[tags_query_buffer[i]]);
            nbh.pos_.emplace_back(x_query_buffer[i][0] - *r,
                                  x_query_buffer[i][1] - *(r+1),
                                  x_query_buffer[i][2] - *(r+2));
            nbh.distances_.push_back(r_query_buffer[i]);
        }
    }
    return nbh;
}

Neighborhood NeighborhoodKD::querybox(double *rlo, double *rhi) {
    alglib::real_1d_array lo, hi;
    lo.setcontent(3, rlo);
    hi.setcontent(3, rhi);
    int nfind = alglib::kdtreequerybox(kdt, lo, hi);
    Neighborhood nbh;
    nbh.centertype = 0;
    nbh.nbhcount_ = nfind;
    if (nfind == 0) {
        return nbh;
    }
    alglib::kdtreequeryresultstags(kdt, tags_query_buffer);
    alglib::kdtreequeryresultsx(kdt, x_query_buffer);
    nbh.indices_.resize(nfind);
    nbh.chemmap_.resize(nfind);
    // nbh.distances_.resize(nfind);
    // nbh.pos_.resize(nfind);

    for (int i = 0; i < nfind; ++i)
    {
        nbh.indices_[i] = tags_query_buffer[i];
        nbh.chemmap_[i] = ref->chemmap_[tags_query_buffer[i]];
        // nbh.pos_[i][0] = x_query_buffer[i][0] - *r;
        // nbh.pos_[i][1] = x_query_buffer[i][1] - *(r+1);
        // nbh.pos_[i][2] = x_query_buffer[i][2] - *(r+2);
        // nbh.distances_[i] = r_query_buffer[i];
    }
    return nbh;
}

int NeighborhoodKD::find_atom(double* r, const double tol)
{
    alglib::real_1d_array pos;
    pos.setcontent(3, r);
    int nfind = alglib::kdtreequeryrnnu(kdt, pos, tol, true);
    if (nfind == 0) return -1;
    alglib::kdtreequeryresultstags(kdt, tags_query_buffer);
    return tags_query_buffer[0];
}
