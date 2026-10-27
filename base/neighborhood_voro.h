// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#pragma once

#include "anyneighborhoods.h"
#include "structures.h"
#include <voro++/voro++.hh>
#include <string>
#define VORO_INIT_MEM 14
#define TOL 10e-10
#include "tools/tools.h"
#include "tools/procstr.h"
#include <iomanip>
//#define HASHED_VORO
#include "voro_index_codec.h"
#if UINTPTR_MAX == UINT32_MAX
#define HASH_VORO_PACK 4
#define HASH_VORO_MASK 0xf
#define HASH_VORO_UPTO_MOST 10
#else
#define HASHED_VORO_PACK_5
#define HASH_VORO_PACK 5
#define HASH_VORO_MASK 0x1f
#define HASH_VORO_UPTO_MOST 12
#endif

#ifdef HASHED_VORO
using voro_hash=VoroIndex::NativeUInt;
using voro_index=VoroIndex::Native;
struct voro_index_bond : voro_index {
    int type;
    voro_index_bond(std::vector<int> histogram,int value)
        : voro_index(histogram),type(value) {}
    bool operator<(const voro_index_bond& other)const {
        if(voro_index::operator<(other))return true;
        if(other.voro_index::operator<(*this))return false;
        return type<other.type;
    }
};
#else  // HASHED_VORO
struct voro_index
{
    std::vector<int> index;

    voro_index(std::vector<int> vec) : index(vec)
    {
    }

    bool operator<(const voro_index &opt) const
    {
        for (int i = 0; i < index.size(); ++i)
        {
            if (index[i] < opt.index[i])
                return true;
            else if (index[i] > opt.index[i])
                return false;
        }
        return false;
    }

    std::string tostring(int a) const
    {
        std::string retstr = "0";
        for (int i = 2; i < index.size(); ++i)
        {
            retstr +=(std::string)"-" + std::to_string(index[i]);
        }
        return retstr;
    }
};
struct voro_index_bond
{
    std::vector<int> index;
    int type;

    voro_index_bond(std::vector<int> vec, int _type) : index(vec), type(_type)
    {
    }

    bool operator<(const voro_index_bond &opt) const
    {
        for (int i = 0; i < index.size(); ++i)
        {
            if (index[i] < opt.index[i])
                return true;
            else if (index[i] > opt.index[i])
                return false;
        }
        if (type < opt.type)
            return true;
        return false;
    }

    std::string tostring(int a) const
    {
        std::string retstr = "0";
        for (int i = 2; i < index.size(); ++i)
        {
            retstr +=(std::string)"-" + std::to_string(index[i]);
        }
        return retstr;
    }
};
#endif // HASHED_VORO
typedef voro_index voro_index_type;
typedef voro_index_bond voro_index_bond_type;
typedef std::map<voro_index_type, int> vorotype;
typedef std::map<voro_index_bond_type, int> vorotype_bond;
struct VoroInfo
{
    vorotype tessencount_all;
    vorotype_bond tessencount_partial;
    size_t total_count = 0;
#ifndef NO_SAMPLE_STR
    std::map<voro_index_type, std::vector<Eigen::RowVector3d>> str;
#endif // NO_SAMPLE_STR
    std::map<voro_index_type, double> meanvol;

    void operator+=(VoroInfo &opt)
    {
        for (auto &it : opt.tessencount_all)
        {
            tessencount_all[it.first] += it.second;
        }
        for (auto &it : opt.tessencount_partial)
        {
            tessencount_partial[it.first] += it.second;
        }
#ifndef NO_SAMPLE_STR
        str.insert(opt.str.begin(), opt.str.end());
#endif
        for (auto &it : opt.meanvol)
        {
            meanvol[it.first] += it.second;
        }
        total_count += opt.total_count;
    }
    // sort and print
    void outall(std::ostream &ofs, int edges)
    {
        std::vector<std::pair<voro_index_type, int>> temp_sort;
        for (auto &it : tessencount_all)
        {
            temp_sort.emplace_back(it);
        }
        std::sort(temp_sort.begin(), temp_sort.end(),
                  [](std::pair<voro_index_type, int> &a, std::pair<voro_index_type, int> &b) { return a.second > b.second; });
        for (auto &it : temp_sort)
        {
            ofs << it.first.tostring(edges) << std::setw(15) << it.second
                << std::setw(15) << static_cast<double>(it.second) / static_cast<double>(total_count) << '\n';
        }
    }
    // sort and print, partial
    void outpart(std::ostream &ofs, int type, int edges)
    {
        std::vector<std::pair<voro_index_bond_type, int>> temp_sort;
        for (auto &it : tessencount_partial)
        {
            if (it.first.type == type)
                temp_sort.emplace_back(it);
        }
        std::sort(temp_sort.begin(), temp_sort.end(),
                  [](std::pair<voro_index_bond_type, int> &a, std::pair<voro_index_bond_type, int> &b) { return a.second > b.second; });
        for (auto &it : temp_sort)
        {
            ofs << it.first.tostring(edges) << std::setw(15) << it.second
                << std::setw(15) << static_cast<double>(it.second) / static_cast<double>(total_count) << '\n';
        }
    }
    // all, select
    void outallcomm(std::ostream &ofs, int edges, int num)
    {
        std::vector<std::pair<voro_index_type, int>> temp_sort;
        for (auto &it : tessencount_all)
        {
            temp_sort.emplace_back(it);
        }
        num = __min(num, temp_sort.size());
        size_t num_present = 0;
        std::sort(temp_sort.begin(), temp_sort.end(),
                  [](std::pair<voro_index_type, int> &a, std::pair<voro_index_type, int> &b) { return a.second > b.second; });
        for (int i = 0; i < num; ++i)
        {
            ofs << temp_sort[i].first.tostring(edges) << std::setw(15) << temp_sort[i].second
                << std::setw(15) << static_cast<double>(temp_sort[i].second) / static_cast<double>(total_count) << '\n';
            num_present += temp_sort[i].second;
        }
        ofs << "ALL_PRESENT" << std::setw(15) << num_present << std::setw(15) << static_cast<double>(num_present) / static_cast<double>(total_count) << '\n';
    }
    // sort and print, partial
    void outpartcomm(std::ostream &ofs, int type, int edges, int num)
    {
        std::vector<std::pair<voro_index_bond_type, int>> temp_sort;
        for (auto &it : tessencount_partial)
        {
            if (it.first.type == type)
                temp_sort.emplace_back(it);
        }
        num = __min(num, temp_sort.size());
        size_t num_present = 0;
        std::sort(temp_sort.begin(), temp_sort.end(),
                  [](std::pair<voro_index_bond_type, int> &a, std::pair<voro_index_bond_type, int> &b) { return a.second > b.second; });
        for (int i = 0; i < num; ++i)
        {
            ofs << temp_sort[i].first.tostring(edges) << std::setw(15) << temp_sort[i].second
                << std::setw(15) << static_cast<double>(temp_sort[i].second) / static_cast<double>(total_count) << '\n';
            num_present += temp_sort[i].second;
        }
        ofs << "ALL_PRESENT" << std::setw(15) << num_present << std::setw(15) << static_cast<double>(num_present) / static_cast<double>(total_count) << '\n';
    }
    // output volume
    void outvol(std::ostream &ofs, int edges)
    {
        for (auto &it : meanvol)
        {
            ofs << it.first.tostring(edges) << std::setw(15) << it.second / tessencount_all[it.first] << '\n';
        }
    }
};

class NeighborhoodsVoro : public AnyNeighborhoods
{
private:
    ;

protected:
    std::vector<int> index;
    int voro_edge_upto_; //count 3,4,5,...,voro_edge_upto_

public:
    vorotype tessencount_;
    vorotype_bond tessencount_type_;
    std::map<voro_index_type, double> vorovol_;
    std::vector<double> voro_volume_each_; // each local environment's Voronoi volume
#ifndef NO_SAMPLE_STR
    std::map<voro_index_type, std::vector<Eigen::RowVector3d>> str_;
#endif

#define N_V_ZERO 1.e-10
    void RefineEdge(int num_faces, voro::voronoicell_neighbor *cell, std::vector<int> &index, double edgetol)
    {
        std::vector<double> vpos;
        std::vector<int> vlst;
        cell->vertices(vpos);
        cell->face_vertices(vlst);

        double edge_min = fabs(edgetol);
        bool flag_edge_ratio = false; // True: use ratio, false: use absolute value
        if (edgetol < 0.)
            flag_edge_ratio = true;

        int nv = vlst.size();
        double perim = 0., edges[nv];
        int k = 0, is = 0, ie = 0;
        // What's this??
        while (k < nv)
        {
            int ned = vlst[k++];
            for (int ii = 0; ii < ned; ++ii)
            {
                int jj = (ii + 1) % ned;
                int v1 = vlst[k + ii], v2 = vlst[k + jj];
                double dx = vpos[v1 * 3] - vpos[v2 * 3];
                double dy = vpos[v1 * 3 + 1] - vpos[v2 * 3 + 1];
                double dz = vpos[v1 * 3 + 2] - vpos[v2 * 3 + 2];
                double r = sqrt(dx * dx + dy * dy + dz * dz);
                edges[ie++] = r;
                perim += r;
            }
            ++is;
            k += ned;
        }
        int ford[num_faces];
        k = is = ie = 0;
        double inv_perim = 2. / perim;
        double ed_scale = 1.;
        if (flag_edge_ratio)
            ed_scale = inv_perim;

        while (k < nv)
        {
            int ned = vlst[k++];

            int nuc = 0;
            for (int ii = 0; ii < ned; ++ii)
            {
                // double rwt = edges[ie] * inv_perim;
                // if (fpedge)
                //     fprintf(fpedge, "%lg %g\n", rwt, edges[ie]);
                if ((edges[ie] * ed_scale) <= edge_min)
                    ++nuc;
                ++ie;
            }
            ford[is++] = ned - nuc;
            k += ned;
        }
        if (edge_min > N_V_ZERO)
        {
            for (int i = 3; i < index.size(); ++i)
                index[i] = 0;
            for (int i = 0; i < num_faces; ++i)
            {
                if (ford[i] < index.size())
                    index[ford[i]] += 1;
            }
        }

        return;
    }
#undef N_V_ZERO

    NeighborhoodsVoro(){};
    ~NeighborhoodsVoro(){};
};
