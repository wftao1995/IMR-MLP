// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#include "io/gpio.h"
#include "symmetry/structure_comparison.h"
#include <map>
#include <cstdio>
using namespace Eigen;
using namespace std;

int remove_duplicate_structures(int argc, char *argv[])
{
    auto &cmdopts = CMDOption::global_cmdopts;
    string inputfile, outputfile = "output-dedup.xyz";
    string outputid = "output-nodup.id";
    string idfile = "index.map";
    GPIO iobj;
    Atoms atom;
    vector<Atoms> atoms;
    bool overwrite = false;
    double symprec = 1e-4;
    double volprec = 1e-3;
    double pos_tol = 0.1;       // separate tolerance for atomic position matching
    bool preserve = true;
    bool allow_reflections = true;
    bool fastalgo = true;
    double desc_cmp_factor = 1.0;
    string informat, outformat;

    cmdopts.addop(allow_reflections, "allow-reflections", "", "include mirror images, default true")
        .addop(idfile, "index", "id", "index file")
        .addop(symprec, "sym-prec", "sprec", "symmetry tolerance (cell/spg)")
        .addop(volprec, "vol-prec", "vprec", "tolerance for volume comparison")
        .addop(pos_tol, "pos-tol", "ptol", "tolerance for atomic position matching (A), default 0.1")
        .addop(preserve, "preserve", "p", "preserve original cell, default true")
        .addop(overwrite, "overwrite", "ov", "overwrite existing output file", false, 0)
        .addop(informat, "input-format", "if")
        .addop(outformat, "output-format", "of")
        .addop(fastalgo, "fastalgo")
        .addop(desc_cmp_factor, "desc-factor", "", "descriptor factor for fastalgo, typical 10.0")
        .addop(inputfile, "input", "in", "input structure file", true)
        .addop(outputfile, "output", "out", "output file", true)
        .add_request("input")
        .phrase_options(argc, argv);

    if (GPIO::file_exist(outputfile) && !overwrite) {
        cerr << "Error: output file " << outputfile << " exists, use --overwrite to overwrite" << endl;
        exit(128);
    }

    if (!GPIO::file_exist(idfile))
        iobj.setmap();
    else
        iobj.readmap(idfile);

    if (!iobj.read(atoms, inputfile, informat)) {
        cerr << "Failed to read input file" << endl;
        exit(128);
    }
    cout << atoms.size() << " structures read" << endl;

    Symmetry3D::StructureComparisonOptions options;
    options.symmetry_tolerance = symprec;
    options.position_tolerance = pos_tol;
    options.volume_tolerance = volprec;
    options.use_descriptors = fastalgo;
    options.descriptor_tolerance = symprec;
    options.anchor_descriptor_tolerance = desc_cmp_factor * symprec;
    options.allow_reflections = allow_reflections;
    Symmetry3D::StructureComparator comparator(options);
    vector<Symmetry3D::ComparableStructure> prepared;
    prepared.reserve(atoms.size());
    map<map<int, int>, vector<int>> groups;
    for (int i = 0; i < atoms.size(); ++i) {
        prepared.push_back(comparator.prepare(atoms[i]));
        prepared.back().atoms.id = i;
        prepared.back().atoms.global_attribs_i_["oid"] = i;
        groups[prepared.back().composition].push_back(i);
    }
    // Group by composition only. Quantizing descriptors into hash buckets can
    // split equivalent structures at bucket boundaries; compare cached values instead.
    vector<vector<int>> atom_by_group;
    for (auto &[composition, indices] : groups) atom_by_group.push_back(std::move(indices));
    cout << "Preprocess finished, " << atom_by_group.size() << " groups" << endl;

    // ── Deduplicate within each group (parallel) ──
    vector<int> nodup_ids;

    #pragma omp parallel
    {
        vector<int> nodup_private;
        #pragma omp for schedule(dynamic)
        for (int iit = 0; iit < (int)atom_by_group.size(); ++iit) {
            auto &queue = atom_by_group[iit];
            vector<int> kept = {queue[0]};
            for (int lid = 1; lid < (int)queue.size(); ++lid) {
                bool is_unique = true;
                for (int kid : kept) {
                    if (comparator.equivalent(prepared[kid], prepared[queue[lid]])) {
                        is_unique = false;
                        break;
                    }
                }
                if (is_unique) kept.push_back(queue[lid]);
            }
            for (int kid : kept)
                nodup_private.push_back(kid);

            #pragma omp critical
            cout << "Group " << iit << " → " << kept.size() << " kept" << endl;
        }
        #pragma omp critical
        nodup_ids.insert(nodup_ids.end(), nodup_private.begin(), nodup_private.end());
    }

    std::ranges::sort(nodup_ids);
    cout << nodup_ids.size() << " unique structures" << endl;

    // ── Write output ──
    vector<Atoms> outputs(nodup_ids.size());
    for (int i = 0; i < (int)nodup_ids.size(); ++i) {
        int oid = nodup_ids[i];
        outputs[i] = preserve ? atoms[oid] : prepared[oid].atoms;
    }
    if (!iobj.write(outputs, outputfile, outformat.empty() ? "xyz" : outformat))
        GPERROR("Failed to write structures: " + outputfile);

    {
        std::ofstream ofs(outputid);
        if (!ofs) GPERROR("Cannot open output index file: " + outputid);
        fmt::print(ofs, "{}", fmt::join(nodup_ids, " "));
        ofs.close();
        if (!ofs) GPERROR("Failed to write indices: " + outputid);
    }

    return 0;
}
