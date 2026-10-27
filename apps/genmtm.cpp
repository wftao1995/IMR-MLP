// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#include "mtmbasis/mtm.h"

using namespace std;

int mtmbasis(int argc, char *argv[])
{
    auto &cmdopts = CMDOption::global_cmdopts;
    bool readpool = false;
    bool convertold = false;
    bool genpoolonly = false;
    bool samplepool = false;
    /**
     * 1. will preserve all mt scalars
     * 2. for each dim, have different sample rate
     * 3. read tag: sample_rates
     */
    bool advanced_sample = false;
    vector<double> advanced_rate;
    double sample_rate = 1.0;
    string alpha_pool = "";
    string mt_pool = "";
    string outpool_file = "sampled_alphas.dat";
    string output_file = "potential.mtp";
    string settingfile = "settings.ini";
    GPSettings settings;

    cmdopts
        .addop(sample_rate, "sample-rate", "rate", "random sample in alpha_pool")
        .addop(alpha_pool, "alphapool", "pool", "pool to be read")
        .addop(advanced_sample, "advanced-sample", "as", "advanced sample")
        .addop(mt_pool, "moments", "mts", "{mu, nu} pairs to be used")
        .addop(output_file, "potfile", "pot", "output file")
        .addop(genpoolonly, "generate-only", "g", "generate pool only")
        .addop(samplepool, "sample", "s", "sample alpha pool")
        .addop(convertold, "convert", "", "convert old to new")
        .addop(settingfile, "settings", "", "settings file", true)
        .attach_settings("", "MTP basis generation settings",
            [](std::ostream &os) { PickTensors().ListSettings(os, ""); });
    cmdopts.phrase_options(argc, argv);
    settings.Load(settingfile);
    apply_settings_to_cmdopts(cmdopts, settings);
    
    if (alpha_pool != "")
    {
        readpool = true;
    }

    PickTensors pick(settings);
    pick.output_file = output_file;
    // not working
    if (convertold) {
        pick.gen_alpha_pool_by_shapeev_level(mt_pool);
        pick.create_hugepool_and_connections();
        pick.generate_trees();
        pick.regen_from_old();
        exit(0);
    }
    // 1. generate alpha pool
    if (!readpool) {
        pick.gen_alpha_pool_by_shapeev_level(mt_pool);
    } else {
        pick.load_alpha_pool_from_file(alpha_pool);
    }
    Message("alphas generated");
    if (samplepool) {
        if (sample_rate > 1) sample_rate = 1;
        if (sample_rate < 0) sample_rate = 1;
        int norig = pick.alphas_gen.size();
        int nnew = sample_rate * norig;
        random_device rd;
        mt19937 mtd(rd());
        ranges::shuffle(pick.alphas_gen, mtd);
        pick.alphas_gen.resize(nnew);
        stable_sort(pick.alphas_gen.begin(), pick.alphas_gen.end());
        pick.save_pool(outpool_file, pick.alphas_gen);
        Message("alphas sampled");
    }
    if (advanced_sample) {
        // per-dimension sampling rates, dims 2 3 4 5 6 7
        string ratesstr = settings["sample_rates"];
        vector<string> vstr;
        str_split(ratesstr, vstr);
        if (vstr.empty()) {
            GPERROR((std::string) "advanced-sample requires the 'sample_rates' setting (rates for dims 2..7)");
        }
        if (vstr.size() > 8) {
            GPERROR((std::string) "sample_rates has too many entries (expected up to 8 for dims 2..9)");
        }
        advanced_rate.resize(10, 0);
        for (int i = 0; i < vstr.size(); ++i) {
            try {
                advanced_rate[i+2] = stod(vstr[i]);
            } catch (const std::exception &err) {
                GPERROR((std::string) "invalid sample_rates entry '" + vstr[i] + "': " + err.what());
            }
            // rates above 1 would read past the sampled dimension block
            if (advanced_rate[i+2] < 0.0 || advanced_rate[i+2] > 1.0) {
                GPERROR((std::string) "sample_rates entries must be within [0, 1]");
            }
        }
        random_device rd;
        mt19937 mtd(rd());
        stable_sort(pick.alphas_gen.begin(), pick.alphas_gen.end(), 
            [](const alpha_index_type &b1, const alpha_index_type &b2) {
                if (b1.size() < b2.size()) return true;
                return false;
            });
        vector<int> vnodes, dimcount;
        dimcount.resize(8, 0);
        vnodes.push_back(0);
        int cdim = 1;
        int maxdim = 1;
        for (int i = 0; i < pick.alphas_gen.size(); ++i) {
            if (pick.alphas_gen[i].size() > cdim) {
                cdim = pick.alphas_gen[i].size();
                dimcount[cdim] = 1;
                vnodes.push_back(i);
            } else {
                dimcount[cdim]++;
            }
            if (pick.alphas_gen[i].size() > maxdim) maxdim = pick.alphas_gen[i].size();
        }
        vnodes.push_back(pick.alphas_gen.size());
        // _DEBUG_VECTOR_(vnodes);
        // _DEBUG_VECTOR_(dimcount);
        vector<alpha_index_type> alphas_new;
        // mt scalar
        for (int i = 0; i < vnodes[1]; ++i) {
            alphas_new.push_back(pick.alphas_gen[i]);
        }
        vector<int> nsample;
        nsample.resize(maxdim+1, 0);
        for (int adim = 2; adim <= maxdim; ++adim) {
            nsample[adim] = dimcount[adim] * advanced_rate[adim];
            std::cout << "Sample count: " << adim << " " << nsample[adim] << std::endl;
        }
        // sample each part
        for (int adim = 2; adim <= maxdim; ++adim) {
            // _DEBUG_OUTPUT_(adim);
            // _DEBUG_OUTPUT_(vnodes[adim-1]);
            // _DEBUG_OUTPUT_(vnodes[adim]);
            shuffle(pick.alphas_gen.begin() + vnodes[adim-1], pick.alphas_gen.begin() + vnodes[adim], mtd);
            for (int i = 0; i < nsample[adim]; ++i) {
                alphas_new.push_back(pick.alphas_gen[vnodes[adim-1] + i]);
            }
        }
        pick.clean_pool(alphas_new);
        pick.save_pool(outpool_file, alphas_new);
        Message("alphas sampled");
        exit(0);
    }
    if (genpoolonly) {
        pick.clean_pool(pick.alphas_gen);
        pick.save_pool("alphapool.dat", pick.alphas_gen);
        exit(0);
    }
    // 2. generate potential
    pick.gen_pot_by_alphapool_GA();
    // 3. output extra info
    {
        ofstream ofs("conns_exp.log");
        for (auto &conn : pick.alpha_moments_count_add_count) {
            ofs << conn.first[0] << " " << conn.first[1] << " " << conn.first[2] << " " << conn.second << " " << pick.alpha_index_times_count_add_count[conn.first] << '\n';
        }
        ofs.close();
    }
    Message("Finished");
    return 0;
}
