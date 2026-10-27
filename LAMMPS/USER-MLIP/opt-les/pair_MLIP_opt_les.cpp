// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: GPL-2.0-only
// Adapted for the IMR-MLP public source layout, 2026-09-22.

/* -*- c++ -*- ----------------------------------------------------------
   MTP+LES optimized pair style for LAMMPS
   Based on pair_MLIP_opt, adds Latent Ewald Summation for long-range
   electrostatics.

   PairStyle: mlip/opt-les
------------------------------------------------------------------------- */

#include "pair_MLIP_opt_les.h"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include "atom.h"
#include "force.h"
#include "comm.h"
#include "neighbor.h"
#include "neigh_request.h"
#include "neigh_list.h"
#include "memory.h"
#include "error.h"
#include "domain.h"
#include "update.h"
#include "mpi.h"

using namespace LAMMPS_NS;

// ═══════════════════════════════════════════════════════════════════════
PairMLIPOPTLES::PairMLIPOPTLES(LAMMPS *lmp) : Pair(lmp)
{
    restartinfo = 0;
    manybody_flag = 1;
    no_virial_fdotr_compute = 1; // Reciprocal energy requires an explicit cell derivative.
    single_enable = 0;
    inited = false;
    allocated = 0;
}

PairMLIPOPTLES::~PairMLIPOPTLES()
{
    if (copymode) return;
    deallocate();
}

// ═══════════════════════════════════════════════════════════════════════
void PairMLIPOPTLES::settings(int narg, char **arg)
{
    try {
        if (metadata_arguments_.defer(narg, arg, narg > 0 && std::string(arg[0]) == "phase")) return;
    } catch (const std::exception &e) { error->all(FLERR, e.what()); }
    if(narg>=3 && strcmp(arg[narg-2],"phase")==0){
        if(strcmp(arg[narg-1],"direct") && strcmp(arg[narg-1],"recurrence"))
            error->all(FLERR,"LES phase must be direct or recurrence");
        use_recurrence=strcmp(arg[narg-1],"recurrence")==0;narg-=2;
    }
    if (narg == 0) error->all(FLERR, "INCORRECT PAIR STYLE");
    if (narg > MLIP_G_S + 1)
        error->all(FLERR, "Too many atom-type mappings for mlip/opt-les");
    if (strlen(arg[0]) > 999) error->all(FLERR, "MLIP settings file name too long");

    for (int i = 0; i <= MLIP_G_S; ++i) spmap[i] = -1;
    try {
        metadata_arguments_.load(narg, arg, narg, spmap, sizeof(spmap)/sizeof(spmap[0]), true);
    } catch (const std::exception &e) { error->all(FLERR, e.what()); }
    strcpy(MLIPpot_filename, arg[0]);
    LoadPot();
    metadata_arguments_.loaded = true;
}

void PairMLIPOPTLES::coeff(int narg, char **arg)
{
    try {
        auto arguments = metadata_arguments_.coefficients(narg, arg, atom->ntypes);
        if (!arguments.empty()) {
            std::vector<char *> pointers;
            for (auto &value : arguments) pointers.push_back(value.data());
            settings(static_cast<int>(pointers.size()), pointers.data());
        }
        metadata_arguments_.validate(atom->ntypes);
    } catch (const std::exception &e) { error->all(FLERR, e.what()); }
    if (!allocated) allocate();
}

void PairMLIPOPTLES::init_style()
{
    try { metadata_arguments_.validate(atom->ntypes); }
    catch (const std::exception &e) { error->all(FLERR, e.what()); }
    if (force->newton_pair != 1)
        error->all(FLERR, "Pair style MLIP requires Newton pair on");
    if (!allocated) allocate();
    if (strcmp(update->unit_style, "metal") != 0)
        error->all(FLERR, "Pair style mlip/opt-les requires units metal");
    if (!domain->xperiodic || !domain->yperiodic || !domain->zperiodic)
        error->all(FLERR, "Pair style mlip/opt-les requires periodic boundaries in all dimensions");
    if (atom->ntypes > MLIP_G_S)
        error->all(FLERR, "Too many LAMMPS atom types for mlip/opt-les");
    for (int t = 1; t <= atom->ntypes; ++t)
        if (spmap[t] < 0 || spmap[t] >= species_count)
            error->all(FLERR, "Missing or invalid LAMMPS-type to MTP-species mapping");
    cutoffsq = cutoff * cutoff;
    int n = atom->ntypes;
    for (int i = 1; i <= n; i++)
        for (int j = 1; j <= n; j++)
            cutsq[i][j] = cutoffsq;
    inited = true;
    neighbor->add_request(this, NeighConst::REQ_FULL);
}

double PairMLIPOPTLES::init_one(int, int) { return cutoff; }

// ═══════════════════════════════════════════════════════════════════════
void PairMLIPOPTLES::allocate()
{
    int n = atom->ntypes;
    memory->create(cutsq, n + 1, n + 1, "pair:cutsq");
    memory->create(setflag, n + 1, n + 1, "pair:setflag");
    for (int i = 1; i <= n; i++)
        for (int j = 1; j <= n; j++)
            setflag[i][j] = 1;
    allocated = 1;
}

void PairMLIPOPTLES::deallocate()
{
    clear_model();
    if (allocated) {
        memory->destroy(cutsq); memory->destroy(setflag);
        allocated = 0;
    }
}

void PairMLIPOPTLES::clear_model()
{
    memory->destroy(rb_vals); memory->destroy(rb_ders);
    memory->destroy(val_mu); memory->destroy(der_mu);
    memory->destroy(moment_vals);
    memory->destroy(site_energy_ders_wrt_moments_);
    memory->destroy(alpha_moment_mapping);
    memory->destroy(alpha_index_times);
    memory->destroy(alpha_index_basic);
    memory->destroy(radial_coeffs);
    memory->destroy(specie_coeffs);
    memory->destroy(moment_coeffs);
    memory->destroy(moment_jacobian);
    memory->destroy(jlist_true);
    memory->destroy(dist_powers);
    memory->destroy(coords_powers_x); memory->destroy(coords_powers_y);
    memory->destroy(coords_powers_z);
    memory->destroy(buff_site_energy_ders);
    memory->destroy(nu_arr_list);
    memory->destroy(latent_charges);
    memory->destroy(latent_channels);
    memory->destroy(site_les_desc);
    memory->destroy(les_species_coeffs);
    memory->destroy(les_moment_coeffs);
    memory->destroy(cached_moment_vals);
    max_alpha_index_basic = max_mu = jnum_old = 0;
    latent_charges_size = site_les_desc_size = cached_moment_vals_size = 0;
    sog_channels = 1; active_channel = 0;
    channel_values.clear(); channel_derivatives.clear();
    kvecs.clear(); grid_valid = false;
    sog = Potentials::SOGModel{};
    lr_les = use_sog = inited = false;
    les_version = 0; les_response_parity = 1;
    lr_sigma = 0.5; dl = 2.0; taper_fraction = 0.1;
    smooth_cutoff = remove_self_interaction = true;
    scaling = 0.01; scaling2 = 2.0 * scaling; rb_b = -1.5;
}

// ═══════════════════════════════════════════════════════════════════════
//  LoadPot — MTP parsing + optional LES sections
// ═══════════════════════════════════════════════════════════════════════
void PairMLIPOPTLES::LoadPot(void)
{
    clear_model();
    std::ifstream ifs(MLIPpot_filename);
    std::string rbasis_type;
    if (!ifs.is_open()) error->all(FLERR, "Cannot open potential file");

    std::string tmpstr;
    char tmpline[1000];

    // Header: "MTP"
    ifs.getline(tmpline, 1000);
    int len = (int)((std::string)tmpline).length();
    if (len && tmpline[len - 1] == '\r') tmpline[len - 1] = '\0';
    if ((std::string)tmpline != "MTP") error->all(FLERR, "Can only read MTP format potentials");

    // Version
    ifs.getline(tmpline, 1000);
    len = (int)((std::string)tmpline).length();
    if (len && tmpline[len - 1] == '\r') tmpline[len - 1] = '\0';
    if ((std::string)tmpline != "version = 1.1.0")
        error->all(FLERR, "MTP file must have version \"1.1.0\"");

    // name
    ifs >> tmpstr;
    if (tmpstr == "potential_name") { ifs.ignore(2); ifs >> tmpstr; ifs >> tmpstr; }
    if (tmpstr == "scaling") { ifs.ignore(2); ifs >> scaling; scaling2 = 2. * scaling; ifs >> tmpstr; }

    if (tmpstr != "species_count") error->all(FLERR, "Error reading .mtp file");
    ifs.ignore(2); ifs >> species_count;
    if (species_count <= 0 || species_count > MLIP_G_S)
        error->all(FLERR, "MTP species_count exceeds mlip/opt-les limit");

    ifs >> tmpstr;
    if (tmpstr == "potential_tag") { getline(ifs, tmpstr); ifs >> tmpstr; }

    // Radial basis
    if (tmpstr != "radial_basis_type") error->all(FLERR, "Error reading .mtp file");
    ifs.ignore(2); ifs >> rbasis_type;

    if (rbasis_type == "RBChebyshev" || rbasis_type == "RBChebyCos" ||
        rbasis_type == "RBChebyshevPoly5" || rbasis_type == "RBChebyshevPoly3") {

        if (rbasis_type == "RBChebyshev") rbcalc = &PairMLIPOPTLES::RB_CalcDer;
        else if (rbasis_type == "RBChebyCos") rbcalc = &PairMLIPOPTLES::RB_CalcDerCos;
        else if (rbasis_type == "RBChebyshevPoly5") rbcalc = &PairMLIPOPTLES::RB_CalcDerPoly5;
        else rbcalc = &PairMLIPOPTLES::RB_CalcDerPoly3;

        ifs >> tmpstr; if (tmpstr != "min_dist") error->all(FLERR, "Error reading mtp file");
        ifs.ignore(2); ifs >> min_dist;
        ifs >> tmpstr; if (tmpstr != "max_dist") error->all(FLERR, "Error reading mtp file");
        ifs.ignore(2); ifs >> max_dist;
        cutoff = max_dist;
        mult = 2.0 / (max_dist - min_dist);
        plus_r = min_dist + max_dist; sub_r = max_dist - min_dist;
        inv_subr = 1.0 / sub_r; irpi = M_PI / max_dist;
        p0 = 1.0 - min_dist / max_dist; pp = 1.0 / p0 / max_dist;

        ifs >> tmpstr; if (tmpstr != "radial_basis_size") error->all(FLERR, "Error reading mtp file");
        ifs.ignore(2); ifs >> rb_size;
    } else error->all(FLERR, "Only RBChebyshev variants supported");

    ifs >> tmpstr;
    if (tmpstr == "rb_b") { ifs.ignore(2); ifs >> rb_b; ifs >> tmpstr; }

    // Radial funcs
    if (tmpstr != "radial_funcs_count") error->all(FLERR, "Error reading .mtp file");
    ifs.ignore(2); ifs >> radial_func_count;
    int pairs_count = species_count * species_count;
    char foo = ' ';
    int radial_size = pairs_count * radial_func_count * rb_size;
    memory->create(radial_coeffs, radial_size, "pair::radial_coeffs");

    ifs >> tmpstr;
    if (tmpstr == "radial_coeffs") {
        for (int s1 = 0; s1 < species_count; s1++)
            for (int s2 = 0; s2 < species_count; s2++) {
                ifs >> foo >> foo >> foo; double t;
                for (int i = 0; i < radial_func_count; i++) {
                    ifs >> foo;
                    for (int j = 0; j < rb_size; j++) { ifs >> t >> foo;
                        radial_coeffs[(s1*species_count+s2)*radial_func_count*rb_size + i*rb_size + j] = t; }
                }
            }
        ifs >> tmpstr;
    } else error->all(FLERR, "error reading mtp file");

    // alpha_index_basic
    if (tmpstr != "alpha_moments_count") error->all(FLERR, "Error reading .mtp file");
    ifs.ignore(2); ifs >> alpha_moments_count;
    ifs >> tmpstr; if (tmpstr != "alpha_index_basic_count") error->all(FLERR, "Error reading .mtp file");
    ifs.ignore(2); ifs >> alpha_index_basic_count;
    ifs >> tmpstr; if (tmpstr != "alpha_index_basic") error->all(FLERR, "Error reading .mtp file");
    ifs.ignore(4);

    memory->create(alpha_index_basic, alpha_index_basic_count, "pair::alpha_index_basic");
    memory->create(nu_arr_list, alpha_index_basic_count, "pair::nu_arr_list");
    int radial_func_max = -1;
    for (int i = 0; i < alpha_index_basic_count; i++) {
        char tmpch; ifs.ignore(1000, '{');
        ifs >> alpha_index_basic[i].mu >> tmpch >> alpha_index_basic[i].b1 >> tmpch
            >> alpha_index_basic[i].b2 >> tmpch >> alpha_index_basic[i].b3;
        nu_arr_list[i] = alpha_index_basic[i].b1 + alpha_index_basic[i].b2 + alpha_index_basic[i].b3;
        if (alpha_index_basic[i].mu > radial_func_max) radial_func_max = alpha_index_basic[i].mu;
    }
    if (radial_func_max != radial_func_count - 1) error->all(FLERR, "Wrong radial func count");
    ifs.ignore(1000, '\n');

    // alpha_index_times
    ifs >> tmpstr; if (tmpstr != "alpha_index_times_count") error->all(FLERR, "Error reading .mtp file");
    ifs.ignore(2); ifs >> alpha_index_times_count;
    ifs >> tmpstr; if (tmpstr != "alpha_index_times") error->all(FLERR, "Error reading .mtp file");
    ifs.ignore(4);
    memory->create(alpha_index_times, alpha_index_times_count, "pair::alpha_index_times");
    for (int i = 0; i < alpha_index_times_count; i++) {
        char tmpch; ifs.ignore(1000, '{');
        ifs >> alpha_index_times[i].l1 >> tmpch >> alpha_index_times[i].l2 >> tmpch
            >> alpha_index_times[i].c >> tmpch >> alpha_index_times[i].id;
    }
    ifs.ignore(1000, '\n');

    // scalar moments
    ifs >> tmpstr; if (tmpstr != "alpha_scalar_moments") error->all(FLERR, "Error reading .mtp file");
    ifs.ignore(2); ifs >> alpha_scalar_moments;
    memory->create(alpha_moment_mapping, alpha_scalar_moments, "pair::alpha_moment_mapping");
    ifs >> tmpstr; if (tmpstr != "alpha_moment_mapping") error->all(FLERR, "Error reading .mtp file");
    ifs.ignore(4);
    for (int i = 0; i < alpha_scalar_moments; i++) { char tmpch; ifs >> alpha_moment_mapping[i] >> tmpch; }
    ifs.ignore(1000, '\n');

    // linear coeffs
    ifs >> tmpstr;
    if (tmpstr != "species_coeffs") error->all(FLERR, "species_coeffs not found");
    ifs.ignore(4);
    memory->create(specie_coeffs, species_count, "pair::specie_coeffs");
    for (int i = 0; i < species_count; i++) ifs >> specie_coeffs[i] >> foo;

    ifs >> tmpstr; if (tmpstr != "moment_coeffs") error->all(FLERR, "Cannot read linear coeffs");
    ifs.ignore(2);
    memory->create(moment_coeffs, alpha_scalar_moments, "pair::moment_coeffs");
    ifs.ignore(10, '{');
    for (int i = 0; i < alpha_scalar_moments; i++) ifs >> moment_coeffs[i] >> foo;

    // Read portable SOG metadata independently of the MTP coefficient layout.
    const auto lr_start=ifs.tellg();
    use_sog=false;
    while (ifs>>tmpstr) {
        if (tmpstr=="model_coeffs")
            error->all(FLERR,"mlip/opt-les supports the linear MTP short-range head only");
        if (tmpstr=="les_kernel") {
            ifs.ignore(1000,'=');ifs>>tmpstr;
            if (tmpstr!="coulomb" && tmpstr!="sog") error->all(FLERR,"Unknown long-range kernel");
            use_sog=use_sog || tmpstr=="sog";
        }
        use_sog=use_sog || tmpstr.rfind("sog_",0)==0;
    }
    ifs.clear();ifs.seekg(lr_start);
    if (use_sog) {
        try { sog=Potentials::SOGModel::read(ifs); }
        catch (const std::exception& e) { error->all(FLERR,e.what()); }
        ifs.clear();ifs.seekg(lr_start);
        dl=sog.dl;taper_fraction=sog.taper;smooth_cutoff=sog.smooth;
        remove_self_interaction=sog.onsite;
    }

    sog_channels=use_sog ? sog.channels : 1;
    // ─── LES parameters (optional) ───
    lr_les = false;
    les_response_parity = 1;
    bool found_les_species = false, found_les_moment = false;
    memory->create(les_species_coeffs, sog_channels*species_count, "pair::les_species_coeffs");
    memory->create(les_moment_coeffs, sog_channels*alpha_scalar_moments, "pair::les_moment_coeffs");
    for (int i = 0; i < species_count; i++) les_species_coeffs[i] = 0.0;
    for (int i = 0; i < alpha_scalar_moments; i++) les_moment_coeffs[i] = 0.0;
    const auto read_latent=[&](double* values,int count) {
        char c;
        const auto expect=[&](char wanted) {
            if (!(ifs>>c) || c!=wanted) error->all(FLERR,"Malformed LES latent coefficient block");
        };
        expect('=');expect('{');
        for (int i=0;i<count;++i) {
            if (i) expect(',');
            if (!(ifs>>values[i]) || !std::isfinite(values[i]))
                error->all(FLERR,"Invalid LES latent coefficient");
        }
        ifs>>std::ws;if (ifs.peek()==',') ifs.get();expect('}');
    };

    while (ifs >> tmpstr) {
        if (tmpstr == "les_version") { ifs.ignore(1000, '='); ifs >> les_version; }
        else if (tmpstr == "les_sigma") { ifs.ignore(1000, '='); ifs >> lr_sigma; }
        else if (tmpstr == "les_dl") { ifs.ignore(1000, '='); ifs >> dl; }
        else if (tmpstr == "les_taper_fraction") { ifs.ignore(1000, '='); ifs >> taper_fraction; }
        else if (tmpstr == "les_cutoff_mode") {
            ifs.ignore(1000, '='); ifs >> tmpstr;
            if (tmpstr == "smooth") smooth_cutoff = true;
            else if (tmpstr == "reference_hard") smooth_cutoff = false;
            else error->all(FLERR, "Unknown les_cutoff_mode in MTP+LES potential");
        }
        else if (tmpstr == "les_charge_treatment") {
            ifs.ignore(1000, '='); ifs >> tmpstr;
            if (tmpstr != "uniform_background")
                error->all(FLERR, "LAMMPS LES supports uniform_background charge treatment only");
        }
        else if (tmpstr == "remove_self_interaction") { ifs.ignore(1000, '='); ifs >> remove_self_interaction; }
        else if (tmpstr == "les_response_parity") { ifs.ignore(1000, '='); ifs >> les_response_parity; }
        else if (tmpstr == "les_species_coeffs") {
            if (found_les_species) error->all(FLERR,"Duplicate les_species_coeffs");
            lr_les = true; found_les_species = true;
            read_latent(les_species_coeffs,sog_channels*species_count);
        } else if (tmpstr == "les_moment_coeffs") {
            if (found_les_moment) error->all(FLERR,"Duplicate les_moment_coeffs");
            lr_les = true; found_les_moment = true;
            read_latent(les_moment_coeffs,sog_channels*alpha_scalar_moments);
        }
    }
    if (lr_les && (!found_les_species || !found_les_moment))
        error->all(FLERR, "Incomplete LES parameters in potential file");
    if (use_sog && !lr_les) error->all(FLERR,"SOG requires latent coefficient blocks");
    if (lr_les && les_version != (use_sog ? sog.version() : 2))
        error->all(FLERR, "LAMMPS mlip/opt-les requires LES v2 or SOG v3/v4");
    if (!(lr_sigma > 0.0) || !(dl > 0.0) || (smooth_cutoff && !(taper_fraction > 0.0 && taper_fraction < 1.0)))
        error->all(FLERR, "Invalid LES sigma, dl, or taper fraction");
    if (les_response_parity != 1 && les_response_parity != -1)
        error->all(FLERR, "LES response parity must be +1 or -1");
    ifs.close();

    // Allocate MTP buffers
    for (int i = 0; i < alpha_index_basic_count; i++) {
        max_alpha_index_basic = std::max(max_alpha_index_basic,
            alpha_index_basic[i].b1 + alpha_index_basic[i].b2 + alpha_index_basic[i].b3);
        max_mu = std::max(max_mu, alpha_index_basic[i].mu);
    }
    max_alpha_index_basic++; max_mu++;

    memory->create(val_mu, max_mu, "pair::val_mu");
    memory->create(der_mu, max_mu, "pair::der_mu");
    memory->create(moment_vals, alpha_moments_count, "pair::moment_vals");
    memory->create(site_energy_ders_wrt_moments_, alpha_moments_count, "pair::site_energy_ders_wrt_moments_");
    memory->create(rb_vals, rb_size, "pair::rb_vals");
    memory->create(rb_ders, rb_size, "pair::rb_ders");
    memory->create(dist_powers, max_alpha_index_basic, "pair::dist_powers"); dist_powers[0] = 1;
    memory->create(coords_powers_x, max_alpha_index_basic + 1, "pair::coords_powers_x");
    memory->create(coords_powers_y, max_alpha_index_basic + 1, "pair::coords_powers_y");
    memory->create(coords_powers_z, max_alpha_index_basic + 1, "pair::coords_powers_z");
    coords_powers_x[0] = 0; coords_powers_y[0] = 0; coords_powers_z[0] = 0;
    coords_powers_x[1] = 1; coords_powers_y[1] = 1; coords_powers_z[1] = 1;
    memory->create(buff_site_energy_ders, jnum_old * 3, "pair::buff_site_energy_ders");
    memory->create(jlist_true, jnum_old, "pair::jlist_true");
    memory->create(moment_jacobian, alpha_index_basic_count * jnum_old * 3, "pair::moment_jacobian");
}

// ═══════════════════════════════════════════════════════════════════════
//  generate_k_vectors — from LAMMPS simulation box
// ═══════════════════════════════════════════════════════════════════════
void PairMLIPOPTLES::generate_k_vectors()
{
    bool same=grid_valid;
    for(int a=0;a<6;++a)same=same && cached_h[a]==domain->h[a];
    if(same)return;
    for(int a=0;a<6;++a)cached_h[a]=domain->h[a];
    kvecs.clear();
    const double twopi = 2.0 * M_PI;
    const double a[3] = {domain->h[0], 0.0, 0.0};
    const double b[3] = {domain->h[5], domain->h[1], 0.0};
    const double c[3] = {domain->h[4], domain->h[3], domain->h[2]};
    const double volume = a[0] * (b[1]*c[2]-b[2]*c[1])
        - a[1] * (b[0]*c[2]-b[2]*c[0])
        + a[2] * (b[0]*c[1]-b[1]*c[0]);
    if (!(volume > 1e-12)) error->all(FLERR, "Invalid LES simulation cell");
    const auto cross_recip = [&](const double u[3], const double v[3], int d) {
        const double cross[3] = {u[1]*v[2]-u[2]*v[1],
                                 u[2]*v[0]-u[0]*v[2],
                                 u[0]*v[1]-u[1]*v[0]};
        return twopi * cross[d] / volume;
    };
    const double r1[3] = {cross_recip(b,c,0),cross_recip(b,c,1),cross_recip(b,c,2)};
    const double r2[3] = {cross_recip(c,a,0),cross_recip(c,a,1),cross_recip(c,a,2)};
    const double r3[3] = {cross_recip(a,b,0),cross_recip(a,b,1),cross_recip(a,b,2)};
    const auto norm = [](const double v[3]) { return std::sqrt(v[0]*v[0]+v[1]*v[1]+v[2]*v[2]); };
    const auto bound = [&](const double v[3]) {
        const double x = norm(v) / dl;
        return std::max(1, smooth_cutoff ? (int)std::ceil(x) : (int)x);
    };
    const int nmax1=bound(a), nmax2=bound(b), nmax3=bound(c);
    phase_bounds={nmax1,nmax2,nmax3};
    for(int d=0;d<3;++d){reciprocal[0][d]=r1[d];reciprocal[1][d]=r2[d];reciprocal[2][d]=r3[d];}
    std::vector<double> inverse_widths(sog.log_bandwidths.size());
    for(size_t m=0;m<inverse_widths.size();++m)inverse_widths[m]=std::exp(-2*sog.log_bandwidths[m]);
    const double kmax = twopi / dl;
    const double kon = kmax * (1.0-taper_fraction);
    const double width = kmax-kon;

    if (use_sog && sog.include_zero) {
        double value,derivative;sog.value(0.,value,derivative);
        kvecs.push_back({0.,0.,0.,value,derivative,{0,0,0}});
    }
    for (int n1 = -nmax1; n1 <= nmax1; n1++)
        for (int n2 = -nmax2; n2 <= nmax2; n2++)
            for (int n3 = -nmax3; n3 <= nmax3; n3++) {
                if (n1 == 0 && n2 == 0 && n3 == 0) continue;
                int sign = n1 != 0 ? n1 : (n2 != 0 ? n2 : n3);
                if (sign <= 0) continue;
                double kx=n1*r1[0]+n2*r2[0]+n3*r3[0];
                double ky=n1*r1[1]+n2*r2[1]+n3*r3[1];
                double kz=n1*r1[2]+n2*r2[2]+n3*r3[2];
                double ksq = kx*kx + ky*ky + kz*kz;
                double kmag=std::sqrt(ksq);
                if (smooth_cutoff ? kmag >= kmax : kmag > kmax) continue;
                double w=1.0, dwdk2=0.0;
                if (smooth_cutoff && kmag > kon) {
                    double x=(kmag-kon)/width, x2=x*x, x3=x2*x, x4=x3*x, x5=x4*x;
                    w=1.0-10.0*x3+15.0*x4-6.0*x5;
                    double dwdx=-30.0*x2+60.0*x3-30.0*x4;
                    dwdk2=dwdx/(width*2.0*kmag);
                }
                double base,basep;
                if (use_sog) {
                    base=basep=0.;
                    for(int m=0;m<sog.terms();++m){const double v=sog.weights[m]*std::exp(-ksq*inverse_widths[m]);base+=v;basep-=inverse_widths[m]*v;}
                }
                else {
                    base=4.0*M_PI*coulomb_constant*std::exp(-0.5*lr_sigma*lr_sigma*ksq)/ksq;
                    basep=base*(-0.5*lr_sigma*lr_sigma-1.0/ksq);
                }
                kvecs.push_back({kx,ky,kz,2.0*w*base,2.0*(w*basep+dwdk2*base),{n1,n2,n3}});
            }
    channel_values.resize(kvecs.size()*sog_channels);channel_derivatives.resize(kvecs.size()*sog_channels);
    for(size_t ik=0;ik<kvecs.size();++ik){
        const auto& kv=kvecs[ik];channel_values[ik*sog_channels]=kv.kernel;channel_derivatives[ik*sog_channels]=kv.kernel_derivative;
        const double ksq=kv.x*kv.x+kv.y*kv.y+kv.z*kv.z,kmag=std::sqrt(ksq);
        double w=1.,dw=0.;
        if(smooth_cutoff && kmag>kon){double z=(kmag-kon)/width;w=1-10*std::pow(z,3)+15*std::pow(z,4)-6*std::pow(z,5);dw=(-30*z*z+60*z*z*z-30*z*z*z*z)/(width*2*kmag);}
        for(int channel=1;channel<sog_channels;++channel){
            double value=0.,derivative=0.;
            for(int m=channel*sog.terms();m<(channel+1)*sog.terms();++m){double v=sog.weights[m]*std::exp(-ksq*inverse_widths[m]);value+=v;derivative-=inverse_widths[m]*v;}
            const double factor=ksq==0.?1.:2.;
            channel_values[ik*sog_channels+channel]=factor*w*value;
            channel_derivatives[ik*sog_channels+channel]=factor*(w*derivative+dw*value);
        }
    }
    grid_valid=true;
}

// ═══════════════════════════════════════════════════════════════════════
//  ewald_lr_compute — Ewald energy + direct forces + indirect forces
// ═══════════════════════════════════════════════════════════════════════
void PairMLIPOPTLES::ewald_lr_compute(double **x, double **f, int *type, int nlocal, int nghost,
                                       double &eng_vdwl)
{
    (void) nghost;
    const double volume = domain->xprd * domain->yprd * domain->zprd;
    if (!(volume > 1.0e-12)) error->all(FLERR, "Invalid LES simulation cell volume");

    const double energy_pref = 0.5 / volume;
    const double force_pref = 2.0 * energy_pref;
    const double self_pref = use_sog ? 0.5*sog.self(active_channel)
        : coulomb_constant / (lr_sigma * std::sqrt(2.0*M_PI));

    double reciprocal_sum = 0.0;
    double reciprocal_virial[6] = {0.0,0.0,0.0,0.0,0.0,0.0};
    std::vector<double> dEdq(nlocal, 0.0);
    std::vector<double> lr_fx(nlocal, 0.0), lr_fy(nlocal, 0.0), lr_fz(nlocal, 0.0);

    Potentials::ReciprocalPhases phases;
    phases.prepare(nlocal,phase_bounds,reciprocal,[&](int i,int d){return x[i][d];},use_recurrence);
    std::vector<double> cos_phase(nlocal),sin_phase(nlocal),structure(2*kvecs.size(),0.);
    for(size_t ik=0;ik<kvecs.size();++ik){
        const auto& kv=kvecs[ik];
        phases.evaluate(kv.index,{kv.x,kv.y,kv.z},cos_phase.data(),sin_phase.data());
        for(int i=0;i<nlocal;++i){structure[2*ik]+=latent_charges[i]*cos_phase[i];structure[2*ik+1]+=latent_charges[i]*sin_phase[i];}
    }
    // Bounded counts, identical chunks on every rank including empty ranks.
    for(size_t start=0;start<structure.size();start+=131072){
        const int count=std::min(size_t(131072),structure.size()-start);
        MPI_Allreduce(MPI_IN_PLACE,structure.data()+start,count,MPI_DOUBLE,MPI_SUM,world);
    }
    for(size_t ik=0;ik<kvecs.size();++ik){
        const auto& kv=kvecs[ik];
        const double kx=kv.x,ky=kv.y,kz=kv.z;
        const double S_real=structure[2*ik],S_imag=structure[2*ik+1];
        phases.evaluate(kv.index,{kx,ky,kz},cos_phase.data(),sin_phase.data());
        const double kernel=channel_values[ik*sog_channels+active_channel],kernel_derivative=channel_derivatives[ik*sog_channels+active_channel];
        const double Sk_sq = S_real*S_real + S_imag*S_imag;
        const double mode_energy = energy_pref * kernel * Sk_sq;
        reciprocal_sum += kernel * Sk_sq;
        reciprocal_virial[0] += mode_energy + 2.0*energy_pref*kernel_derivative*Sk_sq*kx*kx;
        reciprocal_virial[1] += mode_energy + 2.0*energy_pref*kernel_derivative*Sk_sq*ky*ky;
        reciprocal_virial[2] += mode_energy + 2.0*energy_pref*kernel_derivative*Sk_sq*kz*kz;
        reciprocal_virial[3] += 2.0*energy_pref*kernel_derivative*Sk_sq*kx*ky;
        reciprocal_virial[4] += 2.0*energy_pref*kernel_derivative*Sk_sq*kx*kz;
        reciprocal_virial[5] += 2.0*energy_pref*kernel_derivative*Sk_sq*ky*kz;

        for (int i = 0; i < nlocal; i++) {
            const double cos_kr=cos_phase[i],sin_kr=sin_phase[i];
            const double re_part = cos_kr*S_real + sin_kr*S_imag;
            dEdq[i] += force_pref * kernel * re_part;
            double im_part = sin_kr * S_real - cos_kr * S_imag;
            double pref = force_pref * kernel * latent_charges[i] * im_part;
            lr_fx[i] += pref * kx; lr_fy[i] += pref * ky; lr_fz[i] += pref * kz;
            const double atom_mode_energy=energy_pref*kernel*latent_charges[i]*re_part;
            if (eflag_atom) eatom[i] += atom_mode_energy;
            if (vflag_atom) {
                const double atom_hp=2.0*energy_pref*kernel_derivative
                    *latent_charges[i]*re_part;
                vatom[i][0]+=atom_mode_energy+atom_hp*kx*kx;
                vatom[i][1]+=atom_mode_energy+atom_hp*ky*ky;
                vatom[i][2]+=atom_mode_energy+atom_hp*kz*kz;
                vatom[i][3]+=atom_hp*kx*ky;
                vatom[i][4]+=atom_hp*kx*kz;
                vatom[i][5]+=atom_hp*ky*kz;
            }
        }
    }

    int nprocs_ew = 1;
    MPI_Comm_size(world, &nprocs_ew);
    eng_vdwl += energy_pref * reciprocal_sum / nprocs_ew;
    for (int i = 0; i < nlocal; i++) {
        f[i][0] += lr_fx[i]; f[i][1] += lr_fy[i]; f[i][2] += lr_fz[i];
    }
    if (remove_self_interaction) {
        for (int i=0; i<nlocal; ++i) {
            const double self_energy=-self_pref*latent_charges[i]*latent_charges[i];
            eng_vdwl += self_energy;
            if (eflag_atom) eatom[i] += self_energy;
            dEdq[i] -= 2.0*self_pref*latent_charges[i];
        }
    }

    if (vflag_global)
        for (int m=0;m<6;++m) virial[m]+=reciprocal_virial[m]/nprocs_ew;

    // ── Indirect forces ──
    for (int ii = 0; ii < list->inum; ii++) {
        int ati = list->ilist[ii];
        if (ati >= nlocal) continue;
        int type_central = spmap[type[ati]];
        int jnum = list->numneigh[ati];
        double *ivec = x[ati];

        // dE/d(moment) = dEdq[ati] * les_moment_coeffs[a]
        memset(site_energy_ders_wrt_moments_, 0, alpha_moments_count * sizeof(double));
        // Restore full moment tree for this atom from cache
        memcpy(moment_vals, &cached_moment_vals[ati*alpha_moments_count], alpha_moments_count*sizeof(double));
        for (int a = 0; a < alpha_scalar_moments; a++)
            site_energy_ders_wrt_moments_[alpha_moment_mapping[a]] =
                dEdq[ati] * (use_sog ? 1 : les_response_parity) * les_moment_coeffs[active_channel*alpha_scalar_moments+a];

        // Backprop through product rules
        for (int i = alpha_index_times_count - 1; i >= 0; i--) {
            double val0 = moment_vals[alpha_index_times[i].l1];
            double val1 = moment_vals[alpha_index_times[i].l2];
            int mulc = alpha_index_times[i].c;
            site_energy_ders_wrt_moments_[alpha_index_times[i].l2] +=
                site_energy_ders_wrt_moments_[alpha_index_times[i].id] * mulc * val0;
            site_energy_ders_wrt_moments_[alpha_index_times[i].l1] +=
                site_energy_ders_wrt_moments_[alpha_index_times[i].id] * mulc * val1;
        }

        // Recompute moment Jacobian for this atom.  The charge is a local MTP
        // scalar, so its coordinate derivative uses the same graph as SR MTP.
        memset(moment_jacobian, 0, alpha_index_basic_count*jnum*3*sizeof(double));
        int jnum_true = 0;
        for (int jj = 0; jj < jnum; jj++) {
            int atj = list->firstneigh[ati][jj] & NEIGHMASK;
            double *jvec = x[atj];
            int jtype = spmap[type[atj]];
            double rvec[3] = {jvec[0]-ivec[0], jvec[1]-ivec[1], jvec[2]-ivec[2]};
            double rijsq = rvec[0]*rvec[0] + rvec[1]*rvec[1] + rvec[2]*rvec[2];
            if (rijsq < cutoffsq) {
                double rij = sqrt(rijsq);
                double rijinv = 1.0 / rij;
                jlist_true[jnum_true] = atj;
                (this->*rbcalc)(rij);
                coords_powers_x[0]=coords_powers_y[0]=coords_powers_z[0]=0.0;
                coords_powers_x[1]=coords_powers_y[1]=coords_powers_z[1]=1.0;
                for (int k = 1; k < max_alpha_index_basic; k++) {
                    dist_powers[k] = dist_powers[k-1] * rijinv;
                    coords_powers_x[k+1]=coords_powers_x[k]*rvec[0];
                    coords_powers_y[k+1]=coords_powers_y[k]*rvec[1];
                    coords_powers_z[k+1]=coords_powers_z[k]*rvec[2];
                }
                for (int mu = 0; mu < max_mu; mu++) {
                    val_mu[mu] = 0; der_mu[mu] = 0;
                    int off = (type_central*species_count + jtype)*radial_func_count*rb_size + mu*rb_size;
                    for (int xi = 0; xi < rb_size; xi++) { val_mu[mu] += radial_coeffs[off+xi]*rb_vals[xi]; }
                    for (int xi = 0; xi < rb_size; xi++) { der_mu[mu] += radial_coeffs[off+xi]*rb_ders[xi]; }
                }
                for (int i = 0; i < alpha_index_basic_count; i++) {
                    int mu = alpha_index_basic[i].mu;
                    double val = val_mu[mu], der = der_mu[mu];
                    int k = nu_arr_list[i]; double rpowk = dist_powers[k];
                    val *= rpowk; der = der*rpowk - k*val*rijinv;
                    double pow0 = coords_powers_x[alpha_index_basic[i].b1+1];
                    double pow1 = coords_powers_y[alpha_index_basic[i].b2+1];
                    double pow2 = coords_powers_z[alpha_index_basic[i].b3+1];
                    double mult0 = pow0*pow1*pow2;
                    int base = i*jnum*3 + 3*jnum_true;
                    moment_jacobian[base+0] += mult0*der*rijinv*rvec[0];
                    moment_jacobian[base+1] += mult0*der*rijinv*rvec[1];
                    moment_jacobian[base+2] += mult0*der*rijinv*rvec[2];
                    moment_jacobian[base+0] += alpha_index_basic[i].b1*val * coords_powers_x[alpha_index_basic[i].b1]*pow1*pow2;
                    moment_jacobian[base+1] += alpha_index_basic[i].b2*val * coords_powers_y[alpha_index_basic[i].b2]*pow0*pow2;
                    moment_jacobian[base+2] += alpha_index_basic[i].b3*val * coords_powers_z[alpha_index_basic[i].b3]*pow0*pow1;
                }
                jnum_true++;
            }
        }

        // Contract with dE/d(moments)
        double *buf = new double[jnum_true*3]();
        for (int i = 0; i < alpha_index_basic_count; i++) {
            double dEdM = site_energy_ders_wrt_moments_[i];
            if (fabs(dEdM) < 1e-40) continue;
            int off1 = i*jnum*3;
            for (int j = 0; j < jnum_true; j++) {
                int off2 = j*3;
                buf[off2+0] += dEdM * moment_jacobian[off1+off2+0];
                buf[off2+1] += dEdM * moment_jacobian[off1+off2+1];
                buf[off2+2] += dEdM * moment_jacobian[off1+off2+2];
            }
        }
        // Apply
        for (int j = 0; j < jnum_true; j++) {
            int atj = jlist_true[j], off = j*3;
            f[ati][0] += buf[off+0]; f[ati][1] += buf[off+1]; f[ati][2] += buf[off+2];
            f[atj][0] -= buf[off+0]; f[atj][1] -= buf[off+1]; f[atj][2] -= buf[off+2];
            if (vflag_global) {
                const double rx=x[atj][0]-x[ati][0], ry=x[atj][1]-x[ati][1], rz=x[atj][2]-x[ati][2];
                virial[0]-=buf[off]*rx;virial[1]-=buf[off+1]*ry;virial[2]-=buf[off+2]*rz;
                virial[3]-=buf[off+1]*rx;virial[4]-=buf[off+2]*rx;virial[5]-=buf[off+2]*ry;
            }
            if (vflag_atom) {
                const double rx=x[atj][0]-x[ati][0], ry=x[atj][1]-x[ati][1];
                const double rz=x[atj][2]-x[ati][2];
                vatom[ati][0]-=buf[off+0]*rx; vatom[ati][1]-=buf[off+1]*ry;
                vatom[ati][2]-=buf[off+2]*rz; vatom[ati][3]-=buf[off+1]*rx;
                vatom[ati][4]-=buf[off+2]*rx; vatom[ati][5]-=buf[off+2]*ry;
            }
        }
        delete[] buf;
    }

}

// ═══════════════════════════════════════════════════════════════════════
//  compute — main force routine
// ═══════════════════════════════════════════════════════════════════════
void PairMLIPOPTLES::compute(int eflag, int vflag)
{
    if (eflag || vflag) ev_setup(eflag, vflag);
    else ev_unset();

    double en = 0.0;
    double *p_site_en = eflag_atom ? &eatom[0] : NULL;

    double **x = atom->x, **f = atom->f;
    int *type = atom->type;
    const int nlocal = atom->nlocal;

    // Ensure LES buffers sized
    if (lr_les) {
        int nall = std::max(1,atom->nlocal + atom->nghost);
        if (latent_charges_size < nall) { memory->grow(latent_charges, nall, "pair::latent_charges"); memory->destroy(latent_channels);memory->create(latent_channels,nall,sog_channels,"pair::latent_channels"); latent_charges_size = nall; }
        if (site_les_desc_size < nall * alpha_scalar_moments) { memory->grow(site_les_desc, nall*alpha_scalar_moments, "pair::site_les_desc"); site_les_desc_size = nall*alpha_scalar_moments; }
        if (cached_moment_vals_size < nall * alpha_moments_count) { memory->grow(cached_moment_vals, nall*alpha_moments_count, "pair::cached_moment_vals"); cached_moment_vals_size = nall*alpha_moments_count; }
    }

    // ═══ SR MTP loop ═══
    for (int ii = 0; ii < list->inum; ii++) {
        int ati = list->ilist[ii], jnum = list->numneigh[ati];
        if (ati >= nlocal) continue;
        int type_central = spmap[type[ati]];
        double *ivec = x[ati];
        double buff_site_energy = 0;

        if (jnum_old < jnum) {
            memory->grow(moment_jacobian, alpha_index_basic_count*jnum*3, "pair::moment_jacobian");
            memory->grow(jlist_true, jnum, "pair::jlist_true");
            memory->grow(buff_site_energy_ders, jnum*3, "pair::buff_site_energy_ders");
            jnum_old = jnum;
        }
        memset(moment_jacobian, 0, alpha_index_basic_count*jnum*3*sizeof(double));
        memset(buff_site_energy_ders, 0, jnum*3*sizeof(double));
        memset(moment_vals, 0, alpha_moments_count*sizeof(double));

        int jnum_true = 0;
        for (int jj = 0; jj < jnum; jj++) {
            int atj = list->firstneigh[ati][jj] & NEIGHMASK;
            double *jvec = x[atj]; int jtype = spmap[type[atj]];
            double rvec[3] = {jvec[0]-ivec[0], jvec[1]-ivec[1], jvec[2]-ivec[2]};
            double rijsq = rvec[0]*rvec[0] + rvec[1]*rvec[1] + rvec[2]*rvec[2];
            if (rijsq < cutoffsq) {
                double rij = sqrt(rijsq), rijinv = 1.0/rij;
                jlist_true[jnum_true] = atj;
                (this->*rbcalc)(rij);
                for (int k = 1; k < max_alpha_index_basic; k++) { dist_powers[k]=dist_powers[k-1]*rijinv;
                    coords_powers_x[k+1]=coords_powers_x[k]*rvec[0];
                    coords_powers_y[k+1]=coords_powers_y[k]*rvec[1];
                    coords_powers_z[k+1]=coords_powers_z[k]*rvec[2]; }
                for (int mu = 0; mu < max_mu; ++mu) { val_mu[mu]=0; der_mu[mu]=0;
                    int off=(type_central*species_count+jtype)*radial_func_count*rb_size+mu*rb_size;
                    for (int xi=0; xi<rb_size; ++xi) val_mu[mu]+=radial_coeffs[off+xi]*rb_vals[xi];
                    for (int xi=0; xi<rb_size; ++xi) der_mu[mu]+=radial_coeffs[off+xi]*rb_ders[xi]; }
                for (int i=0; i<alpha_index_basic_count; i++) {
                    int mu=alpha_index_basic[i].mu;
                    double val=val_mu[mu], der=der_mu[mu];
                    int k=nu_arr_list[i]; double rpowk=dist_powers[k];
                    val*=rpowk; der=der*rpowk - k*val*rijinv;
                    double pow0=coords_powers_x[alpha_index_basic[i].b1+1],
                           pow1=coords_powers_y[alpha_index_basic[i].b2+1],
                           pow2=coords_powers_z[alpha_index_basic[i].b3+1];
                    double mult0=pow0*pow1*pow2;
                    moment_vals[i] += val*mult0; mult0*=der*rijinv;
                    int off=i*jnum*3+3*jnum_true;
                    moment_jacobian[off+0]+=mult0*rvec[0]; moment_jacobian[off+1]+=mult0*rvec[1]; moment_jacobian[off+2]+=mult0*rvec[2];
                    moment_jacobian[off+0]+=alpha_index_basic[i].b1*val*coords_powers_x[alpha_index_basic[i].b1]*pow1*pow2;
                    moment_jacobian[off+1]+=alpha_index_basic[i].b2*val*coords_powers_y[alpha_index_basic[i].b2]*pow0*pow2;
                    moment_jacobian[off+2]+=alpha_index_basic[i].b3*val*coords_powers_z[alpha_index_basic[i].b3]*pow0*pow1;
                }
                jnum_true++;
            }
        }
        for (int i=0; i<alpha_index_times_count; i++) {
            double v0=moment_vals[alpha_index_times[i].l1], v1=moment_vals[alpha_index_times[i].l2];
            moment_vals[alpha_index_times[i].id]+=alpha_index_times[i].c*v0*v1; }

        // Cache full moment tree + scalar descriptors for LES (before backprop mutates)
        if (lr_les) {
            for (int a=0; a<alpha_scalar_moments; a++)
                site_les_desc[ati*alpha_scalar_moments+a]=moment_vals[alpha_moment_mapping[a]];
            memcpy(&cached_moment_vals[ati*alpha_moments_count], moment_vals,
                   alpha_moments_count*sizeof(double));
        }

        // BP1
        buff_site_energy+=specie_coeffs[type_central];
        memset(site_energy_ders_wrt_moments_,0,alpha_moments_count*sizeof(double));
        for (int i=0; i<alpha_scalar_moments; i++) { buff_site_energy+=moment_coeffs[i]*moment_vals[alpha_moment_mapping[i]];
            site_energy_ders_wrt_moments_[alpha_moment_mapping[i]]=moment_coeffs[i]; }
        // BP2
        for (int i=alpha_index_times_count-1; i>=0; i--) {
            double v0=moment_vals[alpha_index_times[i].l1], v1=moment_vals[alpha_index_times[i].l2];
            int mulc=alpha_index_times[i].c;
            site_energy_ders_wrt_moments_[alpha_index_times[i].l2]+=site_energy_ders_wrt_moments_[alpha_index_times[i].id]*mulc*v0;
            site_energy_ders_wrt_moments_[alpha_index_times[i].l1]+=site_energy_ders_wrt_moments_[alpha_index_times[i].id]*mulc*v1; }
        // BP3
        for (int i=0; i<alpha_index_basic_count; i++) { int o1=i*jnum*3;
            for (int j=0; j<jnum_true; j++) { int o2=j*3;
                buff_site_energy_ders[o2+0]+=site_energy_ders_wrt_moments_[i]*moment_jacobian[o1+o2+0];
                buff_site_energy_ders[o2+1]+=site_energy_ders_wrt_moments_[i]*moment_jacobian[o1+o2+1];
                buff_site_energy_ders[o2+2]+=site_energy_ders_wrt_moments_[i]*moment_jacobian[o1+o2+2]; }}
        // Extract
        en+=buff_site_energy; if (p_site_en) p_site_en[ati]+=buff_site_energy;
        for (int i=0; i<jnum_true; i++) { int off=i*3; int atj=jlist_true[i];
            f[ati][0]+=buff_site_energy_ders[off+0]; f[ati][1]+=buff_site_energy_ders[off+1]; f[ati][2]+=buff_site_energy_ders[off+2];
            f[atj][0]-=buff_site_energy_ders[off+0]; f[atj][1]-=buff_site_energy_ders[off+1]; f[atj][2]-=buff_site_energy_ders[off+2];
            if (vflag_global) {
                const double rx=x[atj][0]-x[ati][0], ry=x[atj][1]-x[ati][1], rz=x[atj][2]-x[ati][2];
                virial[0]-=buff_site_energy_ders[off]*rx;virial[1]-=buff_site_energy_ders[off+1]*ry;virial[2]-=buff_site_energy_ders[off+2]*rz;
                virial[3]-=buff_site_energy_ders[off+1]*rx;virial[4]-=buff_site_energy_ders[off+2]*rx;virial[5]-=buff_site_energy_ders[off+2]*ry;
            }
            if (vflag_atom) {
                const double rx=x[atj][0]-x[ati][0], ry=x[atj][1]-x[ati][1];
                const double rz=x[atj][2]-x[ati][2];
                vatom[ati][0]-=buff_site_energy_ders[off+0]*rx;
                vatom[ati][1]-=buff_site_energy_ders[off+1]*ry;
                vatom[ati][2]-=buff_site_energy_ders[off+2]*rz;
                vatom[ati][3]-=buff_site_energy_ders[off+1]*rx;
                vatom[ati][4]-=buff_site_energy_ders[off+2]*rx;
                vatom[ati][5]-=buff_site_energy_ders[off+2]*ry;
            }
        }
    }

    // ═══ LES ═══
    if (lr_les) {
        generate_k_vectors();
        for(active_channel=0;active_channel<sog_channels;++active_channel){
            for(int i=0;i<atom->nlocal;++i){
                int t=spmap[type[i]];double q=les_species_coeffs[active_channel*species_count+t];
                const double* d=&site_les_desc[i*alpha_scalar_moments];
                for(int a=0;a<alpha_scalar_moments;++a)q+=les_moment_coeffs[active_channel*alpha_scalar_moments+a]*d[a];
                latent_charges[i]=(use_sog?1:les_response_parity)*q;
                latent_channels[i][active_channel]=latent_charges[i];
            }
            ewald_lr_compute(x,f,type,atom->nlocal,atom->nghost,en);
        }
        active_channel=0;
        for(int i=0;i<atom->nlocal;++i)latent_charges[i]=latent_channels[i][0];
    }

    if (eflag_global) eng_vdwl += en;
    if (vflag_fdotr) virial_fdotr_compute();
}

void *PairMLIPOPTLES::extract(const char *name, int &dim)
{
    if (strcmp(name, "les_charge") == 0 && !use_sog) {
        dim = 1;
        return latent_charges;
    }
    if(strcmp(name,"sog_channels")==0){dim=0;return &sog_channels;}
    if(strcmp(name,"sog_latents")==0){dim=2;return latent_channels;}
    if (strcmp(name,"les_latent")==0) { dim=1;return latent_charges; }
    return nullptr;
}

// ═══════════════════════════════════════════════════════════════════════
//  Radial basis functions
// ═══════════════════════════════════════════════════════════════════════
void PairMLIPOPTLES::RB_CalcDer(const double &r) {
    double ksi=(2*r-plus_r)*inv_subr, rm=r-max_dist, rm2=rm*rm;
    rb_vals[0]=scaling*rm2; rb_vals[1]=scaling*ksi*rm2;
    rb_ders[0]=scaling2*rm; rb_ders[1]=scaling2*(ksi*rm+rm2*inv_subr);
    for (int i=2; i<rb_size; i++) { rb_vals[i]=2*ksi*rb_vals[i-1]-rb_vals[i-2]; rb_ders[i]=2*(mult*rb_vals[i-1]+ksi*rb_ders[i-1])-rb_ders[i-2]; }
}
void PairMLIPOPTLES::RB_CalcDerCos(const double &r) {
    double ksi=(2*r-plus_r)/sub_r, rc=0.5*(1.0+cos(r*irpi)), drc=-0.5*irpi*sin(r*irpi);
    rb_vals[0]=scaling*rc; rb_vals[1]=scaling*ksi*rc; rb_ders[0]=scaling*drc; rb_ders[1]=scaling*(mult*rc+ksi*drc);
    for (int i=2; i<rb_size; i++) { rb_vals[i]=2*ksi*rb_vals[i-1]-rb_vals[i-2]; rb_ders[i]=2*(mult*rb_vals[i-1]+ksi*rb_ders[i-1])-rb_ders[i-2]; }
}
void PairMLIPOPTLES::RB_CalcDerPoly5(const double &r) {
    double ksi=(2*r-plus_r)/sub_r, p=1.0-r/cutoff, q=p/p0, q2=q*q, q3=q*q2, q4=q*q3, q5=q*q4;
    double fc=6.0*q5-15.0*q4+10.0*q3, dfc=-(30.0*q4-60.0*q3+30.0*q2)*pp;
    rb_vals[0]=scaling*fc; rb_vals[1]=scaling*ksi*fc; rb_ders[0]=scaling*dfc; rb_ders[1]=scaling*(mult*fc+ksi*dfc);
    for (int i=2; i<rb_size; i++) { rb_vals[i]=2*ksi*rb_vals[i-1]-rb_vals[i-2]; rb_ders[i]=2*(mult*rb_vals[i-1]+ksi*rb_ders[i-1])-rb_ders[i-2]; }
}
void PairMLIPOPTLES::RB_CalcDerPoly3(const double &r) {
    double ksi=(2*r-plus_r)/sub_r, x=(r-min_dist)/sub_r, vx=1.0-x, vxx=vx*vx;
    double rc=vxx*(1+(2+rb_b)*x), drc=vx*(rb_b*(1.-3.0*x)-6.0*x)/sub_r;
    rb_vals[0]=scaling*rc; rb_vals[1]=scaling*ksi*rc; rb_ders[0]=scaling*drc; rb_ders[1]=scaling*(mult*rc+ksi*drc);
    for (int i=2; i<rb_size; i++) { rb_vals[i]=2*ksi*rb_vals[i-1]-rb_vals[i-2]; rb_ders[i]=2*(mult*rb_vals[i-1]+ksi*rb_ders[i-1])-rb_ders[i-2]; }
}
