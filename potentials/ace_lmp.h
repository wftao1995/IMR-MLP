// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#ifndef IMRMLP_POT_ACE_LMP_H
#define IMRMLP_POT_ACE_LMP_H


#ifdef WITH_ACELMP
#include "potential.h"

namespace Potentials {
    class PotentialACELmp : public AnyPotential {
        friend struct GPMCodecAccess;
        std::string serialized_basis_;
    private:
        struct ACEImpl *aceimpl;
        std::vector<std::string> elenames;
        bool recursive = true;
        double cutlim;

        double **buffer_x;
        double *buffer_x_internal;
        int *buffer_type;
        int *buffer_jlist;
        int buffer_size = 10;
        double allocate_ampifer = 1.2; 
    public:
        PotentialACELmp(){};
        PotentialACELmp(const std::string &, std::vector<std::string> &, bool recursive = true);
        ~PotentialACELmp();
        virtual void Load(const std::string &potname);
        virtual void getEFS(Atoms &atom);
    };
};


#endif

#endif
