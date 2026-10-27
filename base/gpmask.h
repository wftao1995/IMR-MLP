// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#ifndef _IMRMLP_GPMASK_H
#define _IMRMLP_GPMASK_H
#include <vector>
#include <algorithm>
// 0 or 1, 0 is unmasked, 1 is masked
class GPMask
{
protected:
    std::vector<int> mask;
    int masksize;

public:
    GPMask(int nsize) : mask(nsize, 0), masksize(nsize) {}
    GPMask() : mask(10, 0), masksize(10) {}
    GPMask(const GPMask &opt) : mask(opt.mask), masksize(opt.masksize) {}
    GPMask(GPMask &&opt) noexcept : mask(std::move(opt.mask)), masksize(opt.masksize)
    {
        opt.masksize = 0;
    }

    void mask_all()
    {
        std::fill(mask.begin(), mask.begin() + masksize, 1);
    }

    void revert()
    {
        for (int i = 0; i < masksize; ++i)
        {
            mask[i] = 0x0001 ^ mask[i];
        }
    }

    int size() const
    {
        return masksize;
    }

    int capacity() const
    {
        return mask.capacity();
    }

    void resize(int i)
    {
        if (i <= static_cast<int>(mask.size()))
            masksize = i;
        else
        {
            mask.resize(i, 0);
            masksize = i;
        }
    }

    void reset()
    {
        std::fill(mask.begin(), mask.begin() + masksize, 0);
    }

    int operator[](const int n) const
    {
        return mask[n];
    }

    int &operator[](const int n)
    {
        return mask[n];
    }
};

#endif
