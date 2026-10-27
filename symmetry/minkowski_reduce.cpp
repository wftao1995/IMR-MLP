// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#include "minkowski_reduce.h"
#include "tools/tools.h"

namespace Symmetry3D
{
    bool gaussian_reduce_two_vectors(Eigen::RowVector3d &U, Eigen::RowVector3d &V, double prec = 1e-4, int max_iter = 10)
    {
        if (U.norm() > V.norm() - prec)
        {
            std::swap(U, V);
        }
        Eigen::RowVector3d R;
        for (int i = 0; i < max_iter; ++i)
        {
            R = V - round(U.dot(V) / U.dot(U) + 1e-10) * U;
            V = U;
            U = R;
            if (U.norm() > V.norm() - prec)
            {
                std::swap(U, V);
                return true;
            }
        }
        return false;
    }
    bool reduce_C_in_ABC(Eigen::Matrix3d &basis, double prec = 1e-4)
    {
        Eigen::RowVector3d A, B, C, pA, pB, pC;
        Eigen::Matrix3d oldbasis = basis;
        A = basis.row(0);
        B = basis.row(1);
        C = basis.row(2);
        pA = A;
        pB = B;
        pC = C;
        gaussian_reduce_two_vectors(A, B, prec);
        Eigen::RowVector3d cpdAB = A.cross(B) / (A.cross(B)).norm();
        Eigen::RowVector3d T = C - C.dot(cpdAB) * cpdAB;
        if (fabs(T.dot(A.cross(B))) > prec)
        {
            _DEBUG_OUTPUT_("ERR1");
            return false;
        }
        basis.row(0) = A;
        basis.row(1) = B;
        Eigen::Matrix3d rbasis = basis.inverse();
        Eigen::RowVector3d LC = T * rbasis;
        for (int i = 0; i < 3; ++i)
        {
            LC(i) = floor(LC(i) + prec);
        }
        //_DEBUG_OUTPUT_(LC);
        std::array<Eigen::RowVector3d, 4> corners;
        for (int i = 0; i < 4; ++i)
        {
            corners[i] << 0, 0, 0;
        }
        corners[1](0) = 1;
        corners[2](1) = 1;
        corners[3](0) = 1;
        corners[3](1) = 1;
        std::array<double, 4> dist;
        Eigen::RowVector3d tmp1, tmp2;
        int idx = -1;
        double mindist = 9e10;
        for (int i = 0; i < 4; ++i)
        {
            //tmp1 = corners[i] + LC;
            //tmp2 = T - (corners[i] + LC) * basis;
            dist[i] = (T - (corners[i] + LC) * basis).norm();
            if (mindist > dist[i])
            {
                mindist = dist[i];
                idx = i;
            }
        }
        //_DEBUG_OUTPUT_(idx);
        //_DEBUG_VECTOR_(dist);
        if (idx == -1) // ????
        {
            _DEBUG_OUTPUT_("ERR2");
            return false;
        }
        else
        {
            tmp1 = corners[idx] + LC;
            tmp2 = tmp1 * basis;
            C = C - tmp2;
        }
        //_DEBUG_OUTPUT_(C);
        basis.row(0) = A;
        basis.row(1) = B;
        basis.row(2) = C;
        rbasis = basis.inverse();
        Eigen::Matrix3d temp = rbasis.transpose() * oldbasis.transpose();
        //_DEBUG_OUTPUT_(temp);
        for (int i = 0; i < 3; ++i)
        {
            for (int j = 0; j < 3; ++j)
            {
                if (fabs(temp(i, j) - round(temp(i, j))) > prec)
                {
                    _DEBUG_OUTPUT_("ERR3");
                    return false;
                }
            }
        }
        //_DEBUG_OUTPUT_("T");
        return true;
    }
    bool minkowski_conditions_check(const Eigen::Matrix3d &basis, double prec = 1e-4)
    {
        Eigen::RowVector3d b1, b2, b3;
        b1 = basis.row(0);
        b2 = basis.row(1);
        b3 = basis.row(2);
        bool minkowski_check = true;
        double nb1, nb2, nb3;
        nb1 = b1.norm();
        nb2 = b2.norm();
        nb3 = b3.norm();

        if (nb1 > nb2 + prec || nb2 > nb3 + prec)
        {
            //1 2
            return false;
        }
        else if (nb2 > (b1 + b2).norm() + prec)
        {
            //3
            return false;
        }
        else if (nb2 > (b1 - b2).norm() + prec)
        {
            //4
            return false;
        }
        else if (nb3 > (b1 + b3).norm() + prec)
        {
            //5
            return false;
        }
        else if (nb3 > (b3 - b1).norm() + prec)
        {
            //6
            return false;
        }
        else if (nb3 > (b2 + b3).norm() + prec)
        {
            //7
            return false;
        }
        else if (nb3 > (b3 - b2).norm() + prec)
        {
            //8
            return false;
        }
        else if (nb3 > (b1 + b2 + b3).norm() + prec)
        {
            //9
            return false;
        }
        else if (nb3 > (b1 - b2 + b3).norm() + prec)
        {
            //10
            return false;
        }
        else if (nb3 > (b1 + b2 - b3).norm() + prec)
        {
            //11
            return false;
        }
        else if (nb3 > (b1 - b2 - b3).norm() + prec)
        {
            //12
            return false;
        }
        else
        {
            return true;
        }
        return true;
    }
    bool minkowski_reduce_basis(Eigen::Matrix3d &basis, double prec, int max_iter = 10)
    {
        // ignore this for faster run, assume true
        // if(fabs(basis.determinant()) < prec)
        // {
        //     return false;
        // }
        std::array<Eigen::RowVector3d, 3> vas;
        //Eigen::Matrix3d temp = basis;
        for (int ii = 0; ii < max_iter; ii++)
        {
            for (int i = 0; i < 3; ++i)
            {
                vas[i] = basis.row(i);
            }
            std::sort(vas.begin(), vas.end(),
                      [](Eigen::RowVector3d a, Eigen::RowVector3d b) -> bool { return a.norm() < b.norm(); });
            for (int i = 0; i < 3; ++i)
            {
                basis.row(i) = vas[i];
            }
            //_DEBUG_OUTPUT_(basis);
            reduce_C_in_ABC(basis, prec);
            if (basis.row(2).norm() >= basis.row(1).norm() - prec)
            {
                break;
            }
        }
        if (!minkowski_conditions_check(basis, prec))
        {
            return false;
        }
        if (basis.determinant() < 0)
        {
            //std::swap(basis.row(1), basis.row(2));
            Eigen::RowVector3d tp = basis.row(1);
            basis.row(1) = basis.row(2);
            basis.row(2) = tp;
        }
        return true;
    }
}
