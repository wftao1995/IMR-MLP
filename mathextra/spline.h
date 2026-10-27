// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#ifndef IMRMLP_SPLINE_H
#define IMRMLP_SPLINE_H
#include <functional>
#include <cstring>
#include <cmath>
#include <vector>
#include <algorithm>
#include <map>
#include <limits>
#include <stdexcept>
#include <Eigen/Dense>
#include "tools/tools.h"

/**
 * Binary format:
 *   uint32(num) float64array(xval) float64array(yval)
 * One may use boost library whek seeking for thread safe impl.
 * **/
// real value
namespace Spline
{
    namespace detail {
    template<class G> std::vector<G> axis(G* data,int count) {
        if(!data || count<2) throw std::invalid_argument("Interpolation requires at least two nodes");
        std::vector<G> result(data,data+count);
        for(int i=0;i<count;++i)
            if(!std::isfinite(result[i]) || (i && !(result[i]>result[i-1])))
                throw std::invalid_argument("Interpolation nodes must be finite and strictly increasing");
        return result;
    }
    template<class G> std::vector<G> uniform_axis(G lo,G hi,int count) {
        if(count<2 || !std::isfinite(lo) || !std::isfinite(hi) || !(hi>lo) || !std::isfinite(hi-lo))
            throw std::invalid_argument("Invalid interpolation range or node count");
        std::vector<G> result(count);
        for(int i=0;i<count;++i)result[i]=lo+(hi-lo)*(G(i)/G(count-1));
        result.back()=hi;
        return axis(result.data(),count);
    }
    template<class G> int interval(const std::vector<G>& nodes,G x) {
        if(nodes.empty())throw std::logic_error("Interpolation is not initialized");
        if(!std::isfinite(x) || x<nodes.front() || x>nodes.back())
            throw std::out_of_range("Interpolation query is outside its closed domain");
        return std::min(int(std::upper_bound(nodes.begin(),nodes.end(),x)-nodes.begin())-1,int(nodes.size())-2);
    }
    template<class G> class LinearTable {
    protected:
        std::vector<G> xval,yval,kval;
        void install(std::vector<G> x,std::vector<G> y,G* slopes=nullptr) {
            if(y.size()!=x.size())throw std::invalid_argument("Interpolation value count mismatch");
            std::vector<G> k(x.size()-1);
            for(size_t i=0;i<y.size();++i)
                if(!std::isfinite(y[i]))throw std::invalid_argument("Nonfinite interpolation values");
            for(size_t i=0;i<k.size();++i) {
                k[i]=slopes ? slopes[i] : (y[i+1]-y[i])/(x[i+1]-x[i]);
                if(!std::isfinite(k[i]))throw std::invalid_argument("Nonfinite interpolation slope");
            }
            xval.swap(x);yval.swap(y);kval.swap(k);
        }
    public:
        int findindex(G x)const{return interval(xval,x);}
        G eval(G x)const {
            const int i=findindex(x);
            if(x==xval.back())return yval.back();
            return std::fma(kval[i],x-xval[i],yval[i]);
        }
        size_t size()const{return xval.size();}
    };
    }
    // Closed-domain, inclusive-endpoint piecewise linear interpolation. Failed
    // initialization leaves the last valid flat table intact; copies own data.
    template <typename G>
    class LinearInterpolationUnivar : public detail::LinearTable<G> {
    public:
        LinearInterpolationUnivar()=default;
        LinearInterpolationUnivar(G lo,G hi,int n,std::function<G(G)> fun){init(lo,hi,n,fun);}
        LinearInterpolationUnivar(G* x,G* y,int n,G* k=nullptr){init(x,y,n,k);}
        void init(G lo,G hi,int n,std::function<G(G)> fun) {
            auto x=detail::uniform_axis(lo,hi,n);std::vector<G> y(n);
            for(int i=0;i<n;++i)y[i]=fun(x[i]);
            this->install(std::move(x),std::move(y));
        }
        void init(G* x,G* y,int n,G* k=nullptr) {
            auto nodes=detail::axis(x,n);
            if(!y)throw std::invalid_argument("Null interpolation values");
            const G step=nodes[1]-nodes[0];
            for(int i=2;i<n;++i)
                if(std::abs((nodes[i]-nodes[i-1])-step)>
                    64*std::numeric_limits<G>::epsilon()*std::max(std::abs(nodes[i]),std::abs(step)))
                    throw std::invalid_argument("Uniform interpolation requires uniformly spaced nodes");
            this->install(std::move(nodes),std::vector<G>(y,y+n),k);
        }
    };

    // Build in an ordered standard container, then commit a flat sorted table.
    // Quarter/midpoint tests enforce sampled absolute error, not a rigorous
    // supremum bound for arbitrary callbacks. Queries use binary search.
    template <typename G>
    class LinearInterpolationAdp : public detail::LinearTable<G> {
    public:
        LinearInterpolationAdp()=default;
        LinearInterpolationAdp(G* x,G* y,int n,G* k=nullptr){init(x,y,n,k);}
        LinearInterpolationAdp(G lo,G hi,double dmin,double dmax,std::function<G(G)> fun,G prec=1e-3) {
            if(!init(lo,hi,dmin,dmax,fun,prec))
                throw std::runtime_error("Adaptive interpolation cannot meet sampled precision at minimum step");
        }
        void init(G* x,G* y,int n,G* k=nullptr) {
            auto nodes=detail::axis(x,n);
            if(!y)throw std::invalid_argument("Null interpolation values");
            this->install(std::move(nodes),std::vector<G>(y,y+n),k);
        }
        bool init(G lo,G hi,double dmin,double dmax,std::function<G(G)> fun,G prec=1e-3) {
            if(!std::isfinite(lo) || !std::isfinite(hi) || !(hi>lo) || !std::isfinite(hi-lo) ||
                !std::isfinite(dmin) || !std::isfinite(dmax) || !(dmin>0) || dmax<dmin ||
                !std::isfinite(prec) || !(prec>0))
                throw std::invalid_argument("Invalid adaptive interpolation settings");
            auto evaluate=[&](G x) {
                G y=fun(x);if(!std::isfinite(y))throw std::invalid_argument("Nonfinite adaptive callback");return y;
            };
            std::map<G,G> table;
            G left=lo,left_value=evaluate(lo);table.emplace(left,left_value);
            double step=dmax;
            while(left<hi) {
                const G right=std::min(hi,G(left+step));
                if(!(right>left))return false;
                const G width=right-left,right_value=evaluate(right);
                G error=0;
                for(G fraction:{G(.25),G(.5),G(.75)}) {
                    const G actual=evaluate(left+width*fraction);
                    const G linear=(G(1)-fraction)*left_value+fraction*right_value;
                    error=std::max(error,std::abs(actual-linear));
                }
                if(error>prec) {
                    if(width<=dmin || left+std::max(dmin,double(width)*.5)==left)return false;
                    step=std::max(dmin,double(width)*.5);continue;
                }
                table.emplace(right,right_value);left=right;left_value=right_value;
                if(table.size()>size_t(std::numeric_limits<int>::max()))throw std::length_error("Adaptive table too large");
                step=std::min(dmax,step*1.2);
            }
            std::vector<G> x,y;x.reserve(table.size());y.reserve(table.size());
            for(const auto& [node,value]:table){x.push_back(node);y.push_back(value);}
            this->install(std::move(x),std::move(y));return true;
        }
    };

    // Row-major grid values z[ix*ny+iy]; nonuniform strictly increasing axes
    // are accepted. Bilinear interpolation includes all four closed boundaries.
    template <typename G>
    class BiLinearInterpolationUnivar {
        std::vector<G> xval,yval,zval;
    public:
        BiLinearInterpolationUnivar()=default;
        BiLinearInterpolationUnivar(G* x,G* y,G* z,int nx,int ny){init(x,y,z,nx,ny);}
        BiLinearInterpolationUnivar(int nx,int ny,G xmin,G ymin,G xmax,G ymax,std::function<G(G,G)> fun) {
            init(nx,ny,xmin,ymin,xmax,ymax,fun);
        }
        void init(G* x,G* y,G* z,int nx,int ny) {
            auto xs=detail::axis(x,nx),ys=detail::axis(y,ny);
            if(!z || size_t(nx)>std::numeric_limits<size_t>::max()/size_t(ny))
                throw std::invalid_argument("Invalid bilinear values or dimensions");
            std::vector<G> zs(z,z+size_t(nx)*ny);
            for(G value:zs)if(!std::isfinite(value))throw std::invalid_argument("Nonfinite bilinear value");
            xval.swap(xs);yval.swap(ys);zval.swap(zs);
        }
        void init(int nx,int ny,G xmin,G ymin,G xmax,G ymax,std::function<G(G,G)> fun) {
            auto xs=detail::uniform_axis(xmin,xmax,nx),ys=detail::uniform_axis(ymin,ymax,ny);
            if(size_t(nx)>std::numeric_limits<size_t>::max()/size_t(ny))throw std::length_error("Bilinear grid too large");
            std::vector<G> zs(size_t(nx)*ny);
            for(int i=0;i<nx;++i)for(int j=0;j<ny;++j)zs[size_t(i)*ny+j]=fun(xs[i],ys[j]);
            init(xs.data(),ys.data(),zs.data(),nx,ny);
        }
        G eval(G x,G y)const {
            const auto i=detail::interval(xval,x),j=detail::interval(yval,y);
            const G tx=(x-xval[i])/(xval[i+1]-xval[i]),ty=(y-yval[j])/(yval[j+1]-yval[j]);
            const size_t n=yval.size();
            const G a=(1-tx)*zval[size_t(i)*n+j]+tx*zval[size_t(i+1)*n+j];
            const G b=(1-tx)*zval[size_t(i)*n+j+1]+tx*zval[size_t(i+1)*n+j+1];
            return (1-ty)*a+ty*b;
        }
    };

    template <typename G>
    class TriLinearInterpolationUnivar
    {
    };

    template <typename G>
    class CubicSplineUnivar
    {
    private:
        G *xval = nullptr;
        G *yval = nullptr;
        G *c0 = nullptr, *c1 = nullptr, *c2 = nullptr, *c3 = nullptr;
        G minval = 0, maxval = 0, delta = 0;
        int npoints = 0;

        void freemem() {
            if (xval) delete[] xval;
            if (yval) delete[] yval;
            if (c0) delete[] c0;
            if (c1) delete[] c1;
            if (c2) delete[] c2;
            if (c3) delete[] c3;
        }

    public:
        CubicSplineUnivar() {};
        CubicSplineUnivar(G *_xval, G *_yval, int _npoints) { init(_xval, _yval, _npoints); }
        ~CubicSplineUnivar() { freemem(); }

        void init(G *_xval, G *_yval, int _npoints) {
            freemem();
            npoints = _npoints;
            minval = _xval[0];
            maxval = _xval[_npoints - 1];
            delta = _xval[1] - _xval[0];
            xval = new G[npoints]; memcpy(xval, _xval, npoints * sizeof(G));
            yval = new G[npoints]; memcpy(yval, _yval, npoints * sizeof(G));
            c0  = new G[npoints - 1];
            c1  = new G[npoints - 1];
            c2  = new G[npoints - 1];
            c3  = new G[npoints - 1];
            // Natural cubic spline: solve tridiagonal for m (second derivatives at knots)
            std::vector<G> h(npoints - 1), b(npoints), u(npoints), v(npoints), m(npoints, 0);
            for (int i = 0; i < npoints - 1; ++i) h[i] = xval[i+1] - xval[i];
            for (int i = 1; i < npoints - 1; ++i) {
                b[i] = 6.0 * ((yval[i+1] - yval[i]) / h[i] - (yval[i] - yval[i-1]) / h[i-1]) / (h[i] + h[i-1]);
                u[i] = 2.0 * (h[i] + h[i-1]);
                v[i] = h[i];
            }
            for (int i = 2; i < npoints - 1; ++i) {
                double w = h[i-1] / u[i-1];
                u[i] -= w * v[i-1];
                b[i] -= w * b[i-1];
            }
            for (int i = npoints - 3; i >= 0; --i) {
                m[i+1] = (b[i+1] - v[i+1] * m[i+2]) / u[i+1];
            }
            for (int i = 0; i < npoints - 1; ++i) {
                c0[i] = yval[i];
                c1[i] = (yval[i+1] - yval[i]) / h[i] - h[i] * (2.0 * m[i] + m[i+1]) / 6.0;
                c2[i] = m[i] / 2.0;
                c3[i] = (m[i+1] - m[i]) / (6.0 * h[i]);
            }
        }

        G eval(G x) const {
            if (x < minval) x = minval;
            if (x > maxval) x = maxval;
            int idx = static_cast<int>((x - minval) / delta);
            if (idx >= npoints - 1) idx = npoints - 2;
            G dx = x - xval[idx];
            return c0[idx] + dx * (c1[idx] + dx * (c2[idx] + dx * c3[idx]));
        }

        G deriv(G x) const {
            if (x < minval) x = minval;
            if (x > maxval) x = maxval;
            int idx = static_cast<int>((x - minval) / delta);
            if (idx >= npoints - 1) idx = npoints - 2;
            G dx = x - xval[idx];
            return c1[idx] + dx * (2.0 * c2[idx] + dx * 3.0 * c3[idx]);
        }
    };

    // LAMMPS-compatible cubic spline coefficient builder.
    // Builds 7 coefficients per grid interval stored in an Eigen matrix.
    // f: function values (size n+1, indexed 1..n; f[0] is padding for 1-based LAMMPS convention)
    // delta: uniform grid spacing
    // spline: output matrix (n+1 rows x 7 columns), rows indexed 1..n
    //   Column 6: f[m] (function value)
    //   Column 5: first derivative estimate at knot
    //   Column 4: 3*(f[m+1]-f[m]) - 2*f'[m] - f'[m+1]
    //   Column 3: f'[m] + f'[m+1] - 2*(f[m+1]-f[m])
    //   Column 2: f'[m] / delta
    //   Column 1: 2*col4 / delta
    //   Column 0: 3*col3 / delta
    void build_cubic_spline(int n, double delta, const std::vector<double> &f,
                            Eigen::MatrixXd &spline);

    // Inline evaluation of LAMMPS-format cubic spline at position r.
    // rmin, inv_dr: grid parameters (r = rmin + m * dr; inv_dr = 1/dr)
    // spline: pre-built coefficient matrix (n+1 x 7, 1-indexed)
    // npoints: number of grid intervals
    // val: output function value; deriv: output derivative
    inline void eval_cubic_spline(double r, double rmin, double inv_dr,
                                  const Eigen::MatrixXd &spline, int npoints,
                                  double &val, double &deriv) {
        if (r <= rmin) { val = spline(1, 6); deriv = 0.0; return; }
        double p = (r - rmin) * inv_dr;
        int m = static_cast<int>(p);
        if (m >= npoints) {
            val = 0.0; deriv = 0.0; return;
        }
        p -= m;
        m += 1; // convert to 1-based
        p = p > 1.0 ? 1.0 : p;
        val = ((spline(m, 3) * p + spline(m, 4)) * p + spline(m, 5)) * p + spline(m, 6);
        deriv = (spline(m, 0) * p + spline(m, 1)) * p + spline(m, 2);
    }
};

#endif
