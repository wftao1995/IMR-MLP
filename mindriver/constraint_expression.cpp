#include "constraint_expression.h"
#include "analyse/trajectory_transform.h"
#include <array>
#include <cctype>
#include <numbers>

namespace Constraints {
namespace {
struct Dual {
    double v;
    Eigen::VectorXd g;
    Dual(double x, int n) : v(x), g(Eigen::VectorXd::Zero(n)) {}
    Dual(double x, Eigen::VectorXd d) : v(x), g(std::move(d)) {}
};
Dual operator+(const Dual &a, const Dual &b) { return {a.v+b.v, (a.g+b.g).eval()}; }
Dual operator-(const Dual &a, const Dual &b) { return {a.v-b.v, (a.g-b.g).eval()}; }
Dual operator*(const Dual &a, const Dual &b) { return {a.v*b.v, (a.g*b.v+b.g*a.v).eval()}; }
Dual operator/(const Dual &a, const Dual &b) {
    if (b.v == 0) throw std::runtime_error("Constraint expression division by zero");
    return {a.v/b.v, ((a.g-b.g*(a.v/b.v))/b.v).eval()};
}
Dual unary(const Dual &a, double value, double derivative) { return {value, (derivative*a.g).eval()}; }
using Vec = std::array<Dual,3>;
Vec sub(const Vec &a, const Vec &b) { return {a[0]-b[0],a[1]-b[1],a[2]-b[2]}; }
Dual dot(const Vec &a, const Vec &b) { return a[0]*b[0]+a[1]*b[1]+a[2]*b[2]; }
Vec cross(const Vec &a, const Vec &b) {
    return {a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]};
}
Dual norm(const Vec &a) {
    auto s = dot(a,a);
    if (!(s.v > 0)) throw std::runtime_error("Constraint expression has a zero geometry vector");
    const double r = std::sqrt(s.v); return unary(s,r,.5/r);
}
Dual atan2(const Dual &y, const Dual &x) {
    const double r2=x.v*x.v+y.v*y.v;
    if (!(r2 > 1e-28)) throw std::runtime_error("Constraint dihedral has collinear arms");
    return {std::atan2(y.v,x.v), ((x.v*y.g-y.v*x.g)/r2).eval()};
}
}

struct Expression::Impl {
    struct Node {
        std::string op;
        double number = 0;
        std::vector<int> children, ids;
        std::array<Eigen::RowVector3i,3> images{Eigen::RowVector3i::Zero(),Eigen::RowVector3i::Zero(),Eigen::RowVector3i::Zero()};
        double branch = 0;
        bool constant = false, affine = false;
    };
    std::vector<Node> nodes;
    int root = -1, natoms = 0;

    struct Parser {
        Impl &out;
        const std::string &s;
        size_t pos = 0;
        [[noreturn]] void fail(const std::string &m) { throw std::invalid_argument("Constraint expression at " + std::to_string(pos) + ": " + m); }
        void space() { while (pos<s.size() && std::isspace(static_cast<unsigned char>(s[pos]))) ++pos; }
        bool take(char c) { space(); if (pos<s.size() && s[pos]==c) {++pos; return true;} return false; }
        void expect(char c) { if (!take(c)) fail(std::string("expected ")+c); }
        int add(Node n) { out.nodes.push_back(std::move(n)); return out.nodes.size()-1; }
        int binary(std::string op,int a,int b) {
            Node n; n.op=op; n.children={a,b}; const auto &x=out.nodes[a], &y=out.nodes[b];
            n.constant=x.constant && y.constant;
            n.affine = n.constant || ((op=="+" || op=="-") && x.affine && y.affine) ||
                (op=="*" && ((x.constant && y.affine)||(y.constant && x.affine))) ||
                (op=="/" && y.constant && x.affine);
            if (op=="^" && !y.constant) fail("power exponent must be constant");
            return add(std::move(n));
        }
        int expr() { int a=term(); for (;;) { if(take('+')) a=binary("+",a,term()); else if(take('-')) a=binary("-",a,term()); else return a; } }
        int term() { int a=sign(); for (;;) { if(take('*')) a=binary("*",a,sign()); else if(take('/')) a=binary("/",a,sign()); else return a; } }
        int sign() {
            if(take('+')) return sign();
            if(take('-')) { Node zero; zero.op="number"; zero.constant=zero.affine=true; int z=add(zero); return binary("-",z,sign()); }
            int a=atom(); if(take('^')) a=binary("^",a,sign()); return a;
        }
        int integer() {
            space(); size_t start=pos; while(pos<s.size() && std::isdigit(static_cast<unsigned char>(s[pos]))) ++pos;
            if(start==pos) fail("primitive indices must be literal nonnegative integers");
            try { return std::stoi(s.substr(start,pos-start)); } catch(...) { fail("index outside integer range"); }
        }
        int atom() {
            if(take('(')) { int a=expr(); expect(')'); return a; }
            space(); if(pos==s.size()) fail("missing operand");
            if(std::isdigit(static_cast<unsigned char>(s[pos])) || s[pos]=='.') {
                char *end=nullptr; const double v=std::strtod(s.c_str()+pos,&end);
                if(end==s.c_str()+pos || !std::isfinite(v)) fail("invalid constant");
                pos=end-s.c_str(); Node n; n.op="number"; n.number=v; n.constant=n.affine=true; return add(n);
            }
            size_t start=pos;
            while(pos<s.size() && (std::isalnum(static_cast<unsigned char>(s[pos])) || s[pos]=='_')) ++pos;
            const auto name=s.substr(start,pos-start); if(name.empty()) fail("expected function");
            if(name=="pi") { Node n; n.op="number"; n.number=std::numbers::pi; n.constant=n.affine=true; return add(n); }
            expect('('); Node n; n.op=name;
            if(name=="sqrt" || name=="exp" || name=="log" || name=="sin" || name=="cos") {
                n.children={expr()}; expect(')'); n.constant=out.nodes[n.children[0]].constant; n.affine=n.constant; return add(n);
            }
            int count=0;
            if(name=="x" || name=="y" || name=="z" || name=="lattice_length") count=1;
            else if(name=="distance" || name=="lattice_angle" || name=="lattice_ratio") count=2;
            else if(name=="angle") count=3;
            else if(name=="dihedral") count=4;
            else if(name!="volume") fail("unknown function "+name);
            for(int i=0;i<count;++i) { if(i) expect(','); n.ids.push_back(integer()); }
            expect(')');
            const bool lattice=name.starts_with("lattice_");
            for(int id:n.ids) if(id>=(lattice?3:out.natoms)) fail("primitive index out of range");
            if(!lattice && count>1) { auto ids=n.ids; std::sort(ids.begin(),ids.end()); if(std::adjacent_find(ids.begin(),ids.end())!=ids.end()) fail("geometry atoms must differ"); }
            n.affine=name=="x" || name=="y" || name=="z";
            return add(n);
        }
    };

    Vec position(const Atoms &a,int id,int n,const Eigen::RowVector3i &image) const {
        Vec p={Dual(0,n),Dual(0,n),Dual(0,n)};
        for(int c=0;c<3;++c) {
            p[c].v=a.pos_[id][c]; p[c].g[3*id+c]=1;
            for(int b=0;b<3;++b) { p[c].v+=image[b]*a.cell_(b,c); p[c].g[3*natoms+3*b+c]=image[b]; }
        }
        return p;
    }
    Vec lattice(const Atoms &a,int id,int n) const {
        Vec p={Dual(0,n),Dual(0,n),Dual(0,n)};
        for(int c=0;c<3;++c) {p[c].v=a.cell_(id,c); p[c].g[3*natoms+3*id+c]=1;} return p;
    }
    Dual evaluate(const Node &node,const Atoms &a,const std::vector<Dual> &vals,int n) const {
        const auto &op=node.op;
        if(op=="number") return {node.number,n};
        if(!node.children.empty()) {
            const auto &x=vals[node.children[0]];
            if(node.children.size()==2) {
                const auto &y=vals[node.children[1]];
                if(op=="+") return x+y; if(op=="-") return x-y; if(op=="*") return x*y; if(op=="/") return x/y;
                const double v=std::pow(x.v,y.v);
                return unary(x,v,y.v==0?0:y.v*std::pow(x.v,y.v-1));
            }
            if(op=="sqrt") { if(!(x.v>0)) throw std::runtime_error("sqrt constraint requires positive argument"); const double v=std::sqrt(x.v); return unary(x,v,.5/v); }
            if(op=="log") {if(!(x.v>0)) throw std::runtime_error("log constraint requires positive argument");return unary(x,std::log(x.v),1/x.v);}
            if(op=="exp") return unary(x,std::exp(x.v),std::exp(x.v));
            if(op=="sin") return unary(x,std::sin(x.v),std::cos(x.v));
            return unary(x,std::cos(x.v),-std::sin(x.v));
        }
        if(op=="x" || op=="y" || op=="z") return position(a,node.ids[0],n,Eigen::RowVector3i::Zero())[op[0]-'x'];
        if(op=="volume") return dot(lattice(a,0,n),cross(lattice(a,1,n),lattice(a,2,n)));
        if(op=="lattice_length") return norm(lattice(a,node.ids[0],n));
        if(op=="lattice_ratio") return norm(lattice(a,node.ids[0],n))/norm(lattice(a,node.ids[1],n));
        Vec u={Dual(0,n),Dual(0,n),Dual(0,n)}, v=u;
        if(op=="lattice_angle") {u=lattice(a,node.ids[0],n);v=lattice(a,node.ids[1],n);}
        else if(op=="distance") return norm(sub(position(a,node.ids[1],n,node.images[0]),position(a,node.ids[0],n,Eigen::RowVector3i::Zero())));
        else if(op=="angle") {u=sub(position(a,node.ids[0],n,node.images[0]),position(a,node.ids[1],n,Eigen::RowVector3i::Zero()));v=sub(position(a,node.ids[2],n,node.images[1]),position(a,node.ids[1],n,Eigen::RowVector3i::Zero()));}
        else {
            const auto p0=position(a,node.ids[0],n,node.images[0]),p1=position(a,node.ids[1],n,Eigen::RowVector3i::Zero());
            const auto p2=position(a,node.ids[2],n,node.images[1]),p3=position(a,node.ids[3],n,node.images[2]);
            const auto b0=sub(p0,p1), b1=sub(p2,p1), b2=sub(p3,p2);
            const auto n0=cross(b0,b1), n1=cross(b2,b1);
            auto phi=atan2(dot(cross(n0,n1),b1)/norm(b1),dot(n0,n1));
            phi.v+=2*std::numbers::pi*std::round((node.branch-phi.v)/(2*std::numbers::pi));
            return unary(phi,phi.v*180/std::numbers::pi,180/std::numbers::pi);
        }
        auto cos=dot(u,v)/(norm(u)*norm(v));
        if(!(std::abs(cos.v)<1)) throw std::runtime_error("Constraint angle has collinear arms");
        return unary(cos,std::acos(cos.v)*180/std::numbers::pi,-180/std::numbers::pi/std::sqrt(1-cos.v*cos.v));
    }
};

Expression::Expression(const std::string &source,const Atoms &initial) {
    if(source.empty() || source.size()>4096) throw std::invalid_argument("Constraint expression is empty or exceeds 4096 bytes");
    if(!initial.cartesian) throw std::invalid_argument("Constraint expressions bind Cartesian input positions");
    auto impl=std::make_shared<Impl>(); impl->natoms=initial.size();
    Impl::Parser parser{*impl,source}; impl->root=parser.expr(); parser.space();
    if(parser.pos!=source.size()) parser.fail("trailing expression text");
    const auto map=trajectory_periodic_coordinate_map(initial);
    Atoms image_frame = initial;
    const auto image=[&](int i,int j) -> Eigen::RowVector3i {
        const Eigen::RowVector3d translation = image_frame.get_rab(i,j) - (initial.pos_[j]-initial.pos_[i]);
        return (translation*map).array().round().cast<int>();
    };
    for(auto &node:impl->nodes) {
        if(node.op=="distance") node.images[0]=image(node.ids[0],node.ids[1]);
        else if(node.op=="angle") {node.images[0]=image(node.ids[1],node.ids[0]);node.images[1]=image(node.ids[1],node.ids[2]);}
        else if(node.op=="dihedral") {
            node.images[0]=image(node.ids[1],node.ids[0]);node.images[1]=image(node.ids[1],node.ids[2]);
            node.images[2]=node.images[1]+image(node.ids[2],node.ids[3]);
            node.branch=impl->evaluate(node,initial,{},3*initial.size()+9).v*std::numbers::pi/180;
        }
    }
    impl_=std::move(impl); (void)evaluate(initial);
}
Expression::Value Expression::evaluate(const Atoms &a) const {
    if(a.size()!=impl_->natoms || !a.cartesian) throw std::invalid_argument("Constraint expression atom layout changed");
    std::vector<Dual> vals; vals.reserve(impl_->nodes.size());
    for(const auto &node:impl_->nodes) vals.push_back(impl_->evaluate(node,a,vals,3*a.size()+9));
    const auto &v=vals[impl_->root];
    if(!std::isfinite(v.v) || !v.g.allFinite()) throw std::runtime_error("Constraint expression returned nonfinite value or derivative");
    Value out; out.value=v.v; out.positions.resize(a.size());
    for(int i=0;i<a.size();++i) out.positions[i]=v.g.segment<3>(3*i);
    for(int i=0;i<9;++i) out.cell(i/3,i%3)=v.g[3*a.size()+i];
    return out;
}
bool Expression::affine_positions() const {return impl_->nodes[impl_->root].affine;}
bool Expression::uses_volume() const {return std::any_of(impl_->nodes.begin(),impl_->nodes.end(),[](const auto &node){return node.op=="volume";});}
}
