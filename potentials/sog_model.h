// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#pragma once

#include <cmath>
#include <istream>
#include <ostream>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

namespace Potentials {
// Portable SOG v3/v4 metadata shared by IMRMLP and the standalone LAMMPS adapter.
// No dependency on Eigen, MPI, or IMRMLP runtime state.
struct SOGModel {
    std::vector<double> weights, log_bandwidths;
    int channels=1;
    int terms() const {return channels>0 ? weights.size()/channels : 0;}
    int version() const {return channels==1 ? 3 : 4;}
    double lower=-4.0, upper=1.0, dl=2.0, taper=.1;
    bool include_zero=true, onsite=true, smooth=true;
    static constexpr double pi=3.14159265358979323846;

    void validate() const {
        if (channels<1 || channels>16 || weights.empty() || weights.size()%channels || terms()>256 || weights.size()!=log_bandwidths.size()
            || !std::isfinite(lower) || !std::isfinite(upper) || lower<-30 || upper>30 || lower>=upper
            || !std::isfinite(dl) || dl<=0 || !std::isfinite(taper)
            || (smooth && !(taper>0 && taper<1)))
            throw std::runtime_error("Invalid SOG dimensions, bounds or reciprocal cutoff");
        for (size_t m=0;m<weights.size();++m)
            if (!std::isfinite(weights[m]) || !std::isfinite(log_bandwidths[m])
                || !(log_bandwidths[m]>lower && log_bandwidths[m]<upper))
                throw std::runtime_error("Invalid SOG weight or bandwidth");
    }
    static SOGModel read(std::istream& in) {
        SOGModel p;
        std::set<std::string> seen;
        std::string field;
        int terms=0, version=0;
        auto expect=[&](char c){char actual;if (!(in>>actual) || actual!=c) throw std::runtime_error("Malformed SOG field: "+field);};
        while (in>>field) {
            const bool related=field=="les_version" || field=="les_kernel" || field.rfind("sog_",0)==0
                || field=="les_dl" || field=="les_cutoff_mode" || field=="les_taper_fraction";
            if (field=="remove_self_interaction" || field=="les_charge_treatment")
                throw std::runtime_error("SOG rejects legacy self and charge fields");
            if (!related) continue;
            if (!seen.insert(field).second) throw std::runtime_error("Duplicate SOG field: "+field);
            expect('=');
            if (field=="sog_weights" || field=="sog_log_bandwidths") {
                if (terms<1 || terms>256) throw std::runtime_error("SOG arrays require preceding sog_terms in [1,256]");
                auto& v=field=="sog_weights" ? p.weights : p.log_bandwidths;
                v.resize(terms*p.channels);expect('{');
                for(int i=0;i<terms*p.channels;++i){if(i)expect(',');if(!(in>>v[i]) || !std::isfinite(v[i]))throw std::runtime_error("Invalid SOG array");}
                in>>std::ws;if(in.peek()==',')in.get();expect('}');
            } else if(field=="sog_terms") in>>terms;
            else if(field=="sog_log_bandwidth_min") in>>p.lower;
            else if(field=="sog_log_bandwidth_max") in>>p.upper;
            else if(field=="les_dl") in>>p.dl;
            else if(field=="les_taper_fraction") in>>p.taper;
            else if(field=="les_version" || field=="sog_channels") {
                int value=0;in>>value;
                if(field=="les_version") {version=value;if(value!=3 && value!=4)throw std::runtime_error("Unsupported SOG version or channel count");}
                else {if(value<1 || value>16 || (value>1 && (seen.count("sog_weights") || seen.count("sog_log_bandwidths"))))throw std::runtime_error("Unsupported SOG version or channel count/order");p.channels=value;}
            } else {
                std::string value;in>>value;
                if(field=="les_kernel") {if(value!="sog")throw std::runtime_error("Expected sog kernel");}
                else if(field=="sog_zero_mode") {if(value!="include" && value!="omit")throw std::runtime_error("Unknown SOG zero mode");p.include_zero=value=="include";}
                else if(field=="sog_self_mode") {if(value!="onsite" && value!="none")throw std::runtime_error("Unknown SOG self mode");p.onsite=value=="onsite";}
                else if(field=="les_cutoff_mode") {if(value!="smooth" && value!="reference_hard")throw std::runtime_error("Unknown SOG cutoff mode");p.smooth=value=="smooth";}
                else throw std::runtime_error("Unknown SOG field: "+field);
            }
            if(in.fail())throw std::runtime_error("Missing SOG field value: "+field);
        }
        for(const auto* f:{"les_version","les_kernel","sog_terms","sog_weights","sog_log_bandwidths",
                "sog_log_bandwidth_min","sog_log_bandwidth_max","sog_zero_mode","sog_self_mode"})
            if(!seen.count(f))throw std::runtime_error(std::string("Missing SOG field: ")+f);
        if(version!=p.version())throw std::runtime_error("Unsupported SOG version or channel count");
        p.validate();return p;
    }
    void write(std::ostream& out) const {
        validate();
        out<<"les_version = "<<version()<<"\nles_kernel = sog\nsog_channels = "<<channels<<"\nsog_terms = "<<terms()<<'\n'
           <<"sog_log_bandwidth_min = "<<lower<<"\nsog_log_bandwidth_max = "<<upper<<'\n'
           <<"sog_zero_mode = "<<(include_zero?"include":"omit")<<'\n'
           <<"sog_self_mode = "<<(onsite?"onsite":"none")<<'\n'
           <<"les_dl = "<<dl<<"\nles_cutoff_mode = "<<(smooth?"smooth":"reference_hard")<<'\n'
           <<"les_taper_fraction = "<<taper<<"\nsog_weights = { ";
        for(double w:weights)out<<w<<", ";out<<"}\nsog_log_bandwidths = { ";
        for(double u:log_bandwidths)out<<u<<", ";out<<"}\n";
    }
    void value(double x,double& g,double& gp,int channel=0) const {
        g=gp=0.;
        for(int m=channel*terms();m<(channel+1)*terms();++m){double a=std::exp(-2*log_bandwidths[m]),v=weights[m]*std::exp(-x*a);g+=v;gp-=a*v;}
    }
    double self(int channel=0) const {
        double c=0;for(int m=channel*terms();m<(channel+1)*terms();++m)c+=weights[m]*std::exp(3*log_bandwidths[m])/(8*std::pow(pi,1.5));return c;
    }
};
}
