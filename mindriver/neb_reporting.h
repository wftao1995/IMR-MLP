#pragma once
#include <chrono>
#include <cmath>
#include <iomanip>
#include <ostream>
#include <string>

namespace NEBReporting {
using Clock = std::chrono::steady_clock;
inline double seconds(Clock::time_point start) {
    return std::chrono::duration<double>(Clock::now()-start).count();
}
// No clock access when disabled. The destination must outlive the scope.
class Timer {
    double *value_;
    Clock::time_point start_;
public:
    explicit Timer(double *value): value_(value) { if(value_) start_=Clock::now(); }
    ~Timer() { if(value_) *value_+=seconds(start_); }
};
inline void string(std::ostream &out, const std::string &value) {
    out<<'"';
    for(unsigned char c:value) {
        if(c=='"' || c=='\\') out<<'\\'<<c;
        else if(c<32) {
            constexpr char digits[]="0123456789abcdef";
            out<<"\\u00"<<digits[c>>4]<<digits[c&15];
        } else out<<c;
    }
    out<<'"';
}
inline void number(std::ostream &out, double value) {
    if(std::isfinite(value)) out<<std::setprecision(17)<<value;
    else out<<"null";
}
}
