#ifndef GPML_UNITS_H
#define GPML_UNITS_H
#include <string>
#include <cmath>

// https://www.unitconverters.net/common-converters.html

/*
    Time: picosecond
    Velocity: Angstrom/picosecond
    Mass: grams/mole (AU)
    Energy: eV
    Force: eV/Angstrom
    Distance: Angstrom
    Temperature: Kelvin
    Pressure: bar
    Angle: REG
    --> Stress/Plusstress/virual:   eV in energy unit

*/

namespace GPUnits {
    // Fixed, explicitly typed conversion table. Canonical pressure is eV/A^3;
    // Atoms::stress_ is an energy-dimensional virial, not a pressure tensor.
    enum class Quantity { Length, Energy, Time, Mass, Angle, Force, Velocity,
                          Pressure, Virial, Hessian };
    Quantity quantity_from_string(const std::string &quantity);
    double conversion_factor(Quantity quantity, const std::string &from,
                             const std::string &to);
    std::string canonical_unit(Quantity quantity);
    double units_factor(const std::string &unitstr); // guess unit
    double units_factor(const std::string &typestr, const std::string &unitstr); // to standard interal units

    enum UType {
        ENERGY, FORCE, PRESS, LENGTH, TIME, ANGLE, MASS, STRESS,
        UNKNOWN
    };
    // energy
    enum UEnergy {
        JOUL, KJ,
        EV,
        HA,
        CAL, KCAL,
    };
    // distance
    enum ULength {
        ANG,
        NM,
    };
    // force
    enum UForce {
        EVPEA, // eV/A
        NEWTON, // N
        JOULPEMETER, // J/m
    };
    // pressure
    enum UPress {
        PA, KPA, GPA,// Pa
        BAR, KBAR, // bar
        ATM, // standard atomsphere
        EVPENM3 // eV/(nm^3)
    };
    // time
    enum UTime {
        PS, // picosecond
        NS, // nano second
        FS, // femo second
    };
    // angle
    enum UAngle {
        DEG,
        REG,
    };
    // mass
    enum UMass {
        KG, GRAM,
        AU, // atomic mass unit
        PROTON, // mass of proton
        NEUTROM, // mass of neutron
    };
    namespace Constants {
        // energy
        constexpr double ev2j = 1.60217733e-19; // eV to joules
        constexpr double j2ev = 6.241506363094e18; // joules to eV
        constexpr double ev2kj = 1.60217733e-22; // eV to joules
        constexpr double kj2ev = 6.241506363094e21; // joules to eV
        constexpr double ev2ha = 0.03674930813664888; // eV to hartree (matches tools/tools.h)
        constexpr double ha2ev = 27.211396641308; // hartree to eV (matches tools/tools.h)
        constexpr double deg2reg = M_PI / 180.0;
        constexpr double reg2deg = 180.0 / M_PI;
        // static double ev2kcal;
        // static double kcal2ev;
        // time
        constexpr double fs2ps = 1e-3;
        constexpr double ps2fs = 1e3;
        // pressure
        constexpr double bar2gpa = 1e-4;
        constexpr double gpa2bar = 1e4;
        // force

    };

    UType guess_unit(const std::string &unitstr);
    UType get_type(const std::string &typestr);
    int get_unit(const std::string &unitstr);
    double units_factor(int &id);
}







#endif
