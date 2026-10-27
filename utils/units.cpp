#include "units.h"
#include <stdexcept>

namespace GPUnits {
    #define RIGEST(a, c) if (a == instr) return c;
    UType guess_unit(const std::string &instr) {
        RIGEST("eV", ENERGY)
        RIGEST("Hartree", ENERGY)
        RIGEST("Angstrom", LENGTH)
        RIGEST("angstrom", LENGTH)
        RIGEST("A", LENGTH)
        RIGEST("degree", ANGLE)
        RIGEST("eV/A", FORCE)
        RIGEST("picosecond", TIME)
        return UNKNOWN;
    }

    UType get_type(const std::string &instr) {
        RIGEST("energy", ENERGY)
        RIGEST("force", FORCE)
        RIGEST("pressure", PRESS)
        RIGEST("press", PRESS)
        RIGEST("length", LENGTH)
        RIGEST("distance", LENGTH)
        RIGEST("time", TIME);
        RIGEST("angle", ANGLE)
        RIGEST("mass", MASS)
        RIGEST("weight", MASS)
        RIGEST("stress", STRESS)
        return UNKNOWN;
    }
    int get_unit(const std::string &instr) {
        RIGEST("eV", EV)
        RIGEST("Hartree", HA)
        RIGEST("Angstrom", ANG)
        RIGEST("angstrom", ANG)
        RIGEST("A", ANG)
        RIGEST("eV/A", EVPEA)
        RIGEST("picosecond", PS)
        RIGEST("degree", DEG)
        return UNKNOWN;
    }
    // to standard type
    double units_factor(const std::string &unitstr) {
        int type = get_unit(unitstr);
        UType utyp = guess_unit(unitstr);
        // Unit IDs are local to each physical quantity (e.g. JOUL and ANG
        // are both zero), so dispatch on the quantity before interpreting ID.
        switch (utyp) {
        case ENERGY:
            return type == HA ? Constants::ha2ev : 1.0;
        case ANGLE:
            return Constants::deg2reg;
        case LENGTH:
        case TIME:
        case FORCE:
            return 1.0;
        default:
            throw std::invalid_argument("Unknown unit: " + unitstr);
        };
    }
}

namespace GPUnits {
namespace {
    double canonical_factor(Quantity quantity, const std::string &unit) {
        // Keep the established Hartree and joule factors; precision migration
        // of existing codecs/constants is deliberately separate.
        const double bohr = 0.529177210903;
        const double avogadro = 6.02214076e23;
        const double amu_kg = 1.66053906660e-27;
        const auto energy = [&](const std::string &u) {
            if (u == "eV") return 1.;
            if (u == "Hartree" || u == "Ha") return Constants::ha2ev;
            if (u == "kJ/mol" || u == "kJ mol^-1") return Constants::kj2ev / avogadro;
            if (u == "kcal/mol") return 4.184 * Constants::kj2ev / avogadro;
            if (u == "J") return Constants::j2ev;
            throw std::invalid_argument("Unknown energy unit: " + u);
        };
        // Derived units are drawn from the fixed base tables below. This is
        // an exact spelling lookup, not a parser for arbitrary expressions.
        if (quantity == Quantity::Force || quantity == Quantity::Hessian ||
            quantity == Quantity::Pressure) {
            for (const char *e : {"eV", "Hartree", "Ha", "kJ/mol", "kcal/mol", "J"})
                for (const char *l : {"A", "Angstrom", "Bohr", "bohr", "nm", "m"}) {
                    const std::string base = std::string(e) + "/" + l;
                    const int power = quantity == Quantity::Force ? 1 :
                                      quantity == Quantity::Hessian ? 2 : 3;
                    if ((power == 1 && unit == base) ||
                        (power > 1 && (unit == base + "^" + std::to_string(power) ||
                                       unit == base + std::to_string(power)))) {
                        const double length = canonical_factor(Quantity::Length, l);
                        return energy(e) / std::pow(length, power);
                    }
                }
        }
        if (quantity == Quantity::Velocity)
            for (const char *l : {"A", "Angstrom", "Bohr", "bohr", "nm", "m"})
                for (const char *t : {"ps", "fs", "ns", "s", "atomic_time"})
                    if (unit == std::string(l) + "/" + t)
                        return canonical_factor(Quantity::Length, l) /
                               canonical_factor(Quantity::Time, t);
        switch (quantity) {
        case Quantity::Length:
            if (unit == "A" || unit == "Angstrom" || unit == "angstrom") return 1.;
            if (unit == "Bohr" || unit == "bohr") return bohr;
            if (unit == "nm") return 10.;
            if (unit == "m") return 1e10;
            break;
        case Quantity::Energy: case Quantity::Virial: return energy(unit);
        case Quantity::Time:
            if (unit == "ps" || unit == "picosecond") return 1.;
            if (unit == "fs") return Constants::fs2ps;
            if (unit == "ns") return 1e3;
            if (unit == "s") return 1e12;
            if (unit == "atomic_time") return 2.4188843265857e-5;
            break;
        case Quantity::Mass:
            if (unit == "amu" || unit == "u" || unit == "g/mol") return 1.;
            if (unit == "kg") return 1. / amu_kg;
            break;
        case Quantity::Angle:
            if (unit == "rad" || unit == "radian") return 1.;
            if (unit == "deg" || unit == "degree") return Constants::deg2reg;
            break;
        case Quantity::Force:
            if (unit == "N") return Constants::j2ev / 1e10;
            break;
        case Quantity::Velocity: break;
        case Quantity::Pressure:
            if (unit == "eV/A^3" || unit == "eV/A3" || unit == "eV A^-3") return 1.;
            if (unit == "Pa") return Constants::j2ev * 1e-30;
            if (unit == "bar") return Constants::j2ev * 1e-25;
            if (unit == "GPa") return Constants::j2ev * 1e-21;
            break;
        case Quantity::Hessian: break;
        }
        throw std::invalid_argument("Unit '" + unit + "' is not supported for " + canonical_unit(quantity));
    }
}

Quantity quantity_from_string(const std::string &quantity) {
    if (quantity == "length" || quantity == "distance") return Quantity::Length;
    if (quantity == "energy") return Quantity::Energy;
    if (quantity == "time") return Quantity::Time;
    if (quantity == "mass") return Quantity::Mass;
    if (quantity == "angle") return Quantity::Angle;
    if (quantity == "force") return Quantity::Force;
    if (quantity == "velocity") return Quantity::Velocity;
    if (quantity == "pressure") return Quantity::Pressure;
    if (quantity == "virial") return Quantity::Virial;
    if (quantity == "hessian" || quantity == "hvp_target") return Quantity::Hessian;
    throw std::invalid_argument("Unknown physical quantity: " + quantity);
}

std::string canonical_unit(Quantity quantity) {
    switch (quantity) {
    case Quantity::Length: return "A";
    case Quantity::Energy: case Quantity::Virial: return "eV";
    case Quantity::Time: return "ps";
    case Quantity::Mass: return "amu";
    case Quantity::Angle: return "rad";
    case Quantity::Force: return "eV/A";
    case Quantity::Velocity: return "A/ps";
    case Quantity::Pressure: return "eV/A^3";
    case Quantity::Hessian: return "eV/A^2";
    }
    throw std::invalid_argument("Unknown physical quantity");
}

double conversion_factor(Quantity quantity, const std::string &from, const std::string &to) {
    return canonical_factor(quantity, from) / canonical_factor(quantity, to);
}

// Previously declared typed legacy overload. Preserve the one-argument API's
// historical values; new explicit pressure conversions use conversion_factor.
double units_factor(const std::string &typestr, const std::string &unitstr) {
    if (typestr == "stress")
        throw std::invalid_argument("Ambiguous stress: specify pressure or energy-dimensional virial");
    const auto quantity = quantity_from_string(typestr);
    return conversion_factor(quantity, unitstr, canonical_unit(quantity));
}
}
