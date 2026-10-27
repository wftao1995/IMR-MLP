#pragma once
#include "gpio.h"
#include "atoms_units.h"
#include <mutex>
#include <optional>

// Each session owns its parser, settings bindings and stream. Configure the
// source GPIO before opening sessions; do not mutate that configuration while
// another thread snapshots it. next/write/close on one session serialize.
class GPIOReader {
    std::unique_ptr<GPIO> io_;
    std::unique_ptr<std::istream> input_;
    std::string format_;
    std::optional<GPUnits::SourceUnits> source_units_;
    std::mutex mutex_;
    void initialize(bool detect_content);
    void close_unlocked();
public:
    GPIOReader(const GPIO& configuration,const std::string& filename,
               std::string format="",bool detect_content=false,
               std::optional<GPUnits::SourceUnits> source_units=std::nullopt);
    GPIOReader(const GPIO& configuration,std::unique_ptr<std::istream> input,
               std::string format,bool detect_content=false,
               std::optional<GPUnits::SourceUnits> source_units=std::nullopt);
    std::optional<Atoms> next();
    void close();
    const std::string& format() const { return format_; }
};

class GPIOWriter {
    std::unique_ptr<GPIO> io_;
    std::unique_ptr<std::ostream> output_;
    std::string filename_,format_;
    bool closed_=false;
    std::size_t count_=0;
    std::mutex mutex_;
    void close_unlocked();
public:
    GPIOWriter(const GPIO& configuration,const std::string& filename,std::string format="");
    ~GPIOWriter();
    void write(const Atoms& atom);
    void close();
};
