#include "gpio_session.h"
#include "format_detection.h"
#include "utils/compressed_stream.h"
#include <stdexcept>

namespace {
std::string read_format_name(GPIO::supported_format value) {
    switch(value) {
    case GPIO::EXTXYZ:return "extxyz";
    case GPIO::MLPCFG:return "mlpcfg";
    case GPIO::MLIPBIN:return "bin";
    case GPIO::N2P2DAT:return "n2p2";
    case GPIO::XSF:return "xsf";
    case GPIO::OUTCAR:return "outcar";
    case GPIO::XDATCAR:return "xdatcar";
    case GPIO::XDATCAR6:return "xdatcar6";
    default:throw std::invalid_argument("Independent streaming readers support EXTXYZ, CFG, BIN, N2P2, XSF, OUTCAR and XDATCAR/6");
    }
}
}

GPIOReader::GPIOReader(const GPIO& configuration,const std::string& filename,
    std::string format,bool detect_content,std::optional<GPUnits::SourceUnits> source_units)
    :io_(configuration.new_session_state()),input_(open_seq_ifile(filename)),
     format_(std::move(format)),source_units_(std::move(source_units)) {
    if(format_.empty() && !detect_content)format_=read_format_name(GPIO::guess_format(filename));
    initialize(detect_content);
}

GPIOReader::GPIOReader(const GPIO& configuration,std::unique_ptr<std::istream> input,
    std::string format,bool detect_content,std::optional<GPUnits::SourceUnits> source_units)
    :io_(configuration.new_session_state()),input_(std::move(input)),format_(std::move(format)),
     source_units_(std::move(source_units)) { initialize(detect_content); }

void GPIOReader::initialize(bool detect_content) {
    if(!input_)throw std::runtime_error("Cannot open streaming input");
    if(format_.empty() && detect_content) {
        auto probed=GPIOFormat::probe(std::move(input_));
        input_=std::move(probed.input);format_=probed.detection.format;
        if(format_.empty())throw std::invalid_argument(probed.detection.candidates.empty()
            ? "Cannot detect input content; specify a format" : "Ambiguous input content; specify a format");
    }
    format_=read_format_name(GPIO::format2enum(format_));
    if(io_->read_startfrom<0 || io_->read_endto<io_->read_startfrom || io_->read_irate<=0)
        throw std::invalid_argument("Invalid session frame range");
    if(source_units_) {
        if(format_!="extxyz")throw std::invalid_argument("Input unit overrides are supported only for EXTXYZ");
        Atoms empty;GPUnits::canonicalize_atoms(empty,*source_units_);
    }
    if(!io_->seq_read_init(*input_,format_))throw std::runtime_error("Cannot initialize streaming input");
}

std::optional<Atoms> GPIOReader::next() {
    std::lock_guard lock(mutex_);
    if(!input_)return std::nullopt;
    try {
        Atoms atom;
        while(io_->seq_read(*input_,atom)) {
            if(atom.id>io_->read_endto) {close_unlocked();return std::nullopt;}
            if(atom.id<io_->read_startfrom || (atom.id-io_->read_startfrom)%io_->read_irate)continue;
            if(source_units_)GPUnits::canonicalize_atoms(atom,*source_units_);
            const int last=io_->read_endto-(io_->read_endto-io_->read_startfrom)%io_->read_irate;
            if(atom.id==last)close_unlocked();
            return atom;
        }
        if(input_->bad())throw std::runtime_error("Streaming input read failed");
        close_unlocked();return std::nullopt;
    } catch(...) {close_unlocked();throw;}
}

void GPIOReader::close_unlocked() {io_->clean_state();input_.reset();}
void GPIOReader::close() {std::lock_guard lock(mutex_);close_unlocked();}

GPIOWriter::GPIOWriter(const GPIO& configuration,const std::string& filename,std::string format)
    :io_(configuration.new_session_state()),filename_(filename),format_(std::move(format)) {
    if(format_.empty())format_=read_format_name(GPIO::guess_format(filename));
    const auto value=GPIO::format2enum(format_);
    if(value!=GPIO::EXTXYZ && value!=GPIO::MLPCFG && value!=GPIO::MLIPBIN && value!=GPIO::N2P2DAT &&
       value!=GPIO::XDATCAR && value!=GPIO::XDATCAR6)
        throw std::invalid_argument("Independent streaming writers support EXTXYZ, CFG, BIN, N2P2 and XDATCAR/6");
    if(io_->append && value!=GPIO::EXTXYZ && value!=GPIO::MLPCFG && value!=GPIO::MLIPBIN)
        throw std::invalid_argument("Session append supports EXTXYZ, CFG and BIN only");
    if(io_->outrate<=0 || io_->out_startfrom<0 || io_->out_endto<io_->out_startfrom)
        throw std::invalid_argument("Invalid session output range");
}

GPIOWriter::~GPIOWriter() {try {close();}catch(...) {}}

void GPIOWriter::write(const Atoms& source) {
    std::lock_guard lock(mutex_);
    if(closed_)throw std::logic_error("Cannot write to a closed session");
    const auto index=count_++;
    if(index<static_cast<std::size_t>(io_->out_startfrom) || index>static_cast<std::size_t>(io_->out_endto) ||
       (index-io_->out_startfrom)%io_->outrate)return;
    try {
        Atoms atom=source;if(!atom.cartesian)atom.to_cartesian();
        if(!output_) {
            output_=open_ofile(filename_,io_->append,GPIO::format2enum(format_)==GPIO::MLIPBIN);
            if(!output_)throw std::runtime_error("Cannot open streaming output: "+filename_);
            if(!io_->seq_write_init(*output_,format_))throw std::runtime_error("Cannot initialize streaming output");
        }
        if(!io_->seq_write(*output_,atom) || !*output_)
            throw std::runtime_error("Cannot write streaming output: "+filename_);
    } catch(...) {try {close_unlocked();}catch(...) {}throw;}
}

void GPIOWriter::close_unlocked() {
    closed_=true;
    if(output_ && !close_ofile(output_))throw std::runtime_error("Cannot close streaming output: "+filename_);
}
void GPIOWriter::close() {std::lock_guard lock(mutex_);close_unlocked();}
