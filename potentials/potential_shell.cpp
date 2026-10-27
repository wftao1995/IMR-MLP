// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#include "potential_shell.h"
#include <filesystem>
#include <exception>
#include <omp.h>
#include <utility>
#include <chrono>
#include <cerrno>
#include <cstring>
#include <spawn.h>
#include <sys/wait.h>
#include <unistd.h>
#include "utils/comm.h"
#include "utils/compressed_stream.h"

namespace fs = std::filesystem;

Potentials::PotShellCalc::PotShellCalc(GPSettings settings) {
    InitSettings();
    ApplySettings(settings);
    str_split(settings["dependencies"],gpm_dependencies_);
    auto capture = [](const std::string &path) {
        auto file = open_ifile(path);
        if (!file) throw std::runtime_error("Cannot read shell resource: " + path);
        std::ostringstream out; out << file->rdbuf();
        if (file->bad()) throw std::runtime_error("Cannot read shell resource: " + path);
        return out.str();
    };
    for (const auto &[key,path] : std::map<std::string,std::string>{{"pre",prescript},{"run",runscript},{"post",postscript}})
        if (!path.empty() && fs::is_regular_file(path)) gpm_scripts_[key] = capture(path);
    for (const auto &[key,path] : settings.settings) if (key.starts_with("resource:")) {
        fs::path relative(key.substr(9));
        if (relative.empty() || relative.is_absolute()) throw std::runtime_error("Shell resource name must be relative");
        for (const auto &part : relative) if (part == "..") throw std::runtime_error("Shell resource escapes job directory");
        gpm_inputs_[relative.string()] = capture(path);
    }

    if (!fs::exists(indexfile)) {
        std::cerr << "WARNING: PotShellCalc index file not found: " << indexfile << std::endl;
    }
    if (fs::exists(indexfile))
        iobj.readmap(indexfile);
    set_element_map(iobj.getmap());
}

int Potentials::PotShellCalc::exec_pre(std::string script) {
    if (script.empty()) return 0;
    int rstate = std::system(("bash " + script).c_str());
    if (rstate != 0) {
        std::cerr << "WARNING: PotShellCalc pre-script failed (exit " << rstate
                  << "): bash " << script << std::endl;
    }
    return rstate;
}

int Potentials::PotShellCalc::exec_cmd(std::string script) {
    int rstate = std::system(("bash " + script).c_str());
    if (rstate != 0) {
        std::cerr << "WARNING: PotShellCalc shell command failed (exit " << rstate
                  << "): bash " << script << std::endl;
    }
    return rstate;
}

int Potentials::PotShellCalc::exec_post(std::string script) {
    if (script.empty()) return 0;
    int rstate = std::system(("bash " + script).c_str());
    if (rstate != 0) {
        std::cerr << "WARNING: PotShellCalc post-script failed (exit " << rstate
                  << "): bash " << script << std::endl;
    }
    return rstate;
}

int Potentials::PotShellCalc::read_output(Atoms &atom, std::string path) {
    std::string outfile = path + "/" + read_outfile;
    if (!fs::exists(outfile)) {
        std::cerr << "ERROR: PotShellCalc output file not found: " << outfile << std::endl;
        return 1;
    }
    std::vector<Atoms> atoms;
    if (!iobj.read(atoms, outfile, read_outformat)) {
        std::cerr << "ERROR: PotShellCalc failed to read output file: " << outfile << std::endl;
        return 1;
    }
    if (atoms.empty()) {
        std::cerr << "ERROR: PotShellCalc output file is empty: " << outfile << std::endl;
        return 1;
    }
    Atoms output_atom = std::move(atoms.back());
    if (read_energy_sep) {
        std::string engpath = path + "/" + energy_file;
        if (!fs::exists(engpath)) {
            std::cerr << "ERROR: PotShellCalc energy file not found: " << engpath << std::endl;
            return 1;
        }
        auto ifs = open_ifile(engpath);
        if (!ifs) {
            std::cerr << "ERROR: PotShellCalc failed to open energy file: " << engpath << std::endl;
            return 1;
        }
        double energy;
        if (!(*ifs >> energy)) {
            std::cerr << "ERROR: PotShellCalc failed to read energy file: " << engpath << std::endl;
            return 1;
        }
        output_atom.energy_ = energy;
        output_atom.have_energy = true;
    }
    atom = std::move(output_atom);
    return 0;
}

/**
 * Must be one of POSCAR, xsf, cif, etc. that only handles one 
 * structure per file.
*/
void Potentials::PotShellCalc::write_input(Atoms &atom, std::string path) {
    for(const auto &[name,contents]:gpm_inputs_) {
        const auto target=fs::path(path)/name;fs::create_directories(target.parent_path());
        std::ofstream out(target,std::ios::binary);out.write(contents.data(),contents.size());
        if(!out)throw std::runtime_error("Cannot write embedded shell resource "+target.string());
    }
    // iobj.set(atom);
    if (!iobj.write_one(atom, path + "/" + write_posfile, write_posformat))
        throw std::runtime_error("PotShellCalc failed to write input: " + path + "/" + write_posfile);
}
// Scripts receive an absolute job-directory argument; the process working directory is unchanged.
// TODO: use asyncio
void Potentials::PotShellCalc::getEFS(Atoms &atom) {
    // get OpenMP thread id
    int myid = omp_get_thread_num();
    std::string path = "run-job-omp-thread-" + std::to_string(myid);
    #ifdef IMRMLP_MPI
    path += "mpi-rank-" + std::to_string(mpi.rank);
    #endif
    if (!fs::exists(path)) {
        bool created = fs::create_directory(path);
        if (!created) {
            throw std::runtime_error("PotShellCalc failed to create directory: " + path);
        }
    }
    // copy files 
    auto rootpath = fs::current_path();
    auto runpath = rootpath;
    runpath.append(path);
    // fs::current_path(runpath);
    const auto quote_path = [](const fs::path &value) {
        std::string quoted = "'";
        for (char ch : value.string())
            quoted += ch == '\'' ? "'\\''" : std::string(1, ch);
        return quoted + "'";
    };
    const auto runarg = quote_path(runpath);
    std::string exec_prescript, exec_postscript, exec_runscript;
    if (!prescript.empty()) {
        auto scriptpath = rootpath;
        scriptpath.append(prescript);
        exec_prescript = quote_path(scriptpath) + " " + runarg;
    }
    if (!postscript.empty()) {
        auto scriptpath = rootpath;
        scriptpath.append(postscript);
        exec_postscript = quote_path(scriptpath) + " " + runarg;
    }
    {
        auto scriptpath = rootpath;
        scriptpath.append(runscript);
        exec_runscript = quote_path(scriptpath) + " " + runarg;
    }

    std::exception_ptr io_error;
    #pragma omp critical
    {
        try { write_input(atom, runpath.string()); }
        catch (...) { io_error = std::current_exception(); }
    }
    if (io_error) std::rethrow_exception(io_error);
    if (exec_pre(exec_prescript) != 0)
        throw std::runtime_error("PotShellCalc pre-script failed: " + exec_prescript);
    if (exec_cmd(exec_runscript) != 0)
        throw std::runtime_error("PotShellCalc run-script failed: " + exec_runscript);
    #pragma omp critical
    {
        try {
            if (read_output(atom, runpath.string()) != 0)
                throw std::runtime_error("PotShellCalc failed to read output: " + runpath.string());
        } catch (...) { io_error = std::current_exception(); }
    }
    if (io_error) std::rethrow_exception(io_error);
    if (exec_post(exec_postscript) != 0)
        throw std::runtime_error("PotShellCalc post-script failed: " + exec_postscript);
    // fs::current_path(rootpath);
}

namespace {
// The script protocol is unchanged: bash SCRIPT ABS_JOBDIR, running relative
// resources from the submission directory. Only the spawned child changes cwd.
void run_async_script(const std::string &script, const std::string &directory,
                      const std::string &submission_directory, const char *stage) {
    if (script.empty()) throw std::runtime_error(std::string("PotShellCalc ")+stage+" script is empty");
    posix_spawn_file_actions_t actions;
    int error = posix_spawn_file_actions_init(&actions);
    if (error) throw std::runtime_error("PotShellCalc spawn actions: "+std::string(std::strerror(error)));
    error = posix_spawn_file_actions_addchdir_np(&actions,submission_directory.c_str());
    pid_t pid = -1;
    if (!error) {
        char name[] = "bash";
        char *arguments[] = {name,const_cast<char *>(script.c_str()),const_cast<char *>(directory.c_str()),nullptr};
        error = posix_spawnp(&pid,name,&actions,nullptr,arguments,environ);
    }
    posix_spawn_file_actions_destroy(&actions);
    if (error) throw std::runtime_error(std::string("PotShellCalc ")+stage+" spawn: "+std::strerror(error));
    int status = 0;
    while (waitpid(pid,&status,0) < 0) {
        if (errno == EINTR) continue;
        throw std::runtime_error("PotShellCalc waitpid: "+std::string(std::strerror(errno)));
    }
    if (!WIFEXITED(status) || WEXITSTATUS(status) != 0)
        throw std::runtime_error(std::string("PotShellCalc ")+stage+" script failed (status "+std::to_string(status)+"): "+script);
}
}

Potentials::ShellJob::~ShellJob() noexcept {
    try { if (completion_.valid()) completion_.wait(); } catch (...) {}
}

bool Potentials::ShellJob::ready() const {
    return completion_.wait_for(std::chrono::seconds(0)) == std::future_status::ready;
}

Atoms Potentials::ShellJob::result() const {
    return completion_.get(); // Copy the immutable stored result on every call.
}

std::shared_ptr<Potentials::ShellJob> Potentials::PotShellCalc::submit(const Atoms &atom) const {
    const fs::path submission_directory = fs::current_path();
    // GPIO constructors register options and initialize stream settings; prepare
    // the owned I/O object here, never from the background calculation.
    auto snapshot = std::make_shared<PotShellCalc>();
    snapshot->InitSettings();
    snapshot->ApplySettings(CurrentSettings());
    snapshot->iobj.ApplySettings(iobj.CurrentSettings());
    snapshot->iobj.setmap(iobj.getmap());
    snapshot->iobj.atomradii = iobj.atomradii;
    snapshot->iobj.atomic_vals = iobj.atomic_vals;
    auto absolute_script = [&](const std::string &script) {
        return script.empty() ? script : (submission_directory/fs::path(script)).lexically_normal().string();
    };
    snapshot->gpm_inputs_ = gpm_inputs_;
    snapshot->gpm_scripts_ = gpm_scripts_;
    snapshot->gpm_dependencies_ = gpm_dependencies_;
    snapshot->gpm_resources = gpm_resources;
    snapshot->prescript = absolute_script(prescript);
    snapshot->runscript = absolute_script(runscript);
    snapshot->postscript = absolute_script(postscript);
    std::string pattern = (submission_directory/"run-job-async-XXXXXX").string();
    if (!mkdtemp(pattern.data()))
        throw std::runtime_error("PotShellCalc cannot create asynchronous job directory: "+std::string(std::strerror(errno)));
    auto completion = std::async(std::launch::async,
        [snapshot,input=atom,directory=pattern,root=submission_directory.string()]() mutable {
            snapshot->write_input(input,directory);
            if (!snapshot->prescript.empty()) run_async_script(snapshot->prescript,directory,root,"pre");
            run_async_script(snapshot->runscript,directory,root,"run");
            Atoms output;
            if (snapshot->read_output(output,directory) != 0)
                throw std::runtime_error("PotShellCalc failed to read asynchronous output: "+directory);
            if (!snapshot->postscript.empty()) run_async_script(snapshot->postscript,directory,root,"post");
            return output;
        }).share();
    return std::shared_ptr<ShellJob>(new ShellJob(std::move(completion),pattern));
}
