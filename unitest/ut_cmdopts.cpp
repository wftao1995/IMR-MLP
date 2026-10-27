// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#include <catch2/catch_all.hpp>
#include "utils/cmdopts.h"
#include "utils/gputils.h"
#include <climits>
#ifdef USE_OMP
#include <omp.h>
#endif

using namespace CMDOption;

TEST_CASE("settings num_threads updates OpenMP runtime", "[openmp][settings]")
{
    const int saved_threads = omp_num_threads;
    struct RestoreThreads {
        int value;
        ~RestoreThreads()
        {
            omp_num_threads = value;
            global_omp_init();
        }
    } restore{saved_threads};

    CommandLineOptions cmd;
    GPSettings settings;
    settings.Modify("num_threads", "2");
    apply_settings_to_cmdopts(cmd, settings);

    REQUIRE(omp_num_threads == 2);
#ifdef USE_OMP
    REQUIRE(omp_get_max_threads() == 2);
#endif
}

// Helper: build argc/argv from a vector of strings
struct TestArgs {
    std::vector<std::string> args_storage;
    std::vector<char*> argv;
    int argc;

    explicit TestArgs(std::initializer_list<std::string> il) {
        args_storage.push_back("test_binary");
        for (auto& s : il) args_storage.push_back(s);
        argc = static_cast<int>(args_storage.size());
        argv.reserve(args_storage.size());
        for (auto& s : args_storage) argv.push_back(s.data());
    }
    explicit TestArgs(const std::vector<std::string>& args) {
        args_storage.push_back("test_binary");
        for (auto& s : args) args_storage.push_back(s);
        argc = static_cast<int>(args_storage.size());
        argv.reserve(args_storage.size());
        for (auto& s : args_storage) argv.push_back(s.data());
    }
};

TEST_CASE("explicit command-line value takes precedence over settings",
          "[settings][precedence]")
{
    CommandLineOptions cmd;
    cmd.set_helponexit(false);
    int value = 1;
    cmd.addop(value, "value", "", "test value");
    TestArgs args{"--value=7"};
    cmd.phrase_options(args.argc, args.argv.data());

    GPSettings settings;
    settings.Modify("value", "3");
    apply_settings_to_cmdopts(cmd, settings);

    REQUIRE(value == 7);
    REQUIRE(cmd.was_provided("value"));
}

// ==========================================================================
// 1. Basic scalar types via --tag=val (narg=1)
// ==========================================================================

TEST_CASE("=int positive",      "[equals][int]") {
    CommandLineOptions cmd; cmd.set_helponexit(false);
    int v=0; cmd.addop(v,"opt","","");
    TestArgs ta{"--opt=42"}; cmd.phrase_options(ta.argc, ta.argv.data());
    REQUIRE(v == 42);
}
TEST_CASE("=int zero",          "[equals][int]") {
    CommandLineOptions cmd; cmd.set_helponexit(false);
    int v=1; cmd.addop(v,"opt","","");
    TestArgs ta{"--opt=0"}; cmd.phrase_options(ta.argc, ta.argv.data());
    REQUIRE(v == 0);
}
TEST_CASE("=int negative",      "[equals][int]") {
    CommandLineOptions cmd; cmd.set_helponexit(false);
    int v=0; cmd.addop(v,"opt","","");
    TestArgs ta{"--opt=-5"}; cmd.phrase_options(ta.argc, ta.argv.data());
    REQUIRE(v == -5);
}
TEST_CASE("=int max",           "[equals][int]") {
    CommandLineOptions cmd; cmd.set_helponexit(false);
    int v=0; cmd.addop(v,"opt","","");
    TestArgs ta{std::string("--opt=") + std::to_string(INT_MAX)}; cmd.phrase_options(ta.argc, ta.argv.data());
    REQUIRE(v == INT_MAX);
}

TEST_CASE("=double positive",   "[equals][double]") {
    CommandLineOptions cmd; cmd.set_helponexit(false);
    double v=0; cmd.addop(v,"opt","","");
    TestArgs ta{"--opt=3.14"}; cmd.phrase_options(ta.argc, ta.argv.data());
    REQUIRE(v == Catch::Approx(3.14));
}
TEST_CASE("=double zero",       "[equals][double]") {
    CommandLineOptions cmd; cmd.set_helponexit(false);
    double v=1; cmd.addop(v,"opt","","");
    TestArgs ta{"--opt=0.0"}; cmd.phrase_options(ta.argc, ta.argv.data());
    REQUIRE(v == Catch::Approx(0.0));
}
TEST_CASE("=double negative",   "[equals][double]") {
    CommandLineOptions cmd; cmd.set_helponexit(false);
    double v=0; cmd.addop(v,"opt","","");
    TestArgs ta{"--opt=-2.718"}; cmd.phrase_options(ta.argc, ta.argv.data());
    REQUIRE(v == Catch::Approx(-2.718));
}
TEST_CASE("=double sci 1e-3",   "[equals][double]") {
    CommandLineOptions cmd; cmd.set_helponexit(false);
    double v=0; cmd.addop(v,"opt","","");
    TestArgs ta{"--opt=1e-3"}; cmd.phrase_options(ta.argc, ta.argv.data());
    REQUIRE(v == Catch::Approx(1e-3));
}
TEST_CASE("=double sci 1.5e+2", "[equals][double]") {
    CommandLineOptions cmd; cmd.set_helponexit(false);
    double v=0; cmd.addop(v,"opt","","");
    TestArgs ta{"--opt=1.5e+2"}; cmd.phrase_options(ta.argc, ta.argv.data());
    REQUIRE(v == Catch::Approx(150.0));
}
TEST_CASE("=double sci -2.5E-4","[equals][double]") {
    CommandLineOptions cmd; cmd.set_helponexit(false);
    double v=0; cmd.addop(v,"opt","","");
    TestArgs ta{"--opt=-2.5E-4"}; cmd.phrase_options(ta.argc, ta.argv.data());
    REQUIRE(v == Catch::Approx(-0.00025));
}
TEST_CASE("=double hex 0x1p3",  "[equals][double]") {
    CommandLineOptions cmd; cmd.set_helponexit(false);
    double v=0; cmd.addop(v,"opt","","");
    TestArgs ta{"--opt=0x1p3"}; cmd.phrase_options(ta.argc, ta.argv.data());
    REQUIRE(v == Catch::Approx(8.0));
}

TEST_CASE("=float positive",    "[equals][float]") {
    CommandLineOptions cmd; cmd.set_helponexit(false);
    float v=0; cmd.addop(v,"opt","","");
    TestArgs ta{"--opt=2.5f"}; cmd.phrase_options(ta.argc, ta.argv.data());
    REQUIRE(v == Catch::Approx(2.5f));
}
TEST_CASE("=float negative",    "[equals][float]") {
    CommandLineOptions cmd; cmd.set_helponexit(false);
    float v=0; cmd.addop(v,"opt","","");
    TestArgs ta{"--opt=-1.25"}; cmd.phrase_options(ta.argc, ta.argv.data());
    REQUIRE(v == Catch::Approx(-1.25f));
}

TEST_CASE("=string simple",     "[equals][string]") {
    CommandLineOptions cmd; cmd.set_helponexit(false);
    std::string v; cmd.addop(v,"opt","","");
    TestArgs ta{"--opt=hello"}; cmd.phrase_options(ta.argc, ta.argv.data());
    REQUIRE(v == "hello");
}
TEST_CASE("=string with_underscore","[equals][string]") {
    CommandLineOptions cmd; cmd.set_helponexit(false);
    std::string v; cmd.addop(v,"opt","","");
    TestArgs ta{"--opt=hello_world"}; cmd.phrase_options(ta.argc, ta.argv.data());
    REQUIRE(v == "hello_world");
}
TEST_CASE("=string path-like",  "[equals][string]") {
    CommandLineOptions cmd; cmd.set_helponexit(false);
    std::string v; cmd.addop(v,"opt","","");
    TestArgs ta{"--opt=/home/user/file.txt"}; cmd.phrase_options(ta.argc, ta.argv.data());
    REQUIRE(v == "/home/user/file.txt");
}
TEST_CASE("=string with dots",  "[equals][string]") {
    CommandLineOptions cmd; cmd.set_helponexit(false);
    std::string v; cmd.addop(v,"opt","","");
    TestArgs ta{"--opt=config.ini"}; cmd.phrase_options(ta.argc, ta.argv.data());
    REQUIRE(v == "config.ini");
}
TEST_CASE("=string with hyphens","[equals][string]") {
    CommandLineOptions cmd; cmd.set_helponexit(false);
    std::string v; cmd.addop(v,"opt","","");
    TestArgs ta{"--opt=some-file-name"}; cmd.phrase_options(ta.argc, ta.argv.data());
    REQUIRE(v == "some-file-name");
}
TEST_CASE("=string numeric",    "[equals][string]") {
    CommandLineOptions cmd; cmd.set_helponexit(false);
    std::string v; cmd.addop(v,"opt","","");
    TestArgs ta{"--opt=12345"}; cmd.phrase_options(ta.argc, ta.argv.data());
    REQUIRE(v == "12345");
}
TEST_CASE("=string empty via next","[equals][string]") {
    // --opt= with empty after = falls back to reading next arg
    CommandLineOptions cmd; cmd.set_helponexit(false);
    std::string v; cmd.addop(v,"opt","","");
    TestArgs ta{"--opt=", "actual_value"};
    cmd.phrase_options(ta.argc, ta.argv.data());
    REQUIRE(v == "actual_value");
}

TEST_CASE("=uint simple",       "[equals][uint]") {
    CommandLineOptions cmd; cmd.set_helponexit(false);
    unsigned int v=0; cmd.addop(v,"opt","","");
    TestArgs ta{"--opt=42"}; cmd.phrase_options(ta.argc, ta.argv.data());
    REQUIRE(v == 42u);
}
TEST_CASE("=ulong simple",      "[equals][ulong]") {
    CommandLineOptions cmd; cmd.set_helponexit(false);
    unsigned long v=0; cmd.addop(v,"opt","","");
    TestArgs ta{"--opt=9999999999"}; cmd.phrase_options(ta.argc, ta.argv.data());
    REQUIRE(v == 9999999999ul);
}
TEST_CASE("=long simple",       "[equals][long]") {
    CommandLineOptions cmd; cmd.set_helponexit(false);
    long v=0; cmd.addop(v,"opt","","");
    TestArgs ta{"--opt=-1234567890"}; cmd.phrase_options(ta.argc, ta.argv.data());
    REQUIRE(v == -1234567890l);
}

// ==========================================================================
// 2. Boolean flags via --flag=value (narg=0)
// ==========================================================================

TEST_CASE("boolean flag values accept supported capitalization", "[equals][bool]") {
    const int i = GENERATE(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11);
    const std::pair<const char *, bool> cases[] = {
        {"yes", true}, {"no", false}, {"true", true}, {"false", false},
        {"Yes", true}, {"No", false}, {"True", true}, {"False", false},
        {"YES", true}, {"NO", false}, {"TRUE", true}, {"FALSE", false},
    };
    const auto [word, expected] = cases[i];
    CAPTURE(word);
    CommandLineOptions cmd; cmd.set_helponexit(false);
    bool value = !expected; cmd.addop(value, "f", "", "", false, 0);
    TestArgs args{std::string("--f=") + word};
    cmd.phrase_options(args.argc, args.argv.data());
    REQUIRE(value == expected);
}

// ==========================================================================
// 3. Vector types via --vec=val1 val2 ... (narg>1, = gives first)
// ==========================================================================

TEST_CASE("=intvec 3 elems", "[equals][intvec]") {
    CommandLineOptions cmd; cmd.set_helponexit(false);
    std::vector<int> v; cmd.addop(v,"v","","",false,3);
    TestArgs ta{"--v=10","20","30"}; cmd.phrase_options(ta.argc, ta.argv.data());
    REQUIRE(v.size() == 3);
    REQUIRE(v[0] == 10); REQUIRE(v[1] == 20); REQUIRE(v[2] == 30);
}
TEST_CASE("=intvec first neg","[equals][intvec]") {
    CommandLineOptions cmd; cmd.set_helponexit(false);
    std::vector<int> v; cmd.addop(v,"v","","",false,3);
    TestArgs ta{"--v=-5","10","15"}; cmd.phrase_options(ta.argc, ta.argv.data());
    REQUIRE(v.size() == 3);
    REQUIRE(v[0] == -5); REQUIRE(v[1] == 10); REQUIRE(v[2] == 15);
}
TEST_CASE("=realvec 3 elems","[equals][realvec]") {
    CommandLineOptions cmd; cmd.set_helponexit(false);
    std::vector<double> v; cmd.addop(v,"v","","",false,3);
    TestArgs ta{"--v=1.1","2.2","3.3"}; cmd.phrase_options(ta.argc, ta.argv.data());
    REQUIRE(v.size() == 3);
    REQUIRE(v[0] == Catch::Approx(1.1));
    REQUIRE(v[1] == Catch::Approx(2.2));
    REQUIRE(v[2] == Catch::Approx(3.3));
}
TEST_CASE("=strvec 3 elems", "[equals][strvec]") {
    CommandLineOptions cmd; cmd.set_helponexit(false);
    std::vector<std::string> v; cmd.addop(v,"v","","",false,3);
    TestArgs ta{"--v=aa","bb","cc"}; cmd.phrase_options(ta.argc, ta.argv.data());
    REQUIRE(v.size() == 3);
    REQUIRE(v[0] == "aa"); REQUIRE(v[1] == "bb"); REQUIRE(v[2] == "cc");
}
TEST_CASE("=uintvec 3 elems","[equals][uintvec]") {
    CommandLineOptions cmd; cmd.set_helponexit(false);
    std::vector<unsigned int> v; cmd.addop(v,"v","","",false,3);
    TestArgs ta{"--v=1","2","3"}; cmd.phrase_options(ta.argc, ta.argv.data());
    REQUIRE(v.size() == 3);
    REQUIRE(v[0] == 1u); REQUIRE(v[1] == 2u); REQUIRE(v[2] == 3u);
}
TEST_CASE("fixed int vector rejects an incomplete group before the next flag","[equals][intvec]") {
    CommandLineOptions cmd; cmd.set_helponexit(false);
    std::vector<int> v; cmd.addop(v,"v","","",false,10);
    int other=0; cmd.addop(other,"x","","");
    TestArgs ta{"--v=100","200","--x=5"};
    REQUIRE_THROWS_AS(cmd.phrase_options(ta.argc, ta.argv.data()), ParseError);
    REQUIRE(v.empty());
    REQUIRE(other == 0);
}

// ==========================================================================
// 4. Set types via --set=val (narg>1, = gives first)
// ==========================================================================

TEST_CASE("=intset multiple","[equals][intset]") {
    CommandLineOptions cmd; cmd.set_helponexit(false);
    std::set<int> v; cmd.addop(v,"s","","",false,3);
    TestArgs ta{"--s=5","10","15"}; cmd.phrase_options(ta.argc, ta.argv.data());
    REQUIRE(v.size() == 3);
    REQUIRE(v.count(5)); REQUIRE(v.count(10)); REQUIRE(v.count(15));
}
TEST_CASE("=uintset multiple","[equals][uintset]") {
    CommandLineOptions cmd; cmd.set_helponexit(false);
    std::set<unsigned int> v; cmd.addop(v,"s","","",false,3);
    TestArgs ta{"--s=1","2","4"}; cmd.phrase_options(ta.argc, ta.argv.data());
    REQUIRE(v.size() == 3);
    REQUIRE(v.count(1u)); REQUIRE(v.count(2u)); REQUIRE(v.count(4u));
}

// ==========================================================================
// 5. Backward compatibility: --opt val (space, no =)
// ==========================================================================

TEST_CASE("space int",        "[space][int]") {
    CommandLineOptions cmd; cmd.set_helponexit(false);
    int v=0; cmd.addop(v,"opt","","");
    TestArgs ta{"--opt","42"}; cmd.phrase_options(ta.argc, ta.argv.data());
    REQUIRE(v == 42);
}
TEST_CASE("space double",     "[space][double]") {
    CommandLineOptions cmd; cmd.set_helponexit(false);
    double v=0; cmd.addop(v,"opt","","");
    TestArgs ta{"--opt","3.14"}; cmd.phrase_options(ta.argc, ta.argv.data());
    REQUIRE(v == Catch::Approx(3.14));
}
TEST_CASE("space string",     "[space][string]") {
    CommandLineOptions cmd; cmd.set_helponexit(false);
    std::string v; cmd.addop(v,"opt","","");
    TestArgs ta{"--opt","hello"}; cmd.phrase_options(ta.argc, ta.argv.data());
    REQUIRE(v == "hello");
}
TEST_CASE("space bool toggle","[space][bool]") {
    CommandLineOptions cmd; cmd.set_helponexit(false);
    bool v=false; cmd.addop(v,"f","","",false,0);
    TestArgs ta{"--f"}; cmd.phrase_options(ta.argc, ta.argv.data());
    REQUIRE(v == true);
}
TEST_CASE("space bool toggle twice","[space][bool]") {
    CommandLineOptions cmd; cmd.set_helponexit(false);
    bool v=false; cmd.addop(v,"f","","",false,0);
    TestArgs ta{"--f","--f"}; cmd.phrase_options(ta.argc, ta.argv.data());
    REQUIRE(v == false); // toggle twice = back to original
}
TEST_CASE("space intvec",     "[space][intvec]") {
    CommandLineOptions cmd; cmd.set_helponexit(false);
    std::vector<int> v; cmd.addop(v,"v","","",false,3);
    TestArgs ta{"--v","1","2","3"}; cmd.phrase_options(ta.argc, ta.argv.data());
    REQUIRE(v.size() == 3);
    REQUIRE(v[0] == 1); REQUIRE(v[1] == 2); REQUIRE(v[2] == 3);
}

// ==========================================================================
// 6. Short options unchanged (-x val)
// ==========================================================================

TEST_CASE("short int",        "[short][int]") {
    CommandLineOptions cmd; cmd.set_helponexit(false);
    int v=0; cmd.addop(v,"longname","s","");
    TestArgs ta{"-s","99"}; cmd.phrase_options(ta.argc, ta.argv.data());
    REQUIRE(v == 99);
}
TEST_CASE("short double",     "[short][double]") {
    CommandLineOptions cmd; cmd.set_helponexit(false);
    double v=0; cmd.addop(v,"longname","d","");
    TestArgs ta{"-d","2.718"}; cmd.phrase_options(ta.argc, ta.argv.data());
    REQUIRE(v == Catch::Approx(2.718));
}
TEST_CASE("short string",     "[short][string]") {
    CommandLineOptions cmd; cmd.set_helponexit(false);
    std::string v; cmd.addop(v,"longname","n","");
    TestArgs ta{"-n","myname"}; cmd.phrase_options(ta.argc, ta.argv.data());
    REQUIRE(v == "myname");
}
TEST_CASE("short normal space-separated unchanged","[short]") {
    // -s=val should NOT be parsed as equals form (short opts unchanged).
    // -s=val → buffer="-s=val", b1='-', b2='s' NOT '-' → is_long=false
    // sname="s=val", maps2l["s=val"] not found → unknown or positional.
    // This test verifies normal short opts still work correctly.
    CommandLineOptions cmd; cmd.set_helponexit(false);
    cmd.set_unknownexit(false);
    int v=0; cmd.addop(v,"longname","s","");
    TestArgs ta{"-s","42"};
    cmd.phrase_options(ta.argc, ta.argv.data());
    REQUIRE(v == 42);
}

// ==========================================================================
// 7. Mixed: = and space in same invocation
// ==========================================================================

TEST_CASE("mixed = then space","[mixed]") {
    CommandLineOptions cmd; cmd.set_helponexit(false);
    int a=0; double b=0.0;
    cmd.addop(a,"intopt","","");
    cmd.addop(b,"dblopt","","");
    TestArgs ta{"--intopt=7","--dblopt","2.718"};
    cmd.phrase_options(ta.argc, ta.argv.data());
    REQUIRE(a == 7);
    REQUIRE(b == Catch::Approx(2.718));
}
TEST_CASE("mixed space then =","[mixed]") {
    CommandLineOptions cmd; cmd.set_helponexit(false);
    int a=0; double b=0.0;
    cmd.addop(a,"intopt","","");
    cmd.addop(b,"dblopt","","");
    TestArgs ta{"--intopt","7","--dblopt=2.718"};
    cmd.phrase_options(ta.argc, ta.argv.data());
    REQUIRE(a == 7);
    REQUIRE(b == Catch::Approx(2.718));
}
TEST_CASE("mixed = = =","[mixed]") {
    CommandLineOptions cmd; cmd.set_helponexit(false);
    int a=0,b=0,c=0;
    cmd.addop(a,"a","","");
    cmd.addop(b,"b","","");
    cmd.addop(c,"c","","");
    TestArgs ta{"--a=1","--b=2","--c=3"};
    cmd.phrase_options(ta.argc, ta.argv.data());
    REQUIRE(a == 1); REQUIRE(b == 2); REQUIRE(c == 3);
}
TEST_CASE("mixed many types","[mixed]") {
    CommandLineOptions cmd; cmd.set_helponexit(false);
    int i=0; double d=0.0; std::string s; bool b=false; float f=0.0f;
    cmd.addop(i,"intopt","","");
    cmd.addop(d,"dblopt","","");
    cmd.addop(s,"stropt","","");
    cmd.addop(b,"boolopt","b","",false,0);
    cmd.addop(f,"floatopt","","");

    TestArgs ta{"--intopt=10","--dblopt=1.5","--stropt=xyz","--boolopt=yes","--floatopt=3.25"};
    cmd.phrase_options(ta.argc, ta.argv.data());
    REQUIRE(i == 10);
    REQUIRE(d == Catch::Approx(1.5));
    REQUIRE(s == "xyz");
    REQUIRE(b == true);
    REQUIRE(f == Catch::Approx(3.25f));
}

// ==========================================================================
// 8. Short + long mixed
// ==========================================================================

TEST_CASE("short+long mixed","[short][mixed]") {
    CommandLineOptions cmd; cmd.set_helponexit(false);
    int a=0,b=0;
    cmd.addop(a,"alpha","a","");
    cmd.addop(b,"beta","b","");
    TestArgs ta{"-a","10","--beta=20"};
    cmd.phrase_options(ta.argc, ta.argv.data());
    REQUIRE(a == 10); REQUIRE(b == 20);
}
TEST_CASE("long= + short","[short][mixed]") {
    CommandLineOptions cmd; cmd.set_helponexit(false);
    int a=0,b=0;
    cmd.addop(a,"alpha","a","");
    cmd.addop(b,"beta","b","");
    TestArgs ta{"--alpha=10","-b","20"};
    cmd.phrase_options(ta.argc, ta.argv.data());
    REQUIRE(a == 10); REQUIRE(b == 20);
}
TEST_CASE("short + long= + short","[short][mixed]") {
    CommandLineOptions cmd; cmd.set_helponexit(false);
    int a=0,b=0,c=0;
    cmd.addop(a,"alpha","a","");
    cmd.addop(b,"beta","b","");
    cmd.addop(c,"gamma","c","");
    TestArgs ta{"-a","1","--beta=2","-c","3"};
    cmd.phrase_options(ta.argc, ta.argv.data());
    REQUIRE(a == 1); REQUIRE(b == 2); REQUIRE(c == 3);
}

// ==========================================================================
// 9. Positional arguments mixed with --tag=val
// ==========================================================================

TEST_CASE("positional before =","[positional]") {
    CommandLineOptions cmd; cmd.set_helponexit(false);
    std::string input; double tol=0;
    cmd.addop(input,"input","","",true);
    cmd.addop(tol,"tol","t","");
    TestArgs ta{"myfile.cfg","--tol=1e-4"};
    cmd.phrase_options(ta.argc, ta.argv.data());
    REQUIRE(input == "myfile.cfg");
    REQUIRE(tol == Catch::Approx(1e-4));
}
TEST_CASE("positional after =","[positional]") {
    CommandLineOptions cmd; cmd.set_helponexit(false);
    std::string output; int n=0;
    cmd.addop(n,"n","","");
    cmd.addop(output,"output","","",true);
    TestArgs ta{"--n=5","result.cfg"};
    cmd.phrase_options(ta.argc, ta.argv.data());
    REQUIRE(n == 5);
    REQUIRE(output == "result.cfg");
}
TEST_CASE("positional both sides","[positional]") {
    CommandLineOptions cmd; cmd.set_helponexit(false);
    std::string in, out; double tol=0;
    cmd.addop(in,"input","","",true);
    cmd.addop(out,"output","","",true);
    cmd.addop(tol,"tol","","");
    TestArgs ta{"input.cfg","--tol=1e-3","output.cfg"};
    cmd.phrase_options(ta.argc, ta.argv.data());
    REQUIRE(in == "input.cfg");
    REQUIRE(tol == Catch::Approx(1e-3));
    REQUIRE(out == "output.cfg");
}

// ==========================================================================
// 10. Multiple option occurrences (vector/set accumulation)
// ==========================================================================

TEST_CASE("multi = same scalar overrides","[multi]") {
    CommandLineOptions cmd; cmd.set_helponexit(false);
    int v=0; cmd.addop(v,"x","","");
    TestArgs ta{"--x=1","--x=2","--x=3"};
    cmd.phrase_options(ta.argc, ta.argv.data());
    REQUIRE(v == 3); // last value wins
}
TEST_CASE("multi space same scalar overrides","[multi]") {
    CommandLineOptions cmd; cmd.set_helponexit(false);
    int v=0; cmd.addop(v,"x","","");
    TestArgs ta{"--x","1","--x","2"};
    cmd.phrase_options(ta.argc, ta.argv.data());
    REQUIRE(v == 2);
}
TEST_CASE("multi =+space same scalar","[multi]") {
    CommandLineOptions cmd; cmd.set_helponexit(false);
    int v=0; cmd.addop(v,"x","","");
    TestArgs ta{"--x=1","--x","2"};
    cmd.phrase_options(ta.argc, ta.argv.data());
    REQUIRE(v == 2);
}

// ==========================================================================
// 11. Edge cases for --opt=value
// ==========================================================================

TEST_CASE("edge = empty falls back","[edge]") {
    CommandLineOptions cmd; cmd.set_helponexit(false);
    int v=0; cmd.addop(v,"opt","","");
    TestArgs ta{"--opt=","55"};
    cmd.phrase_options(ta.argc, ta.argv.data());
    REQUIRE(v == 55);
}
TEST_CASE("edge = empty bool toggles","[edge][bool]") {
    CommandLineOptions cmd; cmd.set_helponexit(false);
    bool v=false; cmd.addop(v,"f","","",false,0);
    TestArgs ta{"--f="};
    cmd.phrase_options(ta.argc, ta.argv.data());
    REQUIRE(v == true); // empty = toggles bool
}
TEST_CASE("edge double = in value","[edge]") {
    // --opt=key=val → name="opt", value="key=val"
    CommandLineOptions cmd; cmd.set_helponexit(false);
    std::string v; cmd.addop(v,"opt","","");
    TestArgs ta{"--opt=key=val"};
    cmd.phrase_options(ta.argc, ta.argv.data());
    REQUIRE(v == "key=val");
}
TEST_CASE("edge name with hyphens","[edge]") {
    CommandLineOptions cmd; cmd.set_helponexit(false);
    int v=0; cmd.addop(v,"my-long-option","","");
    TestArgs ta{"--my-long-option=42"};
    cmd.phrase_options(ta.argc, ta.argv.data());
    REQUIRE(v == 42);
}
TEST_CASE("edge name with underscore","[edge]") {
    CommandLineOptions cmd; cmd.set_helponexit(false);
    double v=0; cmd.addop(v,"my_option","","");
    TestArgs ta{"--my_option=3.14"};
    cmd.phrase_options(ta.argc, ta.argv.data());
    REQUIRE(v == Catch::Approx(3.14));
}
TEST_CASE("edge very long value","[edge]") {
    CommandLineOptions cmd; cmd.set_helponexit(false);
    std::string v; cmd.addop(v,"opt","","");
    std::string longval(1000, 'x');
    TestArgs ta{"--opt=" + longval};
    cmd.phrase_options(ta.argc, ta.argv.data());
    REQUIRE(v == longval);
}
TEST_CASE("edge value with spaces via next arg","[edge]") {
    CommandLineOptions cmd; cmd.set_helponexit(false);
    std::string v; cmd.addop(v,"opt","","");
    TestArgs ta{"--opt","value with spaces"};
    cmd.phrase_options(ta.argc, ta.argv.data());
    REQUIRE(v == "value with spaces");
}
TEST_CASE("edge leading zeros int","[edge]") {
    CommandLineOptions cmd; cmd.set_helponexit(false);
    int v=0; cmd.addop(v,"opt","","");
    TestArgs ta{"--opt=007"}; cmd.phrase_options(ta.argc, ta.argv.data());
    REQUIRE(v == 7);
}
TEST_CASE("edge double no fractional","[edge][double]") {
    CommandLineOptions cmd; cmd.set_helponexit(false);
    double v=0; cmd.addop(v,"opt","","");
    TestArgs ta{"--opt=42"}; cmd.phrase_options(ta.argc, ta.argv.data());
    REQUIRE(v == Catch::Approx(42.0));
}
TEST_CASE("edge double leading dot","[edge][double]") {
    CommandLineOptions cmd; cmd.set_helponexit(false);
    double v=0; cmd.addop(v,"opt","","");
    TestArgs ta{"--opt=.5"}; cmd.phrase_options(ta.argc, ta.argv.data());
    REQUIRE(v == Catch::Approx(0.5));
}

// ==========================================================================
// 12. Boundary / stress values
// ==========================================================================

TEST_CASE("stress int min","[stress]") {
    CommandLineOptions cmd; cmd.set_helponexit(false);
    int v=0; cmd.addop(v,"opt","","");
    TestArgs ta{"--opt=-2147483648"};
    cmd.phrase_options(ta.argc, ta.argv.data());
    REQUIRE(v == INT_MIN);
}
TEST_CASE("stress uint max","[stress]") {
    CommandLineOptions cmd; cmd.set_helponexit(false);
    unsigned int v=0; cmd.addop(v,"opt","","");
    TestArgs ta{"--opt=4294967295"};
    cmd.phrase_options(ta.argc, ta.argv.data());
    REQUIRE(v == UINT_MAX);
}
TEST_CASE("stress double min","[stress]") {
    CommandLineOptions cmd; cmd.set_helponexit(false);
    double v=0; cmd.addop(v,"opt","","");
    TestArgs ta{"--opt=-1.7976931348623157e308"};
    cmd.phrase_options(ta.argc, ta.argv.data());
    REQUIRE(v == Catch::Approx(-1.7976931348623157e308));
}
TEST_CASE("stress zero values all types","[stress]") {
    CommandLineOptions cmd; cmd.set_helponexit(false);
    int i=1; double d=1; float f=1; bool b=true;
    cmd.addop(i,"i","","");
    cmd.addop(d,"d","","");
    cmd.addop(f,"f","","");
    cmd.addop(b,"b","","",false,0);
    TestArgs ta{"--i=0","--d=0.0","--f=0.0","--b=no"};
    cmd.phrase_options(ta.argc, ta.argv.data());
    REQUIRE(i == 0); REQUIRE(d == Catch::Approx(0.0));
    REQUIRE(f == Catch::Approx(0.0f)); REQUIRE(b == false);
}

// ==========================================================================
// 13. Many options in sequence
// ==========================================================================

TEST_CASE("many opts 10 ints","[many]") {
    CommandLineOptions cmd; cmd.set_helponexit(false);
    int v0=0,v1=0,v2=0,v3=0,v4=0,v5=0,v6=0,v7=0,v8=0,v9=0;
    cmd.addop(v0,"a0","",""); cmd.addop(v1,"a1","","");
    cmd.addop(v2,"a2","",""); cmd.addop(v3,"a3","","");
    cmd.addop(v4,"a4","",""); cmd.addop(v5,"a5","","");
    cmd.addop(v6,"a6","",""); cmd.addop(v7,"a7","","");
    cmd.addop(v8,"a8","",""); cmd.addop(v9,"a9","","");
    TestArgs ta{
        "--a0=0","--a1=1","--a2=2","--a3=3","--a4=4",
        "--a5=5","--a6=6","--a7=7","--a8=8","--a9=9"
    };
    cmd.phrase_options(ta.argc, ta.argv.data());
    REQUIRE(v0==0); REQUIRE(v1==1); REQUIRE(v2==2); REQUIRE(v3==3);
    REQUIRE(v4==4); REQUIRE(v5==5); REQUIRE(v6==6); REQUIRE(v7==7);
    REQUIRE(v8==8); REQUIRE(v9==9);
}
TEST_CASE("many opts mixed types","[many]") {
    CommandLineOptions cmd; cmd.set_helponexit(false);
    int i1=0,i2=0; double d1=0,d2=0; std::string s1,s2; bool b1=false,b2=false;
    cmd.addop(i1,"i1","",""); cmd.addop(i2,"i2","","");
    cmd.addop(d1,"d1","",""); cmd.addop(d2,"d2","","");
    cmd.addop(s1,"s1","",""); cmd.addop(s2,"s2","","");
    cmd.addop(b1,"b1","","",false,0); cmd.addop(b2,"b2","","",false,0);
    TestArgs ta{
        "--i1=10","--d1=1.5","--s1=hello","--b1=true",
        "--i2=20","--d2=2.5","--s2=world","--b2=false"
    };
    cmd.phrase_options(ta.argc, ta.argv.data());
    REQUIRE(i1==10); REQUIRE(d1==Catch::Approx(1.5));
    REQUIRE(s1=="hello"); REQUIRE(b1==true);
    REQUIRE(i2==20); REQUIRE(d2==Catch::Approx(2.5));
    REQUIRE(s2=="world"); REQUIRE(b2==false);
}

// ==========================================================================
// 14. Interleaved: space and = for same option
// ==========================================================================

TEST_CASE("interleave = then space same opt","[interleave]") {
    CommandLineOptions cmd; cmd.set_helponexit(false);
    std::vector<int> v; cmd.addop(v,"v","","",false,-1);
    // First value via =, remaining values via space continuation.
    TestArgs ta{"--v=10","20","30"};
    cmd.phrase_options(ta.argc, ta.argv.data());
    REQUIRE(v.size() == 3);
    REQUIRE(v[0]==10); REQUIRE(v[1]==20); REQUIRE(v[2]==30);
}
TEST_CASE("interleave = then = for vec","[interleave]") {
    // second --v=40 starts a new option parse (state 0 sees -v as flag)
    CommandLineOptions cmd; cmd.set_helponexit(false);
    std::vector<int> v; cmd.addop(v,"v","","",false,-1);
    TestArgs ta{"--v=10","20","--v=30","40"};
    cmd.phrase_options(ta.argc, ta.argv.data());
    // state 0 sees "--v=10" → vassign(10), state2. "20" → vassign(20)
    // then "--v=30" → state 0 again, vassign(30), state2. "40" → vassign(40)
    REQUIRE(v.size() == 4);
    REQUIRE(v[0]==10); REQUIRE(v[1]==20); REQUIRE(v[2]==30); REQUIRE(v[3]==40);
}

// ==========================================================================
// 15. Combined with negative numbers as positional detection
// ==========================================================================

TEST_CASE("negative after = not positional","[negative]") {
    CommandLineOptions cmd; cmd.set_helponexit(false);
    int v=0; cmd.addop(v,"opt","","");
    TestArgs ta{"--opt=-42"};
    cmd.phrase_options(ta.argc, ta.argv.data());
    REQUIRE(v == -42);
}
TEST_CASE("negative double after =","[negative]") {
    CommandLineOptions cmd; cmd.set_helponexit(false);
    double v=0; cmd.addop(v,"opt","","");
    TestArgs ta{"--opt=-1.5"};
    cmd.phrase_options(ta.argc, ta.argv.data());
    REQUIRE(v == Catch::Approx(-1.5));
}
TEST_CASE("negative int via space","[negative]") {
    // "-5" after --opt is NOT treated as a flag because b1='-', b2='5' → numeric → positional
    // But since we're in state 1, we read it as a value
    CommandLineOptions cmd; cmd.set_helponexit(false);
    int v=0; cmd.addop(v,"opt","","");
    TestArgs ta{"--opt","-5"};
    cmd.phrase_options(ta.argc, ta.argv.data());
    REQUIRE(v == -5);
}

// ==========================================================================
// 16. Bool toggle mixed with = bool
// ==========================================================================

TEST_CASE("bool =yes then toggle","[bool][multi]") {
    CommandLineOptions cmd; cmd.set_helponexit(false);
    bool v=false; cmd.addop(v,"f","","",false,0);
    TestArgs ta{"--f=yes","--f"};
    cmd.phrase_options(ta.argc, ta.argv.data());
    REQUIRE(v == false); // yes→true, toggle→false
}
TEST_CASE("bool toggle then =no","[bool][multi]") {
    CommandLineOptions cmd; cmd.set_helponexit(false);
    bool v=false; cmd.addop(v,"f","","",false,0);
    TestArgs ta{"--f","--f=no"};
    cmd.phrase_options(ta.argc, ta.argv.data());
    REQUIRE(v == false); // toggle→true, =no→false
}

// ==========================================================================
// 17. String edge cases
// ==========================================================================

TEST_CASE("string with equals sign","[string][edge]") {
    CommandLineOptions cmd; cmd.set_helponexit(false);
    std::string v; cmd.addop(v,"opt","","");
    TestArgs ta{"--opt=a=b=c"};
    cmd.phrase_options(ta.argc, ta.argv.data());
    REQUIRE(v == "a=b=c"); // only first = splits name/value
}
TEST_CASE("string with special chars","[string][edge]") {
    CommandLineOptions cmd; cmd.set_helponexit(false);
    std::string v; cmd.addop(v,"opt","","");
    TestArgs ta{"--opt=@#$%^&*()"};
    cmd.phrase_options(ta.argc, ta.argv.data());
    REQUIRE(v == "@#$%^&*()");
}
TEST_CASE("string URL-like","[string][edge]") {
    CommandLineOptions cmd; cmd.set_helponexit(false);
    std::string v; cmd.addop(v,"opt","","");
    TestArgs ta{"--opt=https://example.com/path?q=1"};
    cmd.phrase_options(ta.argc, ta.argv.data());
    REQUIRE(v == "https://example.com/path?q=1");
}
TEST_CASE("string file extension","[string][edge]") {
    CommandLineOptions cmd; cmd.set_helponexit(false);
    std::string v; cmd.addop(v,"opt","","");
    TestArgs ta{"--opt=data.extxyz"};
    cmd.phrase_options(ta.argc, ta.argv.data());
    REQUIRE(v == "data.extxyz");
}

// ==========================================================================
// 18. Reordered options (different order from registration)
// ==========================================================================

TEST_CASE("reorder reverse","[reorder]") {
    CommandLineOptions cmd; cmd.set_helponexit(false);
    int a=0,b=0,c=0;
    cmd.addop(a,"a","",""); cmd.addop(b,"b","",""); cmd.addop(c,"c","","");
    TestArgs ta{"--c=3","--b=2","--a=1"};
    cmd.phrase_options(ta.argc, ta.argv.data());
    REQUIRE(a==1); REQUIRE(b==2); REQUIRE(c==3);
}
TEST_CASE("reorder random","[reorder]") {
    CommandLineOptions cmd; cmd.set_helponexit(false);
    int a=0,b=0,c=0,d=0,e=0;
    cmd.addop(a,"a","",""); cmd.addop(b,"b","","");
    cmd.addop(c,"c","",""); cmd.addop(d,"d","","");
    cmd.addop(e,"e","","");
    TestArgs ta{"--d=4","--a=1","--e=5","--c=3","--b=2"};
    cmd.phrase_options(ta.argc, ta.argv.data());
    REQUIRE(a==1); REQUIRE(b==2); REQUIRE(c==3); REQUIRE(d==4); REQUIRE(e==5);
}

// ==========================================================================
// 19. Large test: 50 options
// ==========================================================================

TEST_CASE("stress 50 int opts","[stress][many]") {
    CommandLineOptions cmd; cmd.set_helponexit(false);
    const int N = 50;
    int vals[N] = {};
    std::vector<std::string> args;
    for (int i = 0; i < N; ++i) {
        std::string name = "opt" + std::to_string(i);
        cmd.addop(vals[i], name, "", "");
        args.push_back("--" + name + "=" + std::to_string(i * 2));
    }
    TestArgs ta(args);
    cmd.phrase_options(ta.argc, ta.argv.data());
    for (int i = 0; i < N; ++i) {
        REQUIRE(vals[i] == i * 2);
    }
}

// ==========================================================================
// 20. Bool with short names + equals
// ==========================================================================

TEST_CASE("bool short + long =","[bool][short]") {
    CommandLineOptions cmd; cmd.set_helponexit(false);
    bool v=false; cmd.addop(v,"verbose","v","",false,0);
    // short flag (no =)
    TestArgs ta{"-v"};
    cmd.phrase_options(ta.argc, ta.argv.data());
    REQUIRE(v == true);
}
TEST_CASE("bool long = then short toggle","[bool][short]") {
    CommandLineOptions cmd; cmd.set_helponexit(false);
    bool v=false; cmd.addop(v,"verbose","v","",false,0);
    TestArgs ta{"--verbose=true","-v"};
    cmd.phrase_options(ta.argc, ta.argv.data());
    REQUIRE(v == false); // true then toggle → false
}

// ==========================================================================
// 21. Complex scenario: real-world-like invocation
// ==========================================================================

TEST_CASE("realworld relax command","[realworld]") {
    CommandLineOptions cmd; cmd.set_helponexit(false);
    std::string input, output, settings;
    double ftol=0.01, etol=1e-5;
    int maxiter=200;
    bool verbose=false, dump=true;
    cmd.addop(input,"input","i","",true);
    cmd.addop(output,"output","o","",true);
    cmd.addop(settings,"settings","s","");
    cmd.addop(ftol,"ftol","","");
    cmd.addop(etol,"etol","","");
    cmd.addop(maxiter,"maxiter","","");
    cmd.addop(verbose,"verbose","v","",false,0);
    cmd.addop(dump,"dump","","",false,0);
    TestArgs ta{
        "input.cfg","--settings=relax.ini","--ftol=0.001","--etol=1e-6",
        "--maxiter=500","--verbose=yes","--dump=false","output.cfg"
    };
    cmd.phrase_options(ta.argc, ta.argv.data());
    REQUIRE(input=="input.cfg");
    REQUIRE(settings=="relax.ini");
    REQUIRE(ftol==Catch::Approx(0.001));
    REQUIRE(etol==Catch::Approx(1e-6));
    REQUIRE(maxiter==500);
    REQUIRE(verbose==true);
    REQUIRE(dump==false);
    REQUIRE(output=="output.cfg");
}

TEST_CASE("realworld train command","[realworld]") {
    CommandLineOptions cmd; cmd.set_helponexit(false);
    std::string trainSet, potFile;
    double energyWeight=1.0, forceWeight=0.01, stressWeight=0.001;
    int maxIter=1000, batchSize=32;
    bool shuffle=true, saveBest=true;
    cmd.addop(trainSet,"trainset","","",true);
    cmd.addop(potFile,"potfile","p","");
    cmd.addop(energyWeight,"eweight","","");
    cmd.addop(forceWeight,"fweight","","");
    cmd.addop(stressWeight,"sweight","","");
    cmd.addop(maxIter,"maxiter","","");
    cmd.addop(batchSize,"batch","","");
    cmd.addop(shuffle,"shuffle","","",false,0);
    cmd.addop(saveBest,"savebest","","",false,0);
    TestArgs ta{
        "training.xyz","--potfile=pot.mtp","--eweight=1.0","--fweight=0.05",
        "--sweight=0.002","--maxiter=2000","--batch=64",
        "--shuffle=true","--savebest=yes"
    };
    cmd.phrase_options(ta.argc, ta.argv.data());
    REQUIRE(trainSet=="training.xyz");
    REQUIRE(potFile=="pot.mtp");
    REQUIRE(energyWeight==Catch::Approx(1.0));
    REQUIRE(forceWeight==Catch::Approx(0.05));
    REQUIRE(stressWeight==Catch::Approx(0.002));
    REQUIRE(maxIter==2000);
    REQUIRE(batchSize==64);
    REQUIRE(shuffle==true);
    REQUIRE(saveBest==true);
}

// ==========================================================================
// 22. Exception handling: verify exit(0) → ParseError throw
// ==========================================================================

TEST_CASE("exception unknown long option throws ParseError","[exception]") {
    CommandLineOptions cmd;
    cmd.set_helponexit(false);
    int v=0; cmd.addop(v,"known","","");
    TestArgs ta{"--unknown=value"};
    REQUIRE_THROWS_AS(cmd.phrase_options(ta.argc, ta.argv.data()), CMDOption::ParseError);
}
TEST_CASE("exception unknown short option throws ParseError","[exception]") {
    CommandLineOptions cmd;
    cmd.set_helponexit(false);
    int v=0; cmd.addop(v,"known","k","");
    TestArgs ta{"-z"};
    REQUIRE_THROWS_AS(cmd.phrase_options(ta.argc, ta.argv.data()), CMDOption::ParseError);
}
TEST_CASE("exception narg=0 non-bool throws ParseError","[exception]") {
    CommandLineOptions cmd;
    cmd.set_helponexit(false);
    int v=0; cmd.addop(v,"opt","","",false,0); // narg=0 with INT, not STRBOOL
    TestArgs ta{"--opt"};
    REQUIRE_THROWS_AS(cmd.phrase_options(ta.argc, ta.argv.data()), CMDOption::ParseError);
}
TEST_CASE("exception missing required throws ParseError","[exception]") {
    CommandLineOptions cmd;
    cmd.set_helponexit(false);
    std::string required_val;
    cmd.addop(required_val,"musthave","","",true);
    cmd.add_request("musthave");
    cmd.check();
    TestArgs ta{}; // no args → "musthave" not provided
    REQUIRE_THROWS_AS(cmd.phrase_options(ta.argc, ta.argv.data()), CMDOption::ParseError);
}

TEST_CASE("CLI rejects an option whose value is missing at end of input", "[exception]") {
    CommandLineOptions cmd;
    cmd.set_helponexit(false);
    int value = 7;
    cmd.addop(value, "number", "n");
    TestArgs args{GENERATE("--number", "--number=", "-n")};
    REQUIRE_THROWS_WITH(cmd.phrase_options(args.argc, args.argv.data()),
                        Catch::Matchers::ContainsSubstring("Missing value"));
}

TEST_CASE("CLI rejects numeric suffixes for scalars and vectors", "[exception]") {
    CommandLineOptions cmd;
    cmd.set_helponexit(false);
    int integer = 0;
    double real = 0;
    std::vector<int> values;
    std::string positional;
    cmd.addop(integer, "integer").addop(real, "real")
       .addop(values, "values", "", "", false, -1)
       .addop(positional, "filename", "", "", true);
    TestArgs args{GENERATE("--integer=2junk", "--integer=1.5", "--real=1e-3junk", "--values=2junk")};
    REQUIRE_THROWS_AS(cmd.phrase_options(args.argc, args.argv.data()), ParseError);
}

TEST_CASE("CLI does not reinterpret an invalid numeric vector value as a filename", "[exception]") {
    CommandLineOptions cmd;
    cmd.set_helponexit(false);
    std::vector<int> values;
    std::string filename;
    cmd.addop(values, "values", "", "", false, -1)
       .addop(filename, "filename", "", "", true);
    TestArgs args{"--values", "1", "2junk"};
    REQUIRE_THROWS_AS(cmd.phrase_options(args.argc, args.argv.data()), ParseError);
}

TEST_CASE("Numeric settings reject trailing junk in both application paths", "[settings][exception]") {
    const auto value = GENERATE("2junk", "1.5junk");
    GPSettings settings;
    settings.Modify("number", value);
    SECTION("ConnectSettings") {
        ConnectSettings connected;
        double number = 0;
        connected.Connect(number, "number");
        REQUIRE_THROWS_AS(connected.ApplySettings(settings), GPException);
    }
    SECTION("command-line settings overlay") {
        CommandLineOptions cmd;
        double number = 0;
        cmd.addop(number, "number");
        REQUIRE_THROWS_AS(apply_settings_to_cmdopts(cmd, settings), ParseError);
    }
}
TEST_CASE("exception add_request unknown name throws ParseError","[exception]") {
    CommandLineOptions cmd;
    cmd.set_helponexit(false);
    REQUIRE_THROWS_AS(cmd.add_request("nonexistent"), CMDOption::ParseError);
}
TEST_CASE("exception extra positional throws ParseError","[exception]") {
    CommandLineOptions cmd;
    cmd.set_helponexit(false);
    // Register zero positional ops, then pass extra positional args
    int v=0; cmd.addop(v,"opt","","");
    TestArgs ta{"extra_positional_arg"};
    REQUIRE_THROWS_AS(cmd.phrase_options(ta.argc, ta.argv.data()), CMDOption::ParseError);
}
TEST_CASE("exception parseError message is descriptive","[exception]") {
    CommandLineOptions cmd;
    cmd.set_helponexit(false);
    int v=0; cmd.addop(v,"known","","");
    TestArgs ta{"--badflag=123"};
    try {
        cmd.phrase_options(ta.argc, ta.argv.data());
        FAIL("Expected ParseError was not thrown");
    } catch (const CMDOption::ParseError& e) {
        std::string msg(e.what());
        REQUIRE(msg.find("Unknown option") != std::string::npos);
        REQUIRE(msg.find("badflag") != std::string::npos);
    }
}

TEST_CASE("duplicate option warning suppressed below verbose threshold","[warning]") {
    const int saved_level = options_list_global_level;
    options_list_global_level = 100;
    std::ostringstream err;
    auto *saved_buf = std::cerr.rdbuf(err.rdbuf());

    CommandLineOptions cmd;
    int first=0, second=0;
    cmd.addop(first,"dup","d","");
    cmd.addop(second,"dup","e","");
    cmd.addop(second,"other","d","");

    std::cerr.rdbuf(saved_buf);
    options_list_global_level = saved_level;
    REQUIRE(err.str().empty());
}

TEST_CASE("duplicate option warning shown above verbose threshold","[warning]") {
    const int saved_level = options_list_global_level;
    options_list_global_level = 101;
    std::ostringstream err;
    auto *saved_buf = std::cerr.rdbuf(err.rdbuf());

    CommandLineOptions cmd;
    int first=0, second=0;
    cmd.addop(first,"dup","d","");
    cmd.addop(second,"dup","e","");
    cmd.addop(second,"other","d","");

    std::cerr.rdbuf(saved_buf);
    options_list_global_level = saved_level;
    const std::string output = err.str();
    REQUIRE(output.find("Option dup has been used before") != std::string::npos);
    REQUIRE(output.find("Option d has been used before") != std::string::npos);
}
