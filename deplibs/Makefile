# Bundled libraries retain their own licenses; see THIRD_PARTY_NOTICES.md.
CXX ?= g++
CC ?= gcc
CXXFLAGS ?= -O3 -fPIC -std=c++17
CFLAGS ?= -O3 -fPIC
SPG_SRC := $(wildcard spglib/*.c)
SPG_OBJ := $(SPG_SRC:.c=.o)
VORO_SRC := $(addprefix voro++/,cell.cc common.cc container.cc unitcell.cc v_compute.cc c_loops.cc v_base.cc wall.cc pre_container.cc container_prd.cc)
VORO_OBJ := $(VORO_SRC:.cc=.o)
all: libtinyxml2.a libfmt.a libsymspg.a libvoro++.a
libtinyxml2.a: tinyxml2/tinyxml2.o
	$(AR) rcs $@ $^
libfmt.a: fmt/format.o fmt/os.o
	$(AR) rcs $@ $^
libsymspg.a: $(SPG_OBJ)
	$(AR) rcs $@ $^
libvoro++.a: $(VORO_OBJ)
	$(AR) rcs $@ $^
%.o: %.cpp
	$(CXX) $(CXXFLAGS) -I. -c $< -o $@
fmt/%.o: fmt/%.cc
	$(CXX) $(CXXFLAGS) -Ifmt/include -c $< -o $@
voro++/%.o: voro++/%.cc
	$(CXX) $(CXXFLAGS) -Ivoro++ -c $< -o $@
spglib/%.o: spglib/%.c
	$(CC) $(CFLAGS) -DSPG_BUILD -c $< -o $@
clean:
	rm -f libtinyxml2.a libfmt.a libsymspg.a libvoro++.a tinyxml2/*.o fmt/*.o spglib/*.o voro++/*.o
.PHONY: all clean
