.DEFAULT_GOAL := all
include Makefile.inc

# ── Build directory (mode-specific, isolated) ───────────────────────────
BUILD_DIR := build/$(BUILD_MODE)
bd         = $(addprefix $(BUILD_DIR)/,$(1))

# ── Source auto-discovery ──────────────────────────────────────────────
SRCDIRS := utils base io tools apps external basis \
           symmetry potentials analyse descriptors \
           mathextra mindriver train NNBase mtmbasis maxvol active_learn \
           basis_analysis basis_analysis/strategies runtime/mtp runtime/mtp/cpu

SRCS     := $(foreach d,$(SRCDIRS),$(wildcard $(d)/*.cpp))
OBJS     := $(addprefix $(BUILD_DIR)/, $(notdir $(SRCS:.cpp=.o)))
OBJDEPS  := $(OBJS:.o=.d)

# ── Compilation: explicit rules for each source → flat .o in build dir ──
CXXFLAGS += -I. -I runtime/mtp -MMD -MP

# Optional native-interface host profiling; absent from normal builds.
ifeq ($(NATIVE_HOST_PROFILE),1)
CXXFLAGS += -DIMRMLP_NATIVE_HOST_PROFILE
endif

# ── Precompiled header (PCH) — skip Eigen+fftw3+fmt re-parsing ──────
# Capture CXXFLAGS before adding -include, so PCH build doesn't include itself
CXXFLAGS_NO_PCH := $(CXXFLAGS)

PCH_H := $(BUILD_DIR)/stdafx.h
PCH   := $(PCH_H).gch

$(PCH): utils/stdafx.h | $(BUILD_DIR)
	cp utils/stdafx.h $(PCH_H)
	$(CXX) $(CXXFLAGS_NO_PCH) -x c++-header -c $(PCH_H) -o $@

# Force-include the header — GCC auto-detects the .gch alongside it
CXXFLAGS += -include $(PCH_H)

# ── Compile-flag tracking ──────────────────────────────────────────────────
# make does not notice when compile flags change (e.g. toggling USE_GZIP), so objects would silently keep their old codec support. The
# stamp file's mtime advances only when CXXFLAGS actually differs; objects
# and the PCH depend on it, forcing a rebuild on flag changes.
FLAGS_STAMP := $(BUILD_DIR)/.flags
.PHONY: flags-force
$(FLAGS_STAMP): flags-force | $(BUILD_DIR)
	@new="$(CXXFLAGS)"; if [ -f "$@" ]; then old="$$(cat "$@")"; \
	  if [ "$$old" = "$$new" ]; then exit 0; fi; fi; printf '%s' "$$new" > "$@"

# Objects also depend on the PCH so parallel builds never compile against a
# half-written .gch (stale/empty objects), and on the flag stamp.
$(OBJS): $(PCH) $(FLAGS_STAMP)
$(PCH): $(FLAGS_STAMP)

# Fresh checkouts do not contain the ignored generated version header.  Make
# the regular application targets self-contained instead of relying on a
# workstation-local version.sh helper.
VERSION_HEADER := config/version.h
$(VERSION_HEADER):
	@commit="$$(git rev-parse --short HEAD 2>/dev/null || printf unknown)"; \
	  build_date="$$(date +%Y-%m-%d)"; build_stamp="$$(date +%s)"; \
	  build_target="$$(uname -m)"; \
	  printf '%s\n' '#pragma once' \
	    '#define PROJECT_VERSION "1.2.4"' \
	    "#define GIT_COMMIT_ID \"$$commit\"" \
	    "#define BUILD_DATE \"$$build_date\"" \
	    "#define BUILD_TARGET \"$$build_target\"" \
	    "#define BUILD_STAMP \"$$build_stamp\"" > "$@"

$(BUILD_DIR)/cmdopts.o $(BUILD_DIR)/imrmlp.o: $(VERSION_HEADER)

deps: $(PCH)

define _COMPILE_
$(BUILD_DIR)/$(notdir $(basename $(1))).o: $(1) | $(BUILD_DIR)
	$$(CXX) $$(CXXFLAGS) -c $$< -o $$@
endef
$(foreach s,$(SRCS),$(eval $(call _COMPILE_,$(s))))

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)
	mkdir -p $(BUILD_DIR)/unitest

# ── Standard object lists ──────────────────────────────────────────────
OBJAPP = model_bundle_app.o apps.o convstr.o calc_efs.o calc_err.o \
		mlpcalcfc.o ti_reference_app.o ti_fit_fc.o calc_eos.o mlpcalccellhess.o \
		split_set.o calc_ferr.o \
		calc_neb.o calc_dimer.o mlpnroptimize.o mtp2spline.o adjustpot.o testpot.o \
		mlpphdisp.o nnmtp.o calc_prop.o calc_descriptors.o \
		mtpcoeffsremap.o getidealpos.o \
		relaxstr2.o genmtm.o modefollowing.o trjsmooth_fft.o \
		removedup.o trainmtp.o train_nlpot.o \
		initmtpm.o cellutils.o sameq.o \
		remove_duplicate_structures.o fastfit.o \
		findint.o maxvol_select.o fps_select.o \
		train_linear.o \
	pot2table.o attach_hessian.o calc_crowding.o

OBJPOT = potential.o potential_eam_zhou_2004.o potential_mindist.o \
		potential_mtp.o basis.o cutoff.o xtrans.o ace_lmp.o potential_shell.o \
		potential_mtpnn1.o potential_hybmtp.o \
		potential_hybrid.o potential_hybrid_scale.o potential_hybrid_scale_curvature.o potential_mtp_symmetry.o potential_eamalloy_spline.o potential_lj.o potential_eamfs_spline.o \
		potential_mtp_m.o les.o quick_mtp.o potential_eam_any.o potential_eam_poly.o \
		potential_mtp_LES.o potential_descriptor_linear.o potential_descriptor_linear_curvature.o \
		potential_pair_table.o potential_threebody_table.o potential_sw.o \
		potential_tersoff.o potential_zbl.o \
		potential_openkim.o potential_wrapper.o standalone_export.o model_bundle.o gpm_codec.o \
		native_inference.o native_long_range.o native_profile.o mlip.o mtp_cpu.o radial_basis_setup.o

OBJDESC = descriptor_Behler.o descriptor_Behler_curvature.o \
		descriptor_BispectrumSO3.o descriptor_BispectrumSO3_curvature.o \
		descriptor_BispectrumSO4.o descriptor_BispectrumSO4_curvature.o descriptor_BispectrumSO4_global.o so4_wigner_spline.o \
		descriptor_BondOrder.o descriptor_BondOrder_curvature.o descriptor_MTM.o descriptor_wrapper.o

OBJIO = gpio.o gpio_session.o format_detection.o atoms_units.o xsf_fields.o \
		gpio_read.o gpio_write.o gpio_read_lmp.o gpio_read_cfg.o gpio_read_vasp.o \
		gpio_read_ms.o

OBJCP = compute_ha.o compute_voro.o compute_bondangle.o compute_rdf.o \
		compute_msd.o compute_boo.o trajectory_transform.o compute_vacf.o \
		compute_posavg.o compute_fftfilter.o \
		compute_sro.o compute_hessian.o compute_cna.o \
		lowest_eigenmodes.o \
		compute_wcsro.o

OBJBASE = structures.o \
		neighborhood.o anyneighborhoods.o \
		neighborhood_voro_radi.o neighborhood_cluster.o \
		neighborhood_cluster_simple.o neighborhood_cluster_shell_simple.o \
		neighborhood_shell.o neighborhood_ghost.o neighborhoodKD.o neighborhood_voro_direct.o \
		symmetry.o structure_comparison.o

OBJMISC = units.o lower_energy_envelope.o chem.o procstr.o gputils.o comm.o mpi_stubs.o spglib_driver.o \
		compressed_stream.o cmdopts.o minkowski_reduce.o \
		nn1hidden.o nn2hidden.o CutoffFuncs.o \
		combinatorics.o angular_functions.o intersitials.o symbasis.o \
		hessian_io.o maxvol.o online_maxvol.o active_selector.o b_matrix_builder.o \
		hessian_utils.o ti_harmonic_reference.o ti_reference_io.o elastic_utils.o \
		cur_selection.o fps_selector.o crowding.o spline.o

OBJMTM = mtm.o basisgen.o

OBJTRAIN = error_monitor.o training_data.o train_mtp.o train_nonlinear_potential.o train_linear_potential.o linear_refit.o linear_efs_system.o hvp_fit.o varpro.o train_varpro.o nonlinear_workflow.o nonlinear_schedule.o fixed_core.o sgd_scheduler.o

OBJAL = al_potential.o al_scorer.o gpm_al.o

OBJREL = dimer_new.o neb.o neb_solver.o neb_symmetry.o neb_perturbation.o neb_release.o relax_new.o relax_sqp.o constraint_expression.o nonlinear_coordinates.o relax_nonlinear_filter.o relax_preconditioner.o relax_metric.o relax_constraints_io.o constraints.o relax_modifiers.o relax_symm_constraint.o al_modifier.o

OBJCORR = CorrelationAnalyzer.o DescriptorCollector.o Decorrelator.o \
          ThresholdPruning.o ClusterRepresentative.o IterativeGreedyVIF.o \
          DiagnosticReport.o BasisAnalyzer.o MTPBasisEquivalence.o

OBJLIB = $(OBJPOT) $(OBJDESC) $(OBJIO) $(OBJCP) $(OBJBASE) $(OBJMISC) $(OBJREL) $(OBJAPP) $(OBJMTM) $(OBJTRAIN) $(OBJAL) $(OBJCORR)

CUDA_CORE_SRC := runtime/mtp/gpu-direct/mtp_gpu_direct.cu
CUDA_CORE_OBJ := mtp_gpu_direct_cuda.o
SOG_GPU_SRC := runtime/mtp/gpu-direct/sog_gpu.cu
SOG_CUDA_OBJ := sog_gpu_cuda.o
SOG_HIP_OBJ := sog_gpu_hip.o
CUDA_BUILD_DEPS :=
HIP_CORE_OBJ := mtp_gpu_direct_hip.o
ifneq ($(filter cuda mpi-cuda,$(BUILD_MODE)),)
OBJLIB += mtp_gpu_adapter.o $(CUDA_CORE_OBJ) $(SOG_CUDA_OBJ) native_gpu_inference.o native_long_range_gpu.o native_long_range_gpu_kernels_cuda.o native_gpu_ops_cuda.o
CUDA_BUILD_DEPS += $(BUILD_DIR)/$(CUDA_CORE_OBJ) $(BUILD_DIR)/$(SOG_CUDA_OBJ) $(BUILD_DIR)/native_long_range_gpu_kernels_cuda.o $(BUILD_DIR)/native_gpu_ops_cuda.o
endif

ifneq ($(filter hip mpi-hip,$(BUILD_MODE)),)
OBJLIB += mtp_gpu_adapter.o $(HIP_CORE_OBJ) $(SOG_HIP_OBJ) native_gpu_inference.o native_long_range_gpu.o native_long_range_gpu_kernels_hip.o native_gpu_ops_hip.o
CUDA_BUILD_DEPS += $(BUILD_DIR)/$(HIP_CORE_OBJ) $(BUILD_DIR)/$(SOG_HIP_OBJ) $(BUILD_DIR)/native_long_range_gpu_kernels_hip.o $(BUILD_DIR)/native_gpu_ops_hip.o
endif

GPU_HEADERS := runtime/mtp/gpu-direct/mtp_gpu_direct.h \
               runtime/mtp/gpu-direct/sog_gpu.h \
               runtime/mtp/gpu-direct/gpu_cache.h \
               runtime/mtp/gpu-direct/gpu_backend.h \
               runtime/mtp/gpu-direct/gpu_runtime.h \
               runtime/mtp/gpu-direct/radial_gpu_math.h potentials/native_profile.h
GPU_FLAGS_STAMP := $(BUILD_DIR)/.gpu-flags
$(GPU_FLAGS_STAMP): flags-force | $(BUILD_DIR)
	@new='$(NVCC) $(CUDA_ARCH) $(HIPCC) $(HIP_ARCH) $(HIP_CXXFLAGS) $(MTP_GPU_ENABLE_NVTX)'; \
	if test -f "$@" && test "$$(cat "$@")" = "$$new"; then exit 0; fi; \
	printf '%s' "$$new" > "$@"

$(BUILD_DIR)/$(HIP_CORE_OBJ): $(CUDA_CORE_SRC) $(GPU_HEADERS) $(GPU_FLAGS_STAMP) \
        runtime/mtp/mlip.h | $(BUILD_DIR)
	$(HIPCC) -x hip -std=c++20 -O3 -gline-tables-only -fPIC -DIMRMLP_HIP \
		$(HIP_CXXFLAGS) $(foreach arch,$(HIP_ARCH),--offload-arch=$(arch)) \
		-I. -I runtime/mtp -c $< -o $@

$(BUILD_DIR)/$(SOG_HIP_OBJ): $(SOG_GPU_SRC) $(GPU_HEADERS) $(GPU_FLAGS_STAMP) \
        runtime/mtp/mlip.h | $(BUILD_DIR)
	$(HIPCC) -x hip -std=c++20 -O3 -gline-tables-only -fPIC -DIMRMLP_HIP \
		$(HIP_CXXFLAGS) $(foreach arch,$(HIP_ARCH),--offload-arch=$(arch)) \
		-I. -I runtime/mtp -c $< -o $@

$(BUILD_DIR)/$(CUDA_CORE_OBJ): $(CUDA_CORE_SRC) $(GPU_HEADERS) $(GPU_FLAGS_STAMP) \
		runtime/mtp/gpu-direct/mtp_gpu_direct.h \
		runtime/mtp/mlip.h | $(BUILD_DIR)
	$(NVCC) -DIMRMLP_CUDA -std=c++20 -O3 -lineinfo -Xcompiler=-fPIC $(if $(CUDA_ARCH),-arch=$(CUDA_ARCH),) \
		$(if $(MTP_GPU_ENABLE_NVTX),-DMTP_GPU_ENABLE_NVTX=1,) \
		-I. -I runtime/mtp -c $< -o $@

$(BUILD_DIR)/$(SOG_CUDA_OBJ): $(SOG_GPU_SRC) $(GPU_HEADERS) $(GPU_FLAGS_STAMP) \
		runtime/mtp/gpu-direct/mtp_gpu_direct.h \
		runtime/mtp/mlip.h | $(BUILD_DIR)
	$(NVCC) -DIMRMLP_CUDA -std=c++20 -O3 -lineinfo -Xcompiler=-fPIC $(if $(CUDA_ARCH),-arch=$(CUDA_ARCH),) \
		$(if $(MTP_GPU_ENABLE_NVTX),-DMTP_GPU_ENABLE_NVTX=1,) \
		-I. -I runtime/mtp -c $< -o $@

$(BUILD_DIR)/native_long_range_gpu_kernels_cuda.o: potentials/native_long_range_gpu_kernels.cu potentials/native_long_range_gpu_kernels.h $(GPU_HEADERS) $(GPU_FLAGS_STAMP) | $(BUILD_DIR)
	$(NVCC) -DIMRMLP_CUDA -std=c++20 -O3 -lineinfo -Xcompiler=-fPIC $(if $(CUDA_ARCH),-arch=$(CUDA_ARCH),) -I. -c $< -o $@

$(BUILD_DIR)/native_long_range_gpu_kernels_hip.o: potentials/native_long_range_gpu_kernels.cu potentials/native_long_range_gpu_kernels.h $(GPU_HEADERS) $(GPU_FLAGS_STAMP) | $(BUILD_DIR)
	$(HIPCC) -x hip -DIMRMLP_HIP -std=c++20 -O3 -gline-tables-only -fPIC $(HIP_CXXFLAGS) $(foreach arch,$(HIP_ARCH),--offload-arch=$(arch)) -I. -c $< -o $@

$(BUILD_DIR)/native_gpu_ops_cuda.o: potentials/native_gpu_ops.cu potentials/native_gpu_ops.h $(GPU_HEADERS) $(GPU_FLAGS_STAMP) | $(BUILD_DIR)
	$(NVCC) -DIMRMLP_CUDA -std=c++20 -O3 -lineinfo -Xcompiler=-fPIC $(if $(CUDA_ARCH),-arch=$(CUDA_ARCH),) -I. -c $< -o $@

$(BUILD_DIR)/native_gpu_ops_hip.o: potentials/native_gpu_ops.cu potentials/native_gpu_ops.h $(GPU_HEADERS) $(GPU_FLAGS_STAMP) | $(BUILD_DIR)
	$(HIPCC) -x hip -DIMRMLP_HIP -std=c++20 -O3 -gline-tables-only -fPIC $(HIP_CXXFLAGS) $(foreach arch,$(HIP_ARCH),--offload-arch=$(arch)) -I. -c $< -o $@

# ── Vendored libs ──────────────────────────────────────────────────────

DEPLIBS := deplibs/libtinyxml2.a \
           deplibs/libvoro++.a \
           deplibs/libsymspg.a \
           deplibs/libfmt.a
$(DEPLIBS) &:
	$(MAKE) -C deplibs all

deps: $(DEPLIBS)

LINKOPT := -Wl,-rpath,'$$ORIGIN' -L$(BUILD_DIR) $(LIBS) $(DEPLIBS)

# ── Targets ───────────────────────────────────────────────────────────
# Keep source/object basenames stable, but give MPI and CUDA executable
# artifacts explicit suffixes so installations cannot be confused.
CLI_TARGETS_BASE := calc_fitness trj_rmdrift subtract_pot
MPI_BUILD := $(filter mpi mpi-debug mpi-cuda mpi-hip,$(BUILD_MODE))
CUDA_BUILD := $(filter cuda hip,$(BUILD_MODE))
GPU_SUFFIX := $(if $(filter hip,$(BUILD_MODE)),-hip,-cu)
ifneq ($(MPI_BUILD),)
IMRMLP_LIB_NAME := libimrmlp-mpi.so
IMRMLP_LINK_NAME := imrmlp-mpi
IMRMLP_STAND_NAME := imrmlp-mpi
TARGET_SUFFIX := -mpi
TARGETS := $(addsuffix $(TARGET_SUFFIX), $(CLI_TARGETS_BASE))
else ifneq ($(CUDA_BUILD),)
IMRMLP_LIB_NAME := libimrmlp.so
IMRMLP_LINK_NAME := imrmlp
IMRMLP_STAND_NAME := imrmlp$(GPU_SUFFIX)
TARGET_SUFFIX := $(GPU_SUFFIX)
TARGETS := $(addsuffix $(TARGET_SUFFIX), $(CLI_TARGETS_BASE))
else
IMRMLP_LIB_NAME := libimrmlp.so
IMRMLP_LINK_NAME := imrmlp
IMRMLP_STAND_NAME := imrmlp-stand
TARGET_SUFFIX :=
TARGETS := imrmlp $(CLI_TARGETS_BASE)
endif

TARGETS_BD    := $(addprefix $(BUILD_DIR)/, $(TARGETS))

TARGETSLIB    := $(BUILD_DIR)/$(IMRMLP_LIB_NAME)
TARGETSLIB_A  := $(BUILD_DIR)/libimrmlp.a
TARGET_STAND  := $(BUILD_DIR)/$(IMRMLP_STAND_NAME)
ifneq ($(MPI_BUILD)$(CUDA_BUILD),)
ALL_MAIN_TARGET := $(TARGET_STAND)
INSTALL_MAIN_TARGET := $(TARGET_STAND)
else
ALL_MAIN_TARGET :=
INSTALL_MAIN_TARGET :=
endif

.PHONY: all clean deps depc test install updinst updatelib run_tests clean_tests
.PHONY: clean-debug clean-release clean-asan clean-relwithdebinfo clean-relwithdebinfo-asan clean-mpi clean-mpi-debug clean-cuda clean-mpi-cuda clean-all
.PHONY: libimrmlp.so libimrmlp-mpi.so libimrmlp.a main python imrmlp-stand imrmlp-mpi imrmlp-cu
.PHONY: doc doc-html doc-pdf doc-zh doc-zh-html doc-clean
.PHONY: $(TARGETS) $(CLI_TARGETS_BASE)

# ── Build ──────────────────────────────────────────────────────────────

# Phony alias: make libimrmlp.so → build/<mode>/<mode-specific library>
libimrmlp.so: $(TARGETSLIB)

libimrmlp-mpi.so: $(TARGETSLIB)

$(TARGETSLIB): deps
	$(CXX) -shared $(LFLAG) -o $@ $(call bd,$(OBJLIB)) $(DEPLIBS) $(LIBS)

libimrmlp.a: $(TARGETSLIB_A)

$(TARGETSLIB_A): deps
	$(RM) $@
	ar rcs $@ $(call bd,$(OBJLIB)) $(DEPLIBS)

# Standalone static binary (no libimrmlp.so dependency)
main: $(TARGET_STAND)

$(TARGET_STAND): deps
	$(CXX) -o $@ $(call bd,imrmlp.o) $(call bd,$(OBJLIB)) $(DEPLIBS) $(LIBS) $(LINKOPT) $(LFLAG)

imrmlp-stand: main
imrmlp-mpi: main
imrmlp-cu: main
imrmlp-hip: main

# Binaries that link against the mode-specific shared library. The output name
# may carry an MPI or CUDA suffix, while the corresponding object keeps its
# source basename.
define _LINK_TARGET_
$(BUILD_DIR)/$(1): $(TARGETSLIB)
	$$(CXX) -o $$@ $(BUILD_DIR)/$(2).o $$(LIBS) -L$$(BUILD_DIR) -l$$(IMRMLP_LINK_NAME) $$(LINKOPT) $$(LFLAG)
endef
$(foreach t,$(TARGETS),$(eval $(call _LINK_TARGET_,$(t),$(patsubst %-hip,%,$(patsubst %-cu,%,$(patsubst %-mpi,%,$(t)))))))

# Phony aliases: make imrmlp or make <tool> resolves to the mode-specific
# executable name. MPI and CUDA modes use their static primary executable.
define _PHONY_TARGET_
.PHONY: $(1)
$(1): $(2)
endef
ifneq ($(MPI_BUILD),)
$(eval $(call _PHONY_TARGET_,imrmlp,$(TARGET_STAND)))
$(foreach t,$(CLI_TARGETS_BASE),$(eval $(call _PHONY_TARGET_,$(t),$(BUILD_DIR)/$(t)-mpi)))
$(foreach t,$(TARGETS),$(eval $(call _PHONY_TARGET_,$(t),$(BUILD_DIR)/$(t))))
else ifneq ($(CUDA_BUILD),)
$(eval $(call _PHONY_TARGET_,imrmlp,$(TARGET_STAND)))
$(eval $(call _PHONY_TARGET_,imrmlp$(GPU_SUFFIX),$(TARGET_STAND)))
$(foreach t,$(CLI_TARGETS_BASE),$(eval $(call _PHONY_TARGET_,$(t),$(BUILD_DIR)/$(t)$(GPU_SUFFIX))))
$(foreach t,$(TARGETS),$(eval $(call _PHONY_TARGET_,$(t),$(BUILD_DIR)/$(t))))
else
$(foreach t,$(TARGETS),$(eval $(call _PHONY_TARGET_,$(t),$(BUILD_DIR)/$(t))))
endif

# Compile only sources used by a delivered library or command.
DELIVERED_OBJS := $(sort $(OBJLIB) imrmlp.o $(addsuffix .o,$(CLI_TARGETS_BASE)))
deps: $(call bd,$(DELIVERED_OBJS))
deps: $(CUDA_BUILD_DEPS)

# Clean subdirectories + build dir. The subdir clean loops must not prevent
# the build-dir removal (a failing subdir clean used to leave stale objects
# behind when switching compile flags such as USE_GZIP).
depc:
	-for d in $(SRCDIRS); do $(MAKE) --directory=$$d clean 2>/dev/null; done
	-rm -rf build

PYTHON ?= python3
PYTHON_EXTENSION := $(shell $(PYTHON) -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))")
PYTHON_MODULE := imrmlp/_imrmlp$(PYTHON_EXTENSION)
python: $(PYTHON_MODULE)
$(PYTHON_MODULE): python_bind/_imrmlp_module.cpp $(TARGETSLIB)
	$(CXX) $(CXXFLAGS_NO_PCH) -I. `$(PYTHON) -m pybind11 --includes` $< \
		-shared -o $@ -L$(BUILD_DIR) -l$(IMRMLP_LINK_NAME) $(LIBS) \
		-Wl,-rpath,'$$ORIGIN:$$ORIGIN/../$(BUILD_DIR)' $(LFLAG)

all: deps $(TARGETSLIB) $(TARGETS_BD) $(ALL_MAIN_TARGET)
test: deps $(TARGETSLIB) run_tests

# ── Clean targets ─────────────────────────────────────────────────────
# Clean current mode
clean: clean_tests
	-rm -rf $(BUILD_DIR)

# Clean specific modes
clean-debug:
	-rm -rf build/debug

clean-release:
	-rm -rf build/release

clean-asan:
	-rm -rf build/asan

clean-relwithdebinfo:
	-rm -rf build/relwithdebinfo

clean-relwithdebinfo-asan:
	-rm -rf build/relwithdebinfo-asan

clean-mpi:
	-rm -rf build/mpi

clean-mpi-debug:
	-rm -rf build/mpi-debug

clean-cuda:
	-rm -rf build/cuda

clean-mpi-cuda:
	-rm -rf build/mpi-cuda

.PHONY: clean-hip clean-mpi-hip imrmlp-hip
clean-hip:
	-rm -rf build/hip

clean-mpi-hip:
	-rm -rf build/mpi-hip

# Clean all modes
clean-all: clean_tests
	-rm -rf build

# Keep native installations and Python wheels on the same license manifest.
LICENSE_FILES := $(shell cat config/license-files.txt)
LICENSE_INSTALL_DIR ?= $(PREFIX)/share/licenses/imrmlp
.PHONY: install-licenses
install-licenses: $(LICENSE_FILES)
	@for license in $(LICENSE_FILES); do \
		install -D -m 644 "$$license" "$(LICENSE_INSTALL_DIR)/$$license" || exit 1; \
	done

install updatelib updinst: install-licenses

ifneq ($(MPI_BUILD),)
updatelib: $(TARGETSLIB)
	mkdir -p $(PREFIX)/lib
	cp $(TARGETSLIB) $(PREFIX)/lib
install: all updatelib
	mkdir -p $(PREFIX)/lib $(PREFIX)/bin
	cp $(TARGET_STAND) $(TARGETS_BD) $(PREFIX)/bin
updinst: install
else
updatelib: $(TARGETSLIB)
	mkdir -p $(PREFIX)/lib
	cp $(TARGETSLIB) $(PREFIX)/lib
install: all updatelib
	mkdir -p $(PREFIX)/bin
	cp $(INSTALL_MAIN_TARGET) $(TARGETS_BD) $(PREFIX)/bin
updinst:
	mkdir -p $(PREFIX)/bin
	cp $(INSTALL_MAIN_TARGET) $(TARGETS_BD) $(PREFIX)/bin
	cp $(TARGETSLIB) $(PREFIX)/lib
endif

# ── Documentation ──────────────────────────────────────────────────────
DOXYGEN := doxygen
DOCS_DIR := docs
DOCS_PYTHON ?= build/docs-env/bin/python
# 0 = report command/locale drift without failing; 1 = fail on command,
# option and alias differences and cross-language links.
DOC_CHECK_STRICT ?= 0

.PHONY: doc-deps doc-manual check-doc-html
doc-deps:
	mkdir -p build/docs-deps
	cp $(DOCS_DIR)/package.json $(DOCS_DIR)/package-lock.json build/docs-deps/
	npm ci --prefix build/docs-deps --ignore-scripts --no-audit --no-fund
	python3 -m venv build/docs-env
	$(DOCS_PYTHON) -m pip install -r $(DOCS_DIR)/requirements.txt

doc-manual: doc-html
	$(DOCS_PYTHON) docs/tools/build_manual.py

check-doc-html:
	python3 docs/tools/check_doc_html.py --site-root $(DOCS_DIR) $(DOCS_DIR)/build/html $(DOCS_DIR)/build_zh/html $(DOCS_DIR)/manual-html/en $(DOCS_DIR)/manual-html/zh
	node docs/tools/check_doc_search.cjs $(DOCS_DIR)/manual-html/en $(DOCS_DIR)/manual-html/zh
	python3 docs/tools/check_doc_math.py $(DOCS_DIR)/manual-html/en $(DOCS_DIR)/manual-html/zh

doc: doc-html doc-pdf

doc-html: doc-catalog
	python3 docs/tools/build_docs.py --all --doxygen $(DOXYGEN)

doc-pdf: doc-html
	cd $(DOCS_DIR)/build/latex && $(MAKE) 2>/dev/null || true

doc-zh: doc-zh-html

doc-zh-html: doc-html

doc-clean:
	rm -rf $(DOCS_DIR)/build $(DOCS_DIR)/build_zh $(DOCS_DIR)/manual-html

# ── Documentation consistency checks ────────────────────────────────────
.PHONY: check-docs doc-catalog
doc-catalog:
	python3 docs/tools/doc_catalog.py --write

check-docs:
	python3 docs/tools/test_doc_checks.py
	python3 docs/tools/doc_catalog.py
	python3 docs/tools/check_doc_links.py
	python3 docs/tools/check_doc_consistency.py
	DOC_CHECK_STRICT=$(DOC_CHECK_STRICT) python3 docs/tools/check_doc_commands.py
	DOC_CHECK_STRICT=$(DOC_CHECK_STRICT) python3 docs/tools/check_doc_locale.py

# ── Auto-generated deps + test infra ───────────────────────────────────
-include $(OBJDEPS)
include unitest/tests.mk
