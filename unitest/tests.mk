# Catch2 unit test infrastructure (system-installed Catch2 v3)
CATCH2_LIBS = -lCatch2Main -lCatch2
TEST_CXXFLAGS = $(CXXFLAGS) -I.

TEST_DATA_ROOT ?= $(CURDIR)/tests
UNITEST_DATA_ROOT ?= $(CURDIR)/unitest/test-data

# All Catch2 test sources: unitest/ut_*.cpp
CATCH2_SRCS = $(wildcard unitest/ut_*.cpp)
ifneq ($(findstring mpi,$(BUILD_MODE)),)
CATCH2_SRCS := $(filter-out unitest/ut_mpi_stubs.cpp unitest/ut_mtmbasis_model_equivalence.cpp,$(CATCH2_SRCS))
else
CATCH2_SRCS := $(filter-out unitest/ut_mtp_sgd_mpi.cpp unitest/ut_mtmbasis_transport_mpi.cpp unitest/ut_training_workflow_mpi.cpp,$(CATCH2_SRCS))
endif
# This executable owns MPI initialization and GPU collective assertions.
ifeq ($(filter mpi-cuda mpi-hip,$(BUILD_MODE)),)
CATCH2_SRCS := $(filter-out unitest/ut_gpu_training_mpi.cpp,$(CATCH2_SRCS))
endif
# The VP MPI executable owns MPI initialization.
ifeq ($(findstring mpi,$(BUILD_MODE)),)
CATCH2_SRCS := $(filter-out unitest/ut_varpro_mpi.cpp,$(CATCH2_SRCS))
endif
ifeq ($(filter cuda hip mpi-cuda mpi-hip,$(BUILD_MODE)),)
# This executable contains real-device contracts only.
CATCH2_SRCS := $(filter-out unitest/ut_standalone_gpu.cpp unitest/ut_mtp_cuda.cpp unitest/ut_sog_gpu.cpp unitest/ut_gpu_nonlinear_cg.cpp,$(CATCH2_SRCS))
endif
ifneq ($(findstring mpi,$(BUILD_MODE)),)
# GPU CG tests require a single-process GPU build.
CATCH2_SRCS := $(filter-out unitest/ut_gpu_nonlinear_cg.cpp,$(CATCH2_SRCS))
endif
CATCH2_BINS = $(CATCH2_SRCS:unitest/%.cpp=$(BUILD_DIR)/unitest/%.test)
CATCH2_OBJS = $(CATCH2_SRCS:unitest/%.cpp=$(BUILD_DIR)/unitest/%.o)
.SECONDARY: $(CATCH2_OBJS)
# Honor header and included implementation dependencies emitted by -MMD.
# Several command regressions include standalone .cpp files directly.
-include $(CATCH2_OBJS:.o=.d)

# Link flags for test binaries — link against build-dir libimrmlp.so
TEST_LIBS = $(CATCH2_LIBS) -L$(BUILD_DIR) -l$(IMRMLP_LINK_NAME) $(LIBS) $(DEPLIBS)

# Pre-main stdio init (see unitest/main_init.cpp): linked into every test
# binary so the lazy sync_with_stdio(false) in GPIO's constructor does not
# orphan Catch2's reporter (which captures std::cout's rdbuf at startup).
UT_MAIN_INIT = $(BUILD_DIR)/unitest/main_init.o
$(UT_MAIN_INIT): $(PCH) $(FLAGS_STAMP)
$(UT_MAIN_INIT): unitest/main_init.cpp | $(BUILD_DIR)
	$(CXX) $(TEST_CXXFLAGS) -c $< -o $@

# Build test binary into build/<mode>/unitest/
# Depends on the flag stamp so toggling USE_GZIP rebuilds the tests
# (their per-codec TEST_CASEs are #ifdef'd on the same macros).
$(BUILD_DIR)/unitest/%.o: unitest/%.cpp $(PCH) $(FLAGS_STAMP) | $(BUILD_DIR)
	$(CXX) $(TEST_CXXFLAGS) -c $< -o $@

$(BUILD_DIR)/unitest/%.test: $(BUILD_DIR)/unitest/%.o $(TARGETSLIB) $(UT_MAIN_INIT) | $(BUILD_DIR)
	$(CXX) $< $(UT_MAIN_INIT) $(TEST_LIBS) $(LFLAG) -o $@

# Run all Catch2 tests in a temp directory to prevent CWD pollution.
# Tests/analyse code that writes to hardcoded relative paths (e.g. *.dat)
# will write to the temp dir instead of the project root.
run_tests: $(CATCH2_BINS)
	@echo "=== Running unit tests ($(BUILD_MODE) mode) ==="
	@passed=0; failed=0; \
	tmpdir=$$(mktemp -d /tmp/imrmlp_unitest_XXXXXX); \
	ln -sf $(TEST_DATA_ROOT) $$tmpdir/tests; \
	mkdir -p $$tmpdir/unitest; \
	ln -sf $(UNITEST_DATA_ROOT) $$tmpdir/unitest/test-data; \
	for t in $(CATCH2_BINS); do \
		echo "--- $$t ---"; \
		ret=0; \
		(cd $$tmpdir && LD_LIBRARY_PATH="$(CURDIR)/$(BUILD_DIR):$$LD_LIBRARY_PATH" $(CURDIR)/$$t) || ret=$$?; \
		if [ $$ret -eq 0 ]; then \
			passed=$$((passed + 1)); \
		else \
			failed=$$((failed + 1)); \
		fi; \
		echo ""; \
	done; \
	rm -rf $$tmpdir; \
	echo "=== Results: $$passed passed, $$failed failed ==="; \
	[ $$failed -eq 0 ]

clean_tests:
	-rm -f $(CATCH2_BINS) $(CATCH2_OBJS) $(CATCH2_OBJS:.o=.d)

.PHONY: run_tests clean_tests

# Representative local feedback; make test remains the complete C++ suite.
FAST_TEST_NAMES := ut_atoms ut_cmdopts ut_parameters ut_gpio_binary ut_gpio_extxyz \
                   ut_maxvol ut_neb ut_gpm ut_fastfit ut_modefollowing
FAST_TEST_BINS := $(addprefix $(BUILD_DIR)/unitest/,$(addsuffix .test,$(FAST_TEST_NAMES)))

test-fast: deps $(TARGETSLIB) $(FAST_TEST_BINS)
	@echo "=== Running representative unit tests ($(BUILD_MODE) mode) ==="
	@passed=0; failed=0; \
	tmpdir=$$(mktemp -d /tmp/imrmlp_unitest_fast_XXXXXX); \
	ln -sf $(TEST_DATA_ROOT) $$tmpdir/tests; \
	mkdir -p $$tmpdir/unitest; \
	ln -sf $(UNITEST_DATA_ROOT) $$tmpdir/unitest/test-data; \
	for t in $(FAST_TEST_BINS); do \
		echo "--- $$t ---"; ret=0; \
		(cd $$tmpdir && LD_LIBRARY_PATH="$(CURDIR)/$(BUILD_DIR):$$LD_LIBRARY_PATH" $(CURDIR)/$$t) || ret=$$?; \
		if [ $$ret -eq 0 ]; then passed=$$((passed + 1)); else failed=$$((failed + 1)); fi; \
	done; \
	rm -rf $$tmpdir; \
	echo "=== Results: $$passed passed, $$failed failed ==="; \
	[ $$failed -eq 0 ]

python-test-fast:
	python3 -m pytest -m 'not extended'

.PHONY: test-fast python-test-fast
