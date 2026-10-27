# Build mode (default: release)
#   make BUILD_MODE=<mode> <target>
#   Valid modes: debug, release, asan, relwithdebinfo, relwithdebinfo-asan,
#                cuda, mpi-cuda, hip, mpi-hip
#   MPI modes:  mpi, mpi-debug, mpi-cuda
BUILD_MODE ?= release
CPU_ARCH_FLAGS ?= -march=native

MKL_LIBDIR ?= $(if $(wildcard $(MKLROOT)/lib/intel64),$(MKLROOT)/lib/intel64,$(MKLROOT)/lib)

CUDA_HOME ?= /opt/cuda
NVCC ?= $(CUDA_HOME)/bin/nvcc
HIP_ROOT ?= $(if $(ROCM_PATH),$(ROCM_PATH),/opt/rocm)
HIPCC ?= $(HIP_ROOT)/bin/hipcc
HIP_ARCH ?=
HIP_CXXFLAGS ?=
CUDA_ARCH ?=

IMRMLP_DEFINES_BASE = -DUSE_OMP -DHASHED_VORO -DNO_SAMPLE_STR -DBOXED_NEBH -DUSEXML \
		-DEIGEN_USE_MKL_ALL \
		-DAE_CPU=AE_INTEL -DAE_OS=AE_LINUX \
		-Werror=return-type
ifeq ($(BUILD_MODE),debug)
	IMRMLP_DEFINES = $(IMRMLP_DEFINES_BASE)
	OPTINC = -pipe -fopenmp -fPIC $(CPU_ARCH_FLAGS) -pthread -Og -g
	CXX = g++ -fno-omit-frame-pointer
	LFLAG =
else ifeq ($(BUILD_MODE),release)
	IMRMLP_DEFINES = $(IMRMLP_DEFINES_BASE) -DEIGEN_NO_DEBUG -DNDEBUG
	OPTINC = -pipe -fopenmp -fPIC $(CPU_ARCH_FLAGS) -pthread -O3
	CXX = g++ -fno-omit-frame-pointer
	LFLAG =
else ifeq ($(BUILD_MODE),asan)
	IMRMLP_DEFINES = $(IMRMLP_DEFINES_BASE)
	OPTINC = -pipe -fopenmp -fPIC $(CPU_ARCH_FLAGS) -pthread -Og -g
	CXX = g++ -fsanitize=address -fno-omit-frame-pointer -g
	LFLAG = -fsanitize=address
else ifeq ($(BUILD_MODE),relwithdebinfo)
	IMRMLP_DEFINES = $(IMRMLP_DEFINES_BASE)
	OPTINC = -pipe -fopenmp -fPIC $(CPU_ARCH_FLAGS) -pthread -O2 -g
	CXX = g++ -fno-omit-frame-pointer
	LFLAG =
else ifeq ($(BUILD_MODE),relwithdebinfo-asan)
	IMRMLP_DEFINES = $(IMRMLP_DEFINES_BASE)
	OPTINC = -pipe -fopenmp -fPIC $(CPU_ARCH_FLAGS) -pthread -O2 -g
	CXX = g++ -fsanitize=address -fno-omit-frame-pointer -g
	LFLAG = -fsanitize=address
else ifeq ($(BUILD_MODE),mpi)
	IMRMLP_DEFINES = $(IMRMLP_DEFINES_BASE) -DEIGEN_NO_DEBUG -DNDEBUG -DIMRMLP_MPI -DEIGEN_DONT_PARALLELIZE
	OPTINC = -pipe -fopenmp -fPIC $(CPU_ARCH_FLAGS) -pthread -O3
	CXX = mpicxx -fno-omit-frame-pointer
	LFLAG =
else ifeq ($(BUILD_MODE),mpi-debug)
	IMRMLP_DEFINES = $(IMRMLP_DEFINES_BASE) -DIMRMLP_MPI -DEIGEN_DONT_PARALLELIZE
	OPTINC = -pipe -fopenmp -fPIC $(CPU_ARCH_FLAGS) -pthread -Og -g
	CXX = mpicxx -fno-omit-frame-pointer
	LFLAG =
else ifeq ($(BUILD_MODE),cuda)
	IMRMLP_DEFINES = $(IMRMLP_DEFINES_BASE) -DEIGEN_NO_DEBUG -DNDEBUG -DIMRMLP_CUDA -DIMRMLP_GPU
	OPTINC = -pipe -fopenmp -fPIC $(CPU_ARCH_FLAGS) -pthread -O3
	CXX = g++ -fno-omit-frame-pointer
	LFLAG =
else ifeq ($(BUILD_MODE),mpi-cuda)
	IMRMLP_DEFINES = $(IMRMLP_DEFINES_BASE) -DEIGEN_NO_DEBUG -DNDEBUG -DIMRMLP_MPI -DIMRMLP_CUDA -DIMRMLP_GPU -DEIGEN_DONT_PARALLELIZE
	OPTINC = -pipe -fopenmp -fPIC $(CPU_ARCH_FLAGS) -pthread -O3
	CXX = mpicxx -fno-omit-frame-pointer
	LFLAG =
else ifeq ($(BUILD_MODE),hip)
	IMRMLP_DEFINES = $(IMRMLP_DEFINES_BASE) -DEIGEN_NO_DEBUG -DNDEBUG -DIMRMLP_HIP -DIMRMLP_GPU
	OPTINC = -pipe -fopenmp -fPIC $(CPU_ARCH_FLAGS) -pthread -O3
	CXX = g++ -fno-omit-frame-pointer
	LFLAG =
else ifeq ($(BUILD_MODE),mpi-hip)
	IMRMLP_DEFINES = $(IMRMLP_DEFINES_BASE) -DEIGEN_NO_DEBUG -DNDEBUG -DIMRMLP_MPI -DIMRMLP_HIP -DIMRMLP_GPU -DEIGEN_DONT_PARALLELIZE
	OPTINC = -pipe -fopenmp -fPIC $(CPU_ARCH_FLAGS) -pthread -O3
	CXX = mpicxx -fno-omit-frame-pointer
	LFLAG =
else
	$(error Unknown BUILD_MODE: $(BUILD_MODE). Valid: debug, release, asan, relwithdebinfo, relwithdebinfo-asan, mpi, mpi-debug, cuda, mpi-cuda, hip, mpi-hip)
endif

# MPI modes use sequential MKL (parallelism from multi-process)
ifneq ($(filter mpi mpi-debug mpi-cuda mpi-hip,$(BUILD_MODE)),)
LIBS = -fopenmp \
		-lm -ldl -lpthread \
		-lfftw3 \
		-m64 -L$(MKL_LIBDIR) -Wl,--no-as-needed -lmkl_intel_lp64 -lmkl_sequential -lmkl_core -lpthread -lm -ldl
else
LIBS = -fopenmp \
		-lm -ldl -lpthread \
		-lfftw3 \
		-m64 -L$(MKL_LIBDIR) -Wl,--no-as-needed -lmkl_intel_lp64 -lmkl_gnu_thread -lmkl_core -lgomp -lpthread -lm -ldl
endif

# Optional GPM block decompression. Writers default to uncompressed blocks.
USE_ZSTD ?= 0
ifeq ($(USE_ZSTD),1)
IMRMLP_DEFINES_BASE += -DUSE_ZSTD
LIBS += -lzstd
endif

# Gzip file I/O. Enabled by default; disable with make USE_GZIP=0.
# Manual -DUSE_GZIP in CXXFLAGS also requires -lz in the link flags.
USE_GZIP ?= 1
ifeq ($(USE_GZIP),1)
IMRMLP_DEFINES_BASE += -DUSE_GZIP
LIBS += -lz
endif

ifneq ($(filter cuda mpi-cuda,$(BUILD_MODE)),)
LIBS += -L$(CUDA_HOME)/lib64 -lcudart
ifneq ($(MTP_GPU_ENABLE_NVTX),)
LIBS += -lnvToolsExt
endif
endif

ifneq ($(filter hip mpi-hip,$(BUILD_MODE)),)
LIBS += -L$(HIP_ROOT)/lib -L$(HIP_ROOT)/lib64 -L$(HIP_ROOT)/hip/lib -lamdhip64
ifneq ($(MTP_GPU_ENABLE_NVTX),)
$(error MTP_GPU_ENABLE_NVTX is available only with the CUDA backend)
endif
endif

# OpenKIM is enabled by default and can be disabled on compute systems that do
# not provide KIM API, e.g. `make USE_KIM=0 ...`.
USE_KIM ?= 1
ifeq ($(USE_KIM),1)
IMRMLP_DEFINES_BASE += -DUSE_KIM
LIBINC += -I/usr/include/kim-api
LIBS += -lkim-api
endif

# ── ccache: transparent compilation caching ──────────────────────────────
# Prepend ccache to CXX so unchanged headers/inputs skip recompilation.
# ccache is optional — silently skipped if not installed.
ifneq ($(shell command -v ccache 2>/dev/null),)
CXX := ccache $(CXX)
endif

DEPLIB ?= $(CURDIR)/deplibs
EIGEN_DIR ?= /usr/include/eigen3
DEPS_DIR ?= $(patsubst %/,%,$(dir $(EIGEN_DIR)))
ALGLIB_INCLUDE_ROOT ?= /usr/include
ALGLIB_LIBS ?= -lalglib
LIBS += $(ALGLIB_LIBS)
LIBINC += -I$(ALGLIB_INCLUDE_ROOT) -I$(DEPLIB) -I$(DEPLIB)/fmt/include -I$(DEPLIB)/LBFGSpp/include -I$(EIGEN_DIR) \
	-I$(DEPS_DIR) \
	$(if $(MKLROOT),-I$(MKLROOT)/include,) \
	-I$(DEPLIB)/spglib
PREFIX ?= $(HOME)/.local

CXXFLAGS := $(IMRMLP_DEFINES) $(LIBINC) $(OPTINC) -std=c++23
CPPFLAGS := -MMD -MP
