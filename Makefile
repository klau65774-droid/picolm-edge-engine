.DEFAULT_GOAL := native
.DELETE_ON_ERROR:
.SECONDEXPANSION:

ifeq ($(origin CC),default)
CC = gcc
endif
CROSS_PI_CC ?= aarch64-linux-gnu-gcc
ARM32_CC ?= arm-linux-gnueabihf-gcc

CFLAGS ?= -O3
C11 = -std=c11 -Wall -Wextra -Wpedantic
POSIX = -D_POSIX_C_SOURCE=200809L
POSIX_LIBS = -lm -pthread
ARM64 = -march=armv8-a+simd
ARM5 = -march=armv8.2-a+dotprod+fp16
ARM32 = -march=armv6 -marm -mfpu=vfp -mfloat-abi=hard
ARM32_LIBS = -latomic -static
SCALAR_FLAGS = -DPICOLM_SCALAR -fno-tree-vectorize -ffp-contract=off
TARGET_TRIPLE := $(shell $(CC) -dumpmachine)
WINDOWS_MAKE := $(findstring mingw,$(MAKE_HOST))

ifneq ($(findstring mingw,$(TARGET_TRIPLE)),)
EXE = .exe
HOST_FLAGS = -D_WIN32_WINNT=0x0600
HOST_LINK = -static -lpsapi -lm
CLI_LINK = -municode
else
HOST_FLAGS = $(POSIX)
HOST_LINK = $(POSIX_LIBS)
endif

ifneq ($(filter x86_64% i686%,$(TARGET_TRIPLE)),)
NATIVE_FLAGS = -march=native -mavx2 -mfma
else
NATIVE_FLAGS = -march=native
endif

MODEL = src/model.c src/gguf.c src/kvcache.c src/sha256.c src/tensor.c src/threadpool.c src/quant.c
SOURCES = src/picolm.c $(MODEL) src/tokenizer.c src/sampler.c src/grammar.c
HEADERS = $(wildcard src/*.h)
OPT = picolm_opt$(EXE)
SCALAR = picolm_scalar$(EXE)
DEBUG = picolm_debug$(EXE)
TEST_NAMES = quant quant_scalar threadpool grammar tokenizer sampler gguf_fuzz
TESTS = $(addprefix tests/test_,$(addsuffix $(EXE),$(TEST_NAMES)))

.PHONY: native scalar pi pi5 pi-arm32 cross-pi static debug test clean FORCE_REBUILD
native: $(OPT)
scalar: $(SCALAR)
pi: picolm_pi
pi5: picolm_pi5
pi-arm32: picolm_pi_arm32
cross-pi: picolm_cross_pi
static: picolm_static$(EXE)
debug: $(DEBUG)

# Compare the exact compiler invocation with the last successful invocation.
# Stage the signature first, but publish it only after the compiler succeeds.
command_matches = $(and $(findstring $(2),$(file <$(1).build-command)),$(findstring $(file <$(1).build-command),$(2)))
command_dependency = $(if $(call command_matches,$(1),$(2)),,FORCE_REBUILD)
short_flags = $(filter-out -%,$(firstword $(MAKEFLAGS)))
simulate = $(or $(findstring n,$(short_flags)),$(findstring q,$(short_flags)),$(findstring t,$(short_flags)))
stage_command = $(if $(simulate),,$(file >$(1).build-command.tmp,$(2)))

ifneq ($(WINDOWS_MAKE),)
commit_command = powershell -NoProfile -Command "Move-Item -Force -LiteralPath '$(1).build-command.tmp' -Destination '$(1).build-command' -ErrorAction Stop"
else
commit_command = mv -f $(1).build-command.tmp $(1).build-command
endif

compile = $(1) $(CPPFLAGS) -Isrc $(2) $(C11) $(3) -o $(4) $(5) $(LDFLAGS) $(6)
host_cli = $(call compile,$(CC),$(1),$(HOST_FLAGS) $(2) $(CLI_LINK),$(3),$(SOURCES),$(HOST_LINK) $(4))
arm_cli = $(call compile,$(1),$(CFLAGS),$(POSIX) $(2),$(3),$(SOURCES),$(POSIX_LIBS) $(4))

define executable
command_$(1) = $(3)
$(1): $(2) $(HEADERS) $(4) Makefile $$$$(call command_dependency,$$$$@,$$$$(command_$(1)))
	$$(call stage_command,$$@,$$(command_$(1)))$$(command_$(1))
	@$$(call commit_command,$$@)
endef

$(eval $(call executable,$(OPT),$(SOURCES),$$(call host_cli,$$(CFLAGS),$$(NATIVE_FLAGS),$(OPT))))
$(eval $(call executable,$(SCALAR),$(SOURCES),$$(call host_cli,$$(CFLAGS),$$(SCALAR_FLAGS),$(SCALAR))))
$(eval $(call executable,$(DEBUG),$(SOURCES),$$(call host_cli,-O0 -g,,$(DEBUG))))
$(eval $(call executable,picolm_static$(EXE),$(SOURCES),$$(call host_cli,$$(CFLAGS),$$(NATIVE_FLAGS),picolm_static$(EXE),-static)))
$(eval $(call executable,picolm_pi,$(SOURCES),$$(call arm_cli,$$(CC),$$(ARM64),picolm_pi)))
$(eval $(call executable,picolm_pi5,$(SOURCES),$$(call arm_cli,$$(CC),$$(ARM5),picolm_pi5)))
$(eval $(call executable,picolm_pi_arm32,$(SOURCES),$$(call arm_cli,$$(ARM32_CC),$$(ARM32),picolm_pi_arm32,$$(ARM32_LIBS))))
$(eval $(call executable,picolm_cross_pi,$(SOURCES),$$(call arm_cli,$$(CROSS_PI_CC),$$(ARM64),picolm_cross_pi,-static)))

test_compile = $(call compile,$(CC),$(CFLAGS),$(HOST_FLAGS) $(3),$(1),$(2),$(HOST_LINK))
define register_test
$(call executable,tests/test_$(1)$(EXE),tests/test_$(2).c $(3),$$(call test_compile,tests/test_$(1)$(EXE),tests/test_$(2).c $(3),$(4)),tests/check.h)
endef
$(eval $(call register_test,quant,quant,src/quant.c,$$(NATIVE_FLAGS)))
$(eval $(call register_test,quant_scalar,quant,src/quant.c,$$(SCALAR_FLAGS)))
$(eval $(call register_test,threadpool,threadpool,src/threadpool.c,$$(NATIVE_FLAGS)))
$(eval $(call register_test,grammar,grammar,src/grammar.c src/tokenizer.c,$$(NATIVE_FLAGS)))
$(eval $(call register_test,tokenizer,tokenizer,src/tokenizer.c,$$(NATIVE_FLAGS)))
$(eval $(call register_test,sampler,sampler,src/sampler.c,$$(NATIVE_FLAGS)))
$(eval $(call register_test,gguf_fuzz,gguf_fuzz,$(MODEL),$$(NATIVE_FLAGS)))

test: $(TESTS)
	./tests/test_quant$(EXE)
	./tests/test_quant_scalar$(EXE)
	./tests/test_threadpool$(EXE)
	./tests/test_grammar$(EXE)
	./tests/test_tokenizer$(EXE)
	./tests/test_sampler$(EXE)
	./tests/test_gguf_fuzz$(EXE)

PROGRAMS = $(OPT) $(SCALAR) $(DEBUG) picolm_static$(EXE) picolm_pi picolm_pi5 picolm_pi_arm32 picolm_cross_pi $(TESTS)
OUTPUTS = $(PROGRAMS) $(addsuffix .build-command,$(PROGRAMS)) $(addsuffix .build-command.tmp,$(PROGRAMS))
ifneq ($(WINDOWS_MAKE),)
empty :=
space := $(empty) $(empty)
comma := ,
clean:
	powershell -NoProfile -Command "foreach ($$picoFile in ($(subst $(space),$(comma),$(foreach file,$(OUTPUTS),'$(file)')))) { if (Test-Path -LiteralPath $$picoFile -PathType Leaf) { Remove-Item -LiteralPath $$picoFile -ErrorAction Stop } }"
else
clean:
	rm -f $(OUTPUTS)
endif
