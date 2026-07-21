<p align="center">
  <a href="https://www.khronos.org/openvx/">
    <img src="docs/openvx-logo.svg" alt="OpenVX" width="320">
  </a>
</p>

# rustVX

[![OpenVX Conformance](https://github.com/kiritigowda/rustVX/actions/workflows/conformance.yml/badge.svg?branch=main)](https://github.com/kiritigowda/rustVX/actions/workflows/conformance.yml?query=branch%3Amain)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Rust](https://img.shields.io/badge/rust-stable-orange.svg)](https://www.rust-lang.org/)
[![OpenVX 1.3.1](https://img.shields.io/badge/OpenVX-1.3.1-blue.svg)](https://www.khronos.org/openvx/)

An [OpenVX 1.3.1](https://www.khronos.org/openvx/) implementation written in Rust. rustVX provides the complete OpenVX Vision Feature Set through a standard C API (`libopenvx_ffi`), enabling existing OpenVX applications to use a memory-safe, portable backend with no source changes.

> [!NOTE]
> rustVX targets the **OpenVX 1.3.1** specification. The KHR Streaming extension pipeup APIs were recently added (see PR #54); the streaming CTS test suite is not yet enabled in CI and is therefore excluded from the published conformance totals below.

## Conformance Status

rustVX passes the full [Khronos OpenVX 1.3.1 Conformance Test Suite](https://github.com/KhronosGroup/OpenVX-cts) for all required profiles:

| Profile | Required tests | Passing | Status |
|---------|---------------|---------|--------|
| OpenVX baseline | 5551 | **5551 / 5551** | ✅ |
| Vision conformance profile | 5923 | **5923 / 5923** | ✅ |
| Enhanced Vision conformance profile | 1235 | **1235 / 1235** | ✅ |
| User Data Object extension | 14 | **14 / 14** | ✅ |
| Pipelining extension | 81 | **81 / 81** | ✅ |
| **Total** | **6867** | **6867 / 6867** | ✅ **100%** |

All implemented kernels are exercised in CI with `-DOPENVX_CONFORMANCE_VISION=ON -DOPENVX_USE_ENHANCED_VISION=ON -DOPENVX_USE_USER_DATA_OBJECT=ON -DOPENVX_USE_PIPELINING=ON`.

The following OpenVX 1.3.1 extensions are **not implemented** in rustVX today:

- Neural Network (NN) extension
- Import/Export extension
- Debug extension

If you need one of these, please open an issue to discuss scope and priority.

Latest CTS run results are published on each push and pull request via the [Actions tab](https://github.com/kiritigowda/rustVX/actions/workflows/conformance.yml).

## Project Structure

```
rustVX/
├── openvx-core/       # Core framework: context, graph, node, types, C API
├── openvx-image/      # Image data object and channel operations
├── openvx-buffer/     # Generic buffer for intermediate data
├── openvx-vision/     # All vision kernels (filters, geometric, features, etc.)
├── openvx-ffi/        # C shared library (cdylib) exporting the full OpenVX API
├── include/VX/        # Standard OpenVX C headers
└── OpenVX-cts/        # Khronos Conformance Test Suite (git submodule)
```

The workspace compiles into a single shared library (`libopenvx_ffi.so` / `.dylib` / `.dll`) that any OpenVX application can link against.

## Prerequisites

| Tool | Version |
|------|---------|
| [Rust](https://rustup.rs/) | stable (MSRV not declared; latest stable recommended) |
| C compiler | gcc, clang, or MSVC |
| [CMake](https://cmake.org/) | 3.10+ |
| Make or Ninja | for building the CTS |

## Supported platforms

| Platform | Tier | Notes |
|----------|------|-------|
| Linux x86_64 | Primary | CI-tested on Ubuntu 22.04 with AVX2 |
| macOS x86_64 / Apple Silicon | Supported | Build instructions provided; Apple Silicon uses the NEON path |
| Windows MSVC x64 | Supported | Build instructions provided |
| Linux AArch64 | Best-effort | NEON path exists; not continuously tested in CI |

`openvx-core` and `openvx-vision` compile on any platform Rust supports. SIMD kernels are selected at runtime from CPU feature flags, not vendor strings.

## Build

```bash
# Clone with submodules
git clone --recurse-submodules https://github.com/kiritigowda/rustVX.git
cd rustVX

# Build the library
cargo build --release
```

The shared library is produced at:

| Platform | Path |
|----------|------|
| Linux | `target/release/libopenvx_ffi.so` |
| macOS | `target/release/libopenvx_ffi.dylib` |
| Windows | `target/release/openvx_ffi.dll` |

The standard OpenVX 1.3 C headers are bundled in [`include/VX/`](include/VX/) and can be passed to your C/C++ build directly.

## Installing rustVX

rustVX is currently distributed as a **build-from-source** project. After building once, you can use the shared library and headers directly from the build tree, or copy them to a system or project prefix.

### Use from the build tree (simplest)

```bash
# Linux / macOS
export LD_LIBRARY_PATH=/path/to/rustVX/target/release:$LD_LIBRARY_PATH   # Linux
export DYLD_LIBRARY_PATH=/path/to/rustVX/target/release:$DYLD_LIBRARY_PATH # macOS

# Compile your C/C++ application
gcc app.c -I /path/to/rustVX/include -L /path/to/rustVX/target/release -lopenvx_ffi -o app
./app
```

On Windows, add `target\release` to your `PATH` and link against `openvx_ffi.dll.lib`.

### Install to a local prefix

To keep rustVX isolated from system directories, copy the library and headers into a local prefix such as `/opt/rustVX` or `$HOME/.local/rustVX`:

```bash
PREFIX=/opt/rustVX
mkdir -p "$PREFIX/lib" "$PREFIX/include"
cp target/release/libopenvx_ffi.so* "$PREFIX/lib/"     # Linux
# cp target/release/libopenvx_ffi.dylib* "$PREFIX/lib/"   # macOS
# cp target/release/openvx_ffi.dll "$PREFIX/lib/"         # Windows
cp -r include/VX "$PREFIX/include/"

# Optional: create drop-in Khronos library names
ln -sf libopenvx_ffi.so "$PREFIX/lib/libopenvx.so"
ln -sf libopenvx_ffi.so "$PREFIX/lib/libvxu.so"
```

Then build against the prefix:

```bash
gcc app.c -I "$PREFIX/include" -L "$PREFIX/lib" -lopenvx_ffi -Wl,-rpath,"$PREFIX/lib" -o app
```

### System-wide install (Linux / macOS)

If you prefer a system-wide installation, install to `/usr/local`:

```bash
sudo cp target/release/libopenvx_ffi.so* /usr/local/lib/
sudo cp -r include/VX /usr/local/include/
sudo ldconfig                       # Linux only
```

> [!NOTE]
> rustVX is not yet published on crates.io for `cargo install`, and no OS packages (`.deb`, `.rpm`, Homebrew, etc.) are provided. The shared library produced by `cargo build --release -p openvx-ffi` is the canonical artifact.

### Cargo dependency for Rust projects

For Rust projects that want the native Rust kernel API, add the workspace crates as path dependencies:

```toml
[dependencies]
openvx-core = { path = "/path/to/rustVX/openvx-core" }
openvx-vision = { path = "/path/to/rustVX/openvx-vision", features = ["avx2"] }
```

### Cargo features

Both `openvx-core` (host of the C-API kernel callbacks the OpenVX graph executor invokes) and `openvx-vision` (host of the public Rust API kernels) expose a matching opt-in feature set:

| Feature | Effect |
|---------|--------|
| `simd` | Enables the portable SIMD abstraction layer; required by the architecture-specific features below |
| `sse2` / `avx2` | x86_64 SIMD back-ends (imply `simd`) |
| `neon` | AArch64 SIMD back-end (implies `simd`) |
| `parallel` (`openvx-vision` only) | Enables Rayon-based multi-threaded kernels |

Build with the matching pair on each crate so the FFI graph path and the direct Rust API path both pick up the SIMD kernels:

```bash
cargo build --release -p openvx-ffi \
  --features "openvx-core/sse2 openvx-core/avx2 openvx-vision/sse2 openvx-vision/avx2"
```

### Hardware acceleration

Performance work targets **AMD Zen (Ryzen / EPYC, Zen 2+)** — that's what CI measures and what the *Benchmark & compare* numbers come from. Intel and ARM aren't penalised; the runtime dispatcher reads CPU **flags**, not vendor strings, so any host whose flags match the same gate runs the same path:

- **AMD Zen 2+** (Ryzen 3000+, Threadripper 3000+, EPYC Rome / Milan / Genoa) → AVX2 kernels + `-C target-cpu=x86-64-v3` auto-vec.
- **Intel Haswell and later** → same AVX2 path, parity with Zen.
- **Older x86_64** (pre-AVX2) → SSE2 kernels + `-C target-cpu=x86-64-v2`.
- **AArch64** (Apple Silicon, AWS Graviton, etc.) → NEON path.
- **Anything else / no features** → scalar slice loop (still ~50× faster than the original per-pixel reference loops used during early development).

Dispatch lives in `openvx-core::simd_kernels` (FFI graph path) and `openvx-vision::x86_64_simd` (Rust API). CI auto-detects host flags; for a manual Zen-targeted build:

```bash
RUSTFLAGS="-C target-cpu=x86-64-v3" \
  cargo build --release -p openvx-ffi \
    --features "openvx-core/sse2 openvx-core/avx2 \
                openvx-vision/sse2 openvx-vision/avx2"
```

## Using rustVX from a C application

`libopenvx_ffi` exports the full `vx*` / `vxu*` symbol set defined by the standard OpenVX headers, so existing OpenVX code links against it with no source changes. A minimal example:

```c
#include <VX/vx.h>

int main(void) {
    vx_context ctx = vxCreateContext();
    vx_graph   g   = vxCreateGraph(ctx);
    /* ... build graph, call vxVerifyGraph / vxProcessGraph ... */
    vxReleaseGraph(&g);
    vxReleaseContext(&ctx);
    return 0;
}
```

```bash
# Linux
gcc app.c -I rustVX/include -L rustVX/target/release -lopenvx_ffi -o app
LD_LIBRARY_PATH=rustVX/target/release ./app
```

For drop-in compatibility with build systems that look for `libopenvx` / `libvxu` (e.g. `find_library(NAMES openvx vxu)`), symlink the rustVX library to those names:

```bash
ln -s libopenvx_ffi.so target/release/libopenvx.so
ln -s libopenvx_ffi.so target/release/libvxu.so
```

## Using rustVX from Rust

The workspace crates also expose a native Rust API. Add `openvx-vision` (and `openvx-core` if you need the context directly) to your `Cargo.toml`:

```toml
[dependencies]
openvx-core = { path = "path/to/rustVX/openvx-core" }
openvx-vision = { path = "path/to/rustVX/openvx-vision", features = ["avx2"] }
```

Then register kernels and execute them against a context:

```rust
use openvx_core::{Context, VxResult};
use openvx_vision::register_all_kernels;

fn main() -> VxResult<()> {
    let context = Context::new()?;
    register_all_kernels(&context)?;
    // ... create data objects and call kernel.execute() or use the C FFI graph API ...
    Ok(())
}
```

For graph-mode vision workloads from Rust, the recommended path is to use the C FFI (`openvx-ffi`) through the standard `vx*` / `vxu*` API, which gives full OpenVX graph optimization, pipelining, and streaming support.

## Verify your build

Before running the full Khronos CTS, confirm the workspace builds and passes its own tests:

```bash
# Build the shared library
cargo build --release -p openvx-ffi

# Run Rust unit and integration tests
cargo test --workspace --release

# Run the Criterion micro-benchmarks
cargo bench -p openvx-vision
```

If all three commands succeed, the library is ready for the conformance suite.

## Running Conformance Tests

The [Khronos OpenVX Conformance Test Suite](https://github.com/KhronosGroup/OpenVX-cts) is included as a git submodule. Build and run it against the rustVX library:

> [!NOTE]
> The `-DCMAKE_POLICY_VERSION_MINIMUM=3.5` flag below is needed when configuring with CMake 4.0+, which dropped compatibility with the older `cmake_minimum_required` versions used by the upstream CTS. It is harmless on older CMake.

### Linux

```bash
# Build the CTS
cd OpenVX-cts
mkdir -p build && cd build
cmake .. \
  -DCMAKE_C_FLAGS="-fno-strict-aliasing" \
  -DCMAKE_CXX_FLAGS="-fno-strict-aliasing" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
  -DCMAKE_C_STANDARD_LIBRARIES="-lm" \
  -DCMAKE_CXX_STANDARD_LIBRARIES="-lm" \
  -DOPENVX_INCLUDES="$(pwd)/../../include;$(pwd)/../include" \
  -DOPENVX_LIBRARIES="$(pwd)/../../target/release/libopenvx_ffi.so;m" \
  -DOPENVX_CONFORMANCE_VISION=ON \
  -DOPENVX_USE_ENHANCED_VISION=ON \
  -DOPENVX_USE_USER_DATA_OBJECT=ON \
  -DOPENVX_USE_PIPELINING=ON
make -j$(nproc)

# Run all tests
export LD_LIBRARY_PATH=$(pwd)/../../target/release
export VX_TEST_DATA_PATH=$(pwd)/../test_data/
./bin/vx_test_conformance
```

### macOS

```bash
# Build the CTS
cd OpenVX-cts
mkdir -p build && cd build
cmake .. \
  -DCMAKE_C_FLAGS="-fno-strict-aliasing" \
  -DCMAKE_CXX_FLAGS="-fno-strict-aliasing" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
  -DOPENVX_INCLUDES="$(pwd)/../../include;$(pwd)/../include" \
  -DOPENVX_LIBRARIES="$(pwd)/../../target/release/libopenvx_ffi.dylib" \
  -DOPENVX_CONFORMANCE_VISION=ON \
  -DOPENVX_USE_ENHANCED_VISION=ON \
  -DOPENVX_USE_USER_DATA_OBJECT=ON \
  -DOPENVX_USE_PIPELINING=ON
make -j$(sysctl -n hw.ncpu)

# Run all tests
export DYLD_LIBRARY_PATH=$(pwd)/../../target/release
export VX_TEST_DATA_PATH=$(pwd)/../test_data/
./bin/vx_test_conformance
```

### Windows (MSVC)

```powershell
# Build the CTS
cd OpenVX-cts
mkdir build; cd build
cmake .. `
  -DCMAKE_C_FLAGS="-fno-strict-aliasing" `
  -DCMAKE_CXX_FLAGS="-fno-strict-aliasing" `
  -DCMAKE_BUILD_TYPE=Release `
  -DCMAKE_POLICY_VERSION_MINIMUM=3.5 `
  -DOPENVX_INCLUDES="$PWD\..\..\include;$PWD\..\include" `
  -DOPENVX_LIBRARIES="$PWD\..\..\target\release\openvx_ffi.dll.lib" `
  -DOPENVX_CONFORMANCE_VISION=ON `
  -DOPENVX_USE_ENHANCED_VISION=ON `
  -DOPENVX_USE_USER_DATA_OBJECT=ON `
  -DOPENVX_USE_PIPELINING=ON
cmake --build . --config Release

# Run all tests
$env:PATH = "$PWD\..\..\target\release;$env:PATH"
$env:VX_TEST_DATA_PATH = "$PWD\..\test_data\"
.\bin\Release\vx_test_conformance.exe
```

### Running Specific Test Categories

Use the `--filter` flag to run a subset of tests:

```bash
# Run only filter tests
./bin/vx_test_conformance --filter="Gaussian3x3.*:Median3x3.*:Box3x3.*"

# Run only feature detection tests
./bin/vx_test_conformance --filter="HarrisCorners.*:FastCorners.*:vxCanny.*"

# Run only control flow tests
./bin/vx_test_conformance --filter="ControlFlow.SelectNode*:ControlFlow.ScalarOperationNode*"

# Run only tensor tests
./bin/vx_test_conformance --filter="Tensor.*:TensorOp.*"
```

> [!NOTE]
> The `-fno-strict-aliasing` flag above is needed because the CTS test suite contains intentional (but technically undefined-behavior) type punning between `vx_uint8*` and `vx_char*` inside `own_verify_data_items()`. GCC `-O3` optimizes those comparisons incorrectly without the flag, causing 5 pre-existing `Array.vxCreateArray` failures. The flag is harmless on all compilers and does not affect rustVX runtime performance.

## Unit Tests and Benchmarks

Beyond the Khronos CTS, rustVX has its own Rust-side test and benchmark suites:

```bash
# Run all unit and integration tests
cargo test --workspace --release

# Run the Criterion micro-benchmarks for vision kernels
cargo bench -p openvx-vision
```

End-to-end performance is also tracked against the [Khronos OpenVX sample implementation](https://github.com/KhronosGroup/OpenVX-sample-impl) on every CI run via [openvx-mark](https://github.com/kiritigowda/openvx-mark); see the *Benchmark & compare* job in the [Actions tab](https://github.com/kiritigowda/rustVX/actions/workflows/conformance.yml?query=branch%3Amain) for the latest comparison report.

> [!TIP]
> The *Benchmark & compare* job renders the rustVX-vs-Khronos comparison table directly into the GitHub Actions **job summary** for each run — no need to dig into logs. The summary opens with a headline panel showing the **geomean speedup of rustVX over the Khronos sample** (per-kernel best/worst speedups and a win/loss count) followed by the full per-benchmark detail table. The raw JSON reports are also published as downloadable workflow artifacts (`benchmark-results-rustvx`, `benchmark-results-khronos-sample`, and `benchmark-comparison`) on every push and pull request.

## Continuous Integration

GitHub Actions builds and runs the full CTS on every push and pull request. The workflow splits the suite into parallel jobs for faster feedback:

| Job | Test categories | Pipeline status |
|-----|-----------------|-----------------|
| **baseline** | GraphBase, Logging, SmokeTest, Target | [![baseline](https://img.shields.io/github/check-runs/kiritigowda/rustVX/main?nameFilter=baseline&label=)](https://github.com/kiritigowda/rustVX/actions/workflows/conformance.yml?query=branch%3Amain) |
| **graph** | Graph framework (cycles, virtual data, multi-run, replicate node), GraphCallback, GraphDelay, GraphROI, UserNode | [![graph](https://img.shields.io/github/check-runs/kiritigowda/rustVX/main?nameFilter=graph&label=)](https://github.com/kiritigowda/rustVX/actions/workflows/conformance.yml?query=branch%3Amain) |
| **data-objects** | Scalar, Array, ObjectArray, Matrix, Convolution, Distribution, LUT, Histogram | [![data-objects](https://img.shields.io/github/check-runs/kiritigowda/rustVX/main?nameFilter=data-objects&label=)](https://github.com/kiritigowda/rustVX/actions/workflows/conformance.yml?query=branch%3Amain) |
| **image-ops** | Image, CopyImagePatch, MapImagePatch, CreateImageFromChannel, Remap | [![image-ops](https://img.shields.io/github/check-runs/kiritigowda/rustVX/main?nameFilter=image-ops&label=)](https://github.com/kiritigowda/rustVX/actions/workflows/conformance.yml?query=branch%3Amain) |
| **vision-color** | ColorConvert, ChannelExtract, ChannelCombine, ConvertDepth | [![vision-color](https://img.shields.io/github/check-runs/kiritigowda/rustVX/main?nameFilter=vision-color&label=)](https://github.com/kiritigowda/rustVX/actions/workflows/conformance.yml?query=branch%3Amain) |
| **vision-filters** | Box, Gaussian, Median, Dilate, Erode, Sobel, Magnitude, Phase, NonLinearFilter, Convolve, EqualizeHistogram | [![vision-filters](https://img.shields.io/github/check-runs/kiritigowda/rustVX/main?nameFilter=vision-filters&label=)](https://github.com/kiritigowda/rustVX/actions/workflows/conformance.yml?query=branch%3Amain) |
| **vision-arithmetic** | Add, Subtract, Multiply, Bitwise (And/Or/Xor/Not), WeightedAverage, Threshold | [![vision-arithmetic](https://img.shields.io/github/check-runs/kiritigowda/rustVX/main?nameFilter=vision-arithmetic&label=)](https://github.com/kiritigowda/rustVX/actions/workflows/conformance.yml?query=branch%3Amain) |
| **vision-geometric** | Scale, WarpAffine, WarpPerspective, Remap, HalfScaleGaussian | [![vision-geometric](https://img.shields.io/github/check-runs/kiritigowda/rustVX/main?nameFilter=vision-geometric&label=)](https://github.com/kiritigowda/rustVX/actions/workflows/conformance.yml?query=branch%3Amain) |
| **vision-features** | HarrisCorners, FastCorners, Canny | [![vision-features](https://img.shields.io/github/check-runs/kiritigowda/rustVX/main?nameFilter=vision-features&label=)](https://github.com/kiritigowda/rustVX/actions/workflows/conformance.yml?query=branch%3Amain) |
| **vision-statistics** | MeanStdDev, MinMaxLoc, Integral | [![vision-statistics](https://img.shields.io/github/check-runs/kiritigowda/rustVX/main?nameFilter=vision-statistics&label=)](https://github.com/kiritigowda/rustVX/actions/workflows/conformance.yml?query=branch%3Amain) |
| **vision-pyramid** | GaussianPyramid, LaplacianPyramid, LaplacianReconstruct, OptFlowPyrLK | [![vision-pyramid](https://img.shields.io/github/check-runs/kiritigowda/rustVX/main?nameFilter=vision-pyramid&label=)](https://github.com/kiritigowda/rustVX/actions/workflows/conformance.yml?query=branch%3Amain) |
| **user-data-object** | UserDataObject (14 tests) | [![user-data-object](https://img.shields.io/github/check-runs/kiritigowda/rustVX/main?nameFilter=user-data-object&label=)](https://github.com/kiritigowda/rustVX/actions/workflows/conformance.yml?query=branch%3Amain) |
| **KHR: pipelining fast** | GraphPipeline (fast) | [![KHR extension: pipelining fast](https://img.shields.io/github/check-runs/kiritigowda/rustVX/main?nameFilter=KHR%20extension%3A%20pipelining%20fast&label=)](https://github.com/kiritigowda/rustVX/actions/workflows/conformance.yml?query=branch%3Amain) |
| **KHR: pipelining stress** | GraphPipeline (stress) | [![KHR extension: pipelining stress](https://img.shields.io/github/check-runs/kiritigowda/rustVX/main?nameFilter=KHR%20extension%3A%20pipelining%20stress&label=)](https://github.com/kiritigowda/rustVX/actions/workflows/conformance.yml?query=branch%3Amain) |
| **Enhanced-Vision: Feature Extraction** | HOGCells, HOGFeatures, MatchTemplate, LBP (44 tests) | [![Enhanced-Vision: Feature Extraction](https://img.shields.io/github/check-runs/kiritigowda/rustVX/main?nameFilter=Enhanced-Vision%3A%20Feature%20Extraction&label=)](https://github.com/kiritigowda/rustVX/actions/workflows/conformance.yml?query=branch%3Amain) |
| **Enhanced-Vision: Post-Processing** | Copy, NonMaxSuppression, HoughLinesP (84 tests) | [![Enhanced-Vision: Post-Processing](https://img.shields.io/github/check-runs/kiritigowda/rustVX/main?nameFilter=Enhanced-Vision%3A%20Post-Processing&label=)](https://github.com/kiritigowda/rustVX/actions/workflows/conformance.yml?query=branch%3Amain) |
| **Enhanced-Vision: Tensor Arithmetic** | TensorOp, Min, Max (222 tests) | [![Enhanced-Vision: Tensor Arithmetic](https://img.shields.io/github/check-runs/kiritigowda/rustVX/main?nameFilter=Enhanced-Vision%3A%20Tensor%20Arithmetic&label=)](https://github.com/kiritigowda/rustVX/actions/workflows/conformance.yml?query=branch%3Amain) |
| **Enhanced-Vision: Tensor Transforms** | Tensor, TensorEnhanced (18 tests) | [![Enhanced-Vision: Tensor Transforms](https://img.shields.io/github/check-runs/kiritigowda/rustVX/main?nameFilter=Enhanced-Vision%3A%20Tensor%20Transforms&label=)](https://github.com/kiritigowda/rustVX/actions/workflows/conformance.yml?query=branch%3Amain) |
| **Enhanced-Vision: Advanced Filtering** | BilateralFilter (361 tests) | [![Enhanced-Vision: Advanced Filtering](https://img.shields.io/github/check-runs/kiritigowda/rustVX/main?nameFilter=Enhanced-Vision%3A%20Advanced%20Filtering&label=)](https://github.com/kiritigowda/rustVX/actions/workflows/conformance.yml?query=branch%3Amain) |
| **Enhanced-Vision: Control Flow** | SelectNode, ScalarOperationNode (186 tests) | [![Enhanced-Vision: Control Flow](https://img.shields.io/github/check-runs/kiritigowda/rustVX/main?nameFilter=Enhanced-Vision%3A%20Control%20Flow&label=)](https://github.com/kiritigowda/rustVX/actions/workflows/conformance.yml?query=branch%3Amain) |

See the [Actions tab](https://github.com/kiritigowda/rustVX/actions) for full run history.

## Contributing

Contributions, bug reports, and feature requests are welcome.

- Found a bug or have a question? [Open an issue](https://github.com/kiritigowda/rustVX/issues).
- Want to contribute a fix or new kernel? Fork the repo, create a topic branch, and open a pull request against `main`. CI must pass — both the Khronos CTS jobs and the rustVX-vs-Khronos benchmark comparison run on every PR.
- Please make sure `cargo fmt`, `cargo clippy --workspace --all-targets`, and `cargo test --workspace` pass locally before submitting.

## License

This project is licensed under the [MIT License](LICENSE).

The OpenVX logo is a trademark of [The Khronos Group Inc.](https://www.khronos.org/legal/trademarks) The vector logo file in `docs/openvx-logo.svg` is sourced from [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:OpenVX_logo.svg) and is included for identification purposes only.
