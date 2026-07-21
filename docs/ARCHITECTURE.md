# rustVX Architecture

This document describes the high-level architecture of rustVX. For detailed designs of specific subsystems, see:

- [`pipelining_architecture.md`](pipelining_architecture.md) — Pipelining, Streaming & Batch Processing KHR extension
- [`multicore_pipeline_design.md`](multicore_pipeline_design.md) — Wave-based parallel node execution

## Crate responsibilities

rustVX is organized as a Cargo workspace. Each crate has a focused responsibility:

| Crate | Responsibility |
|-------|--------------|
| `openvx-core` | OpenVX framework: context, reference counting, graph data structures, C API wrappers, node execution, pipelining/streaming scheduling |
| `openvx-image` | `vx_image` data object and image channel operations |
| `openvx-buffer` | Generic buffer (`vx_array`, `vx_matrix`, etc.) and User Data Object support |
| `openvx-vision` | All vision kernels: color, filters, gradients, arithmetic, geometric, features, statistics, pyramid, optical flow, object detection, enhanced-vision tensor ops |
| `openvx-ffi` | Thin `cdylib` crate that re-exports `openvx-core`, `openvx-image`, `openvx-buffer`, and `openvx-vision` symbols as a single shared library (`libopenvx_ffi.so` / `.dylib` / `.dll`) |

The final artifact is one shared library that any OpenVX application can link against, while the internal Rust code stays modular and testable.

## Two API entry points

rustVX exposes two ways to use the implementation:

### 1. C FFI graph API (recommended for applications)

`openvx-ffi` exports the standard OpenVX 1.3.1 C API (`vx*` and `vxu*`). Existing C/C++ OpenVX code links against `libopenvx_ffi` without source changes. This path gives access to:

- Graph construction and verification (`vxCreateGraph`, `vxVerifyGraph`)
- Immediate-mode helpers (`vxuScaleImage`, `vxuGaussian3x3`, ...)
- Pipelining and streaming (`vxSetGraphScheduleConfig`, `vxEnableGraphStreaming`, ...)
- User nodes and user kernels

### 2. Native Rust kernel API

`openvx-core` and `openvx-vision` expose Rust traits and types for direct kernel registration and execution:

```rust
use openvx_core::{Context, VxResult};
use openvx_vision::register_all_kernels;

let context = Context::new()?;
register_all_kernels(&context)?;
```

This path is useful for Rust projects that want type-safe, memory-safe access to the kernel implementations, but it does not provide the full graph optimizer. For graph-mode workloads, use the C FFI.

## Graph execution flow

A typical graph execution follows these steps:

1. **Create context and graph** — `vxCreateContext`, `vxCreateGraph`
2. **Create data objects and nodes** — images, arrays, scalars; nodes reference them as parameters
3. **Verify graph** — `vxVerifyGraph` computes the topological order, validates data types and borders, and pre-computes execution waves for parallel dispatch
4. **Process graph** — `vxProcessGraph` runs synchronously; `vxScheduleGraph` + `vxWaitGraph` runs asynchronously
5. **Release resources** — `vxReleaseGraph`, `vxReleaseContext`

Inside `vxVerifyGraph`, the graph is represented as a DAG of nodes. The topological order is stored in `GraphData.topo_order` and, for pipelining/multicore builds, split into waves in `GraphData.topo_waves`.

## Node execution

Each node is executed by calling its registered kernel's `execute` function. The dispatcher (`execute_node` in `openvx-core/src/unified_c_api.rs`):

1. Resolves node parameters (handles virtual data and substitutions)
2. Maps references to the concrete data buffers
3. Calls the kernel
4. Records the node status and completion event

Vision kernels live in `openvx-vision` and implement the `Kernel` trait. At runtime, `openvx-vision::register_all_kernels` registers them with the context so they can be looked up by name or enum.

## SIMD backend selection

rustVX compiles multiple SIMD backends into the same binary and selects one at runtime based on CPU feature flags:

| CPU features | Backend used | Cargo feature |
|--------------|--------------|---------------|
| x86_64 + AVX2 + BMI2 + FMA | AVX2 intrinsic kernels | `avx2` |
| x86_64 + SSE2 (no AVX2) | SSE2 intrinsic kernels | `sse2` |
| AArch64 + NEON | NEON intrinsic kernels | `neon` |
| None of the above | Scalar slice loops | none / fallback |

The selection is done by reading CPU flags at startup, not by vendor string detection. This means an Intel Haswell CPU and an AMD Zen 2 CPU both run the AVX2 path.

The SIMD dispatch lives in:

- `openvx-core::simd_kernels` — for the FFI graph path
- `openvx-vision::x86_64_simd` / `openvx-vision::aarch64_simd` — for the direct Rust API path

## Threading model

rustVX uses three different levels of concurrency:

### 1. Intra-kernel parallelism

Enabled by the `parallel` feature on `openvx-vision`. Individual kernels use [Rayon](https://github.com/rayon-rs/rayon) to parallelize loops over image rows or tiles. This is orthogonal to graph pipelining.

### 2. Inter-node parallelism (pipelining graphs)

For graphs configured with `VX_GRAPH_SCHEDULE_MODE_QUEUE_AUTO` or `QUEUE_MANUAL`, independent nodes within the same graph are dispatched in parallel across a global thread pool. See [`multicore_pipeline_design.md`](multicore_pipeline_design.md) for the wave-based design.

### 3. Graph streaming

The KHR Streaming extension lets a verified graph run continuously in a background thread until `vxStopGraphStreaming` is called. Pipeup buffers absorb the startup latency between producer and consumer nodes. The streaming scheduler is implemented in `openvx-core/src/pipelining.rs` and `pipelining_executor.rs`.

## Data object ownership

rustVX uses reference counting for all OpenVX objects:

- Every reference (`vx_reference`) is wrapped in an `Arc`
- `vxCreate*` increments the count
- `vxRelease*` decrements the count
- Virtual objects created inside a graph are owned by the graph and released with it

The reference table in `Context` maps the public `vx_reference` handles to the internal Rust objects.

## Error handling

OpenVX status codes (`VX_SUCCESS`, `VX_ERROR_INVALID_VALUE`, etc.) are represented by `vx_status` in the C API and by the `VxResult` / `VxError` types in Rust. Kernel implementations return `VxResult<()>`; the C API layer translates those results back to `vx_status`.

## Files worth reading

| File | What it contains |
|------|------------------|
| `openvx-core/src/context.rs` | `Context` creation and reference table |
| `openvx-core/src/unified_c_api.rs` | Graph verification and execution dispatcher |
| `openvx-core/src/pipelining.rs` | Pipelining state and queues |
| `openvx-core/src/pipelining_executor.rs` | QUEUE_AUTO executor and streaming scheduler |
| `openvx-core/src/simd_kernels.rs` | SIMD dispatch for the FFI path |
| `openvx-vision/src/lib.rs` | Vision kernel registration |
| `openvx-vision/src/register.rs` | Mapping kernel names/enums to implementations |
| `openvx-ffi/src/lib.rs` | Re-exports that produce the final `libopenvx_ffi` |
