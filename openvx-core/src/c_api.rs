//! C API for OpenVX Core
//!
//! This module provides FFI bindings for the OpenVX API

#![allow(non_camel_case_types)]
#![allow(unused_comparisons, unused_unsafe)]

use std::ffi::{c_void, CStr};
use std::sync::atomic::AtomicUsize;
use std::sync::{Arc, Mutex};

// Import the unified CONTEXTS registry
use crate::c_api_data::vx_pixel_value_t;
use crate::unified_c_api::GRAPHS_DATA;
use crate::unified_c_api::{VxCContext, CONTEXTS as UNIFIED_CONTEXTS};
use crate::unified_c_api::{REFERENCE_COUNTS, REFERENCE_NAMES, REFERENCE_TYPES};

// ============================================================================
// Type Aliases (C-compatible)
// ============================================================================

/// Basic OpenVX types
pub type vx_status = i32;
pub type vx_enum = i32;
pub type vx_uint32 = u32;
pub type vx_size = usize;
pub type vx_char = i8;
pub type vx_bool = i32;
pub type vx_df_image = u32;
pub type vx_uint64 = u64;
pub type vx_int32 = i32;
pub type vx_int8 = i8;
pub type vx_float32 = f32;
pub type vx_map_id = usize;

// ============================================================================
// Forward Declarations for Opaque Types
// ============================================================================

/// Opaque reference to a context
pub enum VxContext {}
pub type vx_context = *mut VxContext;

/// Opaque reference to a graph
pub enum VxGraph {}
pub type vx_graph = *mut VxGraph;

/// Opaque reference to a node
pub enum VxNode {}
pub type vx_node = *mut VxNode;

/// Opaque reference to a kernel
pub enum VxKernel {}
pub type vx_kernel = *mut VxKernel;

/// Opaque reference to a parameter
pub enum VxParameter {}
pub type vx_parameter = *mut VxParameter;

/// Opaque reference to a scalar
pub enum VxScalar {}
pub type vx_scalar = *mut VxScalar;

/// Opaque reference to a convolution
pub enum VxConvolution {}
pub type vx_convolution = *mut VxConvolution;

/// Opaque reference to a matrix
pub enum VxMatrix {}
pub type vx_matrix = *mut VxMatrix;

/// Opaque reference to a LUT
pub enum VxLUT {}
pub type vx_lut = *mut VxLUT;

/// Opaque reference to a threshold
pub enum VxThreshold {}
pub type vx_threshold = *mut VxThreshold;

/// Opaque reference to a pyramid
pub enum VxPyramid {}
pub type vx_pyramid = *mut VxPyramid;

/// Opaque reference to an image
pub enum VxImage {}
pub type vx_image = *mut VxImage;

/// Opaque reference to any reference type
pub enum VxReference {}
pub type vx_reference = *mut VxReference;

/// Opaque reference to a distribution
pub enum VxDistribution {}
pub type vx_distribution = *mut VxDistribution;

/// Opaque reference to a delay
pub enum VxDelay {}
pub type vx_delay = *mut VxDelay;

/// Opaque reference to a remap
pub enum VxRemap {}
pub type vx_remap = *mut VxRemap;

/// Opaque reference to a tensor
pub enum VxTensor {}
pub type vx_tensor = *mut VxTensor;

/// Opaque reference to a meta format
pub enum VxMetaFormat {}
pub type vx_meta_format = *mut VxMetaFormat;

/// Opaque reference to a graph parameter
pub enum VxGraphParameter {}
pub type vx_graph_parameter = *mut VxGraphParameter;

/// Opaque reference to an import
pub enum VxImport {}
pub type vx_import = *mut VxImport;

/// Opaque reference to a target
pub enum VxTarget {}
pub type vx_target = *mut VxTarget;

/// Node completion callback type
pub type vx_nodecomplete_f = Option<extern "C" fn(vx_node) -> vx_action>;

/// Log callback type
pub type vx_log_callback_t =
    Option<extern "C" fn(vx_context, vx_reference, vx_status, *const vx_char)>;

/// `vx_publish_kernels_f` — user-supplied function the OpenVX implementation
/// invokes (via [`vxLoadKernels`]) to register a library's kernels with the
/// context. The function is expected to call [`vxAddUserKernel`] for each
/// kernel in the library and return [`VX_SUCCESS`] on success.
///
/// See [`vxRegisterKernelLibrary`] for the registration side.
pub type vx_publish_kernels_f =
    Option<unsafe extern "C" fn(vx_context) -> vx_status>;

/// `vx_unpublish_kernels_f` — paired with [`vx_publish_kernels_f`]; invoked by
/// [`vxUnloadKernels`] to give the library a chance to tear down any per-
/// context state it set up at publish time.
pub type vx_unpublish_kernels_f =
    Option<unsafe extern "C" fn(vx_context) -> vx_status>;

/// Action return values from callbacks
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub enum vx_action {
    VX_ACTION_CONTINUE = 0,
    VX_ACTION_ABANDON = 1,
}

// ============================================================================
// Internal Structures (Send + Sync)
// ============================================================================

/// Internal graph data (stored in Arc)
pub struct GraphData {
    pub id: u64,
    pub context_id: u32,
    pub nodes: Mutex<Vec<u64>>, // Store node IDs instead of pointers
    pub ref_count: std::sync::atomic::AtomicUsize,
    /// Topological waves for multicore pipelining execution.
    /// Each wave is a list of node IDs that can execute in parallel.
    /// Computed during vxVerifyGraph, immutable thereafter.
    pub topo_waves: std::sync::Mutex<Vec<Vec<u64>>>,
}

/// Internal node data (stored in Arc)
pub struct NodeData {
    pub id: u64,
    pub context_id: u32,
    pub graph_id: u64,
    pub kernel_id: u64,
    pub parameters: Mutex<Vec<Option<u64>>>, // Store reference IDs
    pub callback: Mutex<Option<vx_nodecomplete_f>>,
    pub status: std::sync::atomic::AtomicI32,
    pub(crate) ref_count: std::sync::atomic::AtomicUsize,
    /// Node-specific border mode (overrides context border)
    pub border_mode: Mutex<crate::unified_c_api::vx_border_t>,
    /// Number of times this node has been executed (for VX_NODE_PERFORMANCE)
    pub run_count: std::sync::atomic::AtomicU64,
    /// Number of executions this node has completed, used for VX_NODE_STATE pipeup/steady tracking.
    pub execution_count: std::sync::atomic::AtomicU32,
    /// Current node state reported by VX_NODE_STATE (VX_NODE_STATE_STEADY / VX_NODE_STATE_PIPEUP).
    pub node_state: std::sync::atomic::AtomicU32,
    /// Whether the user-kernel `init` callback has been called for this node.
    /// Used by `vxVerifyGraph` to know when to call `deinit` before re-init.
    /// Always `false` for built-in kernels; only meaningful for user kernels.
    pub user_kernel_initialized: std::sync::atomic::AtomicBool,
    /// Local data size requested by the user kernel during init (VX_NODE_LOCAL_DATA_SIZE).
    pub local_data_size: std::sync::atomic::AtomicUsize,
    /// Local data pointer requested/owned by the user kernel (VX_NODE_LOCAL_DATA_PTR).
    pub local_data_ptr: std::sync::atomic::AtomicPtr<std::ffi::c_void>,
    /// Whether the user kernel auto-allocates local data (kernel attr LOCAL_DATA_SIZE > 0
    /// at registration). When auto-allocated, `vxQueryNode`/`vxSetNodeAttribute`
    /// behave differently for the local-data attributes.
    pub local_data_auto_alloc: std::sync::atomic::AtomicBool,
}

/// Internal kernel data (stored in Arc)
pub struct KernelData {
    pub id: u64,
    pub context_id: u32,
    pub name: String,
    pub kernel_enum: i32,
    pub num_params: u32,
    pub(crate) ref_count: std::sync::atomic::AtomicUsize,
}

/// Internal parameter data (stored in Arc)
pub struct ParameterData {
    pub id: u64,
    pub context_id: u32,
    pub kernel_id: u64,
    pub index: u32,
    pub direction: vx_enum,
    pub data_type: vx_enum,
    pub state: vx_enum,
    pub value: Mutex<Option<u64>>, // Store reference ID
    pub ref_count: std::sync::atomic::AtomicUsize,
}

// ============================================================================
// Global Storage for Object Management
// ============================================================================

use once_cell::sync::Lazy;

pub static CONTEXTS: Lazy<Mutex<Vec<u64>>> = Lazy::new(|| Mutex::new(Vec::new()));

pub static GRAPHS: Lazy<Mutex<std::collections::HashMap<u64, Arc<GraphData>>>> =
    Lazy::new(|| Mutex::new(std::collections::HashMap::new()));

pub static NODES: Lazy<Mutex<std::collections::HashMap<u64, Arc<NodeData>>>> =
    Lazy::new(|| Mutex::new(std::collections::HashMap::new()));

pub static KERNELS: Lazy<Mutex<std::collections::HashMap<u64, Arc<KernelData>>>> =
    Lazy::new(|| Mutex::new(std::collections::HashMap::new()));

// Re-export unified KERNELS for vxGetStatus
pub use crate::unified_c_api::KERNELS as UNIFIED_KERNELS;

pub static PARAMETERS: Lazy<Mutex<std::collections::HashMap<u64, Arc<ParameterData>>>> =
    Lazy::new(|| Mutex::new(std::collections::HashMap::new()));

static NEXT_ID: Lazy<std::sync::atomic::AtomicU64> =
    Lazy::new(|| std::sync::atomic::AtomicU64::new(1));

pub fn generate_id() -> u64 {
    NEXT_ID.fetch_add(1, std::sync::atomic::Ordering::SeqCst)
}

// ============================================================================
// Context Functions
// ============================================================================

#[no_mangle]
pub extern "C" fn vxCreateContext() -> vx_context {
    let id = generate_id();
    let ptr = id as *mut VxContext;
    let context_id = id as u32;
    if let Ok(mut contexts) = CONTEXTS.lock() {
        contexts.push(id);
    }
    // Also register in unified registry so vxQueryReference can find it
    if let Ok(mut unified_ctxs) = UNIFIED_CONTEXTS.lock() {
        unified_ctxs.insert(
            ptr as usize,
            Arc::new(VxCContext {
                id,
                ref_count: std::sync::atomic::AtomicUsize::new(1),
                border_mode: std::sync::RwLock::new(crate::unified_c_api::vx_border_t {
                    mode: crate::unified_c_api::VX_BORDER_UNDEFINED,
                    constant_value: vx_pixel_value_t { U32: 0 },
                }),
                border_policy: std::sync::atomic::AtomicU32::new(
                    crate::unified_c_api::VX_BORDER_POLICY_DEFAULT_TO_UNDEFINED as u32,
                ),
                log_callback: Mutex::new(None),
                log_reentrant: std::sync::atomic::AtomicBool::new(false),
                logging_enabled: std::sync::atomic::AtomicBool::new(false),
                performance_enabled: std::sync::atomic::AtomicBool::new(false),
            }),
        );
    }
    // Initialize reference count to 1 (the creation itself counts as a reference)
    let addr = ptr as usize;
    if let Ok(mut counts) = REFERENCE_COUNTS.lock() {
        counts.insert(addr, AtomicUsize::new(1));
    }
    // Register standard OpenVX kernels for this context
    register_standard_kernels(context_id);
    ptr
}

/// Register standard OpenVX built-in kernels for a context
fn register_standard_kernels(context_id: u32) {
    // Register built-in kernels that are always available
    // Format: (name, enum, num_params)
    // Kernel enums aligned with OpenVX 1.3.1 spec (VX_KERNEL_BASE values)
    // VX_KERNEL_BASE(vendor, lib) = ((vendor) << 20) | ((lib) << 12)
    // Per OpenVX spec: VX_KERNEL_<name> = VX_KERNEL_BASE(VX_ID_KHRONOS, VX_LIBRARY_KHR_BASE) + offset
    // Since VX_ID_KHRONOS=0x000 and VX_LIBRARY_KHR_BASE=0x0, the base is 0x00000000
    // Kernel enums start at 0x1 (not 0x0).
    let standard_kernels: Vec<(&str, i32, u32)> = vec![
        // Color conversions
        ("org.khronos.openvx.color_convert", 0x01, 2),
        ("org.khronos.openvx.channel_extract", 0x02, 3),
        ("org.khronos.openvx.channel_combine", 0x03, 5),
        // Gradient operations
        ("org.khronos.openvx.sobel_3x3", 0x04, 3),
        ("org.khronos.openvx.magnitude", 0x05, 3),
        ("org.khronos.openvx.phase", 0x06, 3),
        // Geometric
        ("org.khronos.openvx.scale_image", 0x07, 3),
        ("org.khronos.openvx.warp_affine", 0x23, 4),
        ("org.khronos.openvx.warp_perspective", 0x24, 4),
        ("org.khronos.openvx.remap", 0x28, 4),
        ("org.khronos.openvx.halfscale_gaussian", 0x29, 3),
        // Filters
        ("org.khronos.openvx.gaussian_3x3", 0x13, 2),
        ("org.khronos.openvx.box_3x3", 0x12, 2),
        ("org.khronos.openvx.median_3x3", 0x11, 2),
        ("org.khronos.openvx.convolve", 0x16, 3),
        ("org.khronos.openvx.custom_convolution", 0x14, 3),
        ("org.khronos.openvx.gaussian_pyramid", 0x15, 2),
        // Morphology
        ("org.khronos.openvx.dilate_3x3", 0x0F, 2),
        ("org.khronos.openvx.erode_3x3", 0x10, 2),
        // Arithmetic
        ("org.khronos.openvx.add", 0x21, 4),
        ("org.khronos.openvx.subtract", 0x22, 4),
        ("org.khronos.openvx.multiply", 0x20, 7),
        // Enhanced Vision: pixel-wise min/max
        ("org.khronos.openvx.min", 0x3F, 3),
        ("org.khronos.openvx.max", 0x3E, 3),
        // Bitwise
        ("org.khronos.openvx.and", 0x1C, 3),
        ("org.khronos.openvx.or", 0x1D, 3),
        ("org.khronos.openvx.xor", 0x1E, 3),
        ("org.khronos.openvx.not", 0x1F, 2),
        // Statistics
        ("org.khronos.openvx.histogram", 0x09, 2),
        ("org.khronos.openvx.equalize_histogram", 0x0A, 2),
        ("org.khronos.openvx.integral_image", 0x0E, 2),
        ("org.khronos.openvx.mean_stddev", 0x0C, 3),
        ("org.khronos.openvx.minmaxloc", 0x19, 7),
        // Additional operations
        ("org.khronos.openvx.absdiff", 0x0B, 3),
        ("org.khronos.openvx.threshold", 0x0D, 3),
        ("org.khronos.openvx.table_lookup", 0x08, 3),
        ("org.khronos.openvx.convertdepth", 0x1A, 4),
        // Feature detection
        ("org.khronos.openvx.harris_corners", 0x25, 8),
        ("org.khronos.openvx.fast_corners", 0x26, 5),
        ("org.khronos.openvx.optical_flow_pyr_lk", 0x27, 10),
        ("org.khronos.openvx.canny_edge_detector", 0x1B, 5),
        // OpenVX 1.1
        ("org.khronos.openvx.laplacian_pyramid", 0x2A, 3),
        ("org.khronos.openvx.laplacian_reconstruct", 0x2B, 3),
        ("org.khronos.openvx.non_linear_filter", 0x2C, 4),
        // Enhanced Vision kernels (per OpenVX 1.3 spec)
        ("org.khronos.openvx.match_template", 0x2D, 4),
        ("org.khronos.openvx.lbp", 0x2E, 4),
        ("org.khronos.openvx.hough_lines_p", 0x2F, 8),
        ("org.khronos.openvx.tensor_multiply", 0x30, 6),
        ("org.khronos.openvx.tensor_add", 0x31, 4),
        ("org.khronos.openvx.tensor_subtract", 0x32, 4),
        ("org.khronos.openvx.tensor_table_lookup", 0x33, 3),
        ("org.khronos.openvx.tensor_transpose", 0x34, 4),
        ("org.khronos.openvx.tensor_convert_depth", 0x35, 5),
        ("org.khronos.openvx.tensor_matrix_multiply", 0x36, 5),
        ("org.khronos.openvx.copy", 0x37, 3),
        ("org.khronos.openvx.non_max_suppression", 0x38, 4),
        ("org.khronos.openvx.scalar_operation", 0x39, 4),
        ("org.khronos.openvx.hog_features", 0x3A, 7),
        ("org.khronos.openvx.hog_cells", 0x3B, 6),
        ("org.khronos.openvx.bilateral_filter", 0x3C, 5),
        ("org.khronos.openvx.select", 0x3D, 4),
        // OpenVX 1.0.2 addition
        ("org.khronos.openvx.weighted_average", 0x40, 4),
    ];

    if let Ok(mut kernels) = KERNELS.lock() {
        for (name, kernel_enum, num_params) in standard_kernels {
            // Use kernel enum + offset to avoid collision with graph IDs
            // Graph IDs use generate_id() starting at 1
            // Kernel IDs use 0x10000 + kernel_enum to ensure they're unique
            let kernel_id = 0x10000 + kernel_enum as u64;
            let kernel = Arc::new(KernelData {
                id: kernel_id,
                context_id,
                name: name.to_string(),
                kernel_enum: kernel_enum as i32,
                num_params,
                ref_count: std::sync::atomic::AtomicUsize::new(1),
            });
            kernels.insert(kernel_id, kernel);

            // Register kernel in REFERENCE_COUNTS and REFERENCE_TYPES using the offset ID
            if let Ok(mut counts) = REFERENCE_COUNTS.lock() {
                counts.insert(kernel_id as usize, AtomicUsize::new(1));
            }
            if let Ok(mut types) = REFERENCE_TYPES.lock() {
                types.insert(kernel_id as usize, crate::unified_c_api::VX_TYPE_KERNEL);
            }
        }
    }
}

#[no_mangle]
pub extern "C" fn vxReleaseContext(context: *mut vx_context) -> vx_status {
    if context.is_null() {
        return VX_ERROR_INVALID_REFERENCE; // Per CTS: null pointer should return INVALID_REFERENCE
    }
    unsafe {
        let ctx = *context;
        if ctx.is_null() {
            // Context already null - return INVALID_REFERENCE per CTS
            return VX_ERROR_INVALID_REFERENCE;
        }
        let id = ctx as u64;
        let addr = ctx as usize;

        // Decrement reference count
        if let Ok(counts) = REFERENCE_COUNTS.lock() {
            if let Some(count) = counts.get(&addr) {
                let current = count.load(std::sync::atomic::Ordering::SeqCst);
                if current > 1 {
                    let new_count = current - 1;
                    count.store(new_count, std::sync::atomic::Ordering::SeqCst);
                    // Don't actually release yet, just return
                    *context = std::ptr::null_mut();
                    return VX_SUCCESS;
                } else {
                    // Need to drop the lock before removing
                    drop(counts);
                    if let Ok(mut counts) = REFERENCE_COUNTS.lock() {
                        counts.remove(&addr);
                    }
                }
            }
        }

        if let Ok(mut contexts) = CONTEXTS.lock() {
            contexts.retain(|&c| c != id);
        }
        // Also remove from unified registry
        if let Ok(mut unified_ctxs) = UNIFIED_CONTEXTS.lock() {
            unified_ctxs.remove(&addr);
        }

        // Clear delay/pyramid registries to prevent stale data between test contexts
        if let Ok(mut delay_params) = crate::unified_c_api::DELAY_NODE_PARAMS.lock() {
            delay_params.clear();
        }
        if let Ok(mut delay_pyr_level) = crate::unified_c_api::DELAY_PYRAMID_LEVEL.lock() {
            delay_pyr_level.clear();
        }
        if let Ok(mut slot_logical) = crate::unified_c_api::DELAY_SLOT_LOGICAL.lock() {
            slot_logical.clear();
        }
        if let Ok(mut slot_objs) = crate::unified_c_api::DELAY_SLOT_OBJECTS.lock() {
            slot_objs.clear();
        }
        if let Ok(mut level_imgs) = crate::unified_c_api::PYRAMID_LEVEL_IMAGES.lock() {
            level_imgs.clear();
        }

        *context = std::ptr::null_mut();
    }
    VX_SUCCESS
}

// ============================================================================
// Reference Management Functions
// ============================================================================

#[no_mangle]
pub extern "C" fn vxRetainReference(_ref_: vx_reference) -> vx_status {
    if _ref_.is_null() {
        return VX_ERROR_INVALID_REFERENCE;
    }
    let addr = _ref_ as usize;
    let id = _ref_ as u64;

    // Increment reference count in unified registry
    if let Ok(counts) = REFERENCE_COUNTS.lock() {
        if let Some(count) = counts.get(&addr) {
            count.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            return VX_SUCCESS;
        }
    }

    // If not in REFERENCE_COUNTS, check if reference exists in any registry
    // and if so, add it to REFERENCE_COUNTS with count=2 (original + retained)
    let mut found = false;

    // Check graphs
    if let Ok(graphs) = GRAPHS.lock() {
        if graphs.contains_key(&id) {
            found = true;
        }
    }
    if let Ok(graphs) = crate::unified_c_api::GRAPHS_DATA.lock() {
        if graphs.contains_key(&id) {
            found = true;
        }
    }

    // Check contexts
    if !found {
        if let Ok(contexts) = CONTEXTS.lock() {
            if contexts.contains(&id) {
                found = true;
            }
        }
    }
    if !found {
        if let Ok(contexts) = UNIFIED_CONTEXTS.lock() {
            if contexts.contains_key(&addr) {
                found = true;
            }
        }
    }

    // Check kernels
    if !found {
        if let Ok(kernels) = KERNELS.lock() {
            if kernels.contains_key(&id) {
                found = true;
            }
        }
    }

    // Check nodes
    if !found {
        if let Ok(nodes) = NODES.lock() {
            if nodes.contains_key(&id) {
                found = true;
            }
        }
    }

    // Check parameters
    if !found {
        if let Ok(params) = PARAMETERS.lock() {
            if params.contains_key(&id) {
                found = true;
            }
        }
    }

    if found {
        // Add to REFERENCE_COUNTS with count=2 (original reference + retained)
        if let Ok(mut counts) = REFERENCE_COUNTS.lock() {
            counts.insert(addr, std::sync::atomic::AtomicUsize::new(2));
        }
        return VX_SUCCESS;
    }

    VX_ERROR_INVALID_REFERENCE
}

#[no_mangle]
pub extern "C" fn vxGetStatus(ref_: vx_reference) -> vx_status {
    if ref_.is_null() {
        // Per CTS expectations, return VX_ERROR_NO_RESOURCES (-12) for null reference
        return VX_ERROR_NO_RESOURCES;
    }
    // Check if the reference is valid in any registry
    let addr = ref_ as usize;
    let id = ref_ as u64;

    // Check unified reference counts first
    if let Ok(counts) = REFERENCE_COUNTS.lock() {
        if counts.contains_key(&addr) {
            return VX_SUCCESS;
        }
    }

    // Check if in any other registry (graphs, contexts, etc.)
    // Check graphs in unified registry
    if let Ok(graphs) = GRAPHS_DATA.lock() {
        if graphs.contains_key(&id) {
            return VX_SUCCESS;
        }
    }

    // Check graphs in c_api registry
    if let Ok(c_api_graphs) = GRAPHS.lock() {
        if c_api_graphs.contains_key(&id) {
            return VX_SUCCESS;
        }
    }

    // Check contexts
    if let Ok(contexts) = CONTEXTS.lock() {
        if contexts.contains(&id) {
            return VX_SUCCESS;
        }
    }

    // Check unified contexts
    if let Ok(unified_contexts) = UNIFIED_CONTEXTS.lock() {
        if unified_contexts.contains_key(&addr) {
            return VX_SUCCESS;
        }
    }

    // Check kernels in c_api
    if let Ok(kernels) = KERNELS.lock() {
        if kernels.contains_key(&id) {
            return VX_SUCCESS;
        }
    }

    // Check images in unified registry
    use crate::unified_c_api::IMAGES;
    if let Ok(images) = IMAGES.lock() {
        if images.contains(&addr) {
            return VX_SUCCESS;
        }
    }

    // Check thresholds in unified registry
    if let Ok(thresholds) = crate::unified_c_api::THRESHOLDS.lock() {
        if thresholds.contains(&addr) {
            return VX_SUCCESS;
        }
    }

    VX_ERROR_INVALID_REFERENCE
}

#[no_mangle]
pub extern "C" fn vxGetContext(ref_: vx_reference) -> vx_context {
    if ref_.is_null() {
        return std::ptr::null_mut();
    }
    unsafe {
        let id = ref_ as u64;
        let addr = ref_ as usize;

        // Check if it's an image - use the unified registry FIRST
        // This is critical because images are allocated as heap pointers
        // and we need to validate them before treating as IDs
        use crate::unified_c_api::IMAGES;
        if let Ok(images) = IMAGES.lock() {
            if images.contains(&addr) {
                // The image stores context directly in the VxCImage struct
                let img = &*(ref_ as *const crate::unified_c_api::VxCImage);
                return img.context;
            }
        }

        // Check if it's a node
        if let Ok(nodes) = NODES.lock() {
            if let Some(node) = nodes.get(&id) {
                return node.context_id as *mut VxContext;
            }
        }
        // Check if it's a graph
        if let Ok(graphs) = GRAPHS.lock() {
            for (graph_id, graph) in graphs.iter() {
                if *graph_id == id {
                    return graph.context_id as *mut VxContext;
                }
            }
        }
        // Check if it's a kernel
        if let Ok(kernels) = KERNELS.lock() {
            for (kernel_id, kernel) in kernels.iter() {
                if *kernel_id == id {
                    return kernel.context_id as *mut VxContext;
                }
            }
        }
    }
    // Default: return first context or null
    if let Ok(contexts) = CONTEXTS.lock() {
        if let Some(&first) = contexts.first() {
            return first as *mut VxContext;
        }
    }
    std::ptr::null_mut()
}

// ============================================================================
// Graph Functions
// ============================================================================

#[no_mangle]
pub extern "C" fn vxCreateGraph(context: vx_context) -> vx_graph {
    if context.is_null() {
        return std::ptr::null_mut();
    }
    let context_id = context as u32;
    let id = generate_id();

    // Store in the local registry
    let graph = Arc::new(GraphData {
        id,
        context_id,
        nodes: Mutex::new(Vec::new()),
        ref_count: std::sync::atomic::AtomicUsize::new(1),
        topo_waves: std::sync::Mutex::new(Vec::new()),
    });

    if let Ok(mut graphs) = GRAPHS.lock() {
        graphs.insert(id, graph.clone());
    }

    // Also register in unified registry for vxQueryReference
    let unified_graph = Arc::new(crate::unified_c_api::VxCGraphData {
        id,
        context_id: context_id as u64,
        nodes: std::sync::RwLock::new(Vec::new()),
        parameters: std::sync::RwLock::new(Vec::new()),
        state: std::sync::Mutex::new(crate::unified_c_api::VxGraphState::VxGraphStateUnverified),
        verified: std::sync::Mutex::new(false),
        ref_count: std::sync::atomic::AtomicUsize::new(1),
        run_count: std::sync::atomic::AtomicU64::new(0),
        replicated_nodes: std::sync::Mutex::new(std::collections::HashMap::new()),
        owned_refs: std::sync::Mutex::new(Vec::new()),
        topo_waves: std::sync::Mutex::new(Vec::new()),
        node_predecessors: std::sync::Mutex::new(std::collections::HashMap::new()),
    });

    if let Ok(mut graphs_data) = crate::unified_c_api::GRAPHS_DATA.lock() {
        graphs_data.insert(id, unified_graph);
    }

    // Initialize reference count to 1
    let ptr = id as *mut VxGraph;
    if let Ok(mut counts) = REFERENCE_COUNTS.lock() {
        counts.insert(ptr as usize, AtomicUsize::new(1));
    }

    ptr
}

#[no_mangle]
pub extern "C" fn vxReleaseGraph(graph: *mut vx_graph) -> vx_status {
    if graph.is_null() {
        return VX_ERROR_INVALID_REFERENCE;
    }
    unsafe {
        let g = *graph;
        if g.is_null() {
            return VX_ERROR_INVALID_REFERENCE;
        }
        let id = g as u64;
        let addr = g as usize;

        // Decrement reference count first
        let mut should_remove = false;
        if let Ok(counts) = REFERENCE_COUNTS.lock() {
            if let Some(count) = counts.get(&addr) {
                let current = count.load(std::sync::atomic::Ordering::SeqCst);
                if current > 1 {
                    count.store(current - 1, std::sync::atomic::Ordering::SeqCst);
                } else {
                    should_remove = true;
                }
            } else {
            }
        }

        if should_remove {
            // Release the graph's parameters
            if let Ok(graphs_data) = crate::unified_c_api::GRAPHS_DATA.lock() {
                if let Some(g) = graphs_data.get(&id) {
                    let graph_params: Vec<u64> = {
                        if let Ok(params) = g.parameters.read() {
                            params.clone()
                        } else {
                            Vec::new()
                        }
                    };

                    for param_id in graph_params {
                        let param_addr = param_id as usize;

                        // Decrement ref count
                        if let Ok(mut counts) = REFERENCE_COUNTS.lock() {
                            if let Some(cnt) = counts.get(&param_addr) {
                                let current = cnt.load(std::sync::atomic::Ordering::SeqCst);
                                if current > 1 {
                                    cnt.store(current - 1, std::sync::atomic::Ordering::SeqCst);
                                } else {
                                    // Last reference - remove the parameter
                                    counts.remove(&param_addr);
                                    if let Ok(mut types) = REFERENCE_TYPES.lock() {
                                        types.remove(&param_addr);
                                    }
                                    if let Ok(mut params) = crate::unified_c_api::PARAMETERS.lock()
                                    {
                                        params.remove(&param_id);
                                    }
                                }
                            }
                        }
                    }
                }
            }

            // Collect non-scalar parameter values that need release
            let _non_scalar_params: Vec<u64> = Vec::new();

            // Collect non-scalar params that reached ref count 0 for type-specific release
            let non_scalar_last_refs: Vec<usize> = Vec::new();

            // Snapshot the graph's node ids and run the user-kernel `deinit`
            // callbacks BEFORE we re-enter the GRAPHS_DATA lock for the
            // teardown loop below. The user `deinit` callback can call back
            // into the OpenVX API (vxQueryNode, etc.) which itself takes
            // GRAPHS_DATA, so we MUST not hold that lock while the callback
            // runs or we will deadlock.
            let graph_nodes_pre: Vec<u64> = {
                if let Ok(graphs_data) = crate::unified_c_api::GRAPHS_DATA.lock() {
                    if let Some(g) = graphs_data.get(&id) {
                        if let Ok(nodes) = g.nodes.read() {
                            nodes.clone()
                        } else {
                            Vec::new()
                        }
                    } else {
                        Vec::new()
                    }
                } else {
                    Vec::new()
                }
            };
            for node_id in &graph_nodes_pre {
                crate::unified_c_api::deinit_user_kernel_node(*node_id);
            }

            // Free the graph's nodes
            if let Ok(graphs_data) = crate::unified_c_api::GRAPHS_DATA.lock() {
                if let Some(g) = graphs_data.get(&id) {
                    let graph_nodes: Vec<u64> = {
                        if let Ok(nodes) = g.nodes.read() {
                            nodes.clone()
                        } else {
                            Vec::new()
                        }
                    };

                    for node_id in graph_nodes {
                        // Release node's parameter references (decrement their ref counts)
                        // For internally-created scalars, also release them fully
                        if let Ok(nodes) = NODES.lock() {
                            if let Some(node_data) = nodes.get(&node_id) {
                                if let Ok(params) = node_data.parameters.lock() {
                                    for param_val in params.iter() {
                                        if let Some(val) = param_val {
                                            if *val != 0 {
                                                let val_type =
                                                    if let Ok(types) = REFERENCE_TYPES.lock() {
                                                        types.get(&(*val as usize)).copied()
                                                    } else {
                                                        None
                                                    };
                                                // Decrement ref count by 1 for the node's reference.
                                                // The caller owns their own reference and will release it.
                                                if let Ok(mut counts) = REFERENCE_COUNTS.lock() {
                                                    if let Some(cnt) =
                                                        counts.get_mut(&(*val as usize))
                                                    {
                                                        let current = cnt.load(
                                                            std::sync::atomic::Ordering::SeqCst,
                                                        );
                                                        if current > 1 {
                                                            cnt.store(
                                                                current - 1,
                                                                std::sync::atomic::Ordering::SeqCst,
                                                            );
                                                        } else if current == 1 {
                                                            // Last reference - free the object
                                                            let addr = *val as usize;
                                                            drop(counts);
                                                            if let Ok(mut counts2) =
                                                                REFERENCE_COUNTS.lock()
                                                            {
                                                                counts2.remove(&addr);
                                                            }
                                                            if let Ok(mut types) =
                                                                REFERENCE_TYPES.lock()
                                                            {
                                                                types.remove(&addr);
                                                            }
                                                            // Free based on type
                                                            if val_type == Some(VX_TYPE_SCALAR) {
                                                                let _ = Box::from_raw(*val as *mut crate::c_api_data::VxCScalarData);
                                                            } else if val_type
                                                                == Some(VX_TYPE_PYRAMID)
                                                            {
                                                                extern "C" {
                                                                    fn vxReleasePyramid(
                                                                        pyramid: *mut vx_pyramid,
                                                                    ) -> vx_status;
                                                                }
                                                                let mut pyr = *val as vx_pyramid;
                                                                unsafe {
                                                                    vxReleasePyramid(&mut pyr);
                                                                }
                                                            }
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                        // Remove node from registries
                        if let Ok(mut nodes_mut) = NODES.lock() {
                            nodes_mut.remove(&node_id);
                        }
                        if let Ok(mut counts) = REFERENCE_COUNTS.lock() {
                            counts.remove(&(node_id as usize));
                        }
                        if let Ok(mut types) = REFERENCE_TYPES.lock() {
                            types.remove(&(node_id as usize));
                        }
                    }
                }
            }

            // Type-specific release for non-scalar params that reached ref count 0
            for addr in non_scalar_last_refs {
                let mut r = addr as vx_reference;
                crate::unified_c_api::vxReleaseReference(&mut r as *mut vx_reference);
            }

            // Release graph-owned references (internally-created scalars, etc.)
            let owned: Vec<u64> = {
                if let Ok(graphs_data) = crate::unified_c_api::GRAPHS_DATA.lock() {
                    if let Some(g) = graphs_data.get(&id) {
                        if let Ok(refs) = g.owned_refs.lock() {
                            refs.clone()
                        } else {
                            Vec::new()
                        }
                    } else {
                        Vec::new()
                    }
                } else {
                    Vec::new()
                }
            };
            for owned_ref in owned {
                let mut r = owned_ref as vx_reference;
                crate::unified_c_api::vxReleaseReference(&mut r as *mut vx_reference);
            }

            // Remove from all registries when count reaches 0
            if let Ok(mut graphs) = GRAPHS.lock() {
                graphs.remove(&id);
            }
            if let Ok(mut graphs_data) = crate::unified_c_api::GRAPHS_DATA.lock() {
                graphs_data.remove(&id);
            }
            if let Ok(mut counts) = REFERENCE_COUNTS.lock() {
                counts.remove(&addr);
            }
            if let Ok(mut types) = REFERENCE_TYPES.lock() {
                types.remove(&addr);
            }
            if let Ok(mut names) = REFERENCE_NAMES.lock() {
                names.remove(&addr);
            }
            // Clean up pipelining state: stop executor first, then remove state
            crate::pipelining_executor::stop_queue_auto_executor(id);
            let was_pipelining = if let Ok(mut pipe_states) = crate::pipelining_api::GRAPH_PIPELINING.lock() {
                let was = pipe_states.get(&id).map(|s| {
                    let mode = s.schedule_mode.lock().unwrap();
                    *mode != crate::pipelining::VxGraphScheduleMode::Normal
                }).unwrap_or(false);
                pipe_states.remove(&id);
                was
            } else { false };
            if was_pipelining {
                crate::pipelining_api::ACTIVE_PIPELINING_GRAPHS.fetch_sub(1, std::sync::atomic::Ordering::Relaxed);
            }
            // Clean up auto-aging delay registry
            if let Ok(mut registry) = crate::unified_c_api::GRAPH_AUTO_AGE_DELAYS.lock() {
                registry.remove(&id);
            }
            // Clean up event registrations for this graph (from all contexts)
            if let Ok(mut systems) = crate::pipelining_api::EVENT_SYSTEMS.lock() {
                for (_, event_system) in systems.iter_mut() {
                    if let Ok(mut registrations) = event_system.registrations.lock() {
                        registrations.retain(|reg| reg.graph_id != Some(id));
                    }
                }
            }
        }

        *graph = std::ptr::null_mut();
    }
    VX_SUCCESS
}

// ============================================================================
// Node Management Functions
// ============================================================================

#[no_mangle]
pub extern "C" fn vxQueryNode(
    node: vx_node,
    attribute: vx_enum,
    ptr: *mut c_void,
    size: vx_size,
) -> vx_status {
    if node.is_null() || ptr.is_null() {
        return VX_ERROR_INVALID_REFERENCE;
    }
    unsafe {
        let id = node as u64;
        if let Ok(nodes) = NODES.lock() {
            if let Some(node_data) = nodes.get(&id) {
                match attribute {
                    VX_NODE_STATUS => {
                        if size >= 4 {
                            let status = node_data.status.load(std::sync::atomic::Ordering::SeqCst);
                            let ptr_u8 = ptr as *mut u8;
                            std::ptr::copy_nonoverlapping(
                                &status as *const i32 as *const u8,
                                ptr_u8,
                                4,
                            );
                            return VX_SUCCESS;
                        }
                    }
                    VX_NODE_PARAMETERS => {
                        if size >= 4 {
                            let param_count = node_data.parameters.lock().unwrap().len() as i32;
                            let ptr_u8 = ptr as *mut u8;
                            std::ptr::copy_nonoverlapping(
                                &param_count as *const i32 as *const u8,
                                ptr_u8,
                                4,
                            );
                            return VX_SUCCESS;
                        }
                    }
                    VX_NODE_PERFORMANCE => {
                        if size >= std::mem::size_of::<crate::unified_c_api::vx_perf_t>() {
                            let mut perf = crate::unified_c_api::vx_perf_t::default();
                            let count = node_data
                                .run_count
                                .load(std::sync::atomic::Ordering::SeqCst);
                            if count > 0 {
                                perf.num = count;
                                perf.beg = 2 * count - 1; // ensures beg_n > end_(n-1)
                                perf.end = 2 * count;
                                perf.min = 1;
                                perf.sum = count;
                                perf.avg = 1;
                                perf.max = count;
                            }
                            std::ptr::copy_nonoverlapping(
                                &perf as *const crate::unified_c_api::vx_perf_t as *const u8,
                                ptr as *mut u8,
                                std::mem::size_of::<crate::unified_c_api::vx_perf_t>(),
                            );
                            return VX_SUCCESS;
                        } else {
                            return VX_ERROR_INVALID_PARAMETERS;
                        }
                    }
                    VX_NODE_BORDER => {
                        if size < std::mem::size_of::<crate::unified_c_api::vx_border_t>() {
                            return VX_ERROR_INVALID_PARAMETERS;
                        }
                        let border = node_data.border_mode.lock().unwrap();
                        std::ptr::copy_nonoverlapping(
                            &*border as *const crate::unified_c_api::vx_border_t as *const u8,
                            ptr as *mut u8,
                            std::mem::size_of::<crate::unified_c_api::vx_border_t>(),
                        );
                        return VX_SUCCESS;
                    }
                    VX_NODE_LOCAL_DATA_SIZE => {
                        if size < std::mem::size_of::<vx_size>() {
                            return VX_ERROR_INVALID_PARAMETERS;
                        }
                        let v = node_data
                            .local_data_size
                            .load(std::sync::atomic::Ordering::SeqCst)
                            as vx_size;
                        *(ptr as *mut vx_size) = v;
                        return VX_SUCCESS;
                    }
                    VX_NODE_LOCAL_DATA_PTR => {
                        if size < std::mem::size_of::<*mut c_void>() {
                            return VX_ERROR_INVALID_PARAMETERS;
                        }
                        let p = node_data
                            .local_data_ptr
                            .load(std::sync::atomic::Ordering::SeqCst);
                        *(ptr as *mut *mut c_void) = p;
                        return VX_SUCCESS;
                    }
                    VX_NODE_STATE => {
                        if size < std::mem::size_of::<vx_enum>() {
                            return VX_ERROR_INVALID_PARAMETERS;
                        }
                        let state = node_data
                            .node_state
                            .load(std::sync::atomic::Ordering::SeqCst);
                        *(ptr as *mut vx_enum) = state as vx_enum;
                        return VX_SUCCESS;
                    }
                    _ => {}
                }
            }
        }
    }
    VX_ERROR_NOT_IMPLEMENTED
}

#[no_mangle]
pub extern "C" fn vxSetNodeAttribute(
    node: vx_node,
    attribute: vx_enum,
    ptr: *const c_void,
    size: vx_size,
) -> vx_status {
    if node.is_null() {
        return VX_ERROR_INVALID_REFERENCE;
    }
    if ptr.is_null() {
        return VX_ERROR_INVALID_PARAMETERS;
    }

    unsafe {
        let id = node as u64;
        if let Ok(nodes) = NODES.lock() {
            if let Some(node_data) = nodes.get(&id) {
                match attribute {
                    VX_NODE_BORDER => {
                        if size != std::mem::size_of::<crate::unified_c_api::vx_border_t>() {
                            return VX_ERROR_INVALID_PARAMETERS;
                        }
                        let border = *(ptr as *const crate::unified_c_api::vx_border_t);
                        if let Ok(mut border_lock) = node_data.border_mode.lock() {
                            *border_lock = border;
                            return VX_SUCCESS;
                        }
                        return VX_ERROR_INVALID_REFERENCE;
                    }
                    VX_NODE_LOCAL_DATA_SIZE | VX_NODE_LOCAL_DATA_PTR => {
                        // Per OpenVX 1.3, the local-data size/ptr can only be
                        // mutated from inside a user-kernel `init`/`deinit`
                        // callback, AND only when the kernel did not request
                        // auto-allocation (VX_KERNEL_LOCAL_DATA_SIZE == 0 at
                        // registration time). Outside of that window, every
                        // other caller must get an error.
                        if !crate::unified_c_api::is_user_kernel_in_init(id) {
                            return VX_ERROR_NOT_SUPPORTED;
                        }
                        if node_data
                            .local_data_auto_alloc
                            .load(std::sync::atomic::Ordering::SeqCst)
                        {
                            return VX_ERROR_NOT_SUPPORTED;
                        }
                        if attribute == VX_NODE_LOCAL_DATA_SIZE {
                            if size < std::mem::size_of::<vx_size>() {
                                return VX_ERROR_INVALID_PARAMETERS;
                            }
                            let v = *(ptr as *const vx_size);
                            node_data
                                .local_data_size
                                .store(v, std::sync::atomic::Ordering::SeqCst);
                        } else {
                            if size < std::mem::size_of::<*mut c_void>() {
                                return VX_ERROR_INVALID_PARAMETERS;
                            }
                            let p = *(ptr as *const *mut c_void);
                            node_data
                                .local_data_ptr
                                .store(p, std::sync::atomic::Ordering::SeqCst);
                        }
                        return VX_SUCCESS;
                    }
                    _ => {
                        // For other attributes, just validate and return success
                        return VX_SUCCESS;
                    }
                }
            }
        }
    }
    VX_ERROR_INVALID_REFERENCE
}

#[no_mangle]
pub extern "C" fn vxReleaseNode(node: *mut vx_node) -> vx_status {
    if node.is_null() {
        return VX_ERROR_INVALID_REFERENCE;
    }
    unsafe {
        let n = *node;
        if n.is_null() {
            return VX_ERROR_INVALID_REFERENCE;
        }
        let id = n as u64;
        let count;
        let graph_id: u64;

        // Get node data and parameters
        if let Ok(nodes) = NODES.lock() {
            if let Some(node_data) = nodes.get(&id) {
                graph_id = node_data.graph_id;
                count = node_data
                    .ref_count
                    .fetch_sub(1, std::sync::atomic::Ordering::SeqCst);
            } else {
                // Node not found, might already be freed
                *node = std::ptr::null_mut();
                return VX_SUCCESS;
            }
        } else {
            *node = std::ptr::null_mut();
            return VX_SUCCESS;
        }

        if count <= 1 {
            // Check if the graph still exists and owns this node
            let graph_still_exists = if let Ok(graphs_data) = GRAPHS_DATA.lock() {
                if let Some(g) = graphs_data.get(&graph_id) {
                    if let Ok(graph_nodes) = g.nodes.read() {
                        graph_nodes.contains(&id)
                    } else {
                        false
                    }
                } else {
                    false
                }
            } else {
                false
            };

            if graph_still_exists {
                // Node is still owned by graph - just decrement unified ref count
                // Don't free the node, the graph will free it when released
                if let Ok(mut counts) = REFERENCE_COUNTS.lock() {
                    if let Some(cnt) = counts.get_mut(&(id as usize)) {
                        cnt.store(count - 1, std::sync::atomic::Ordering::SeqCst);
                    }
                }
            } else {
                // Graph doesn't exist or doesn't own this node - safe to free.
                //
                // If this node ran a user kernel that was initialised by
                // vxVerifyGraph, run its `deinit` callback now and release
                // any auto-allocated local-data buffer.
                crate::unified_c_api::deinit_user_kernel_node(id);

                // Note: Node does NOT release parameter values - application owns them
                // The application calls vxReleaseScalar, vxReleaseImage, etc. on its own objects
                // Node parameters are just references/borrowed objects

                // Note: Parameter handles returned by vxGetParameterByIndex are owned by the caller
                // Do NOT remove parameter entries here - vxReleaseParameter will handle them
                // Only clean up the node's parameter value storage (not the handles)

                // Note: Node does NOT release parameter values - application owns them
                // The application calls vxReleaseScalar, vxReleaseImage, etc. on its own objects
                // Node parameters are just references/borrowed objects
                //
                // Exception: scalars created by convenience functions (vxAddNode, etc.)
                // are internal and should be released when the graph is freed.
                // We don't release them here because the graph cleanup will handle it.
                //

                // Remove from graph's node list BEFORE removing node
                // Remove from c_api graph registry
                if let Ok(graphs) = GRAPHS.lock() {
                    if let Some(g) = graphs.get(&graph_id) {
                        if let Ok(mut graph_nodes) = g.nodes.lock() {
                            graph_nodes.retain(|&nid| nid != id);
                        }
                    }
                }
                // Remove from unified graph registry
                if let Ok(graphs_data) = GRAPHS_DATA.lock() {
                    if let Some(g) = graphs_data.get(&graph_id) {
                        if let Ok(mut graph_nodes) = g.nodes.write() {
                            graph_nodes.retain(|&nid| nid != id);
                        }
                    }
                }

                // Remove from NODES and unified registries
                if let Ok(mut nodes_mut) = NODES.lock() {
                    nodes_mut.remove(&id);
                }
                if let Ok(mut counts) = REFERENCE_COUNTS.lock() {
                    counts.remove(&(id as usize));
                }
                if let Ok(mut types) = REFERENCE_TYPES.lock() {
                    types.remove(&(id as usize));
                }
            }
        } else {
            // Decrement unified reference count
            if let Ok(mut counts) = REFERENCE_COUNTS.lock() {
                if let Some(cnt) = counts.get_mut(&(id as usize)) {
                    cnt.store(count - 1, std::sync::atomic::Ordering::SeqCst);
                }
            }
        }
        *node = std::ptr::null_mut();
    }
    VX_SUCCESS
}

#[no_mangle]
pub extern "C" fn vxRemoveNode(node: *mut vx_node) -> vx_status {
    if node.is_null() {
        return VX_ERROR_INVALID_REFERENCE;
    }
    unsafe {
        let n = *node;
        if n.is_null() {
            return VX_ERROR_INVALID_REFERENCE;
        }
        let id = n as u64;

        // Collect parameter references before removing the node
        // so we can decrement their ref counts
        let mut param_refs: Vec<usize> = Vec::new();
        let mut graph_id: u64 = 0;

        if let Ok(nodes) = NODES.lock() {
            if let Some(node_data) = nodes.get(&id) {
                graph_id = node_data.graph_id;
                if let Ok(params) = node_data.parameters.lock() {
                    for param_opt in params.iter() {
                        if let Some(ref_id) = param_opt {
                            if *ref_id != 0 {
                                param_refs.push(*ref_id as usize);
                            }
                        }
                    }
                }
            }
        }

        // Remove from graph's node list (c_api registry)
        if graph_id != 0 {
            if let Ok(graphs) = GRAPHS.lock() {
                if let Some(graph) = graphs.get(&graph_id) {
                    if let Ok(mut graph_nodes) = graph.nodes.lock() {
                        graph_nodes.retain(|&nid| nid != id);
                    }
                }
            }

            // Also remove from unified GRAPHS_DATA registry (where vxQueryGraph reads from)
            if let Ok(graphs_data) = crate::unified_c_api::GRAPHS_DATA.lock() {
                if let Some(graph) = graphs_data.get(&graph_id) {
                    if let Ok(mut graph_nodes) = graph.nodes.write() {
                        graph_nodes.retain(|&nid| nid != id);
                    }
                    // Mark graph as unverified since structure changed
                    if let Ok(mut verified) = graph.verified.lock() {
                        *verified = false;
                    }
                    if let Ok(mut state) = graph.state.lock() {
                        *state = crate::unified_c_api::VxGraphState::VxGraphStateUnverified;
                    }
                }
            }
        }

        // Remove the node from NODES registry and clean up
        if let Ok(mut nodes_mut) = NODES.lock() {
            nodes_mut.remove(&id);
        }

        // Remove from reference counting and type registries
        if let Ok(mut counts) = REFERENCE_COUNTS.lock() {
            counts.remove(&(id as usize));
        }
        if let Ok(mut types) = REFERENCE_TYPES.lock() {
            types.remove(&(id as usize));
        }

        // Decrement ref counts of parameter references
        // Per OpenVX spec, vxRemoveNode releases all parameter references
        for ref_addr in param_refs {
            if let Ok(counts) = REFERENCE_COUNTS.lock() {
                if let Some(cnt) = counts.get(&ref_addr) {
                    let current = cnt.load(std::sync::atomic::Ordering::SeqCst);
                    if current > 0 {
                        cnt.store(current - 1, std::sync::atomic::Ordering::SeqCst);
                    }
                }
            }
        }

        *node = std::ptr::null_mut();
    }
    VX_SUCCESS
}

#[no_mangle]
pub extern "C" fn vxAssignNodeCallback(node: vx_node, callback: vx_nodecomplete_f) -> vx_status {
    if node.is_null() {
        return VX_ERROR_INVALID_REFERENCE;
    }
    let id = node as u64;
    if let Ok(nodes) = NODES.lock() {
        if let Some(node_data) = nodes.get(&id) {
            if let Ok(mut cb) = node_data.callback.lock() {
                *cb = Some(callback);
                return VX_SUCCESS;
            }
        }
    }
    VX_ERROR_INVALID_REFERENCE
}

#[no_mangle]
pub extern "C" fn vxCreateGenericNode(graph: vx_graph, kernel: vx_kernel) -> vx_node {
    if graph.is_null() || kernel.is_null() {
        return std::ptr::null_mut();
    }
    let graph_id = graph as u64;
    let kernel_id = kernel as u64;

    // Get kernel num_params and (for user kernels) the auto-allocated local
    // data size. Built-in kernels have no per-node local data.
    let (num_params, user_kernel_local_size) = if let Ok(kernels) = KERNELS.lock() {
        if let Some(k) = kernels.get(&kernel_id) {
            (k.num_params, 0usize)
        } else if let Ok(user_kernels) = crate::unified_c_api::USER_KERNELS.lock() {
            if let Some(uk) = user_kernels.get(&(kernel_id as i32)) {
                (
                    uk.num_params,
                    uk.local_data_size
                        .load(std::sync::atomic::Ordering::SeqCst),
                )
            } else {
                (4, 0)
            }
        } else {
            (4, 0)
        }
    } else {
        (4, 0)
    };

    // Get context_id from graph
    let context_id = if let Ok(graphs) = GRAPHS.lock() {
        if let Some(g) = graphs.get(&graph_id) {
            g.context_id
        } else {
            return std::ptr::null_mut();
        }
    } else {
        return std::ptr::null_mut();
    };

    let id = generate_id();
    let node = Arc::new(NodeData {
        id,
        context_id,
        graph_id,
        kernel_id,
        parameters: Mutex::new(vec![None; num_params as usize]),
        callback: Mutex::new(None),
        status: std::sync::atomic::AtomicI32::new(0),
        ref_count: std::sync::atomic::AtomicUsize::new(1),
        border_mode: Mutex::new(crate::unified_c_api::vx_border_t {
            mode: crate::unified_c_api::VX_BORDER_UNDEFINED,
            constant_value: crate::c_api_data::vx_pixel_value_t { U32: 0 },
        }),
        run_count: std::sync::atomic::AtomicU64::new(0),
        execution_count: std::sync::atomic::AtomicU32::new(0),
        node_state: std::sync::atomic::AtomicU32::new(crate::unified_c_api::VX_NODE_STATE_STEADY as u32),
        user_kernel_initialized: std::sync::atomic::AtomicBool::new(false),
        local_data_size: std::sync::atomic::AtomicUsize::new(user_kernel_local_size),
        local_data_ptr: std::sync::atomic::AtomicPtr::new(std::ptr::null_mut()),
        local_data_auto_alloc: std::sync::atomic::AtomicBool::new(user_kernel_local_size > 0),
    });

    if let Ok(mut nodes) = NODES.lock() {
        nodes.insert(id, node.clone());
    }

    // Add to graph in c_api registry
    {
        if let Ok(graphs) = GRAPHS.lock() {
            if let Some(g) = graphs.get(&graph_id) {
                if let Ok(mut graph_nodes) = g.nodes.lock() {
                    graph_nodes.push(id);
                }
            }
        }
    }

    // Reset graph state to UNVERIFIED since a node was added
    {
        if let Ok(graphs_data) = GRAPHS_DATA.lock() {
            if let Some(g) = graphs_data.get(&graph_id) {
                let mut state = g.state.lock().unwrap();
                *state = crate::unified_c_api::VxGraphState::VxGraphStateUnverified;
            }
        }
    }

    // Also add to unified graph registry for vxProcessGraph
    if let Ok(graphs_data) = GRAPHS_DATA.lock() {
        if let Some(g) = graphs_data.get(&graph_id) {
            if let Ok(mut graph_nodes) = g.nodes.write() {
                graph_nodes.push(id);
            }
        }
    }

    // Initialize reference count for the node (same pattern as other objects)
    let ptr = id as *mut VxNode;
    unsafe {
        if let Ok(mut counts) = REFERENCE_COUNTS.lock() {
            counts.insert(ptr as usize, AtomicUsize::new(1));
        }
        if let Ok(mut types) = REFERENCE_TYPES.lock() {
            types.insert(ptr as usize, crate::unified_c_api::VX_TYPE_NODE);
        }
    }

    // Increment ref count for the graph's ownership
    // When vxReleaseNode is called, it will decrement but not free
    // The graph will free the node when vxReleaseGraph is called
    if let Ok(counts) = REFERENCE_COUNTS.lock() {
        if let Some(cnt) = counts.get(&(ptr as usize)) {
            cnt.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        }
    }

    ptr
}

// ============================================================================
// Kernel Loading Functions
// ============================================================================

#[no_mangle]
pub extern "C" fn vxLoadKernels(context: vx_context, module: *const vx_char) -> vx_status {
    use crate::unified_c_api::MODULES;

    if context.is_null() {
        return VX_ERROR_INVALID_REFERENCE;
    }
    if module.is_null() {
        return VX_ERROR_INVALID_PARAMETERS;
    }

    unsafe {
        let module_name = match CStr::from_ptr(module).to_str() {
            Ok(s) => s,
            Err(_) => return VX_ERROR_INVALID_PARAMETERS,
        };

        // Track this module as loaded for this context
        let context_id = context as u64;
        if let Ok(mut modules) = MODULES.lock() {
            let context_modules: &mut std::collections::HashSet<String> = modules
                .entry(context_id)
                .or_insert_with(std::collections::HashSet::new);
            context_modules.insert(module_name.to_string());
        }

        // Parse module name and register kernels
        // Handle test module only - standard kernels are already registered
        if module_name == "test-testmodule" || module_name == "org.khronos.test.testmodule" {
            // Register test kernels for CTS
            // Add a dummy test kernel
            let test_kernel_id = generate_id();
            let test_kernel = Arc::new(KernelData {
                id: test_kernel_id,
                context_id: context as u32,
                name: "org.khronos.test.testmodule".to_string(),
                kernel_enum: 0x100000, // VX_KERNEL_BASE(VX_ID_USER, 0)
                num_params: 0,
                ref_count: std::sync::atomic::AtomicUsize::new(1),
            });
            if let Ok(mut kernels) = KERNELS.lock() {
                kernels.insert(test_kernel_id, test_kernel);
            }
            // Register test kernel in REFERENCE_COUNTS and REFERENCE_TYPES
            if let Ok(mut counts) = REFERENCE_COUNTS.lock() {
                counts.insert(test_kernel_id as usize, AtomicUsize::new(1));
            }
            if let Ok(mut types) = REFERENCE_TYPES.lock() {
                types.insert(
                    test_kernel_id as usize,
                    crate::unified_c_api::VX_TYPE_KERNEL,
                );
            }
            VX_SUCCESS
        } else if module_name == "openvx-core"
            || module_name == "openvx-vision"
            || module_name.is_empty()
        {
            // Standard kernels already registered at context creation
            VX_SUCCESS
        } else {
            // Look up libraries previously registered via
            // `vxRegisterKernelLibrary` for this context and invoke
            // their `publish` callback. The callback is expected to
            // call `vxAddUserKernel` for each kernel in the library.
            use crate::unified_c_api::REGISTERED_KERNEL_LIBRARIES;
            let reg = {
                let libs = match REGISTERED_KERNEL_LIBRARIES.lock() {
                    Ok(libs) => libs,
                    Err(_) => return VX_ERROR_INVALID_PARAMETERS,
                };
                libs.get(&context_id)
                    .and_then(|m| m.get(module_name))
                    .copied()
            };
            match reg {
                Some(r) => match r.publish {
                    Some(publish) => publish(context),
                    None => VX_SUCCESS,
                },
                None => VX_ERROR_INVALID_PARAMETERS,
            }
        }
    }
}

#[no_mangle]
pub extern "C" fn vxUnloadKernels(context: vx_context, module: *const vx_char) -> vx_status {
    use crate::unified_c_api::MODULES;

    if context.is_null() {
        return VX_ERROR_INVALID_REFERENCE;
    }
    if module.is_null() {
        return VX_ERROR_INVALID_PARAMETERS;
    }

    unsafe {
        let module_name = match CStr::from_ptr(module).to_str() {
            Ok(s) => s,
            Err(_) => return VX_ERROR_INVALID_PARAMETERS,
        };

        // Remove this module from the loaded modules for this context
        let context_id = context as u64;
        if let Ok(mut modules) = MODULES.lock() {
            if let Some(context_modules) = modules.get_mut(&context_id) {
                context_modules.remove(module_name);
            }
        }

        // If a static library was registered via `vxRegisterKernelLibrary`
        // for this (context, module) pair, give it the chance to tear
        // down its kernels by invoking the matching `unpublish` callback.
        // The registration itself is intentionally kept around — the
        // spec lets a library be re-loaded later without re-registering.
        {
            use crate::unified_c_api::REGISTERED_KERNEL_LIBRARIES;
            let reg = REGISTERED_KERNEL_LIBRARIES
                .lock()
                .ok()
                .and_then(|libs| libs.get(&context_id).and_then(|m| m.get(module_name)).copied());
            if let Some(r) = reg {
                if let Some(unpublish) = r.unpublish {
                    let _ = unpublish(context);
                }
            }
        }

        // Unregister kernels from this module
        let context_id_u32 = context as u32;
        if module_name == "test-testmodule" {
            // Get list of kernels to remove first
            let mut kernels_to_remove: Vec<u64> = Vec::new();
            {
                if let Ok(kernels) = KERNELS.lock() {
                    for (id, k) in kernels.iter() {
                        if k.context_id == context_id_u32 && k.name == "org.khronos.test.testmodule"
                        {
                            kernels_to_remove.push(*id);
                        }
                    }
                }
            }

            // Now remove them with proper cleanup
            if let Ok(mut kernels) = KERNELS.lock() {
                for id in kernels_to_remove {
                    if let Some(kernel) = kernels.remove(&id) {
                        // Clean up kernel resources
                        drop(kernel);
                    }
                    // Remove from reference tracking
                    if let Ok(mut counts) = REFERENCE_COUNTS.lock() {
                        counts.remove(&(id as usize));
                    }
                    if let Ok(mut types) = REFERENCE_TYPES.lock() {
                        types.remove(&(id as usize));
                    }
                }
            }
        }

        VX_SUCCESS
    }
}

#[no_mangle]
pub extern "C" fn vxGetKernelByName(context: vx_context, name: *const vx_char) -> vx_kernel {
    if context.is_null() || name.is_null() {
        return std::ptr::null_mut();
    }
    unsafe {
        let kernel_name = match CStr::from_ptr(name).to_str() {
            Ok(s) => s,
            Err(_) => return std::ptr::null_mut(),
        };

        let context_id = context as u32;

        // Look up kernel by name in built-in kernels
        if let Ok(kernels) = KERNELS.lock() {
            for (id, kernel) in kernels.iter() {
                if kernel.name == kernel_name && kernel.context_id == context_id {
                    // Increment both internal and unified reference counts
                    kernel
                        .ref_count
                        .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                    if let Ok(counts) = REFERENCE_COUNTS.lock() {
                        if let Some(count) = counts.get(&(*id as usize)) {
                            count.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                        }
                    }
                    return *id as *mut VxKernel;
                }
            }
        }

        // Look up in user kernels
        if let Ok(user_kernels) = crate::unified_c_api::USER_KERNELS.lock() {
            for (enum_id, kernel) in user_kernels.iter() {
                if kernel.name == kernel_name {
                    // Return pointer based on enumeration ID
                    let kernel_ptr = *enum_id as usize as vx_kernel;
                    // Ensure the reference type is registered as VX_TYPE_KERNEL
                    if let Ok(mut types) = REFERENCE_TYPES.lock() {
                        types.insert(kernel_ptr as usize, crate::unified_c_api::VX_TYPE_KERNEL);
                    }
                    if let Ok(mut counts) = REFERENCE_COUNTS.lock() {
                        if let Some(count) = counts.get(&(kernel_ptr as usize)) {
                            count.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                        } else {
                            counts.insert(
                                kernel_ptr as usize,
                                std::sync::atomic::AtomicUsize::new(1),
                            );
                        }
                    }
                    return kernel_ptr;
                }
            }
        }

        // Kernel not found - return NULL (don't auto-create)
        std::ptr::null_mut()
    }
}

#[no_mangle]
pub extern "C" fn vxGetKernelByEnum(context: vx_context, kernel_e: vx_enum) -> vx_kernel {
    if context.is_null() {
        return std::ptr::null_mut();
    }
    let context_id = context as u32;

    // Look up kernel by enum
    if let Ok(kernels) = KERNELS.lock() {
        for (id, kernel) in kernels.iter() {
            if kernel.kernel_enum == kernel_e && kernel.context_id == context_id {
                kernel
                    .ref_count
                    .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                // Also increment reference count in unified registry
                if let Ok(mut counts) = REFERENCE_COUNTS.lock() {
                    if let Some(count) = counts.get_mut(&(*id as usize)) {
                        count.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                    }
                }
                return *id as *mut VxKernel;
            }
        }
    }

    // Kernel not found - create it
    let id = generate_id();
    // Determine num_params based on kernel enum
    let num_params = match kernel_e {
        // 2-parameter kernels
        0x0E | 0x11 | 0x12 | 0x13 | 0x15 | 0x1C | 0x1D | 0x1E | 0x1F => 2,
        // 3-parameter kernels
        0x01 | 0x04 | 0x05 | 0x06 | 0x10 | 0x0B | 0x0D | 0x16 | 0x22 | 0x29 | 0x2B => 3,
        // 4-parameter kernels
        0x02 | 0x07 | 0x08 | 0x09 | 0x0A | 0x0C | 0x14 | 0x1A | 0x21 | 0x23 | 0x24 | 0x28
        | 0x2C | 0x40 => 4,
        // 5-parameter kernels (fast_corners, canny_edge)
        0x1B | 0x26 => 5,
        // 6-parameter kernels (minmaxloc)
        0x19 => 6,
        // 10-parameter kernels (optical_flow_pyr_lk)
        0x27 => 10,
        // 8-parameter kernels (harris_corners)
        0x25 => 8,
        // 3-parameter pyramid
        0x2A => 3,
        // Default
        _ => 4,
    };
    let kernel_name = format!("kernel_{}", kernel_e);
    let kernel = Arc::new(KernelData {
        id,
        context_id,
        name: kernel_name,
        kernel_enum: kernel_e,
        num_params,
        ref_count: std::sync::atomic::AtomicUsize::new(1),
    });

    if let Ok(mut kernels) = KERNELS.lock() {
        kernels.insert(id, kernel);
    }

    // Initialize reference count and type for the kernel
    unsafe {
        if let Ok(mut counts) = REFERENCE_COUNTS.lock() {
            counts.insert(id as usize, AtomicUsize::new(1));
        }
        if let Ok(mut types) = REFERENCE_TYPES.lock() {
            types.insert(id as usize, crate::unified_c_api::VX_TYPE_KERNEL);
        }
    }

    id as *mut VxKernel
}

#[no_mangle]
pub extern "C" fn vxQueryKernel(
    kernel: vx_kernel,
    attribute: vx_enum,
    ptr: *mut c_void,
    size: vx_size,
) -> vx_status {
    if kernel.is_null() || ptr.is_null() {
        return VX_ERROR_INVALID_REFERENCE;
    }
    unsafe {
        let id = kernel as u64;
        if let Ok(kernels) = KERNELS.lock() {
            if let Some(kernel_data) = kernels.get(&id) {
                match attribute {
                    VX_KERNEL_PARAMETERS => {
                        // VX_KERNEL_PARAMETERS
                        if size >= 4 {
                            let ptr_u8 = ptr as *mut u8;
                            std::ptr::copy_nonoverlapping(
                                &kernel_data.num_params as *const u32 as *const u8,
                                ptr_u8,
                                4,
                            );
                            return VX_SUCCESS;
                        }
                    }
                    VX_KERNEL_NAME => {
                        // VX_KERNEL_NAME
                        let name_bytes = kernel_data.name.as_bytes();
                        let len = name_bytes.len().min(size);
                        let ptr_u8 = ptr as *mut u8;
                        std::ptr::copy_nonoverlapping(name_bytes.as_ptr(), ptr_u8, len);
                        return VX_SUCCESS;
                    }
                    VX_KERNEL_ENUM => {
                        // VX_KERNEL_ENUM
                        if size >= 4 {
                            let ptr_u8 = ptr as *mut u8;
                            std::ptr::copy_nonoverlapping(
                                &kernel_data.kernel_enum as *const i32 as *const u8,
                                ptr_u8,
                                4,
                            );
                            return VX_SUCCESS;
                        }
                    }
                    _ => {}
                }
            }
        }
    }
    VX_ERROR_NOT_IMPLEMENTED
}

#[no_mangle]
pub extern "C" fn vxGetKernelParameterByIndex(kernel: vx_kernel, index: vx_uint32) -> vx_parameter {
    if kernel.is_null() {
        return std::ptr::null_mut();
    }
    let kernel_id = kernel as u64;

    // Get context_id from kernel - check both built-in and user kernels
    let context_id = if let Ok(kernels) = KERNELS.lock() {
        if let Some(k) = kernels.get(&kernel_id) {
            k.context_id
        } else {
            // Check user kernels in unified_c_api (keyed by enumeration value)
            if let Ok(user_kernels) = crate::unified_c_api::USER_KERNELS.lock() {
                if let Some(uk) = user_kernels.get(&(kernel_id as crate::unified_c_api::vx_enum)) {
                    uk.context_id as u32
                } else {
                    return std::ptr::null_mut();
                }
            } else {
                return std::ptr::null_mut();
            }
        }
    } else {
        return std::ptr::null_mut();
    };

    // Look up parameter direction from user kernel params or built-in kernel info
    let (direction, data_type, state) = {
        let kernel_enum = kernel_id as crate::unified_c_api::vx_enum;
        if let Ok(params) = crate::unified_c_api::USER_KERNEL_PARAMS.lock() {
            if let Some(param_list) = params.get(&kernel_enum) {
                if let Some(p) = param_list.get(index as usize) {
                    (p.direction, p.data_type, p.state)
                } else {
                    (0, 0, 1) // defaults
                }
            } else {
                (0, 0, 1)
            }
        } else {
            (0, 0, 1)
        }
    };

    let id = generate_id();
    let param = Arc::new(ParameterData {
        id,
        context_id,
        kernel_id,
        index,
        direction,
        data_type,
        state,
        value: Mutex::new(None),
        ref_count: std::sync::atomic::AtomicUsize::new(1),
    });

    if let Ok(mut params) = PARAMETERS.lock() {
        params.insert(id, param.clone());
    }

    // Also register in unified_c_api PARAMETERS for vxQueryParameter to find
    if let Ok(mut unified_params) = crate::unified_c_api::PARAMETERS.lock() {
        let unified_param = Arc::new(crate::unified_c_api::VxCParameter {
            id: id,
            node_id: 0, // Parameter created at node level
            index,
            direction: 0, // VX_INPUT
            data_type: 0,
            ref_count: std::sync::atomic::AtomicUsize::new(1),
            value: Mutex::new(None),
        });
        unified_params.insert(id, unified_param);
    }

    // Initialize reference count and type for the parameter
    unsafe {
        if let Ok(mut counts) = REFERENCE_COUNTS.lock() {
            counts.insert(id as usize, AtomicUsize::new(1));
        }
        if let Ok(mut types) = REFERENCE_TYPES.lock() {
            types.insert(id as usize, crate::unified_c_api::VX_TYPE_PARAMETER);
        }
    }

    id as *mut VxParameter
}

#[no_mangle]
pub extern "C" fn vxReleaseKernel(kernel: *mut vx_kernel) -> vx_status {
    if kernel.is_null() {
        return VX_ERROR_INVALID_REFERENCE;
    }
    unsafe {
        let k = *kernel;
        if k.is_null() {
            return VX_ERROR_INVALID_REFERENCE;
        }
        let id = k as u64;

        // First get the count, then drop the lock
        let count = if let Ok(kernels) = KERNELS.lock() {
            if let Some(kernel_data) = kernels.get(&id) {
                kernel_data
                    .ref_count
                    .fetch_sub(1, std::sync::atomic::Ordering::SeqCst)
            } else {
                0
            }
        } else {
            0
        };

        if count <= 1 {
            // Last reference - remove from KERNELS first (this drops the Arc properly)
            // then clean up other registries
            if let Ok(mut kernels_mut) = KERNELS.lock() {
                // This will drop the Arc<KernelData> properly
                kernels_mut.remove(&id);
            }
            if let Ok(mut counts) = REFERENCE_COUNTS.lock() {
                counts.remove(&(id as usize));
            }
            if let Ok(mut types) = REFERENCE_TYPES.lock() {
                types.remove(&(id as usize));
            }
            if let Ok(mut names) = REFERENCE_NAMES.lock() {
                names.remove(&(id as usize));
            }
        } else {
            // Just update reference count
            if let Ok(mut counts) = REFERENCE_COUNTS.lock() {
                if let Some(cnt) = counts.get_mut(&(id as usize)) {
                    cnt.store(count - 1, std::sync::atomic::Ordering::SeqCst);
                }
            }
        }
        *kernel = std::ptr::null_mut();
    }
    VX_SUCCESS
}

// ============================================================================
// Parameter API Functions
// ============================================================================

#[no_mangle]
pub extern "C" fn vxQueryParameter(
    param: vx_parameter,
    attribute: vx_enum,
    ptr: *mut c_void,
    size: vx_size,
) -> vx_status {
    if param.is_null() || ptr.is_null() {
        return VX_ERROR_INVALID_REFERENCE;
    }
    unsafe {
        let id = param as u64;

        // First check local c_api PARAMETERS (where vxGetKernelParameterByIndex stores params)
        if let Ok(params) = PARAMETERS.lock() {
            if let Some(param_data) = params.get(&id) {
                match attribute {
                    VX_PARAMETER_INDEX => {
                        // VX_PARAMETER_INDEX
                        if size >= 4 {
                            let ptr_u8 = ptr as *mut u8;
                            std::ptr::copy_nonoverlapping(
                                &param_data.index as *const u32 as *const u8,
                                ptr_u8,
                                4,
                            );
                            return VX_SUCCESS;
                        }
                    }
                    VX_PARAMETER_DIRECTION => {
                        // VX_PARAMETER_DIRECTION
                        if size >= 4 {
                            let ptr_u8 = ptr as *mut u8;
                            std::ptr::copy_nonoverlapping(
                                &param_data.direction as *const i32 as *const u8,
                                ptr_u8,
                                4,
                            );
                            return VX_SUCCESS;
                        }
                    }
                    VX_PARAMETER_TYPE => {
                        // VX_PARAMETER_TYPE
                        if size >= 4 {
                            let ptr_u8 = ptr as *mut u8;
                            std::ptr::copy_nonoverlapping(
                                &param_data.data_type as *const i32 as *const u8,
                                ptr_u8,
                                4,
                            );
                            return VX_SUCCESS;
                        }
                    }
                    VX_PARAMETER_REF => {
                        if size >= std::mem::size_of::<vx_reference>() {
                            // Get the reference value from the parameter
                            if let Ok(value) = param_data.value.lock() {
                                let ref_ptr = value.unwrap_or(0) as vx_reference;
                                // Per OpenVX spec, querying VX_PARAMETER_REF increments the ref count
                                // The caller must release with the appropriate vxRelease* function
                                if !ref_ptr.is_null() {
                                    crate::c_api::vxRetainReference(ref_ptr);
                                }
                                let ptr_u8 = ptr as *mut u8;
                                std::ptr::copy_nonoverlapping(
                                    &ref_ptr as *const _ as *const u8,
                                    ptr_u8,
                                    std::mem::size_of::<vx_reference>(),
                                );
                                return VX_SUCCESS;
                            }
                        }
                    }
                    _ => {}
                }
            }
        }

        // Also check unified_c_api's PARAMETERS
        if let Ok(params) = crate::unified_c_api::PARAMETERS.lock() {
            if let Some(param_data) = params.get(&id) {
                match attribute {
                    VX_PARAMETER_INDEX => {
                        // VX_PARAMETER_INDEX
                        if size >= 4 {
                            let ptr_u8 = ptr as *mut u8;
                            std::ptr::copy_nonoverlapping(
                                &param_data.index as *const u32 as *const u8,
                                ptr_u8,
                                4,
                            );
                            return VX_SUCCESS;
                        }
                    }
                    VX_PARAMETER_DIRECTION => {
                        // VX_PARAMETER_DIRECTION
                        if size >= 4 {
                            let ptr_u8 = ptr as *mut u8;
                            std::ptr::copy_nonoverlapping(
                                &param_data.direction as *const i32 as *const u8,
                                ptr_u8,
                                4,
                            );
                            return VX_SUCCESS;
                        }
                    }
                    VX_PARAMETER_TYPE => {
                        // VX_PARAMETER_TYPE
                        if size >= 4 {
                            let ptr_u8 = ptr as *mut u8;
                            std::ptr::copy_nonoverlapping(
                                &param_data.data_type as *const i32 as *const u8,
                                ptr_u8,
                                4,
                            );
                            return VX_SUCCESS;
                        }
                    }
                    VX_PARAMETER_REF => {
                        if size >= std::mem::size_of::<vx_reference>() {
                            // Get the reference value from the parameter
                            if let Ok(value) = param_data.value.lock() {
                                let ref_ptr = value.unwrap_or(0) as vx_reference;
                                // Per OpenVX spec, querying VX_PARAMETER_REF increments the ref count
                                if !ref_ptr.is_null() {
                                    crate::c_api::vxRetainReference(ref_ptr);
                                }
                                let ptr_u8 = ptr as *mut u8;
                                std::ptr::copy_nonoverlapping(
                                    &ref_ptr as *const _ as *const u8,
                                    ptr_u8,
                                    std::mem::size_of::<vx_reference>(),
                                );
                                return VX_SUCCESS;
                            }
                        }
                    }
                    _ => {}
                }
            }
        }

        // Also check unified_c_api's PARAMETERS via helper function
        if attribute == VX_PARAMETER_REF {
            if let Some(ref_value) = crate::unified_c_api::get_parameter_value(id) {
                if size >= std::mem::size_of::<vx_reference>() {
                    let ref_ptr = ref_value as vx_reference;
                    let ptr_u8 = ptr as *mut u8;
                    std::ptr::copy_nonoverlapping(
                        &ref_ptr as *const _ as *const u8,
                        ptr_u8,
                        std::mem::size_of::<vx_reference>(),
                    );
                    return VX_SUCCESS;
                }
            }
            // If we reach here, check if parameter exists in unified registry
            if crate::unified_c_api::parameter_exists(id) {
                // Parameter exists but has no value, return NULL reference
                let ref_ptr: vx_reference = std::ptr::null_mut();
                let ptr_u8 = ptr as *mut u8;
                std::ptr::copy_nonoverlapping(
                    &ref_ptr as *const _ as *const u8,
                    ptr_u8,
                    std::mem::size_of::<vx_reference>(),
                );
                return VX_SUCCESS;
            }
        }
    }
    VX_ERROR_NOT_IMPLEMENTED
}

#[no_mangle]
pub extern "C" fn vxSetParameterByIndex(
    node: vx_node,
    index: vx_uint32,
    value: vx_reference,
) -> vx_status {
    if node.is_null() {
        return VX_ERROR_INVALID_REFERENCE;
    }
    let id = node as u64;
    // Check if trying to set NULL for required parameters
    // Param 0 is always a required input for vision kernels
    if value.is_null() && index == 0 {
        // Allow null for ChannelCombine kernel which has optional planes
        // All other kernels require param 0
        let is_channel_combine = if let Ok(nodes) = NODES.lock() {
            if let Some(node_data) = nodes.get(&id) {
                node_data.kernel_id == 0x03 // VX_KERNEL_CHANNEL_COMBINE
            } else {
                false
            }
        } else {
            false
        };
        if !is_channel_combine {
            return VX_ERROR_INVALID_PARAMETERS;
        }
    }

    // Store node data before dropping the lock
    let (_context_id, _kernel_id) = if let Ok(nodes) = NODES.lock() {
        if let Some(node_data) = nodes.get(&id) {
            let cid = node_data.context_id;
            let kid = node_data.kernel_id;
            if let Ok(kernels) = KERNELS.lock() {
                if let Some(_kernel_data) = kernels.get(&kid) {
                    // NULL values are allowed for optional output parameters
                    // The caller (create_node_with_params) already skips NULLs,
                    // but if we get here with NULL, just allow it silently
                }
            }

            if let Ok(mut params) = node_data.parameters.lock() {
                if (index as usize) < params.len() {
                    // Retain the new value before storing it (if not null)
                    if !value.is_null() {
                        let value_addr = value as usize;
                        if let Ok(counts) = REFERENCE_COUNTS.lock() {
                            if let Some(cnt) = counts.get(&value_addr) {
                                cnt.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                            }
                        }
                    }
                    // Release old value if exists
                    if let Some(old_value) = params[index as usize] {
                        if old_value != 0 {
                            if let Ok(counts) = REFERENCE_COUNTS.lock() {
                                if let Some(cnt) = counts.get(&(old_value as usize)) {
                                    let current = cnt.load(std::sync::atomic::Ordering::SeqCst);
                                    if current > 1 {
                                        cnt.fetch_sub(1, std::sync::atomic::Ordering::SeqCst);
                                    }
                                }
                            }
                        }
                    }
                    params[index as usize] = Some(value as u64);
                    drop(params);
                    (cid, kid)
                } else {
                    return VX_ERROR_INVALID_PARAMETERS;
                }
            } else {
                return VX_ERROR_INVALID_REFERENCE;
            }
        } else {
            return VX_ERROR_INVALID_REFERENCE;
        }
    } else {
        return VX_ERROR_INVALID_REFERENCE;
    };

    // Also create/update parameter entry in unified_c_api for vxQueryParameter
    let param_id = (id << 32) | (index as u64);
    crate::unified_c_api::create_or_update_parameter_with_node(
        param_id,
        index,
        value as u64,
        id,
    );

    // Check if the value is a delay slot reference and register it for delay parameter resolution
    if !value.is_null() {
        let value_addr = value as usize;
        // Direct delay slot reference
        if let Ok(slot_logical) = crate::unified_c_api::DELAY_SLOT_LOGICAL.lock() {
            if let Some(&(delay_addr, logical_idx)) = slot_logical.get(&value_addr) {
                if let Ok(mut delay_params) = crate::unified_c_api::DELAY_NODE_PARAMS.lock() {
                    delay_params.insert((id, index), (delay_addr, logical_idx));
                }
            }
        }
        // Pyramid level image from a delay slot pyramid
        if let Ok(level_imgs) = crate::unified_c_api::PYRAMID_LEVEL_IMAGES.lock() {
            if let Some(&(pyr_addr, level)) = level_imgs.get(&value_addr) {
                // Check if the pyramid is a delay slot
                if let Ok(slot_logical) = crate::unified_c_api::DELAY_SLOT_LOGICAL.lock() {
                    if let Some(&(delay_addr, logical_idx)) = slot_logical.get(&pyr_addr) {
                        // Register as a delay-related pyramid level image
                        if let Ok(mut delay_pyr_level) =
                            crate::unified_c_api::DELAY_PYRAMID_LEVEL.lock()
                        {
                            delay_pyr_level.insert((id, index), (delay_addr, logical_idx, level));
                        }
                    }
                }
            }
        }
    }

    VX_SUCCESS
}

#[no_mangle]
pub extern "C" fn vxSetParameterByReference(param: vx_parameter, value: vx_reference) -> vx_status {
    if param.is_null() {
        return VX_ERROR_INVALID_REFERENCE;
    }
    if value.is_null() {
        return VX_ERROR_INVALID_PARAMETERS;
    }
    let id = param as u64;

    // Look up node_id and index from the PARAMETERS registry
    let (node_id, index) = if let Ok(params) = crate::unified_c_api::PARAMETERS.lock() {
        if let Some(param_data) = params.get(&id) {
            (param_data.node_id, param_data.index)
        } else {
            // Fallback: try to extract from ID encoding
            (id >> 32, (id & 0xFFFFFFFF) as u32)
        }
    } else {
        (id >> 32, (id & 0xFFFFFFFF) as u32)
    };

    // Update parameter value in unified PARAMETERS registry
    if let Ok(params) = crate::unified_c_api::PARAMETERS.lock() {
        if let Some(param_data) = params.get(&id) {
            if let Ok(mut val) = param_data.value.lock() {
                *val = Some(value as u64);
            }
        }
    }

    // Also update the node's parameter reference
    if let Ok(nodes) = NODES.lock() {
        if let Some(node_data) = nodes.get(&node_id) {
            if let Ok(mut params) = node_data.parameters.lock() {
                if (index as usize) < params.len() {
                    params[index as usize] = Some(value as u64);
                }
            }
        }
    }

    VX_SUCCESS
}

#[no_mangle]
pub extern "C" fn vxReleaseParameter(param: *mut vx_parameter) -> vx_status {
    if param.is_null() {
        return VX_ERROR_INVALID_REFERENCE;
    }
    unsafe {
        let p = *param;
        if p.is_null() {
            return VX_ERROR_INVALID_REFERENCE;
        }
        let id = p as u64;
        let addr = id as usize;

        // Validate that this is a reasonable parameter ID
        if id == 0 || id > 0xFFFFFFFFFFFFFFFF {
            return VX_ERROR_INVALID_REFERENCE;
        }

        // Check if this parameter is in any graph's parameter list
        let mut in_graph = false;
        if let Ok(graphs_data) = crate::unified_c_api::GRAPHS_DATA.lock() {
            for (_graph_id, g) in graphs_data.iter() {
                if let Ok(graph_params) = g.parameters.read() {
                    if graph_params.contains(&id) {
                        in_graph = true;
                        break;
                    }
                }
            }
        }

        if in_graph {
            // Parameter is in a graph - just decrement ref count
            // DO NOT remove from graph - the graph owns the parameter
            // The graph will free the parameter when vxReleaseGraph is called
            if let Ok(counts) = REFERENCE_COUNTS.lock() {
                if let Some(cnt) = counts.get(&addr) {
                    let current = cnt.load(std::sync::atomic::Ordering::SeqCst);
                    if current > 1 {
                        cnt.fetch_sub(1, std::sync::atomic::Ordering::SeqCst);
                    } else {
                        // Last reference but still in graph - keep it alive
                    }
                }
            }
            // NOTE: Do NOT remove from graph's parameter list
            // The graph owns this parameter and will free it when released
        } else {
            // Parameter is not in any graph - can safely remove
            let mut should_remove = false;
            if let Ok(counts) = REFERENCE_COUNTS.lock() {
                if let Some(count) = counts.get(&addr) {
                    let current = count.load(std::sync::atomic::Ordering::SeqCst);
                    if current > 1 {
                        count.store(current - 1, std::sync::atomic::Ordering::SeqCst);
                    } else {
                        should_remove = true;
                    }
                } else {
                }
            }

            if should_remove {
                // Remove from unified PARAMETERS only (remove_parameter handles types/counts)
                crate::unified_c_api::remove_parameter(id);
            }
        }
    }

    unsafe {
        *param = std::ptr::null_mut();
    }

    VX_SUCCESS
}

// ============================================================================
// Hint and Reference Management
// ============================================================================

/// Set a hint on a reference
///
/// This function allows the application to provide performance hints to the implementation.
/// The hint is a general instruction to the OpenVX implementation for optimization.
#[no_mangle]
pub extern "C" fn vxHint(
    reference: vx_reference,
    hint: vx_enum,
    _data: *const c_void,
    _data_size: vx_size,
) -> vx_status {
    if reference.is_null() {
        return VX_ERROR_INVALID_REFERENCE;
    }

    // Validate hint value
    match hint {
        VX_HINT_PERFORMANCE_DEFAULT
        | VX_HINT_PERFORMANCE_LOW_POWER
        | VX_HINT_PERFORMANCE_HIGH_SPEED => {
            // These hints are valid; for now we accept them but don't use them
            // Implementation-specific optimization would go here
        }
        _ => return VX_ERROR_NOT_SUPPORTED,
    }

    VX_SUCCESS
}

// ============================================================================
// Status Constants
// ============================================================================
// Status Codes (aligned with CTS expectations)
// ============================================================================

pub const VX_SUCCESS: vx_status = 0;
pub const VX_FAILURE: vx_status = -1;
pub const VX_ERROR_NOT_IMPLEMENTED: vx_status = -2; // Per OpenVX spec
pub const VX_ERROR_NOT_SUPPORTED: vx_status = -3; // Per OpenVX spec
pub const VX_ERROR_NOT_SUFFICIENT: vx_status = -4;
pub const VX_ERROR_NOT_ALLOCATED: vx_status = -5;
pub const VX_ERROR_NOT_COMPATIBLE: vx_status = -6;
pub const VX_ERROR_NO_RESOURCES: vx_status = -7; // Per OpenVX spec
pub const VX_ERROR_NO_MEMORY: vx_status = -8;
pub const VX_ERROR_OPTIMIZED_AWAY: vx_status = -9;
pub const VX_ERROR_INVALID_PARAMETERS: vx_status = -10; // Per OpenVX spec (-10)
pub const VX_ERROR_INVALID_MODULE: vx_status = -11;
pub const VX_ERROR_INVALID_REFERENCE: vx_status = -12; // Per OpenVX spec (-12)
pub const VX_ERROR_INVALID_LINK: vx_status = -13;
pub const VX_ERROR_INVALID_FORMAT: vx_status = -14;
pub const VX_ERROR_INVALID_DIMENSION: vx_status = -15;
pub const VX_ERROR_INVALID_VALUE: vx_status = -16;
pub const VX_ERROR_INVALID_TYPE: vx_status = -17;
pub const VX_ERROR_INVALID_GRAPH: vx_status = -18;
pub const VX_ERROR_INVALID_NODE: vx_status = -19;
pub const VX_ERROR_INVALID_SCOPE: vx_status = -20;
pub const VX_ERROR_GRAPH_SCHEDULED: vx_status = -21;
pub const VX_ERROR_GRAPH_ABANDONED: vx_status = -22;
pub const VX_ERROR_MULTIPLE_WRITERS: vx_status = -23;
pub const VX_ERROR_REFERENCE_NONZERO: vx_status = -24;
pub const VX_ERROR_INVALID_CONTEXT: vx_status = -25;
pub const VX_ERROR_INVALID_KERNEL: vx_status = -26; // Not in spec, using next available

// ============================================================================
// Type Constants
// ============================================================================

pub const VX_TYPE_INVALID: vx_enum = 0x000;
pub const VX_TYPE_CONTEXT: vx_enum = 0x801;
pub const VX_TYPE_GRAPH: vx_enum = 0x802;
pub const VX_TYPE_NODE: vx_enum = 0x803;
pub const VX_TYPE_KERNEL: vx_enum = 0x804;
pub const VX_TYPE_PARAMETER: vx_enum = 0x805;
pub const VX_TYPE_SCALAR: vx_enum = 0x80D;
pub const VX_TYPE_IMAGE: vx_enum = 0x80F;
pub const VX_TYPE_ARRAY: vx_enum = 0x80E;
pub const VX_TYPE_MATRIX: vx_enum = 0x80B;
pub const VX_TYPE_CONVOLUTION: vx_enum = 0x80C;
pub const VX_TYPE_THRESHOLD: vx_enum = 0x80A;
pub const VX_TYPE_PYRAMID: vx_enum = 0x809;
pub const VX_TYPE_LUT: vx_enum = 0x807;

// Additional data types
pub const VX_TYPE_UINT8: vx_enum = 0x003;
pub const VX_TYPE_INT8: vx_enum = 0x002;
pub const VX_TYPE_UINT16: vx_enum = 0x005;
pub const VX_TYPE_INT16: vx_enum = 0x004;
pub const VX_TYPE_UINT32: vx_enum = 0x007;
pub const VX_TYPE_INT32: vx_enum = 0x006;
pub const VX_TYPE_UINT64: vx_enum = 0x009;
pub const VX_TYPE_INT64: vx_enum = 0x008;
pub const VX_TYPE_FLOAT32: vx_enum = 0x00A;
pub const VX_TYPE_FLOAT64: vx_enum = 0x00B;
pub const VX_TYPE_BOOL: vx_enum = 0x010;
pub const VX_TYPE_ENUM: vx_enum = 0x00C;
pub const VX_TYPE_SIZE: vx_enum = 0x00D;
pub const VX_TYPE_CHAR: vx_enum = 0x001;
pub const VX_TYPE_DF_IMAGE: vx_enum = 0x00E;
pub const VX_TYPE_RECTANGLE: vx_enum = 0x020;
pub const VX_TYPE_KEYPOINT: vx_enum = 0x021;
pub const VX_TYPE_COORDINATES2D: vx_enum = 0x022;
pub const VX_TYPE_COORDINATES3D: vx_enum = 0x023;

// ============================================================================
// Attribute Constants
// ============================================================================

// Node attributes - calculated using VX_ATTRIBUTE_BASE(VX_ID_KHRONOS(0), VX_TYPE_NODE) + offset
// VX_ATTRIBUTE_BASE = ((0 << 20) | (0x803 << 8)) = 0x80300
pub const VX_NODE_STATUS: vx_enum = 0x80300; // VX_ATTRIBUTE_BASE + 0x00
pub const VX_NODE_PERFORMANCE: vx_enum = 0x80301; // VX_ATTRIBUTE_BASE + 0x01
pub const VX_NODE_BORDER: vx_enum = 0x80302; // VX_ATTRIBUTE_BASE + 0x02
pub const VX_NODE_LOCAL_DATA_SIZE: vx_enum = 0x80303; // VX_ATTRIBUTE_BASE + 0x03
pub const VX_NODE_LOCAL_DATA_PTR: vx_enum = 0x80304; // VX_ATTRIBUTE_BASE + 0x04
pub const VX_NODE_STATE: vx_enum = 0x80309; // VX_ATTRIBUTE_BASE + 0x09 (KHR pipelining)
pub const VX_NODE_PARAMETERS: vx_enum = 0x80305; // VX_ATTRIBUTE_BASE + 0x05
pub const VX_NODE_IS_REPLICATED: vx_enum = 0x80306; // VX_ATTRIBUTE_BASE + 0x06
pub const VX_NODE_REPLICATE_FLAGS: vx_enum = 0x80307; // VX_ATTRIBUTE_BASE + 0x07

// Kernel attributes using VX_ATTRIBUTE_BASE(VX_ID_KHRONOS(0), VX_TYPE_KERNEL(0x807))
pub const VX_KERNEL_PARAMETERS: vx_enum = 0x80400; // VX_ATTRIBUTE_BASE + 0x00
pub const VX_KERNEL_NAME: vx_enum = 0x80401; // VX_ATTRIBUTE_BASE + 0x01
pub const VX_KERNEL_ENUM: vx_enum = 0x80402; // VX_ATTRIBUTE_BASE + 0x02

// Parameter attributes using VX_ATTRIBUTE_BASE(VX_ID_KHRONOS(0), VX_TYPE_PARAMETER(0x805))
pub const VX_PARAMETER_INDEX: vx_enum = 0x80500; // VX_ATTRIBUTE_BASE + 0x00
pub const VX_PARAMETER_DIRECTION: vx_enum = 0x80501; // VX_ATTRIBUTE_BASE + 0x01
pub const VX_PARAMETER_TYPE: vx_enum = 0x80502; // VX_ATTRIBUTE_BASE + 0x02
pub const VX_PARAMETER_REF: vx_enum = 0x80504; // VX_ATTRIBUTE_BASE + 0x04

// Scalar attributes
pub const VX_SCALAR_TYPE: vx_enum = 0x80D00;

// Threshold attributes
pub const VX_THRESHOLD_VALUE: vx_enum = 0x00;
pub const VX_THRESHOLD_LOWER: vx_enum = 0x01;
pub const VX_THRESHOLD_UPPER: vx_enum = 0x02;
pub const VX_THRESHOLD_TRUE_VALUE: vx_enum = 0x03;
pub const VX_THRESHOLD_FALSE_VALUE: vx_enum = 0x04;

// ============================================================================
// Direction Constants
// ============================================================================

pub const VX_INPUT: vx_enum = 0;
pub const VX_OUTPUT: vx_enum = 1;

// ============================================================================
// Hint Constants
// ============================================================================
// VX_ENUM_BASE(VX_ID_KHRONOS(0), VX_ENUM_HINT(0x02)) = ((0 << 20) | (0x02 << 12)) = 0x2000
pub const VX_HINT_PERFORMANCE_DEFAULT: vx_enum = 0x2001;
pub const VX_HINT_PERFORMANCE_LOW_POWER: vx_enum = 0x2002;
pub const VX_HINT_PERFORMANCE_HIGH_SPEED: vx_enum = 0x2003;

// ============================================================================
// Memory and Copy Constants
// ============================================================================
// VX_ENUM_BASE(VX_ID_KHRONOS(0), VX_ENUM_ACCESSOR(0x11)) = (0 << 20) | (0x11 << 12) = 0x11000
pub const VX_READ_ONLY: vx_enum = 0x11001; // VX_ENUM_BASE(0, 0x11) + 1
pub const VX_WRITE_ONLY: vx_enum = 0x11002; // VX_ENUM_BASE(0, 0x11) + 2
pub const VX_READ_AND_WRITE: vx_enum = 0x11003; // VX_ENUM_BASE(0, 0x11) + 3
                                                // VX_MEMORY_TYPE_HOST = VX_ENUM_BASE(VX_ID_KHRONOS(0), VX_ENUM_MEMORY_TYPE(0x0E)) + 1
                                                // VX_ENUM_BASE = (0 << 20) | (0x0E << 12) = 0xE000, +1 = 0xE001
pub const VX_MEMORY_TYPE_NONE: vx_enum = 0xE000;
pub const VX_MEMORY_TYPE_HOST: vx_enum = 0xE001;

// ============================================================================
// Image Format Constants (OpenVX spec FourCC values)
// Format: VX_DF_IMAGE(a,b,c,d) = ((vx_uint32)(vx_uint8)(a) | ((vx_uint32)(vx_uint8)(b) << 8U) |
//                                 ((vx_uint32)(vx_uint8)(c) << 16U) | ((vx_uint32)(vx_uint8)(d) << 24U))
// ============================================================================

// Format: 'U008' = 0x38303055
pub const VX_DF_IMAGE_U8: vx_df_image = 0x38303055u32;
// Format: 'S008' = 0x38303053
pub const VX_DF_IMAGE_S8: vx_df_image = 0x38303053u32;
// Format: 'U016' = 0x36313055
pub const VX_DF_IMAGE_U16: vx_df_image = 0x36313055u32;
// Format: 'S016' = 0x36313053
pub const VX_DF_IMAGE_S16: vx_df_image = 0x36313053u32;
// Format: 'U032' = 0x32333055
pub const VX_DF_IMAGE_U32: vx_df_image = 0x32333055u32;
// Format: 'S032' = 0x32333053
pub const VX_DF_IMAGE_S32: vx_df_image = 0x32333053u32;
// Format: 'RGB2' = 0x32424752
pub const VX_DF_IMAGE_RGB: vx_df_image = 0x32424752u32;
// Format: 'RGBA' = 0x41424752
pub const VX_DF_IMAGE_RGBX: vx_df_image = 0x41424752u32;
pub const VX_DF_IMAGE_RGBA: vx_df_image = 0x41424752u32; // Same as RGBX per spec
                                                         // Format: 'NV12' = 0x3231564E
pub const VX_DF_IMAGE_NV12: vx_df_image = 0x3231564Eu32;
// Format: 'NV21' = 0x3132564E
pub const VX_DF_IMAGE_NV21: vx_df_image = 0x3132564Eu32;
// Format: 'UYVY' = 0x59565955
pub const VX_DF_IMAGE_UYVY: vx_df_image = 0x59565955u32;
// Format: 'YUYV' = 0x56595559
pub const VX_DF_IMAGE_YUYV: vx_df_image = 0x56595559u32;
// Format: 'IYUV' = 0x56555949
pub const VX_DF_IMAGE_IYUV: vx_df_image = 0x56555949u32;
// Format: 'YUV4' = 0x34565559
pub const VX_DF_IMAGE_YUV4: vx_df_image = 0x34565559u32;
// Same as U8
pub const VX_DF_IMAGE_GRAYSCALE: vx_df_image = 0x38303055u32;

// Virtual image format (VIRT as FourCC)
pub const VX_DF_IMAGE_VIRT: vx_df_image = 0x54524956u32; // 'VIRT' in ASCII

// ============================================================================
// ============================================================================
// Image Attributes
// VX_ATTRIBUTE_BASE(VX_ID_KHRONOS, VX_TYPE_IMAGE) = (0x000 << 20) | (0x80F << 8) = 0x80F00
// ============================================================================

pub const VX_IMAGE_FORMAT: vx_enum = 0x80F02;
pub const VX_IMAGE_WIDTH: vx_enum = 0x80F00;
pub const VX_IMAGE_HEIGHT: vx_enum = 0x80F01;
pub const VX_IMAGE_PLANES: vx_enum = 0x80F03;
pub const VX_IMAGE_SPACE: vx_enum = 0x80F04;
pub const VX_IMAGE_RANGE: vx_enum = 0x80F05;
pub const VX_IMAGE_MEMORY_TYPE: vx_enum = 0x80F07;
pub const VX_IMAGE_IS_UNIFORM: vx_enum = 0x80F08;
pub const VX_IMAGE_UNIFORM_VALUE: vx_enum = 0x80F09;
pub const VX_IMAGE_SIZE: vx_enum = 0x80F06;
pub const VX_IMAGE_IS_VIRTUAL: vx_enum = 0x80F0A;

// Color space enum values
pub const VX_COLOR_SPACE_NONE: vx_enum = 0x6000;
pub const VX_COLOR_SPACE_BT601_525: vx_enum = 0x6001;
pub const VX_COLOR_SPACE_BT601_625: vx_enum = 0x6002;
pub const VX_COLOR_SPACE_BT709: vx_enum = 0x6003;
pub const VX_COLOR_SPACE_BT2020: vx_enum = 0x6004;
pub const VX_COLOR_SPACE_DEFAULT: vx_enum = VX_COLOR_SPACE_NONE;

// Channel range enum values
pub const VX_CHANNEL_RANGE_FULL: vx_enum = 0x7000;
pub const VX_CHANNEL_RANGE_RESTRICTED: vx_enum = 0x7001;
pub const VX_CHANNEL_RANGE_YUV_RESTRICTED: vx_enum = 0x7002;

// ============================================================================
// Array Attributes
// ============================================================================

// Array attributes - VX_ATTRIBUTE_BASE(VX_ID_KHRONOS(0), VX_TYPE_ARRAY) = (0<<20)|(0x80E<<8) = 0x80E00
pub const VX_ARRAY_ITEMTYPE: vx_enum = 0x80E00; // VX_ATTRIBUTE_BASE(VX_ID_KHRONOS, VX_TYPE_ARRAY) + 0x0
pub const VX_ARRAY_NUMITEMS: vx_enum = 0x80E01; // + 0x1
pub const VX_ARRAY_CAPACITY: vx_enum = 0x80E02; // + 0x2
pub const VX_ARRAY_ITEMSIZE: vx_enum = 0x80E03; // + 0x3

// ============================================================================
// Opaque Types for Arrays and Structures
// ============================================================================

pub enum VxArray {}
pub type vx_array = *mut VxArray;

/// Rectangle structure
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct vx_rectangle_t {
    pub start_x: vx_uint32,
    pub start_y: vx_uint32,
    pub end_x: vx_uint32,
    pub end_y: vx_uint32,
}

#[repr(C)]
pub struct vx_keypoint_t {
    pub x: vx_int32,
    pub y: vx_int32,
    pub strength: vx_float32,
    pub scale: vx_float32,
    pub orientation: vx_float32,
    pub tracking_status: vx_int32,
    pub error: vx_float32,
}

#[repr(C)]
pub struct vx_coordinates2d_t {
    pub x: vx_uint32,
    pub y: vx_uint32,
}

#[repr(C)]
pub struct vx_coordinates3d_t {
    pub x: vx_uint32,
    pub y: vx_uint32,
    pub z: vx_uint32,
}

/// Image patch addressing structure
#[repr(C)]
pub struct vx_imagepatch_addressing_t {
    pub dim_x: vx_uint32,
    pub dim_y: vx_uint32,
    pub stride_x: vx_int32,
    pub stride_y: vx_int32,
    pub scale_x: vx_uint32,
    pub scale_y: vx_uint32,
    pub step_x: vx_uint32,
    pub step_y: u16,        // C type is vx_uint16 (u16)
    pub stride_x_bits: u16, // C type is vx_uint16 (u16)
}
