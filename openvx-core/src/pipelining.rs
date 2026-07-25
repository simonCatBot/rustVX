//! OpenVX Pipelining, Streaming, and Batch Processing Extension
//!
//! Implements the KHR pipelining extension:
//! - Queueing API: vxSetGraphScheduleConfig, vxGraphParameterEnqueueReadyRef,
//!   vxGraphParameterDequeueDoneRef, vxGraphParameterCheckDoneRef
//! - Event API: vxWaitEvent, vxEnableEvents, vxDisableEvents,
//!   vxSendUserEvent, vxRegisterEvent
//! - Streaming API: vxEnableGraphStreaming, vxStartGraphStreaming,
//!   vxStopGraphStreaming

use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicBool, AtomicU32};
use std::sync::{Arc, Condvar, Mutex, RwLock};

// ============================================================================
// Pipelining Types
// ============================================================================

/// Graph schedule mode (Khronos vendor-encoded values)
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VxGraphScheduleMode {
    Normal = 0x21000,
    QueueAuto = 0x21001,
    QueueManual = 0x21002,
}

impl TryFrom<vx_enum> for VxGraphScheduleMode {
    type Error = ();
    fn try_from(v: vx_enum) -> Result<Self, Self::Error> {
        match v {
            0x21000 => Ok(VxGraphScheduleMode::Normal),
            0x21001 => Ok(VxGraphScheduleMode::QueueAuto),
            0x21002 => Ok(VxGraphScheduleMode::QueueManual),
            _ => Err(()),
        }
    }
}

/// Event type (Khronos vendor-encoded values)
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VxEventType {
    GraphParameterConsumed = 0x22000,
    GraphCompleted = 0x22001,
    NodeCompleted = 0x22002,
    NodeError = 0x22003,
    User = 0x22004,
}

impl TryFrom<vx_enum> for VxEventType {
    type Error = ();
    fn try_from(v: vx_enum) -> Result<Self, Self::Error> {
        match v {
            0x22000 => Ok(VxEventType::GraphParameterConsumed),
            0x22001 => Ok(VxEventType::GraphCompleted),
            0x22002 => Ok(VxEventType::NodeCompleted),
            0x22003 => Ok(VxEventType::NodeError),
            0x22004 => Ok(VxEventType::User),
            _ => Err(()),
        }
    }
}

/// Node state for streaming
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VxNodeState {
    Steady = 0,
    Pipeup = 1,
}

// ============================================================================
// Data Structures
// ============================================================================

/// Per-graph parameter queue
pub struct VxGraphParameterQueue {
    /// Queue of references ready for next execution
    pub ready_refs: Mutex<VecDeque<usize>>,
    /// References currently consumed by an active execution
    pub consumed_refs: Mutex<VecDeque<usize>>,
    /// References that have completed execution and can be dequeued
    pub done_refs: Mutex<VecDeque<usize>>,
    /// Condition variable for dequeue blocking
    pub done_cv: Condvar,
    /// Maximum queue depth
    pub max_depth: usize,
    /// Whether queueing is enabled for this parameter
    pub enabled: AtomicBool,
    /// List of valid references (from vxSetGraphScheduleConfig)
    pub valid_refs: RwLock<Vec<usize>>,
}

impl VxGraphParameterQueue {
    pub fn new(max_depth: usize) -> Self {
        VxGraphParameterQueue {
            ready_refs: Mutex::new(VecDeque::new()),
            consumed_refs: Mutex::new(VecDeque::new()),
            done_refs: Mutex::new(VecDeque::new()),
            done_cv: Condvar::new(),
            max_depth,
            enabled: AtomicBool::new(false),
            valid_refs: RwLock::new(Vec::new()),
        }
    }
}

/// Event data
pub struct VxEvent {
    pub event_type: VxEventType,
    pub timestamp_ns: u64,
    pub app_value: u32,
    pub graph_id: Option<u64>,
    pub node_id: Option<u64>,
    pub graph_parameter_index: Option<u32>,
    pub status: Option<i32>,
    pub user_parameter: Option<usize>,
}

/// Event registration entry
pub struct VxEventRegistration {
    pub ref_id: u64,
    pub event_type: VxEventType,
    pub graph_id: Option<u64>,
    pub graph_parameter_index: Option<u32>,
    pub node_id: Option<u64>,
    pub app_value: u32,
}

/// Per-context event system
pub struct VxEventSystem {
    /// Event queue
    pub events: Mutex<VecDeque<VxEvent>>,
    /// Condition variable for vxWaitEvent
    pub event_cv: Condvar,
    /// Event registrations
    pub registrations: Mutex<Vec<VxEventRegistration>>,
    /// Whether events are enabled
    pub enabled: AtomicBool,
}

impl VxEventSystem {
    pub fn new() -> Self {
        VxEventSystem {
            events: Mutex::new(VecDeque::new()),
            event_cv: Condvar::new(),
            registrations: Mutex::new(Vec::new()),
            enabled: AtomicBool::new(true),
        }
    }
}

/// Pipelining state added to VxCGraphData
pub struct VxGraphPipeliningState {
    /// Graph schedule mode
    pub schedule_mode: Mutex<VxGraphScheduleMode>,
    /// Per-parameter queues (index -> queue)
    pub parameter_queues: Mutex<HashMap<u32, Arc<VxGraphParameterQueue>>>,
    /// Number of graph parameters configured for queueing
    pub num_queue_params: AtomicU32,
    /// Streaming enabled flag
    pub streaming_enabled: AtomicBool,
    /// Trigger node for streaming (node_id)
    pub trigger_node_id: Mutex<Option<u64>>,
    /// Background thread handle for streaming
    pub streaming_thread: Mutex<Option<std::thread::JoinHandle<()>>>,
    /// Signal to stop streaming
    pub streaming_stop: AtomicBool,
    /// Number of pending graph executions
    pub pending_executions: Mutex<u32>,
    /// Condition for pending execution tracking
    pub pending_cv: Condvar,
    /// Background thread handle for QUEUE_AUTO executor
    pub executor_handle: Mutex<Option<std::thread::JoinHandle<()>>>,
    /// Signal to stop QUEUE_AUTO executor
    pub executor_stop: AtomicBool,
    /// Number of active graph executions (for vxWaitGraph in pipelining mode)
    pub active_executions: AtomicU32,
    /// Condition variable for waiting on active_executions to reach 0
    pub active_cv: Condvar,
    /// Mutex paired with active_cv
    pub active_mutex: Mutex<()>,
    /// Mutex to serialize graph execution in pipelining mode
    pub execution_mutex: Mutex<()>,
}

impl VxGraphPipeliningState {
    pub fn new() -> Self {
        VxGraphPipeliningState {
            schedule_mode: Mutex::new(VxGraphScheduleMode::Normal),
            parameter_queues: Mutex::new(HashMap::new()),
            num_queue_params: AtomicU32::new(0),
            streaming_enabled: AtomicBool::new(false),
            trigger_node_id: Mutex::new(None),
            streaming_thread: Mutex::new(None),
            streaming_stop: AtomicBool::new(false),
            pending_executions: Mutex::new(0),
            pending_cv: Condvar::new(),
            executor_handle: Mutex::new(None),
            executor_stop: AtomicBool::new(false),
            active_executions: AtomicU32::new(0),
            active_cv: Condvar::new(),
            active_mutex: Mutex::new(()),
            execution_mutex: Mutex::new(()),
        }
    }
}

// ============================================================================
// Type Aliases for C API
// ============================================================================

#[allow(non_camel_case_types)]
pub type vx_enum = i32;
#[allow(non_camel_case_types)]
pub type vx_uint32 = u32;
#[allow(non_camel_case_types)]
pub type vx_bool = u32;
#[allow(non_camel_case_types)]
pub type vx_graph = *mut std::ffi::c_void;
#[allow(non_camel_case_types)]
pub type vx_reference = *mut std::ffi::c_void;
#[allow(non_camel_case_types)]
pub type vx_node = *mut std::ffi::c_void;
#[allow(non_camel_case_types)]
pub type vx_context = *mut std::ffi::c_void;
#[allow(non_camel_case_types)]
pub type vx_status = i32;

// ============================================================================
// C Struct Definitions (matching vx_khr_pipelining.h)
// ============================================================================

/// Queueing parameters for a specific graph parameter
#[repr(C)]
pub struct vx_graph_parameter_queue_params_t {
    pub graph_parameter_index: u32,
    pub refs_list_size: u32,
    pub refs_list: *mut vx_reference,
}

/// Event info union members
#[repr(C)]
pub struct vx_event_graph_parameter_consumed {
    pub graph: vx_graph,
    pub graph_parameter_index: u32,
}

#[repr(C)]
pub struct vx_event_graph_completed {
    pub graph: vx_graph,
}

#[repr(C)]
pub struct vx_event_node_completed {
    pub graph: vx_graph,
    pub node: vx_node,
}

#[repr(C)]
pub struct vx_event_node_error {
    pub graph: vx_graph,
    pub node: vx_node,
    pub status: vx_status,
}

#[repr(C)]
pub struct vx_event_user_event {
    pub user_event_parameter: *mut std::ffi::c_void,
}

#[repr(C)]
pub union vx_event_info_t {
    pub graph_parameter_consumed: std::mem::ManuallyDrop<vx_event_graph_parameter_consumed>,
    pub graph_completed: std::mem::ManuallyDrop<vx_event_graph_completed>,
    pub node_completed: std::mem::ManuallyDrop<vx_event_node_completed>,
    pub node_error: std::mem::ManuallyDrop<vx_event_node_error>,
    pub user_event: std::mem::ManuallyDrop<vx_event_user_event>,
}

/// Event data structure
#[repr(C)]
pub struct vx_event_t {
    pub event_type: vx_enum,
    pub timestamp: u64,
    pub app_value: u32,
    pub event_info: vx_event_info_t,
}

// ============================================================================
// Constants
// ============================================================================

// Constants (matching Khronos spec values)
pub const VX_SUCCESS: vx_status = 0;
pub const VX_FAILURE: vx_status = -1;
pub const VX_ERROR_NOT_SUPPORTED: vx_status = -3;
pub const VX_ERROR_INVALID_PARAMETERS: vx_status = -10;
pub const VX_ERROR_INVALID_REFERENCE: vx_status = -12;
pub const VX_ERROR_INVALID_NODE: vx_status = -17;
pub const VX_ERROR_INVALID_GRAPH: vx_status = -18;
pub const VX_ERROR_GRAPH_SCHEDULED: vx_status = -19;

// Vendor-encoded pipelining enums
pub const VX_GRAPH_SCHEDULE_MODE_NORMAL: vx_enum = 0x21000; // 135168
pub const VX_GRAPH_SCHEDULE_MODE_QUEUE_AUTO: vx_enum = 0x21001; // 135169
pub const VX_GRAPH_SCHEDULE_MODE_QUEUE_MANUAL: vx_enum = 0x21002; // 135170

pub const VX_EVENT_GRAPH_PARAMETER_CONSUMED: vx_enum = 0x22000; // 139264
pub const VX_EVENT_GRAPH_COMPLETED: vx_enum = 0x22001; // 139265
pub const VX_EVENT_NODE_COMPLETED: vx_enum = 0x22002; // 139266
pub const VX_EVENT_NODE_ERROR: vx_enum = 0x22003; // 139267
pub const VX_EVENT_USER: vx_enum = 0x22004; // 139268

pub const VX_ATTRIBUTE_BASE_KHRONOS_GRAPH: vx_enum = 0x00080200;
pub const VX_GRAPH_SCHEDULE_MODE: vx_enum = VX_ATTRIBUTE_BASE_KHRONOS_GRAPH + 0x5;

pub const VX_TRUE_E: vx_bool = 1;
pub const VX_FALSE_E: vx_bool = 0;
