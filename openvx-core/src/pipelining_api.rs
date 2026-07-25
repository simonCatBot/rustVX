//! OpenVX Pipelining Extension - C API Implementation
//!
//! This module implements the 12 functions from vx_khr_pipelining.h:
//! - Queueing API: vxSetGraphScheduleConfig, vxGraphParameterEnqueueReadyRef,
//!   vxGraphParameterDequeueDoneRef, vxGraphParameterCheckDoneRef
//! - Event API: vxWaitEvent, vxEnableEvents, vxDisableEvents,
//!   vxSendUserEvent, vxRegisterEvent
//! - Streaming API: vxEnableGraphStreaming, vxStartGraphStreaming,
//!   vxStopGraphStreaming

use crate::pipelining::*;
use crate::unified_c_api::{GRAPHS_DATA, VX_NODE_STATE_PIPEUP};
use log::{error, info, warn};
use std::ffi::c_void;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

// ============================================================================
// Global Event Systems (per context)
// ============================================================================

use once_cell::sync::Lazy;
use std::collections::HashMap as StdHashMap;

/// Event systems per context (context_id -> event system)
pub static EVENT_SYSTEMS: Lazy<Mutex<StdHashMap<u64, Arc<VxEventSystem>>>> =
    Lazy::new(|| Mutex::new(StdHashMap::new()));

/// Pipelining state per graph (graph_id -> state)
pub static GRAPH_PIPELINING: Lazy<Mutex<StdHashMap<u64, Arc<VxGraphPipeliningState>>>> =
    Lazy::new(|| Mutex::new(StdHashMap::new()));

/// Fast-path counter: number of graphs with non-Normal schedule mode.
/// Non-pipelining code paths check this with Relaxed ordering first;
/// if zero, the GRAPH_PIPELINING lock can be skipped entirely.
pub static ACTIVE_PIPELINING_GRAPHS: AtomicUsize = AtomicUsize::new(0);

/// Get or create event system for a context
fn get_event_system(context_id: u64) -> Arc<VxEventSystem> {
    let mut systems = EVENT_SYSTEMS.lock().unwrap();
    systems
        .entry(context_id)
        .or_insert_with(|| Arc::new(VxEventSystem::new()))
        .clone()
}

/// Get or create pipelining state for a graph
pub(crate) fn get_pipelining_state(graph_id: u64) -> Arc<VxGraphPipeliningState> {
    let mut states = GRAPH_PIPELINING.lock().unwrap();
    states
        .entry(graph_id)
        .or_insert_with(|| Arc::new(VxGraphPipeliningState::new()))
        .clone()
}

/// Fast-path helper: return true if any graph is in pipelining mode.
/// Uses Relaxed ordering — the caller still validates with a proper lock.
pub fn any_pipelining_active() -> bool {
    ACTIVE_PIPELINING_GRAPHS.load(Ordering::Relaxed) != 0
}

/// Get current timestamp in nanoseconds
fn current_timestamp_ns() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or(Duration::from_secs(0))
        .as_nanos() as u64
}

/// Push an event to the context's event queue
fn push_event(context_id: u64, event: VxEvent) {
    let event_system = get_event_system(context_id);
    let mut events = event_system.events.lock().unwrap();
    events.push_back(event);
    event_system.event_cv.notify_one();
}

// ============================================================================
// Queueing API
// ============================================================================

/// Sets the graph scheduler config
///
/// This API is used to set the graph scheduler config to allow user to
/// schedule multiple instances of a graph for execution.
#[no_mangle]
pub extern "C" fn vxSetGraphScheduleConfig(
    graph: vx_graph,
    graph_schedule_mode: vx_enum,
    graph_parameters_list_size: vx_uint32,
    graph_parameters_queue_params_list: *const vx_graph_parameter_queue_params_t,
) -> vx_status {
    if graph.is_null() {
        error!("vxSetGraphScheduleConfig: graph is NULL");
        return VX_ERROR_INVALID_REFERENCE;
    }

    let graph_id = graph as u64;

    // Validate graph exists
    {
        let graphs = match GRAPHS_DATA.lock() {
            Ok(g) => g,
            Err(_) => return VX_FAILURE,
        };
        if !graphs.contains_key(&graph_id) {
            error!("vxSetGraphScheduleConfig: graph not found");
            return VX_ERROR_INVALID_REFERENCE;
        }
    }

    // Parse schedule mode
    let schedule_mode = match VxGraphScheduleMode::try_from(graph_schedule_mode) {
        Ok(mode) => mode,
        Err(_) => {
            error!("vxSetGraphScheduleConfig: invalid schedule mode {}", graph_schedule_mode);
            return VX_ERROR_INVALID_PARAMETERS;
        }
    };

    // Normal mode: no queue params allowed
    if schedule_mode == VxGraphScheduleMode::Normal {
        if graph_parameters_list_size != 0 || !graph_parameters_queue_params_list.is_null() {
            error!("vxSetGraphScheduleConfig: NORMAL mode requires 0 params");
            return VX_ERROR_INVALID_PARAMETERS;
        }
    }

    // Get pipelining state
    let pipe_state = get_pipelining_state(graph_id);

    // Set schedule mode
    {
        let mut mode = pipe_state.schedule_mode.lock().unwrap();
        let old_mode = *mode;
        *mode = schedule_mode;
        // Update fast-path counter when mode transitions into/out of Normal
        if old_mode == VxGraphScheduleMode::Normal && schedule_mode != VxGraphScheduleMode::Normal {
            ACTIVE_PIPELINING_GRAPHS.fetch_add(1, Ordering::Relaxed);
        } else if old_mode != VxGraphScheduleMode::Normal && schedule_mode == VxGraphScheduleMode::Normal {
            ACTIVE_PIPELINING_GRAPHS.fetch_sub(1, Ordering::Relaxed);
        }
    }

    // Clear existing queues
    {
        let mut queues = pipe_state.parameter_queues.lock().unwrap();
        queues.clear();
    }

    // Configure parameter queues
    if graph_parameters_list_size > 0 && !graph_parameters_queue_params_list.is_null() {
        let params_slice = unsafe {
            std::slice::from_raw_parts(
                graph_parameters_queue_params_list,
                graph_parameters_list_size as usize,
            )
        };

        let mut queues = pipe_state.parameter_queues.lock().unwrap();
        for param in params_slice {
            let index = param.graph_parameter_index;
            let list_size = param.refs_list_size as usize;

            // Validate refs_list_size
            if list_size == 0 {
                error!("vxSetGraphScheduleConfig: refs_list_size must be > 0");
                return VX_ERROR_INVALID_PARAMETERS;
            }

            let queue = Arc::new(VxGraphParameterQueue::new(list_size));
            queue.enabled.store(true, Ordering::SeqCst);

            // If refs_list is provided, store valid references
            if !param.refs_list.is_null() {
                let refs_slice = unsafe {
                    std::slice::from_raw_parts(param.refs_list, list_size)
                };
                let mut valid_refs = queue.valid_refs.write().unwrap();
                for &ref_ptr in refs_slice {
                    valid_refs.push(ref_ptr as usize);
                }
            }

            queues.insert(index, queue);
        }

        pipe_state
            .num_queue_params
            .store(queues.len() as u32, Ordering::SeqCst);
    }

    info!(
        "vxSetGraphScheduleConfig: graph {} set to mode {:?}",
        graph_id, schedule_mode
    );

    // Start background executor for QUEUE_AUTO mode
    if schedule_mode == VxGraphScheduleMode::QueueAuto {
        crate::pipelining_executor::start_queue_auto_executor(graph_id);
    }

    VX_SUCCESS
}

/// Enqueues new references into a graph parameter for processing
#[no_mangle]
pub extern "C" fn vxGraphParameterEnqueueReadyRef(
    graph: vx_graph,
    graph_parameter_index: vx_uint32,
    refs: *mut vx_reference,
    num_refs: vx_uint32,
) -> vx_status {
    if graph.is_null() {
        return VX_ERROR_INVALID_REFERENCE;
    }
    if refs.is_null() && num_refs > 0 {
        return VX_ERROR_INVALID_PARAMETERS;
    }

    let graph_id = graph as u64;
    let pipe_state = get_pipelining_state(graph_id);

    // Check if queueing is enabled for this parameter (auto-create if missing)
    let queue = {
        let mut queues = pipe_state.parameter_queues.lock().unwrap();
        match queues.get(&graph_parameter_index) {
            Some(q) => q.clone(),
            None => {
                // Auto-create queue for this parameter (lazy initialization)
                let q = Arc::new(VxGraphParameterQueue::new(10));
                q.enabled.store(true, Ordering::SeqCst);
                queues.insert(graph_parameter_index, q.clone());
                pipe_state
                    .num_queue_params
                    .store(queues.len() as u32, Ordering::SeqCst);
                info!("vxGraphParameterEnqueueReadyRef: auto-created queue for param {}", graph_parameter_index);
                q
            }
        }
    };

    if !queue.enabled.load(Ordering::SeqCst) {
        return VX_ERROR_INVALID_PARAMETERS;
    }

    // Validate refs against valid_refs list if provided
    let valid_refs = queue.valid_refs.read().unwrap();
    if !valid_refs.is_empty() && num_refs > 0 {
        let refs_slice = unsafe { std::slice::from_raw_parts(refs, num_refs as usize) };
        for &ref_ptr in refs_slice {
            let ref_addr = ref_ptr as usize;
            if !valid_refs.contains(&ref_addr) {
                warn!(
                    "vxGraphParameterEnqueueReadyRef: ref {:p} not in valid_refs list",
                    ref_ptr
                );
            }
        }
    }
    drop(valid_refs);

    // Enqueue references
    let mut ready_refs = queue.ready_refs.lock().unwrap();
    if num_refs > 0 {
        let refs_slice = unsafe { std::slice::from_raw_parts(refs, num_refs as usize) };
        for &ref_ptr in refs_slice {
            ready_refs.push_back(ref_ptr as usize);
        }
    }

    let enqueued_count = ready_refs.len();
    drop(ready_refs);
    info!(
        "vxGraphParameterEnqueueReadyRef: enqueued {} refs at param {} (total ready: {})",
        num_refs,
        graph_parameter_index,
        enqueued_count
    );

    // For QUEUE_AUTO mode, the background executor polls queues continuously.
    // No explicit scheduling needed here.

    VX_SUCCESS
}

/// Dequeues 'consumed' references from a graph parameter
/// This API will block until at least one reference is dequeued.
#[no_mangle]
pub extern "C" fn vxGraphParameterDequeueDoneRef(
    graph: vx_graph,
    graph_parameter_index: vx_uint32,
    refs: *mut vx_reference,
    max_refs: vx_uint32,
    num_refs: *mut vx_uint32,
) -> vx_status {
    if graph.is_null() {
        return VX_ERROR_INVALID_REFERENCE;
    }
    if refs.is_null() || num_refs.is_null() {
        return VX_ERROR_INVALID_PARAMETERS;
    }

    let graph_id = graph as u64;
    let pipe_state = get_pipelining_state(graph_id);

    // Get queue for this parameter (auto-create if missing)
    let queue = {
        let mut queues = pipe_state.parameter_queues.lock().unwrap();
        match queues.get(&graph_parameter_index) {
            Some(q) => q.clone(),
            None => {
                // Auto-create queue for this parameter
                let q = Arc::new(VxGraphParameterQueue::new(10));
                q.enabled.store(true, Ordering::SeqCst);
                queues.insert(graph_parameter_index, q.clone());
                pipe_state
                    .num_queue_params
                    .store(queues.len() as u32, Ordering::SeqCst);
                info!("vxGraphParameterDequeueDoneRef: auto-created queue for param {}", graph_parameter_index);
                q
            }
        }
    };

    let mut done_refs = queue.done_refs.lock().unwrap();

    // Block until at least one reference is available
    while done_refs.is_empty() {
        done_refs = queue.done_cv.wait(done_refs).unwrap();
    }

    // Dequeue up to max_refs
    let mut count: u32 = 0;
    let refs_slice = unsafe { std::slice::from_raw_parts_mut(refs, max_refs as usize) };
    while count < max_refs && !done_refs.is_empty() {
        if let Some(ref_addr) = done_refs.pop_front() {
            refs_slice[count as usize] = ref_addr as vx_reference;
            count += 1;
        }
    }

    unsafe {
        *num_refs = count;
    }

    VX_SUCCESS
}

/// Checks and returns the number of references that are ready for dequeue
#[no_mangle]
pub extern "C" fn vxGraphParameterCheckDoneRef(
    graph: vx_graph,
    graph_parameter_index: vx_uint32,
    num_refs: *mut vx_uint32,
) -> vx_status {
    if graph.is_null() {
        return VX_ERROR_INVALID_REFERENCE;
    }
    if num_refs.is_null() {
        return VX_ERROR_INVALID_PARAMETERS;
    }

    let graph_id = graph as u64;
    let pipe_state = get_pipelining_state(graph_id);

    let queue = {
        let mut queues = pipe_state.parameter_queues.lock().unwrap();
        match queues.get(&graph_parameter_index) {
            Some(q) => q.clone(),
            None => {
                // Auto-create queue for this parameter
                let q = Arc::new(VxGraphParameterQueue::new(10));
                q.enabled.store(true, Ordering::SeqCst);
                queues.insert(graph_parameter_index, q.clone());
                pipe_state
                    .num_queue_params
                    .store(queues.len() as u32, Ordering::SeqCst);
                info!("vxGraphParameterCheckDoneRef: auto-created queue for param {}", graph_parameter_index);
                q
            }
        }
    };

    let done_refs = queue.done_refs.lock().unwrap();
    unsafe {
        *num_refs = done_refs.len() as u32;
    }

    VX_SUCCESS
}

// ============================================================================
// Event API
// ============================================================================

/// Enable event generation
#[no_mangle]
pub extern "C" fn vxEnableEvents(context: vx_context) -> vx_status {
    if context.is_null() {
        return VX_ERROR_INVALID_REFERENCE;
    }

    let context_id = context as u64;
    let event_system = get_event_system(context_id);
    event_system.enabled.store(true, Ordering::SeqCst);

    info!("vxEnableEvents: context {}", context_id);
    VX_SUCCESS
}

/// Disable event generation
#[no_mangle]
pub extern "C" fn vxDisableEvents(context: vx_context) -> vx_status {
    if context.is_null() {
        return VX_ERROR_INVALID_REFERENCE;
    }

    let context_id = context as u64;
    let event_system = get_event_system(context_id);
    event_system.enabled.store(false, Ordering::SeqCst);

    info!("vxDisableEvents: context {}", context_id);
    VX_SUCCESS
}

/// Wait for a single event
#[no_mangle]
pub extern "C" fn vxWaitEvent(
    context: vx_context,
    event: *mut vx_event_t,
    do_not_block: vx_bool,
) -> vx_status {
    if context.is_null() || event.is_null() {
        return VX_ERROR_INVALID_REFERENCE;
    }

    let context_id = context as u64;
    let event_system = get_event_system(context_id);

    let mut events = event_system.events.lock().unwrap();

    if do_not_block == VX_TRUE_E {
        // Non-blocking: check if event available
        if let Some(evt) = events.pop_front() {
            drop(events); // Release lock before conversion
            return convert_event_to_c(evt, event);
        }
        return VX_FAILURE;
    }

    // Blocking wait
    loop {
        if let Some(evt) = events.pop_front() {
            drop(events);
            return convert_event_to_c(evt, event);
        }

        // Wait for new events
        events = event_system.event_cv.wait(events).unwrap();
    }
}

/// Convert internal event to C structure
fn convert_event_to_c(evt: VxEvent, c_event: *mut vx_event_t) -> vx_status {
    unsafe {
        (*c_event).event_type = match evt.event_type {
            VxEventType::GraphParameterConsumed => VX_EVENT_GRAPH_PARAMETER_CONSUMED,
            VxEventType::GraphCompleted => VX_EVENT_GRAPH_COMPLETED,
            VxEventType::NodeCompleted => VX_EVENT_NODE_COMPLETED,
            VxEventType::NodeError => VX_EVENT_NODE_ERROR,
            VxEventType::User => VX_EVENT_USER,
        };
        (*c_event).timestamp = evt.timestamp_ns;
        (*c_event).app_value = evt.app_value;

        // Fill event info based on type
        match evt.event_type {
            VxEventType::GraphParameterConsumed => {
                if let (Some(gid), Some(pidx)) = (evt.graph_id, evt.graph_parameter_index) {
                    (*c_event).event_info.graph_parameter_consumed =
                        std::mem::ManuallyDrop::new(vx_event_graph_parameter_consumed {
                            graph: gid as vx_graph,
                            graph_parameter_index: pidx,
                        });
                }
            }
            VxEventType::GraphCompleted => {
                if let Some(gid) = evt.graph_id {
                    (*c_event).event_info.graph_completed =
                        std::mem::ManuallyDrop::new(vx_event_graph_completed {
                            graph: gid as vx_graph,
                        });
                }
            }
            VxEventType::NodeCompleted => {
                if let (Some(gid), Some(nid)) = (evt.graph_id, evt.node_id) {
                    (*c_event).event_info.node_completed =
                        std::mem::ManuallyDrop::new(vx_event_node_completed {
                            graph: gid as vx_graph,
                            node: nid as vx_node,
                        });
                }
            }
            VxEventType::NodeError => {
                if let (Some(gid), Some(nid), Some(st)) =
                    (evt.graph_id, evt.node_id, evt.status)
                {
                    (*c_event).event_info.node_error =
                        std::mem::ManuallyDrop::new(vx_event_node_error {
                            graph: gid as vx_graph,
                            node: nid as vx_node,
                            status: st,
                        });
                }
            }
            VxEventType::User => {
                if let Some(param) = evt.user_parameter {
                    (*c_event).event_info.user_event =
                        std::mem::ManuallyDrop::new(vx_event_user_event {
                            user_event_parameter: param as *mut c_void,
                        });
                }
            }
        }
    }

    VX_SUCCESS
}

/// Generate user defined event
#[no_mangle]
pub extern "C" fn vxSendUserEvent(
    context: vx_context,
    app_value: u32,
    parameter: *mut c_void,
) -> vx_status {
    if context.is_null() {
        return VX_ERROR_INVALID_REFERENCE;
    }

    let context_id = context as u64;
    let event_system = get_event_system(context_id);

    // Check if events are enabled
    if !event_system.enabled.load(Ordering::SeqCst) {
        return VX_FAILURE;
    }

    let event = VxEvent {
        event_type: VxEventType::User,
        timestamp_ns: current_timestamp_ns(),
        app_value,
        graph_id: None,
        node_id: None,
        graph_parameter_index: None,
        status: None,
        user_parameter: Some(parameter as usize),
    };

    push_event(context_id, event);

    info!("vxSendUserEvent: context {}, app_value {}", context_id, app_value);
    VX_SUCCESS
}

/// Register an event to be generated
#[no_mangle]
pub extern "C" fn vxRegisterEvent(
    ref_: vx_reference,
    event_type: vx_enum,
    param: u32,
    app_value: u32,
) -> vx_status {
    if ref_.is_null() {
        return VX_ERROR_INVALID_REFERENCE;
    }

    let ref_id = ref_ as u64;

    // Resolve the actual context from the reference instead of hardcoding 1
    let mut context_id = crate::c_api::vxGetContext(ref_ as *mut _) as u64;
    if context_id == 0 {
        // Fallback: if ref is a graph, look it up in GRAPHS_DATA
        let mut fallback_ctx: Option<u64> = None;
        if let Ok(graphs) = GRAPHS_DATA.lock() {
            if let Some(g) = graphs.get(&ref_id) {
                fallback_ctx = Some(g.context_id);
            }
        }
        if fallback_ctx.is_none() {
            if let Ok(nodes_map) = crate::c_api::NODES.lock() {
                if let Some(node) = nodes_map.get(&ref_id) {
                    fallback_ctx = Some(node.context_id as u64);
                }
            }
        }
        if let Some(ctx) = fallback_ctx {
            context_id = ctx;
        } else {
            return VX_ERROR_INVALID_REFERENCE;
        }
    }

    let event_system = get_event_system(context_id);

    let evt_type = match VxEventType::try_from(event_type) {
        Ok(t) => t,
        Err(_) => return VX_ERROR_NOT_SUPPORTED,
    };

    // Determine whether this is a graph or node reference by checking
    // if ref_id exists in GRAPHS_DATA or NODES.
    let (graph_id, node_id, graph_param_idx) = {
        let mut gid: Option<u64> = None;
        let mut nid: Option<u64> = None;
        {
            if let Ok(graphs) = GRAPHS_DATA.lock() {
                if graphs.contains_key(&ref_id) {
                    gid = Some(ref_id);
                }
            }
            if gid.is_none() {
                if let Ok(nodes_map) = crate::c_api::NODES.lock() {
                    if nodes_map.contains_key(&ref_id) {
                        nid = Some(ref_id);
                        if let Some(node) = nodes_map.get(&ref_id) {
                            gid = Some(node.graph_id);
                        }
                    }
                }
            }
        }
        let gparam = if evt_type == VxEventType::GraphParameterConsumed && param < 0xFFFFFFFF {
            Some(param)
        } else {
            None
        };
        (gid, nid, gparam)
    };

    let registration = VxEventRegistration {
        ref_id,
        event_type: evt_type,
        graph_id,
        graph_parameter_index: graph_param_idx,
        node_id,
        app_value,
    };

    let mut registrations = event_system.registrations.lock().unwrap();
    registrations.push(registration);

    info!(
        "vxRegisterEvent: ref {} type {:?} graph_id {:?} node_id {:?} graph_param_idx {:?} app_value {}",
        ref_id, evt_type, graph_id, node_id, graph_param_idx, app_value
    );

    VX_SUCCESS
}

// ============================================================================
// Streaming API
// ============================================================================

/// Enable streaming mode of graph execution
#[no_mangle]
pub extern "C" fn vxEnableGraphStreaming(
    graph: vx_graph,
    trigger_node: vx_node,
) -> vx_status {
    if graph.is_null() {
        return VX_ERROR_INVALID_REFERENCE;
    }

    let graph_id = graph as u64;
    let pipe_state = get_pipelining_state(graph_id);

    // Validate graph exists
    {
        let graphs = match GRAPHS_DATA.lock() {
            Ok(g) => g,
            Err(_) => return VX_FAILURE,
        };
        if !graphs.contains_key(&graph_id) {
            return VX_ERROR_INVALID_REFERENCE;
        }
    }

    pipe_state.streaming_enabled.store(true, Ordering::SeqCst);

    // Store trigger node
    if !trigger_node.is_null() {
        let trigger_node_id = trigger_node as u64;
        let mut tid = pipe_state.trigger_node_id.lock().unwrap();
        *tid = Some(trigger_node_id);
    }

    info!(
        "vxEnableGraphStreaming: graph {} trigger_node {:p}",
        graph_id, trigger_node
    );

    VX_SUCCESS
}

/// Start streaming mode of graph execution
#[no_mangle]
pub extern "C" fn vxStartGraphStreaming(graph: vx_graph) -> vx_status {
    if graph.is_null() {
        return VX_ERROR_INVALID_REFERENCE;
    }

    let graph_id = graph as u64;
    let pipe_state = get_pipelining_state(graph_id);

    // Check graph is verified
    {
        let graphs = match GRAPHS_DATA.lock() {
            Ok(g) => g,
            Err(_) => return VX_FAILURE,
        };
        if let Some(g) = graphs.get(&graph_id) {
            if !*g.verified.lock().unwrap() {
                error!("vxStartGraphStreaming: graph not verified");
                return VX_FAILURE;
            }
        } else {
            return VX_ERROR_INVALID_REFERENCE;
        }
    }

    // Check streaming is enabled
    if !pipe_state.streaming_enabled.load(Ordering::SeqCst) {
        error!("vxStartGraphStreaming: streaming not enabled");
        return VX_FAILURE;
    }

    {
        let thread = pipe_state.streaming_thread.lock().unwrap();
        if thread.is_some() {
            error!("vxStartGraphStreaming: already streaming");
            return VX_FAILURE;
        }
    }

    // Reset per-node streaming execution counters and state so that kernels
    // with a pipeup depth observe the pipeup-to-steady transition during this
    // streaming session.
    {
        let graphs = match GRAPHS_DATA.lock() {
            Ok(g) => g,
            Err(_) => return VX_FAILURE,
        };
        if let Some(g) = graphs.get(&graph_id) {
            let nodes = match g.nodes.read() {
                Ok(n) => n.clone(),
                Err(_) => return VX_FAILURE,
            };
            drop(graphs);
            if let Ok(nodes_map) = crate::c_api::NODES.lock() {
                for node_id in nodes.iter() {
                    if let Some(node_data) = nodes_map.get(node_id) {
                        node_data.execution_count.store(0, Ordering::SeqCst);
                        node_data
                            .node_state
                            .store(VX_NODE_STATE_PIPEUP as u32, Ordering::SeqCst);
                    }
                }
            }
        }
    }

    // Reset stop signal
    pipe_state.streaming_stop.store(false, Ordering::SeqCst);

    // Spawn streaming thread
    let graph_ptr = graph as usize;
    let pipe_state_clone = pipe_state.clone();

    let handle = std::thread::spawn(move || {
        streaming_loop(graph_ptr as vx_graph, pipe_state_clone);
    });

    {
        let mut thread = pipe_state.streaming_thread.lock().unwrap();
        *thread = Some(handle);
    }

    info!("vxStartGraphStreaming: graph {} streaming started", graph_id);
    VX_SUCCESS
}

/// Streaming loop - continuously executes the graph until stopped.
fn streaming_loop(graph: vx_graph, pipe_state: Arc<VxGraphPipeliningState>) {
    let graph_id = graph as u64;

    info!("streaming_loop: graph {} started", graph_id);

    while !pipe_state.streaming_stop.load(Ordering::SeqCst) {
        // Mark the graph as running for this iteration. execute_graph_nodes
        // will set it back to COMPLETED on success or ABANDONED on error.
        {
            if let Ok(graphs) = GRAPHS_DATA.lock() {
                if let Some(g) = graphs.get(&graph_id) {
                    if let Ok(mut state) = g.state.lock() {
                        *state = crate::unified_c_api::VxGraphState::VxGraphStateRunning;
                    }
                }
            }
        }

        // Execute one graph iteration synchronously. This reuses the same
        // node-execution path as vxProcessGraph, including queueing support
        // and event emission.
        let graph_ptr = graph as crate::unified_c_api::vx_graph;
        let status = crate::unified_c_api::execute_graph_nodes(graph_ptr);

        if status != VX_SUCCESS {
            error!(
                "streaming_loop: graph {} execution failed with status {}, stopping",
                graph_id, status
            );
            break;
        }

        // Briefly yield so a stop request can be observed promptly without
        // fully busy-waiting the CPU.
        std::thread::sleep(Duration::from_micros(100));
    }

    // Drain any pending execution tracking that the stub implementation used.
    {
        let mut pending = pipe_state.pending_executions.lock().unwrap();
        *pending = 0;
        pipe_state.pending_cv.notify_all();
    }

    info!("streaming_loop: graph {} stopped", graph_id);
}

/// Stop streaming mode of graph execution
#[no_mangle]
pub extern "C" fn vxStopGraphStreaming(graph: vx_graph) -> vx_status {
    if graph.is_null() {
        return VX_ERROR_INVALID_REFERENCE;
    }

    let graph_id = graph as u64;
    let pipe_state = get_pipelining_state(graph_id);

    // Check if streaming was started
    {
        let thread = pipe_state.streaming_thread.lock().unwrap();
        if thread.is_none() {
            error!("vxStopGraphStreaming: graph not in streaming mode");
            return VX_FAILURE;
        }
    }

    // Signal stop
    pipe_state.streaming_stop.store(true, Ordering::SeqCst);

    // Wait for thread to finish
    {
        let mut thread = pipe_state.streaming_thread.lock().unwrap();
        if let Some(handle) = thread.take() {
            drop(thread); // Release lock before join
            let _ = handle.join();
        }
    }

    // Wait for all pending executions
    {
        let mut pending = pipe_state.pending_executions.lock().unwrap();
        while *pending > 0 {
            pending = pipe_state.pending_cv.wait(pending).unwrap();
        }
    }

    // Reset streaming state
    pipe_state.streaming_enabled.store(false, Ordering::SeqCst);
    {
        let mut tid = pipe_state.trigger_node_id.lock().unwrap();
        *tid = None;
    }

    info!("vxStopGraphStreaming: graph {} stopped gracefully", graph_id);
    VX_SUCCESS
}

// ============================================================================
// Integration Helpers
// ============================================================================

/// Called by the graph executor when a node completes to emit events
pub fn notify_node_completed(graph_id: u64, node_id: u64, context_id: u64) {
    let event_system = get_event_system(context_id);
    if !event_system.enabled.load(Ordering::SeqCst) {
        return;
    }

    // Look up app_value from registrations for NODE_COMPLETED events on this node
    let registrations = event_system.registrations.lock().unwrap();
    let app_value = registrations.iter().find_map(|reg| {
        if reg.event_type == VxEventType::NodeCompleted && reg.node_id == Some(node_id) {
            Some(reg.app_value)
        } else {
            None
        }
    });
    drop(registrations);

    // Only emit the event if someone registered for it
    if app_value.is_none() {
        return;
    }

    let event = VxEvent {
        event_type: VxEventType::NodeCompleted,
        timestamp_ns: current_timestamp_ns(),
        app_value: app_value.unwrap(),
        graph_id: Some(graph_id),
        node_id: Some(node_id),
        graph_parameter_index: None,
        status: None,
        user_parameter: None,
    };

    push_event(context_id, event);
}

/// Called by the graph executor when a node errors
pub fn notify_node_error(graph_id: u64, node_id: u64, status: i32, context_id: u64) {
    let event_system = get_event_system(context_id);
    if !event_system.enabled.load(Ordering::SeqCst) {
        return;
    }

    let event = VxEvent {
        event_type: VxEventType::NodeError,
        timestamp_ns: current_timestamp_ns(),
        app_value: 0,
        graph_id: Some(graph_id),
        node_id: Some(node_id),
        graph_parameter_index: None,
        status: Some(status),
        user_parameter: None,
    };

    push_event(context_id, event);
}

/// Called by the graph executor when graph completes
pub fn notify_graph_completed(graph_id: u64, context_id: u64) {
    let event_system = get_event_system(context_id);
    if !event_system.enabled.load(Ordering::SeqCst) {
        return;
    }

    let event = VxEvent {
        event_type: VxEventType::GraphCompleted,
        timestamp_ns: current_timestamp_ns(),
        app_value: 0,
        graph_id: Some(graph_id),
        node_id: None,
        graph_parameter_index: None,
        status: None,
        user_parameter: None,
    };

    push_event(context_id, event);
}

/// Look up the app_value for a graph parameter from event registrations.
/// Returns Some(app_value) if a GRAPH_PARAMETER_CONSUMED event is registered
/// for this graph/parameter, or None if no matching registration.
fn lookup_app_value_for_param(context_id: u64, graph_id: u64, param_index: u32) -> Option<u32> {
    let event_system = get_event_system(context_id);
    let registrations = event_system.registrations.lock().unwrap();
    for reg in registrations.iter() {
        if reg.event_type == VxEventType::GraphParameterConsumed
            && reg.graph_id == Some(graph_id)
            && reg.graph_parameter_index == Some(param_index)
        {
            return Some(reg.app_value);
        }
    }
    None
}

/// Called when a graph parameter is consumed to emit event
pub fn notify_parameter_consumed(
    graph_id: u64,
    param_index: u32,
    app_value: u32,
    context_id: u64,
) {
    let event_system = get_event_system(context_id);
    if !event_system.enabled.load(Ordering::SeqCst) {
        return;
    }

    let event = VxEvent {
        event_type: VxEventType::GraphParameterConsumed,
        timestamp_ns: current_timestamp_ns(),
        app_value,
        graph_id: Some(graph_id),
        node_id: None,
        graph_parameter_index: Some(param_index),
        status: None,
        user_parameter: None,
    };

    push_event(context_id, event);
}

/// Move references from consumed queue to done queue after execution
pub fn move_refs_to_done(graph_id: u64, param_index: u32) {
    let pipe_state = get_pipelining_state(graph_id);

    let queue = {
        let queues = pipe_state.parameter_queues.lock().unwrap();
        match queues.get(&param_index) {
            Some(q) => q.clone(),
            None => return,
        }
    };

    let mut consumed_refs = queue.consumed_refs.lock().unwrap();
    let mut done_refs = queue.done_refs.lock().unwrap();

    // Move all consumed refs to done
    while let Some(ref_addr) = consumed_refs.pop_front() {
        done_refs.push_back(ref_addr);
    }

    // Notify waiters
    queue.done_cv.notify_one();

    // Emit parameter-consumed event (look up app_value from registrations)
    {
        let context_id = {
            if let Ok(graphs) = GRAPHS_DATA.lock() {
                if let Some(g) = graphs.get(&graph_id) {
                    g.context_id
                } else {
                    return;
                }
            } else {
                return;
            }
        };
        if let Some(app_value) = lookup_app_value_for_param(context_id, graph_id, param_index) {
            notify_parameter_consumed(graph_id, param_index, app_value, context_id);
        }
    }
}
