//! Pipelining Executor - Dedicated execution path for pipelining mode
//!
//! Bypasses the standard graph state machine to allow concurrent executions.

use crate::pipelining::{VxGraphPipeliningState, VxGraphScheduleMode};
use crate::pipelining_api::{move_refs_to_done, notify_graph_completed};
use crate::pipelining_api::get_pipelining_state;
use crate::unified_c_api::GRAPHS_DATA;
use log::info;
use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::thread;

/// Start background executor for QUEUE_AUTO mode
pub fn start_queue_auto_executor(graph_id: u64) {
    let pipe_state = get_pipelining_state(graph_id);

    let mut handle = pipe_state.executor_handle.lock().unwrap();
    if handle.is_some() {
        return; // already running
    }

    let pipe_clone = pipe_state.clone();
    let h: std::thread::JoinHandle<()> = thread::spawn(move || {
        executor_loop(graph_id, pipe_clone);
    });

    *handle = Some(h);
    info!("Started QUEUE_AUTO executor for graph {}", graph_id);
}

/// Stop the executor
pub fn stop_queue_auto_executor(graph_id: u64) {
    let pipe_state = get_pipelining_state(graph_id);
    pipe_state.executor_stop.store(true, Ordering::SeqCst);

    let mut handle = pipe_state.executor_handle.lock().unwrap();
    if let Some(h) = handle.take() {
        drop(h.join());
    }
}

/// Executor main loop
fn executor_loop(graph_id: u64, pipe_state: Arc<VxGraphPipeliningState>) {
    let mut loop_count = 0;
    while !pipe_state.executor_stop.load(Ordering::SeqCst) {
        loop_count += 1;
        // Check mode
        {
            let mode = pipe_state.schedule_mode.lock().unwrap();
            if *mode != VxGraphScheduleMode::QueueAuto {
                if loop_count % 1000 == 0 {
                }
                thread::sleep(std::time::Duration::from_millis(5));
                continue;
            }
        }
        
        // Wait until ALL queue parameters have at least one ready ref
        let (all_ready, num_params, _ready_counts) = {
            let queues = pipe_state.parameter_queues.lock().unwrap();
            let n = queues.len();
            let mut ready = true;
            let mut counts = Vec::new();
            for (param_idx, q) in queues.iter() {
                let refs = q.ready_refs.lock().unwrap();
                let count = refs.len();
                counts.push((*param_idx, count));
                if count == 0 {
                    ready = false;
                }
            }
            (ready, n, counts)
        };
        
        if !all_ready || num_params == 0 {
            if loop_count % 1000 == 0 {
            }
            thread::sleep(std::time::Duration::from_millis(1));
            continue;
        }
        
        let status = execute_pipelined_graph(graph_id);
        
        if status != 0 {
            // Node execution failed; loop continues and tries next execution
            // from the queue. The executor keeps retrying.
        } else {
        }
    }
    
}

/// Execute a graph instance in pipelining mode.
/// Consumes refs from queues, executes nodes, moves refs to done.
/// This function is serialized per-graph via execution_mutex to prevent
/// concurrent execution of the same graph.
fn execute_pipelined_graph(graph_id: u64) -> i32 {
    // Clear any stale reference substitutions from previous executions
    crate::unified_c_api::clear_ref_substitutions();

    // Acquire per-graph execution lock to serialize executions
    let pipe_state = get_pipelining_state(graph_id);
    let _exec_lock = pipe_state.execution_mutex.lock().unwrap();

    // Increment active execution count BEFORE starting
    pipe_state.active_executions.fetch_add(1, std::sync::atomic::Ordering::SeqCst);

    // Ensure we always decrement and signal, even on error paths
    struct ActiveExecGuard {
        pipe_state: std::sync::Arc<crate::pipelining::VxGraphPipeliningState>,
    }
    impl Drop for ActiveExecGuard {
        fn drop(&mut self) {
            let prev = self.pipe_state.active_executions.fetch_sub(1, std::sync::atomic::Ordering::SeqCst);
            if prev == 1 {
                // Was the last one, signal waiters
                let _guard = self.pipe_state.active_mutex.lock().unwrap();
                self.pipe_state.active_cv.notify_all();
            }
        }
    }
    let _guard = ActiveExecGuard { pipe_state: pipe_state.clone() };

    // Get graph data (clone Arc to avoid holding GRAPHS_DATA lock during execution)
    let g = {
        let graphs = match GRAPHS_DATA.lock() {
            Ok(g) => g,
            Err(_) => return -1,
        };
        match graphs.get(&graph_id) {
            Some(g) => g.clone(),
            None => return -1,
        }
    };

    // Verify if needed
    {
        let verified = g.verified.lock().unwrap();
        if !*verified {
            drop(verified);
            let graph_ptr = graph_id as *mut std::ffi::c_void;
            let status = unsafe {
                extern "C" { fn vxVerifyGraph(graph: *mut std::ffi::c_void) -> i32; }
                vxVerifyGraph(graph_ptr)
            };
            if status != 0 {
                finish_pipelined_execution(graph_id, g.context_id);
                return status;
            }
        }
    }

    // Get topological waves for multicore execution
    let waves = match g.topo_waves.lock() {
        Ok(w) => w.clone(),
        Err(_) => {
            finish_pipelined_execution(graph_id, g.context_id);
            return -1;
        }
    };

    let nodes = match g.nodes.read() {
        Ok(n) => n.clone(),
        Err(_) => {
            finish_pipelined_execution(graph_id, g.context_id);
            return -1;
        }
    };

    if nodes.is_empty() {
        finish_pipelined_execution(graph_id, g.context_id);
        return 0;
    }

    let context_id = g.context_id;

    // Execute nodes wave by wave
    for wave in waves.iter() {
        if wave.is_empty() {
            continue;
        }

        if wave.len() == 1 {
            // Fast path: single node, execute on caller thread
            let node_id = wave[0];
            if node_id == 0 {
                finish_pipelined_execution(graph_id, context_id);
                return -1;
            }
            match crate::unified_c_api::execute_node(node_id) {
                Some(status) => {
                    if status == 0 {
                        crate::pipelining_api::notify_node_completed(graph_id, node_id, context_id);
                    } else {
                        finish_pipelined_execution(graph_id, context_id);
                        return status;
                    }
                }
                None => {
                    finish_pipelined_execution(graph_id, context_id);
                    return -1;
                }
            }
        } else {
            // Parallel path: use global thread pool for wave execution
            let (tx, rx) = std::sync::mpsc::channel::<(u64, Option<i32>)>();
            let pool = crate::thread_pool::get_global_pool();
            let remaining = wave.len();
            
            for &node_id in wave.iter() {
                if node_id == 0 {
                    finish_pipelined_execution(graph_id, context_id);
                    return -1;
                }
                let nid = node_id;
                let tx2 = tx.clone();
                if let Some(ref p) = pool {
                    p.execute(move || {
                        let result = crate::unified_c_api::execute_node(nid);
                        let _ = tx2.send((nid, result));
                    });
                } else {
                    // No pool available — fall back to sequential for safety
                    let result = crate::unified_c_api::execute_node(nid);
                    let _ = tx.send((nid, result));
                }
            }
            drop(tx); // close sender so recv knows when done

            // Wait for all nodes in wave and check for errors
            for _ in 0..remaining {
                match rx.recv() {
                    Ok((node_id, Some(status))) => {
                        if status == 0 {
                            crate::pipelining_api::notify_node_completed(graph_id, node_id, context_id);
                        } else {
                            finish_pipelined_execution(graph_id, context_id);
                            return status;
                        }
                    }
                    Ok((_, None)) => {
                        finish_pipelined_execution(graph_id, context_id);
                        return -1;
                    }
                    Err(_) => {
                        finish_pipelined_execution(graph_id, context_id);
                        return -1;
                    }
                }
            }
        }
    }

    // Fallback: if waves were empty but nodes exist (shouldn't happen), execute sequentially
    if waves.is_empty() {
        for node_id in nodes.iter() {
            if *node_id == 0 {
                finish_pipelined_execution(graph_id, context_id);
                return -1;
            }
            match crate::unified_c_api::execute_node(*node_id) {
                Some(status) => {
                    if status == 0 {
                        crate::pipelining_api::notify_node_completed(graph_id, *node_id, context_id);
                    } else {
                        finish_pipelined_execution(graph_id, context_id);
                        return status;
                    }
                }
                None => {
                    finish_pipelined_execution(graph_id, context_id);
                    return -1;
                }
            }
        }
    }

    // Success: move consumed refs to done, emit events
    finish_pipelined_execution(graph_id, context_id);

    // Increment run count
    g.run_count.fetch_add(1, std::sync::atomic::Ordering::SeqCst);

    0 // VX_SUCCESS
}

/// Clean up after pipelined execution: move consumed refs to done, emit events
fn finish_pipelined_execution(graph_id: u64, context_id: u64) {
    // Collect param indices first, then release GRAPH_PIPELINING lock
    // before calling move_refs_to_done (which also acquires it)
    let param_indices: Vec<u32> = {
        if let Ok(pipe_states) = crate::pipelining_api::GRAPH_PIPELINING.lock() {
            if let Some(pipe_state) = pipe_states.get(&graph_id) {
                let queues = pipe_state.parameter_queues.lock().unwrap();
                queues.keys().copied().collect()
            } else {
                vec![]
            }
        } else {
            vec![]
        }
    };

    // Now safe to call move_refs_to_done without holding GRAPH_PIPELINING
    for param_idx in param_indices {
        move_refs_to_done(graph_id, param_idx);
    }

    // Auto-age any registered delays
    crate::unified_c_api::auto_age_delays(graph_id);

    // Emit graph completion event
    notify_graph_completed(graph_id, context_id);
}
