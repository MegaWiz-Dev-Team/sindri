//! Milestone 5 — GEMM on the Apple GPU via Metal.
//!
//! A tiled MSL kernel (16×16 threadgroup tiles, using `threadgroup` shared
//! memory) dispatched through the `metal` crate. Buffers use
//! `StorageModeShared` — on Apple Silicon CPU and GPU share one physical
//! memory pool, so there's **no host↔device copy**: we write into the buffer,
//! the GPU reads it, we read the result straight back.
//!
//! Build the pipeline once (`MetalGemm::new`), then call `.run()` per matmul.

use std::ffi::c_void;

use metal::{CommandQueue, ComputePipelineState, Device, MTLResourceOptions, MTLSize};

const TILE: u64 = 16;

const SHADER_SRC: &str = r#"
#include <metal_stdlib>
using namespace metal;

// C[M,N] = A[M,K] * B[K,N], row-major. Tiled with 16x16 threadgroup blocks.
kernel void gemm_tiled(
    device const float* A   [[buffer(0)]],
    device const float* B   [[buffer(1)]],
    device float*       C   [[buffer(2)]],
    constant uint*      dims[[buffer(3)]],   // [M, K, N]
    uint2 gid  [[thread_position_in_grid]],
    uint2 tid  [[thread_position_in_threadgroup]])
{
    const uint M = dims[0], K = dims[1], N = dims[2];
    const uint T = 16;

    threadgroup float As[16][16];
    threadgroup float Bs[16][16];

    uint row = gid.y;   // 0..M
    uint col = gid.x;   // 0..N
    float acc = 0.0;

    uint ntiles = (K + T - 1) / T;
    for (uint t = 0; t < ntiles; t++) {
        uint a_col = t * T + tid.x;
        uint b_row = t * T + tid.y;
        As[tid.y][tid.x] = (row < M && a_col < K) ? A[row * K + a_col] : 0.0;
        Bs[tid.y][tid.x] = (b_row < K && col < N) ? B[b_row * N + col] : 0.0;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint kk = 0; kk < T; kk++) {
            acc += As[tid.y][kk] * Bs[kk][tid.x];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (row < M && col < N) {
        C[row * N + col] = acc;
    }
}
"#;

pub struct MetalGemm {
    device: Device,
    queue: CommandQueue,
    pipeline: ComputePipelineState,
}

impl MetalGemm {
    pub fn new() -> Result<Self, String> {
        let device = Device::system_default().ok_or("no Metal device")?;
        let queue = device.new_command_queue();
        let lib = device.new_library_with_source(SHADER_SRC, &metal::CompileOptions::new())?;
        let func = lib.get_function("gemm_tiled", None)?;
        let pipeline = device.new_compute_pipeline_state_with_function(&func)?;
        Ok(Self { device, queue, pipeline })
    }

    /// GPU name, for the benchmark header.
    pub fn device_name(&self) -> String {
        self.device.name().to_string()
    }

    pub fn run(&self, a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
        let bytes = |v: &[f32]| (v.len() * std::mem::size_of::<f32>()) as u64;
        let shared = MTLResourceOptions::StorageModeShared;

        let buf_a = self.device.new_buffer_with_data(a.as_ptr() as *const c_void, bytes(a), shared);
        let buf_b = self.device.new_buffer_with_data(b.as_ptr() as *const c_void, bytes(b), shared);
        let buf_c = self.device.new_buffer((m * n * 4) as u64, shared);
        let dims: [u32; 3] = [m as u32, k as u32, n as u32];

        let cmd = self.queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&self.pipeline);
        enc.set_buffer(0, Some(&buf_a), 0);
        enc.set_buffer(1, Some(&buf_b), 0);
        enc.set_buffer(2, Some(&buf_c), 0);
        enc.set_bytes(3, 12, dims.as_ptr() as *const c_void);

        // One thread per output element; threadgroups of 16×16.
        let threads_per_group = MTLSize::new(TILE, TILE, 1);
        let groups = MTLSize::new(
            (n as u64).div_ceil(TILE),
            (m as u64).div_ceil(TILE),
            1,
        );
        enc.dispatch_thread_groups(groups, threads_per_group);
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();

        // Shared memory ⇒ read the result directly, no copy-back needed.
        let ptr = buf_c.contents() as *const f32;
        unsafe { std::slice::from_raw_parts(ptr, m * n).to_vec() }
    }
}
