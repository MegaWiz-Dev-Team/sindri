//! Milestone 6 — GEMM via MPS (Metal Performance Shaders), Apple's tuned GPU
//! matrix-multiply kernel.
//!
//! metal-rs (0.29) doesn't wrap `MPSMatrixMultiplication`, so we call it
//! directly via Objective-C message sends (metal-rs re-exports `objc` and
//! `foreign_types`, and we link the MetalPerformanceShaders framework).
//! Buffers are `StorageModeShared` — same unified-memory, no-copy story.
//!
//! This is the "use the tuned library" path: it should beat our hand-written
//! tiled kernel from milestone 5, and challenge the CPU `gemm` crate.

use std::ffi::c_void;

use metal::foreign_types::{ForeignType, ForeignTypeRef};
use metal::objc::rc::autoreleasepool;
use metal::objc::runtime::Object;
use metal::objc::{class, msg_send, sel, sel_impl};
use metal::{CommandQueue, Device, MTLResourceOptions};

// MPSDataType: float bit (0x10000000) | 32 bits.
const MPS_FLOAT32: usize = 0x1000_0000 | 32;

// Link Apple's MPS framework (this module is macOS-only via cfg in lib.rs).
#[link(name = "MetalPerformanceShaders", kind = "framework")]
unsafe extern "C" {}

pub struct MetalMps {
    device: Device,
    queue: CommandQueue,
}

impl MetalMps {
    pub fn new() -> Result<Self, String> {
        let device = Device::system_default().ok_or("no Metal device")?;
        let queue = device.new_command_queue();
        Ok(Self { device, queue })
    }

    pub fn device_name(&self) -> String {
        self.device.name().to_string()
    }

    pub fn run(&self, a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
        let shared = MTLResourceOptions::StorageModeShared;
        let buf_a =
            self.device.new_buffer_with_data(a.as_ptr() as *const c_void, (a.len() * 4) as u64, shared);
        let buf_b =
            self.device.new_buffer_with_data(b.as_ptr() as *const c_void, (b.len() * 4) as u64, shared);
        let buf_c = self.device.new_buffer((m * n * 4) as u64, shared);

        autoreleasepool(|| unsafe {
            let dev = self.device.as_ptr() as *mut Object;

            // Row-major descriptors: rowBytes = columns * sizeof(f32).
            let desc = |rows: usize, cols: usize| -> *mut Object {
                msg_send![class!(MPSMatrixDescriptor),
                    matrixDescriptorWithRows: rows
                    columns: cols
                    rowBytes: cols * 4
                    dataType: MPS_FLOAT32]
            };
            let mat = |buf: *mut Object, d: *mut Object| -> *mut Object {
                let m: *mut Object = msg_send![class!(MPSMatrix), alloc];
                msg_send![m, initWithBuffer: buf descriptor: d]
            };

            let mat_a = mat(buf_a.as_ptr() as *mut Object, desc(m, k));
            let mat_b = mat(buf_b.as_ptr() as *mut Object, desc(k, n));
            let mat_c = mat(buf_c.as_ptr() as *mut Object, desc(m, n));

            // C = 1.0 * A·B + 0.0 * C
            let kern: *mut Object = msg_send![class!(MPSMatrixMultiplication), alloc];
            let kern: *mut Object = msg_send![kern,
                initWithDevice: dev
                transposeLeft: false
                transposeRight: false
                resultRows: m
                resultColumns: n
                interiorColumns: k
                alpha: 1.0f64
                beta: 0.0f64];

            let cmd = self.queue.new_command_buffer();
            let cmd_ptr = cmd.as_ptr() as *mut Object;
            let _: () = msg_send![kern,
                encodeToCommandBuffer: cmd_ptr
                leftMatrix: mat_a
                rightMatrix: mat_b
                resultMatrix: mat_c];
            cmd.commit();
            cmd.wait_until_completed();

            // Release the +1 objects we alloc/init'd (descriptors are autoreleased).
            let _: () = msg_send![mat_a, release];
            let _: () = msg_send![mat_b, release];
            let _: () = msg_send![mat_c, release];
            let _: () = msg_send![kern, release];

            // Shared memory ⇒ read result directly.
            let ptr = buf_c.contents() as *const f32;
            std::slice::from_raw_parts(ptr, m * n).to_vec()
        })
    }
}
