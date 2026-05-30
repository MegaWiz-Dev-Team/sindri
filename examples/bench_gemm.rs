//! Milestone 4 benchmark: naive triple-loop GEMM vs the `gemm` crate (SIMD +
//! multithread) on Apple Silicon. Run with:
//!
//!   cargo run --release --example bench_gemm
//!
//! (must be --release — naive is unbearably slow in debug, and SIMD codegen
//! only kicks in with optimizations on)

use std::time::Instant;
use sindri::ops::{matmul_fast, matmul_naive};

/// Tiny deterministic xorshift PRNG — no deps, reproducible.
fn rand_vec(n: usize, seed: &mut u64) -> Vec<f32> {
    (0..n)
        .map(|_| {
            *seed ^= *seed << 13;
            *seed ^= *seed >> 7;
            *seed ^= *seed << 17;
            ((*seed >> 40) as f32) / ((1u64 << 24) as f32) - 0.5
        })
        .collect()
}

fn best_of<F: FnMut() -> Vec<f32>>(reps: u32, mut f: F) -> (std::time::Duration, Vec<f32>) {
    let mut best = std::time::Duration::MAX;
    let mut out = Vec::new();
    for _ in 0..reps {
        let t = Instant::now();
        out = f();
        let e = t.elapsed();
        if e < best {
            best = e;
        }
    }
    (best, out)
}

#[cfg(target_os = "macos")]
fn bench(
    gpu: &sindri::metal_gemm::MetalGemm,
    mps: &sindri::metal_mps::MetalMps,
    m: usize,
    k: usize,
    n: usize,
) {
    let mut seed = 0x1234_5678_9abc_def0;
    let a = rand_vec(m * k, &mut seed);
    let b = rand_vec(k * n, &mut seed);

    // Naive is slow → fewer reps; the rest are quick → more reps for a stable best.
    let (t_naive, c1) = best_of(3, || matmul_naive(&a, &b, m, k, n));
    let (t_fast, _) = best_of(10, || matmul_fast(&a, &b, m, k, n));
    let (t_gpu, _) = best_of(10, || gpu.run(&a, &b, m, k, n));
    let (t_mps, c4) = best_of(10, || mps.run(&a, &b, m, k, n));

    // Correctness: MPS result vs the naive reference.
    let max_diff = c1
        .iter()
        .zip(&c4)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max);

    let flops = 2.0 * m as f64 * k as f64 * n as f64; // multiply-add = 2 flops
    let gf = |d: std::time::Duration| flops / d.as_secs_f64() / 1e9;

    println!(
        "{:>5}³ | naive {:>5.0} | gemm(CPU) {:>5.0} | metal(hand) {:>5.0} | MPS {:>5.0}  GFLOP/s   (mps diff {:.0e})",
        m,
        gf(t_naive),
        gf(t_fast),
        gf(t_gpu),
        gf(t_mps),
        max_diff,
    );
}

#[cfg(target_os = "macos")]
fn main() {
    let gpu = sindri::metal_gemm::MetalGemm::new().expect("Metal init failed");
    let mps = sindri::metal_mps::MetalMps::new().expect("MPS init failed");
    println!("GEMM benchmark — naive CPU vs `gemm` crate (CPU) vs hand Metal kernel vs MPS");
    println!("GPU: {}\n", gpu.device_name());
    // warm up the GPU (first dispatch pays pipeline/shader warmup)
    let _ = gpu.run(&vec![0.0; 64 * 64], &vec![0.0; 64 * 64], 64, 64, 64);
    let _ = mps.run(&vec![0.0; 64 * 64], &vec![0.0; 64 * 64], 64, 64, 64);
    for n in [128usize, 256, 512, 768, 1024, 2048] {
        bench(&gpu, &mps, n, n, n);
    }
}

#[cfg(not(target_os = "macos"))]
fn main() {
    eprintln!("this benchmark's Metal path is macOS-only");
}
