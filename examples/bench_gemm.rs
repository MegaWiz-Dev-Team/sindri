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

fn bench(m: usize, k: usize, n: usize) {
    let mut seed = 0x1234_5678_9abc_def0;
    let a = rand_vec(m * k, &mut seed);
    let b = rand_vec(k * n, &mut seed);

    // Naive is slow → fewer reps; fast is quick → more reps for a stable best.
    let (t_naive, c1) = best_of(3, || matmul_naive(&a, &b, m, k, n));
    let (t_fast, c2) = best_of(10, || matmul_fast(&a, &b, m, k, n));

    let max_diff = c1
        .iter()
        .zip(&c2)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max);

    let flops = 2.0 * m as f64 * k as f64 * n as f64; // multiply-add = 2 flops
    let gf = |d: std::time::Duration| flops / d.as_secs_f64() / 1e9;
    let ms = |d: std::time::Duration| d.as_secs_f64() * 1e3;

    println!(
        "{:>5}³   naive {:>9.3} ms ({:>6.1} GFLOP/s)   fast {:>8.3} ms ({:>7.1} GFLOP/s)   speedup {:>5.1}×   maxdiff {:.1e}",
        m,
        ms(t_naive),
        gf(t_naive),
        ms(t_fast),
        gf(t_fast),
        t_naive.as_secs_f64() / t_fast.as_secs_f64(),
        max_diff,
    );
}

fn main() {
    println!("GEMM benchmark — naive triple-loop vs `gemm` crate (SIMD+threads)");
    println!("Apple Silicon, square matrices N×N @ N×N\n");
    for n in [128usize, 256, 512, 768, 1024] {
        bench(n, n, n);
    }
}
