//! Operators — the 4 the MNIST FFN needs: Gemm, Relu, Add, Flatten.
//!
//! Milestone 3 = correctness with these naive versions (get the "7").
//! Milestone 4 = swap `gemm` for the `gemm` crate or Accelerate BLAS.

use crate::graph::Attrs;
use crate::tensor::Tensor;

/// ONNX Gemm:  Y = alpha * (A or Aᵀ) @ (B or Bᵀ) + beta * C
///
/// For an MNIST `nn.Linear`, weights export as [out, in] and the node carries
/// transB=1, so Bᵀ is [in, out] and Y = A[1,in] @ [in,out] + bias.
pub fn gemm(a: &Tensor, b: &Tensor, c: Option<&Tensor>, attrs: &Attrs) -> Tensor {
    let (a0, a1) = a.dims2();
    let (b0, b1) = b.dims2();

    // Effective dims after optional transpose.
    let (m, k) = if attrs.trans_a { (a1, a0) } else { (a0, a1) };
    let (kb, n) = if attrs.trans_b { (b1, b0) } else { (b0, b1) };
    assert_eq!(k, kb, "Gemm inner dims mismatch: {k} vs {kb}");

    let a_at = |i: usize, p: usize| -> f32 {
        if attrs.trans_a { a.data[p * a1 + i] } else { a.data[i * a1 + p] }
    };
    let b_at = |p: usize, j: usize| -> f32 {
        if attrs.trans_b { b.data[j * b1 + p] } else { b.data[p * b1 + j] }
    };

    let mut y = vec![0.0f32; m * n];
    for i in 0..m {
        for p in 0..k {
            let aip = a_at(i, p);
            for j in 0..n {
                y[i * n + j] += aip * b_at(p, j); // ikj order = cache-friendlier
            }
        }
    }

    if attrs.alpha != 1.0 {
        for v in &mut y {
            *v *= attrs.alpha;
        }
    }

    if let Some(c) = c {
        let cd = &c.data;
        for i in 0..m {
            for j in 0..n {
                // broadcast bias: [n] (per-column), [m*n] (full), or [1] (scalar)
                let cij = match cd.len() {
                    l if l == n => cd[j],
                    l if l == m * n => cd[i * n + j],
                    1 => cd[0],
                    _ => panic!("Gemm C has un-broadcastable len {} (m={m}, n={n})", cd.len()),
                };
                y[i * n + j] += attrs.beta * cij;
            }
        }
    }

    Tensor::new(y, vec![m, n])
}

// ── Milestone 4: plain C[m,n] = A[m,k] · B[k,n] (row-major), two backends ──
// These are the apples-to-apples kernels the benchmark compares. The full
// ONNX `gemm` above stays as the engine's correctness reference.

/// Naive triple loop (ikj order). Single-threaded, no SIMD intrinsics —
/// the compiler may auto-vectorize the inner loop but that's it.
pub fn matmul_naive(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    let mut c = vec![0.0f32; m * n];
    for i in 0..m {
        for p in 0..k {
            let aip = a[i * k + p];
            let brow = &b[p * n..p * n + n];
            let crow = &mut c[i * n..i * n + n];
            for j in 0..n {
                crow[j] += aip * brow[j];
            }
        }
    }
    c
}

/// Same product via the `gemm` crate — hand-tuned SIMD (NEON on Apple Silicon)
/// + multithreading via rayon. Drop-in replacement for `matmul_naive`.
pub fn matmul_fast(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    let mut c = vec![0.0f32; m * n];
    // Row-major strides: element (i,j) at i*cols + j ⇒ col_stride=1, row_stride=cols.
    // `::gemm` = the external crate (our local `fn gemm` would shadow a bare `gemm`).
    unsafe {
        ::gemm::gemm(
            m, n, k,
            c.as_mut_ptr(), 1, n as isize, // dst m×n
            false,                          // read_dst: false ⇒ dst = beta·A·B
            a.as_ptr(), 1, k as isize,      // lhs m×k
            b.as_ptr(), 1, n as isize,      // rhs k×n
            0.0f32, 1.0f32,                 // alpha (dst scale, unused), beta
            false, false, false,            // no conjugation
            ::gemm::Parallelism::Rayon(0),  // 0 ⇒ use all available threads
        );
    }
    c
}

pub fn relu(x: &Tensor) -> Tensor {
    Tensor::new(x.data.iter().map(|&v| v.max(0.0)).collect(), x.shape.clone())
}

pub fn add(a: &Tensor, b: &Tensor) -> Tensor {
    assert_eq!(a.shape, b.shape, "Add expects equal shapes (no broadcast here)");
    Tensor::new(
        a.data.iter().zip(&b.data).map(|(x, y)| x + y).collect(),
        a.shape.clone(),
    )
}

/// ONNX Flatten: collapse dims [0..axis) and [axis..) into a 2-D [outer, inner].
/// Row-major ⇒ only the shape changes, data is untouched.
pub fn flatten(x: &Tensor, axis: i64) -> Tensor {
    let rank = x.shape.len() as i64;
    let axis = if axis < 0 { axis + rank } else { axis } as usize;
    let outer: usize = x.shape[..axis].iter().product::<usize>().max(1);
    let inner: usize = x.shape[axis..].iter().product::<usize>().max(1);
    Tensor::new(x.data.clone(), vec![outer, inner])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gemm_plain_2x3_3x2() {
        // A[2,3] @ B[3,2], no transpose, no bias
        let a = Tensor::new(vec![1., 2., 3., 4., 5., 6.], vec![2, 3]);
        let b = Tensor::new(vec![7., 8., 9., 10., 11., 12.], vec![3, 2]);
        let y = gemm(&a, &b, None, &Attrs::default());
        // row0: [1*7+2*9+3*11, 1*8+2*10+3*12] = [58, 64]
        // row1: [4*7+5*9+6*11, 4*8+5*10+6*12] = [139, 154]
        assert_eq!(y.shape, vec![2, 2]);
        assert_eq!(y.data, vec![58., 64., 139., 154.]);
    }

    #[test]
    fn gemm_transb_with_bias_like_linear() {
        // x[1,3] @ Wᵀ where W is [2,3] (out=2,in=3), + bias[2]  → y[1,2]
        let x = Tensor::new(vec![1., 2., 3.], vec![1, 3]);
        let w = Tensor::new(vec![1., 0., 0., 0., 1., 0.], vec![2, 3]); // picks x[0], x[1]
        let bias = Tensor::new(vec![10., 20.], vec![2]);
        let attrs = Attrs { trans_b: true, ..Attrs::default() };
        let y = gemm(&x, &w, Some(&bias), &attrs);
        assert_eq!(y.shape, vec![1, 2]);
        assert_eq!(y.data, vec![1. + 10., 2. + 20.]); // [11, 22]
    }

    #[test]
    fn matmul_fast_matches_naive() {
        // 5×7 @ 7×3, deterministic values
        let (m, k, n) = (5, 7, 3);
        let a: Vec<f32> = (0..m * k).map(|i| (i as f32 * 0.13).sin()).collect();
        let b: Vec<f32> = (0..k * n).map(|i| (i as f32 * 0.07).cos()).collect();
        let naive = matmul_naive(&a, &b, m, k, n);
        let fast = matmul_fast(&a, &b, m, k, n);
        for (x, y) in naive.iter().zip(&fast) {
            assert!((x - y).abs() < 1e-4, "{x} vs {y}");
        }
    }

    #[test]
    fn relu_clamps_negatives() {
        let x = Tensor::new(vec![-1., 0., 2., -3.5], vec![4]);
        assert_eq!(relu(&x).data, vec![0., 0., 2., 0.]);
    }

    #[test]
    fn add_elementwise() {
        let a = Tensor::new(vec![1., 2., 3.], vec![3]);
        let b = Tensor::new(vec![10., 20., 30.], vec![3]);
        assert_eq!(add(&a, &b).data, vec![11., 22., 33.]);
    }

    #[test]
    fn flatten_image_to_vector() {
        let img = Tensor::new(vec![0.0; 1 * 28 * 28], vec![1, 28, 28]);
        let f = flatten(&img, 1);
        assert_eq!(f.shape, vec![1, 784]);
    }
}
