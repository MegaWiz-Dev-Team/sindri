//! Tensor — a thin wrapper around `Vec<f32>` with a shape, stored row-major.
//!
//! Same idea as the C++ article's `Tensor<T>` wrapping `std::vector<T>`.
//! Row-major means a Flatten is just a shape change (data never moves).

#[derive(Clone, Debug, PartialEq)]
pub struct Tensor {
    pub data: Vec<f32>,
    pub shape: Vec<usize>,
}

impl Tensor {
    pub fn new(data: Vec<f32>, shape: Vec<usize>) -> Self {
        debug_assert_eq!(
            data.len(),
            shape.iter().product::<usize>(),
            "data length must equal product(shape)"
        );
        Self { data, shape }
    }

    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }

    /// Interpret as a 2-D matrix (rows, cols). Panics if not rank-2.
    pub fn dims2(&self) -> (usize, usize) {
        assert_eq!(self.shape.len(), 2, "expected rank-2 tensor, got {:?}", self.shape);
        (self.shape[0], self.shape[1])
    }
}
