//! Graph — index-based arena (NOT pointer-based!).
//!
//! The #1 Rust lesson vs the C++ article: don't store parent/child pointers.
//! Ownership lives in one `Vec<Node>`; everything else refers to nodes by
//! `usize` index. This sidesteps the borrow checker entirely and is exactly
//! how real engines (and `petgraph`) model graphs.

use std::collections::HashMap;

/// Operator attributes we care about for the MNIST FFN (Gemm/Flatten).
/// ONNX defaults: alpha=1.0, beta=1.0, transA=0, transB=0, Flatten axis=1.
#[derive(Debug, Clone)]
pub struct Attrs {
    pub trans_a: bool,
    pub trans_b: bool,
    pub alpha: f32,
    pub beta: f32,
    pub axis: i64,
}

impl Default for Attrs {
    fn default() -> Self {
        Attrs { trans_a: false, trans_b: false, alpha: 1.0, beta: 1.0, axis: 1 }
    }
}

#[derive(Debug, Clone)]
pub struct Node {
    pub op: String,          // "Gemm" | "Relu" | "Add" | "Flatten"
    pub inputs: Vec<String>, // names of input tensors (weights or activations)
    pub outputs: Vec<String>,
    pub attrs: Attrs,
}

pub struct Graph {
    pub nodes: Vec<Node>,
    /// edges[i] = indices of nodes that consume an output of node i (children).
    pub edges: Vec<Vec<usize>>,
}

impl Graph {
    /// Build adjacency from tensor-name matching: an edge p -> c exists when
    /// an output tensor of node p is an input tensor of node c.
    pub fn build(nodes: Vec<Node>) -> Self {
        // Map each produced tensor name -> the node index that produces it.
        let mut producer: HashMap<String, usize> = HashMap::new();
        for (i, n) in nodes.iter().enumerate() {
            for out in &n.outputs {
                producer.insert(out.clone(), i);
            }
        }

        let mut edges = vec![Vec::new(); nodes.len()];
        for (i, n) in nodes.iter().enumerate() {
            for inp in &n.inputs {
                if let Some(&p) = producer.get(inp) {
                    if p != i {
                        edges[p].push(i); // p produces a tensor that i consumes
                    }
                }
            }
        }
        Graph { nodes, edges }
    }

    /// Topological order via DFS post-order (reversed). Assumes a DAG.
    /// Guarantees every node runs only after all its producers have run.
    pub fn topo_sort(&self) -> Vec<usize> {
        let n = self.nodes.len();
        let mut visited = vec![false; n];
        let mut order = Vec::with_capacity(n);
        for i in 0..n {
            if !visited[i] {
                self.dfs(i, &mut visited, &mut order);
            }
        }
        order.reverse();
        order
    }

    fn dfs(&self, u: usize, visited: &mut [bool], order: &mut Vec<usize>) {
        visited[u] = true;
        for &v in &self.edges[u] {
            if !visited[v] {
                self.dfs(v, visited, order);
            }
        }
        order.push(u); // post-order: children pushed before parent
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn node(op: &str, ins: &[&str], outs: &[&str]) -> Node {
        Node {
            op: op.into(),
            inputs: ins.iter().map(|s| s.to_string()).collect(),
            outputs: outs.iter().map(|s| s.to_string()).collect(),
            attrs: Attrs::default(),
        }
    }

    #[test]
    fn topo_respects_diamond_dependencies() {
        // A -> B -> D, A -> C -> D  (the diamond from the article's animation)
        //   A:  x      -> a
        //   B:  a      -> b
        //   C:  a      -> c
        //   D:  b, c   -> y
        let nodes = vec![
            node("A", &["x"], &["a"]),
            node("B", &["a"], &["b"]),
            node("C", &["a"], &["c"]),
            node("D", &["b", "c"], &["y"]),
        ];
        let g = Graph::build(nodes);
        let order = g.topo_sort();

        let pos = |idx: usize| order.iter().position(|&x| x == idx).unwrap();
        // A(0) before B(1) and C(2); B and C before D(3)
        assert!(pos(0) < pos(1));
        assert!(pos(0) < pos(2));
        assert!(pos(1) < pos(3));
        assert!(pos(2) < pos(3));
    }
}
