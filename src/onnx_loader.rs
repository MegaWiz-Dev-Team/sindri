//! ONNX loader — decode the protobuf, extract weights, build the graph.
//!
//! Maps to the article's "Loading the model" + "Graph construction" steps.

use std::collections::HashMap;
use std::error::Error;

use prost::Message;

use crate::graph::{Attrs, Graph, Node};
use crate::onnx_proto::{ModelProto, NodeProto, TensorProto};
use crate::tensor::Tensor;

pub struct Model {
    pub graph: Graph,
    /// Initializer tensors (trained weights & biases) keyed by name.
    pub weights: HashMap<String, Tensor>,
    pub input_names: Vec<String>,
    pub output_names: Vec<String>,
    /// Precomputed topological execution order (indices into graph.nodes).
    pub order: Vec<usize>,
}

pub fn load(path: &str) -> Result<Model, Box<dyn Error>> {
    let bytes = std::fs::read(path)?;
    let model = ModelProto::decode(&*bytes)?;
    let g = model.graph.ok_or("ONNX model has no graph")?;

    // Weights (initializers).
    let mut weights = HashMap::new();
    for init in &g.initializer {
        let name = init.name.clone().unwrap_or_default();
        weights.insert(name, tensor_from_proto(init)?);
    }

    // Nodes.
    let nodes: Vec<Node> = g
        .node
        .iter()
        .map(|n| Node {
            op: n.op_type.clone().unwrap_or_default(),
            inputs: n.input.clone(),
            outputs: n.output.clone(),
            attrs: attrs_from(n),
        })
        .collect();

    let graph = Graph::build(nodes);
    let order = graph.topo_sort();

    // Graph-level declared inputs that are NOT initializers = the real model inputs.
    let weight_names: std::collections::HashSet<_> = weights.keys().cloned().collect();
    let input_names = g
        .input
        .iter()
        .filter_map(|v| v.name.clone())
        .filter(|n| !weight_names.contains(n))
        .collect();
    let output_names = g.output.iter().filter_map(|v| v.name.clone()).collect();

    Ok(Model { graph, weights, input_names, output_names, order })
}

/// Read a float tensor from a TensorProto. PyTorch export typically uses
/// `raw_data` (little-endian f32 bytes); some exporters use `float_data`.
fn tensor_from_proto(t: &TensorProto) -> Result<Tensor, Box<dyn Error>> {
    let shape: Vec<usize> = t.dims.iter().map(|&d| d as usize).collect();
    let data: Vec<f32> = if !t.float_data.is_empty() {
        t.float_data.clone()
    } else if let Some(raw) = &t.raw_data {
        raw.chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect()
    } else {
        return Err(format!(
            "tensor '{}' has neither float_data nor raw_data (dtype={:?})",
            t.name.clone().unwrap_or_default(),
            t.data_type
        )
        .into());
    };
    Ok(Tensor::new(data, shape))
}

fn attrs_from(n: &NodeProto) -> Attrs {
    let mut a = Attrs::default();
    for attr in &n.attribute {
        match attr.name.as_deref() {
            Some("transA") => a.trans_a = attr.i.unwrap_or(0) != 0,
            Some("transB") => a.trans_b = attr.i.unwrap_or(0) != 0,
            Some("alpha") => a.alpha = attr.f.unwrap_or(1.0),
            Some("beta") => a.beta = attr.f.unwrap_or(1.0),
            Some("axis") => a.axis = attr.i.unwrap_or(1),
            _ => {}
        }
    }
    a
}
