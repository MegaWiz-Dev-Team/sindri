//! Inference — walk the topo-sorted graph, run each op, thread tensors through.
//!
//! Maps to the article's `infer()` loop.

use std::collections::HashMap;
use std::error::Error;

use crate::ops;
use crate::onnx_loader::Model;
use crate::tensor::Tensor;

/// Run the model on the given named inputs. Returns the model's output tensors.
pub fn infer(
    model: &Model,
    inputs: HashMap<String, Tensor>,
) -> Result<HashMap<String, Tensor>, Box<dyn Error>> {
    // Tensor store starts with all weights, then the user inputs layered on top.
    let mut store: HashMap<String, Tensor> = model.weights.clone();
    store.extend(inputs);

    for &idx in &model.order {
        let node = &model.graph.nodes[idx];

        // Gather this node's inputs (clone out so the immutable borrow of `store`
        // ends before we insert the output — keeps the borrow checker happy).
        let ins: Vec<Tensor> = node
            .inputs
            .iter()
            .map(|name| {
                store
                    .get(name)
                    .cloned()
                    .ok_or_else(|| format!("node '{}' missing input tensor '{name}'", node.op))
            })
            .collect::<Result<_, _>>()?;

        let out = match node.op.as_str() {
            "Gemm" => {
                let c = ins.get(2);
                ops::gemm(&ins[0], &ins[1], c, &node.attrs)
            }
            "Relu" => ops::relu(&ins[0]),
            "Add" => ops::add(&ins[0], &ins[1]),
            "Flatten" => ops::flatten(&ins[0], node.attrs.axis),
            other => return Err(format!("unsupported op: '{other}'").into()),
        };

        store.insert(node.outputs[0].clone(), out);
    }

    let mut result = HashMap::new();
    for name in &model.output_names {
        if let Some(t) = store.get(name) {
            result.insert(name.clone(), t.clone());
        }
    }
    Ok(result)
}

/// argmax over a flat logits vector — turns the [1,10] output into a digit.
pub fn argmax(data: &[f32]) -> usize {
    data.iter()
        .enumerate()
        .fold((0usize, f32::MIN), |(bi, bv), (i, &v)| if v > bv { (i, v) } else { (bi, bv) })
        .0
}
