//! tiny-infer CLI — load an ONNX model, print its graph, optionally run an image.
//!
//! Pipeline: load → build graph → topo sort → infer (see the `tiny_infer` lib).
//!
//! Usage:
//!   tiny-infer <model.onnx>                 # load + print graph summary
//!   tiny-infer <model.onnx> <image.ubyte>   # also run a 28x28 MNIST image

use std::collections::HashMap;

use sindri::{infer, onnx_loader, tensor::Tensor};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("usage: {} <model.onnx> [image.ubyte]", args[0]);
        std::process::exit(1);
    }

    let model = onnx_loader::load(&args[1])?;
    println!("✓ loaded: {} nodes, {} weight tensors", model.graph.nodes.len(), model.weights.len());
    println!("  inputs : {:?}", model.input_names);
    println!("  outputs: {:?}", model.output_names);
    print!("  topo   : ");
    for &i in &model.order {
        print!("{} ", model.graph.nodes[i].op);
    }
    println!();

    if let Some(img_path) = args.get(2) {
        let raw = std::fs::read(img_path)?;
        // MNIST 28x28 = 784 bytes, 0..255 → normalize to 0..1.
        // (real normalization is model-specific — match how YOURS was trained)
        let data: Vec<f32> = raw.iter().map(|&b| b as f32 / 255.0).collect();
        let n = data.len();
        let input = Tensor::new(data, vec![1, n]);

        let in_name = model.input_names.first().cloned().ok_or("model declares no input")?;
        let mut inputs = HashMap::new();
        inputs.insert(in_name, input);

        let outputs = infer::infer(&model, inputs)?;
        for (name, t) in &outputs {
            println!("→ output '{name}' shape {:?}", t.shape);
            println!("  prediction = {}", infer::argmax(&t.data));
            println!("  logits     = {:?}", t.data);
        }
    }

    Ok(())
}
