//! The Qwen3.8 kernels' bodies, compiled for the host, against the model crate's
//! functions bit for bit (`qwen_common::cases`): every family through its grid function
//! and the body a GPU compiles, threads in descending order and shared memory starting as
//! garbage. Skipped, saying why, without a C++ compiler.

mod qwen_common;

use qwen_common::cases;

#[test]
fn qwen_kernel_bodies_equal_the_model_crate() {
    let Some(dev) = qwen_common::compiled() else {
        return;
    };
    let mut failed = Vec::new();
    for (name, outcome) in cases::all(&dev) {
        match outcome {
            Ok(()) => eprintln!("ok      {name}"),
            Err(e) => {
                eprintln!("FAILED  {name}: {e}");
                failed.push(name);
            }
        }
    }
    assert!(failed.is_empty(), "{failed:?}");
}
