//! Launch wrappers refuse inputs outside their kernels' limits on the host,
//! before anything reaches the device (the addresses here are never used).

mod common;

use common::setup;
use eidola_engine_cuda::attention::{Attention, AttnLayer};
use eidola_engine_cuda::engine_ops::{EngineOps, QkvArgs};

#[test]
fn out_of_range_launches_are_refused() {
    let Some(su) = setup() else { return };
    let gpu = &su.gpu;
    let arch = su.archs[0];
    let ops = EngineOps::from_module(su.module("engine_ops", arch)).unwrap();
    unsafe {
        assert!(
            ops.quant_fp8(gpu, 0, 0, 0, 4, 200, 4).is_err(),
            "K not whole groups"
        );
        assert!(
            ops.quant_fp8(gpu, 0, 0, 0, 8, 128, 4).is_err(),
            "m_pad below rows"
        );
        assert!(ops.swiglu_quant_f32scale(gpu, 0, 0, 0, 4, 200, 4).is_err());
        assert!(
            ops.swiglu_quant_ue8m0(gpu, 0, 0, 0, 4, 1000, 4, 0).is_err(),
            "not whole words"
        );
        assert!(
            ops.swiglu_quant_ue8m0(gpu, 0, 0, 0, 130, 2048, 4, 128)
                .is_err(),
            "rows not whole experts"
        );
        assert!(
            ops.gather_quant_ue8m0(gpu, 0, 0, 0, 0, 8, 4096, 4, 0)
                .is_err(),
            "rows4 below rows"
        );
        assert!(
            ops.router_topk(gpu, 0, 0, 0, 0, 0, 1, 4096, 300, 8, 1.0)
                .is_err(),
            "experts"
        );
        assert!(
            ops.router_topk(gpu, 0, 0, 0, 0, 0, 1, 4096, 256, 9, 1.0)
                .is_err(),
            "top_k"
        );
        assert!(
            ops.moe_permute(gpu, 0, 0, 0, 0, 200, 8, 128, 0).is_err(),
            "masked capacity"
        );
        assert!(
            ops.moe_combine(gpu, 0, 0, 0, 0, 1, 4096, 0).is_err(),
            "top_k 0"
        );
        assert!(
            ops.rmsnorm(gpu, 0, 0, 0, 100, 0, 4096, 1e-6, 1).is_err(),
            "stride"
        );
        let short = QkvArgs {
            chunks: 4,
            chunk_stride: 3000,
            q_heads_per_chunk: 16,
            kv_heads_per_chunk: 1,
            ..QkvArgs::default()
        };
        assert!(ops.qkv_rope_kv(gpu, short, 1).is_err(), "chunk stride");
    }
    let attn = Attention::from_module(su.module("flashinfer_fa2_sink_paged", arch)).unwrap();
    let plan = attn
        .plan(
            gpu,
            &[eidola_engine_cuda::attention::AttnRequest {
                q_start: 0,
                qo_len: 1,
                pages: vec![1],
                kv_len: 1,
            }],
            16,
            16,
        )
        .unwrap();
    let layer = AttnLayer {
        k_base: 0,
        v_base: 0,
        block_elems: 1,
        num_kv_heads: 4,
        page_size: 16,
        window_left: -1,
        sink: 0,
    };
    assert!(
        unsafe { attn.run(gpu, &plan, &layer, 63, 0, 0) }.is_err(),
        "GQA groups"
    );
    let other_pages = AttnLayer {
        page_size: 32,
        ..layer
    };
    assert!(
        unsafe { attn.run(gpu, &plan, &other_pages, 64, 0, 0) }.is_err(),
        "a plan for another page size"
    );
}
