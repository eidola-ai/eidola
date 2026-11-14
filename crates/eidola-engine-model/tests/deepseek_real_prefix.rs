//! Local measurement on the first four layers of the real
//! DeepSeek-V4.1-Flash-1017 checkpoint, whole (every routed expert): how the
//! residual streams respond to the bits the reference leaves unspecified.
//! Ignored: needs the fetched weights and goldens, see
//! `golden/deepseek_v41/README.md`.
//!
//!     EIDOLA_DEEPSEEK_PREFIX_DIR=<dir> cargo test --release -p eidola-engine-model \
//!         --test deepseek_real_prefix -- --ignored --nocapture
//!
//! Every `golden/reference-*.safetensors` in the directory is compared, with
//! our forward and with each other. `reference-pinned` must be equalled
//! exactly; the others are reported. Without it nothing is asserted.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use eidola_engine_model::WeightSet;
use eidola_engine_model::deepseek::engram::{EngramTable, SparseEngramTable};
use eidola_engine_model::deepseek::forward::LayerOptions;
use eidola_engine_model::deepseek::{DsLoadOptions, DsModel, DsWeights, config_of};

const LAYERS: [usize; 4] = [0, 1, 2, 3];

fn open(path: &Path) -> WeightSet {
    WeightSet::open_files(&[path.to_path_buf()]).unwrap()
}

fn f32s(set: &WeightSet, name: &str) -> Vec<f32> {
    set.get(name).unwrap().to_f32(name).unwrap()
}

/// One run of the forward: per layer, the streams and each token's routed
/// experts as an ascending set.
struct Run {
    streams: Vec<Vec<f32>>,
    routes: Vec<Vec<Vec<usize>>>,
}

fn run(model: &DsModel, tokens: &[u32], reference_prefill: Option<usize>) -> Run {
    let mut state = model.new_state();
    let mut rows = model.embed(&mut state, tokens).unwrap();
    let (mut streams, mut routes) = (Vec::new(), Vec::new());
    for l in 0..model.config().layers.len() {
        let opts = LayerOptions {
            reference_prefill,
            ..Default::default()
        };
        let out = model.layer_forward(l, &mut state, &rows, opts).unwrap();
        routes.push(
            out.routing
                .iter()
                .map(|r| {
                    let mut set = r.experts.clone();
                    set.sort_unstable();
                    set
                })
                .collect(),
        );
        rows = out.rows;
        streams.push(rows.streams.data.clone());
    }
    Run { streams, routes }
}

/// One line of the per-layer table. Returns the count of differing elements.
fn report(
    layer: usize,
    got: &[f32],
    want: &[f32],
    width: usize,
    routes: Option<(usize, usize)>,
) -> usize {
    let n = got.len().min(want.len());
    let (got, want) = (&got[..n], &want[..n]);
    let differing = got.iter().zip(want).filter(|(a, b)| a != b).count();
    let (mut err, mut norm, mut worst) = (0.0f64, 0.0f64, 0.0f32);
    for (a, b) in got.iter().zip(want) {
        err += ((a - b) as f64).powi(2);
        norm += (*b as f64).powi(2);
        worst = worst.max((a - b).abs());
    }
    let positions = got
        .chunks(width)
        .zip(want.chunks(width))
        .filter(|(a, b)| a != b)
        .count();
    let first = got
        .chunks(width)
        .zip(want.chunks(width))
        .position(|(a, b)| a != b);
    let routed = routes.map_or(String::new(), |(same, of)| {
        format!("; router sets equal at {same} of {of} tokens")
    });
    println!(
        "  after layer {layer}: relative error {:.3e}, max |Δ| {worst:.3e}, {:.4} % of elements differ, {positions} of {} positions (first {first:?}){routed}",
        (err / norm.max(f64::MIN_POSITIVE)).sqrt(),
        100.0 * differing as f64 / n as f64,
        n / width,
    );
    differing
}

fn same_routes(a: &[Vec<usize>], b: &[Vec<usize>]) -> (usize, usize) {
    let n = a.len().min(b.len());
    (
        a[..n].iter().zip(&b[..n]).filter(|(x, y)| x == y).count(),
        n,
    )
}

#[test]
#[ignore = "needs the fetched first four layers of the real checkpoint"]
fn whole_early_layers_match_the_reference_and_measure_its_sensitivity() {
    let dir = PathBuf::from(
        std::env::var("EIDOLA_DEEPSEEK_PREFIX_DIR")
            .expect("EIDOLA_DEEPSEEK_PREFIX_DIR names the fetched checkpoint"),
    );
    let store = Arc::new(WeightSet::open_dir(&dir).unwrap());
    let manifest: BTreeMap<String, String> =
        serde_json::from_slice(&std::fs::read(dir.join("local-sha256.json")).unwrap()).unwrap();
    store
        .verify_sha256(&manifest)
        .expect("fetched files match their manifest");

    let mut goldens: Vec<PathBuf> = std::fs::read_dir(dir.join("golden"))
        .expect("the checkpoint directory has a golden/ directory")
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| {
            p.file_name()
                .and_then(|f| f.to_str())
                .is_some_and(|f| f.starts_with("reference-") && f.ends_with(".safetensors"))
        })
        .collect();
    goldens.sort_by_key(|p| (!p.ends_with("reference-pinned.safetensors"), p.clone()));
    assert!(
        !goldens.is_empty(),
        "no golden/reference-*.safetensors to compare against"
    );
    // Every golden holds the tokens and the Engram map it was made with.
    let pinned = open(&goldens[0]);
    let ints = |set: &WeightSet, name: &str| set.get(name).unwrap().to_i64(name).unwrap();
    let tokens: Vec<u32> = ints(&pinned, "tokens").iter().map(|t| *t as u32).collect();
    let token_map = ints(&pinned, "engram.token_map")
        .iter()
        .map(|t| *t as u32)
        .collect();

    let config = config_of(&store).unwrap().truncated(&LAYERS).unwrap();
    assert!(config.drafter.is_none() && config.moe.n_routed == config.moe.source_n_routed);
    let rows = open(&dir.join("golden/engram-rows.safetensors"));
    let spec = config.engram.clone().unwrap();
    let engram_tables = spec
        .layers
        .iter()
        .filter(|l| LAYERS.contains(&l.source_layer))
        .map(|l| {
            let p = format!("layers.{}", l.source_layer);
            let ids: Vec<u64> = ints(&rows, &format!("{p}.rows"))
                .iter()
                .map(|r| *r as u64)
                .collect();
            let table = SparseEngramTable::from_stored(
                l.num_rows,
                spec.head_dim,
                &ids,
                rows.get(&format!("{p}.weight")).unwrap().data,
                rows.get(&format!("{p}.scale")).unwrap().data,
            )
            .unwrap();
            (l.source_layer, Arc::new(table) as Arc<dyn EngramTable>)
        })
        .collect();
    let weights = DsWeights::load(
        store,
        config,
        &DsLoadOptions {
            skip_drafter: true,
            engram_tables,
        },
    )
    .unwrap();
    let model = DsModel::new(weights, Some(token_map)).unwrap();
    let c = model.config();
    let (n, width, layers) = (tokens.len(), c.hc_mult * c.dim, c.layers.len());
    println!(
        "{n} tokens, checkpoint layers {LAYERS:?}, {} routed experts per layer",
        c.moe.n_routed
    );

    // The reference's one prefill chunk in its own layout, and the canonical
    // layout (the reference's for one token per call).
    let prefill = run(&model, &tokens, Some(n));
    let canonical = run(&model, &tokens, None);

    let mut inexact = 0;
    for path in &goldens {
        let name = path.file_stem().unwrap().to_str().unwrap().to_string();
        let golden = open(path);
        println!("our forward (the reference's prefill layout) against {name}, one prefill chunk:");
        let mut differing = 0;
        for l in 0..layers {
            let want_routes: Vec<Vec<usize>> = ints(&golden, &format!("prefill.route.{l}"))
                .chunks(c.moe.n_active)
                .map(|r| r.iter().map(|e| *e as usize).collect())
                .collect();
            let same = same_routes(&prefill.routes[l], &want_routes);
            differing += report(
                l,
                &prefill.streams[l],
                &f32s(&golden, &format!("prefill.layer.{l}")),
                width,
                Some(same),
            );
            differing += same.1 - same.0;
        }
        if golden.contains("step.layer.0") {
            println!("our forward (canonical layout) against {name}, one token per call:");
            for l in 0..layers {
                differing += report(
                    l,
                    &canonical.streams[l],
                    &f32s(&golden, &format!("step.layer.{l}")),
                    width,
                    None,
                );
            }
        }
        if name == "reference-pinned" {
            inexact = differing;
        }
    }
    // The reference against itself: every pair of goldens other than the
    // pinned one (which this forward equals, so its rows are the ones above).
    let names: Vec<String> = goldens
        .iter()
        .map(|p| p.file_stem().unwrap().to_string_lossy().into_owned())
        .collect();
    for i in 0..goldens.len() {
        for j in i + 1..goldens.len() {
            if names[i] == "reference-pinned" {
                continue;
            }
            let (a, b) = (open(&goldens[i]), open(&goldens[j]));
            println!("{} against {}, one prefill chunk:", names[j], names[i]);
            for l in 0..layers {
                let routes = |g: &WeightSet| -> Vec<Vec<usize>> {
                    ints(g, &format!("prefill.route.{l}"))
                        .chunks(c.moe.n_active)
                        .map(|r| r.iter().map(|e| *e as usize).collect())
                        .collect()
                };
                let same = same_routes(&routes(&b), &routes(&a));
                let key = format!("prefill.layer.{l}");
                report(l, &f32s(&b, &key), &f32s(&a, &key), width, Some(same));
            }
        }
    }
    println!(
        "the reference's prefill layout against its decode layout (both ours, exact stand-ins):"
    );
    for l in 0..layers {
        let same = same_routes(&canonical.routes[l], &prefill.routes[l]);
        report(
            l,
            &canonical.streams[l],
            &prefill.streams[l],
            width,
            Some(same),
        );
    }
    if names[0] != "reference-pinned" {
        println!("no reference-pinned golden present: nothing was asserted beyond a completed run");
    }
    assert_eq!(inexact, 0, "the pinned reference must be equalled exactly");
}
