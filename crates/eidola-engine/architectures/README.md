# Architectures

The engine implements one model architecture per entry in `eidola_common::architecture::Architecture`. The architectures are siblings: each has its own reference forward, chat format, CPU executor path and CUDA executor path, and shares the serving core, the seam, the kernels' build, the node and the trust chain.

Each file here says, for one architecture:

- what is implemented and how it is verified;
- where its code lives in every crate (the list to follow when adding a sibling, and to prune when removing one);
- what is outstanding.

| Architecture | File | State |
|---|---|---|
| MiMo-V2.6-Flash (`MimoV26Flash`) | [`mimo-v2.6-flash.md`](mimo-v2.6-flash.md) | In the catalog; pinned to a committed one-B300 deployment |
| DeepSeek-V4.1-Flash (`DeepSeekV41Flash`) | [`deepseek-v4.1-flash.md`](deepseek-v4.1-flash.md) | Verified for correctness on two B300s; not deployed; speed work deferred |
| Qwen3.8-27B (`Qwen38Dense27B`) | [`qwen3.8-27b.md`](qwen3.8-27b.md) | In progress: served by the node; its CUDA executor not yet run on a GPU; in the catalog; pinned to a committed one-B200 deployment |

In the model, chat, CPU and CUDA crates an architecture's code is one module named for it (`mimo`, `deepseek`, `qwen`); what stays at a crate's root is shared by the architectures, and each crate's `AGENTS.md` says which root modules those are. Across the engine crates an architecture's test, example and script files carry its name as a prefix (`mimo_*`, `deepseek_*`, `qwen_*`), and its fixtures and golden tooling sit in a directory of that name; tests of shared machinery keep neutral names. The serving core (`eidola-engine`) has no architecture module: the seam is shared, and only its mock is shaped like each. The node (`eidola-server-engine`) dispatches on the architecture in shared code, one function per architecture where they differ. Kernel sources, and test data named for a kernel or a checkpoint, keep their names: renaming a kernel source could change the built images and so the measured kernel image.

Adding an architecture: a variant in `eidola_common::architecture`, then the crates in the order each file's table lists them. Start the file here with the first commit.
