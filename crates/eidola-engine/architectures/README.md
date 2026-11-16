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

Adding an architecture: a variant in `eidola_common::architecture`, then the crates in the order each file's table lists them. Start the file here with the first commit.
