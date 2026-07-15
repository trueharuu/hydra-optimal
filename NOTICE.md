# Notices and provenance

This repository is distributed under GPL-3.0-only.

It is a Rust port and extension of
[`randomidiot13/hydra`](https://github.com/randomidiot13/hydra), based on upstream version
`v0.4.20240203`. The original solver design, `weights.txt`, and browser decision-tree viewer come
from that GPL-3.0 project.

The included `vstar_l0_f32.bin` is project-generated data from the zxcl Perfect Clear MDP. Its
format, size, and checksum are documented in `README.md`.

`graph.bin` is not included. It remains an external download published with the upstream project;
the expected size and hashes are documented in `README.md`.
