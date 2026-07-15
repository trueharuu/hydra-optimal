# zxcl optimal solver

This is a standalone Rust port of
[randomidiot13/hydra](https://github.com/randomidiot13/hydra), a brute-force solver that finds a
decision tree with the best immediate Perfect Clear (PC) chance.

The port keeps the upstream input format and mode behavior while replacing the in-memory graph
representation and search hot path. The main goals are lower memory use, faster graph loading, and
faster searches without changing the numerical results. It also adds an optional layer-0
`V*`-seeded objective for see-7 decision trees. The compatibility baseline is upstream
`v0.4.20240203`.

## What you need

- A recent stable Rust toolchain
- A 64-bit machine with roughly 2 GiB available memory for the classic modes
- At least 4 GiB available memory for an empty-field `--optimal` tree; 8 GiB is a comfortable
  recommendation across allocators and platforms
- The canonical `graph.bin` described below

The exact layer-0 `V*` table needed by `--optimal` is included as `vstar_l0_f32.bin`. It contains
`1,120,140` ordered boundary states in the `L0F32V1` format and is exactly `8,961,136` bytes:

```text
SHA-256 89f42a16c41cfa1c0fdab28e429d2df00501b378a58af0231c627d61a3840eda
```

This V* table is project-generated data from the companion PC MDP pipeline; it is not an upstream
solver asset. The keyed file is bundled under this repository's GPL-3.0-only terms so optimal mode
works without a separate value-table download.

`graph.bin` is not included in this repository. Download it from the
[original Google Drive link](https://drive.google.com/file/d/1XEYrDFhatN-McOcpTpWuAYq8ubyanJtq/view?usp=sharing)
and place it in the repository root.

The expected file is exactly `510,917,451` bytes:

```text
MD5     1490c9bd252d6bfa5c7d00518737be6b
SHA-1   3add9a11996e386db2482763344eb5062a745afa
SHA-256 7dc434e3611684756911480d6d338d37bf8ca5ade263611d651fa8709e0294f4
```

On Linux, the size and strongest hash can be checked with:

```bash
wc -c graph.bin
sha256sum graph.bin
```

## Build and run

```bash
cargo build --release --locked
./target/release/zxcl-optimal-solver --graph ./graph.bin
```

The solver reads whitespace-separated queries from standard input. With the default `-s 7`, the
first token contains seven pieces in `IJLOSTZ` order: held piece, active piece, and previews. The
second token is the current partial bag.

```text
IJLOSTZ IJLOSTZ
```

This produces a result like:

```text
Result: 838/840
```

The bag may instead be given as a digit from `1` through `7`. In that form, the solver infers the
remaining pieces from the visible queue:

```text
OTJLISO 1
```

```text
Result: 206/210
```

Send more queries to reuse the loaded graph. End input with EOF, or enter a token whose length does
not match the current `see` value.

Almost all human-readable output, including prompts, results, and timings, goes to standard error
for compatibility with the original program. Add `-o` to also write only the numerical result to
standard output. This makes batch use straightforward:

```bash
printf '%s\n' 'IJLOSTZ IJLOSTZ' 'OTJLISO 1' |
  ./target/release/zxcl-optimal-solver --graph ./graph.bin -o > results.txt
```

### Decision-tree output

Use `-d` to generate the optimal decision tree for a query:

```bash
printf '%s\n' 'IJLOSTZ IJLOSTZ' |
  ./target/release/zxcl-optimal-solver --graph ./graph.bin -d
```

This writes `tree_data.js` in the current working directory. Open the included `tree_viewer.html` in
a browser to inspect it; `tree_viewer.html`, `main.js`, and the generated `tree_data.js` need to be
in the same directory. A later decision query replaces `tree_data.js`.

#### V*-optimal mode

Use `--optimal` to evaluate placements by the expected number of future PCs. Without `-d`, it
prints the values without writing a tree:

```bash
printf '%s\n' 'OIJLSTZ IJLOSTZ' |
  ./target/release/zxcl-optimal-solver \
    --graph ./graph.bin \
    --optimal -f 274072600575 -o
```

The diagnostics include both objectives:

```text
Result: 4350.43798828125
Survival: 1/1
```

`Result` is the V*-optimal expected-PC value. `Survival` is the baseline objective computed
independently for the same state: the best probability of completing the current four-line PC. It
is not the survival rate of the V* policy. With `-o`, stdout still contains only the V* number so
existing batch parsers remain simple.

Add `-d` to also write the reveal-conditioned V* policy to `tree_data.js`:

```bash
printf '%s\n' 'OIJLSTZ IJLOSTZ' |
  ./target/release/zxcl-optimal-solver \
    --graph ./graph.bin \
    -d --optimal -f 274072600575
```

`--optimal` requires exactly see 7 and cannot be combined with `-b`, `-t`, or `-w`. The bundled
table is used by default; `--vstar PATH` selects another compatible keyed table.

Optimal mode treats both two-line and four-line PCs as terminal successes. It reveals a new piece
after every placement, including the placement that completes the PC, then scores that resulting
boundary as `1 + V*(next state)`. These extra reveal branches are required because pieces seen late
in one PC become the next PC's queue.

When `-d` is present, the generated file includes `objective="expected_pc"`, `survival_success`,
and `survival_total`. The bundled viewer shows the independently optimized Baseline survival
probability above the V* policy tree, including a percentage rounded to three decimal places. It
uses the objective metadata to display positive expected-PC scores directly; legacy immediate-PC
trees have no objective or survival metadata and retain their original failure-cost display
convention.

An optimal tree can be much larger than an immediate-PC tree. Empty-field queries use an exact
layered-DAG engine; the full oracle below peaks around 1.5–1.6 GiB and writes a 33.1 MB tree on the
development machine. Custom `-f` starts use a general state-memoized fallback because their relative
reveal horizon differs. They are exact too, but a difficult middle layer can use more time and
memory. The late-layer example above is the quickest way to check a build.

### Weighted mode

Weighted mode uses the included `weights.txt`:

```bash
./target/release/zxcl-optimal-solver \
  --graph ./graph.bin \
  --weights ./weights.txt \
  -w
```

Each weight must be an integer in `[0, 2^32]`. As in the original solver, failures have weight
`2^32`; the reported value is the accumulated save cost after subtracting the minimum value from
each applicable weight row. A result of zero means that the minimum possible weight can always be
reached. The Rust implementation keeps these values in exact integer units instead of scaling them
through floating point.

## Command-line options

Options are separate arguments; combined short forms such as `-bo` are intentionally not parsed.

| Option | Meaning |
| --- | --- |
| `-b` | Boolean mode. Reports `1/1` only when the PC is guaranteed and `0/1` otherwise. Faster when hidden pieces exist. Incompatible with `-d` and `-w`. |
| `-d` | Decision mode. Writes the selected policy tree to `tree_data.js`. Considerably slower than a score-only search. Incompatible with `-b` and `-t`. |
| `--optimal` | With `-s 7`, report the V*-optimal expected-PC value and classic current-PC survival probability. Add `-d` to write the V* policy tree. Both 2L and 4L PCs are V* terminals. Incompatible with `-b`, `-t`, and `-w`. |
| `-f HASH` | Start at the field with the given unsigned 40-bit decimal hash instead of the empty field. The hash must exist in the graph. |
| `-m THREADS` | Maximum search threads. Defaults to available hardware parallelism and is clamped to `1..=available_parallelism`. |
| `-o` | Echo each numerical result to stdout. Normal diagnostic output remains on stderr. |
| `-s SEE` | Number of visible pieces, including hold and active. Defaults to `7`; valid values are `2` through `11`. |
| `-t` | Count two-line PCs as successful. In weighted mode they always have weight zero. Incompatible with `-d`. |
| `-v`, `--version` | Print the compatible zxcl optimal solver version and exit. |
| `-w` | Enable composition-save weights from `weights.txt`. Incompatible with `-b`. |
| `--graph PATH` | Read the canonical graph from `PATH` instead of `./graph.bin`. This is a Rust-port extension. |
| `--vstar PATH` | Read the keyed `L0F32V1` table from `PATH` instead of `./vstar_l0_f32.bin`. Used only by `--optimal`. |
| `--weights PATH` | Read weighted-mode data from `PATH` instead of `./weights.txt`. This is a Rust-port extension and only matters with `-w`. |
| `-h`, `--help` | Print the short usage message and exit. |

For compatibility, unknown standalone arguments are ignored. A typo can therefore go unnoticed;
scripts should use only the options listed above.

While the process is running, the following commands may be entered in place of a query:

| Input command | Effect |
| --- | --- |
| `-f HASH` | Change the starting field. |
| `-m THREADS` | Change the thread limit. |
| `-s SEE` | Change the visible-piece count. |

## Graph and field format

The graph contains `15,185,706` fields and `109,562,993` directed placement edges. It has no file
header. Records are stored in strictly increasing field-hash order, and each record is:

```text
5 bytes    field hash, unsigned 40-bit big-endian

for each piece in IJLOSTZ:
    1 byte              number of outgoing placements for this piece
    degree * 3 bytes    target field indices, unsigned 24-bit little-endian
```

The resulting size is:

```text
15,185,706 * (5 + 7) + 109,562,993 * 3 = 510,917,451 bytes
```

Every target is an index into the same sorted field table. Index `0` is the empty field. The final
field is the four-line PC terminal with hash `0xffffffffff`; the two-line PC field has hash
`0x00000fffff`. Piece IDs follow the on-disk loop order `I, J, L, O, S, T, Z`.

A field hash represents the 4-by-10 field as a 40-bit occupancy bitmap. Move any cleared rows to the
bottom first, then read cells left-to-right and top-to-bottom, using `0` for empty and `1` for
filled. For example:

```text
1111110000
1111100000
1111110001
1111111111
```

is:

```text
0b1111110000_1111100000_1111110001_1111111111 = 1083372980223
```

`--graph` changes the file location; it does not make arbitrary graph variants compatible. The
loader expects the canonical counts and terminal fields above. It also rejects unsorted hashes,
out-of-range targets, malformed records, unexpected edge counts, and trailing data.

## V* table format

`vstar_l0_f32.bin` begins with the eight-byte magic `L0F32V1\0`, followed by a little-endian `u64`
record count. Each record is a little-endian `(u32 key, f32 value)` pair, sorted strictly by key.
The key is:

```text
(hold << 24) | (bag_mask << 17) | queue6
```

`queue6` encodes the six visible queue pieces in little-endian base 7, with piece IDs
`I, J, L, O, S, T, Z = 0..6`. The values are position-specific expected future PC counts. The
loader checks the magic, exact record count and file size, key validity and ordering, and finite
non-negative values before searching.

## Compatibility and limitations

- Regular, boolean, decision, weighted, two-line, custom-field, and `see` modes use the original
  input and result conventions.
- Decision output uses the original `tree_data.js` schema and included viewer.
- `--optimal` is a Rust-port extension. It requires `-s 7`, automatically recognizes both PC
  terminal fields, and also reports the baseline four-line survival probability. Add `-d`
  to write positive expected-PC scores with an objective metadata header.
- Empty-field optimal searches use a pruned exact DAG with reveal-vector backups. A non-empty `-f`
  field uses the general full-state fallback so arbitrary relative horizons remain correct.
- Root search may use multiple threads, but numerical results and decision tie-breaking remain
  deterministic.
- The solver can only start from fields present in the pre-generated graph. The graph is limited to
  four-line-PC-capable fields reachable from an empty field by its placement model.
- Timing text naturally differs between runs. Error wording and graph-loading diagnostics are not
  intended to be byte-for-byte identical to the C++ executable.

## Optimization notes

The original graph expands into a very large number of small C++ vectors. This port instead decodes
it into a few contiguous arrays:

- sorted `u64` field hashes;
- one `u32` base offset and seven cumulative `u8` degrees per field;
- one flat `u32` target array.

The input file is mapped read-only while it is validated and decoded, then the mapping is released.
The search hot path can obtain an edge slice directly, without rescanning records or decoding
24-bit targets. Root placements are distributed across scoped worker threads, while deeper recursive
search stays local to avoid nested scheduling overhead. Weighted costs use exact integer arithmetic.

For an empty-field optimal tree, the solver builds a structural
`(depth, field, hold, remaining-bag)` DAG, removes nodes that cannot reach either PC terminal, and
backs up reveal histories as contiguous value vectors. The tree writer keeps only the selected
shallow policy and rebuilds one restricted deep slice at a time, so it can stream the expanded
reveal tree without retaining every branch in memory. Non-empty-field optimal searches retain the
more general full queue state instead.

### Measurements on the development machine

These are approximate measurements from this machine, not portable benchmark claims. The machine
had two Intel Xeon Gold 6126 CPUs (24 physical cores total, SMT off), 251 GiB visible RAM, and Linux
5.15. The graph was already in the OS page cache. The C++ executable was built with GCC 11.4.0 and
`g++ -O3 -std=c++14`; the Rust executable was built with Rust 1.95.0 and
`cargo build --release`. Query times exclude graph loading, and memory is the maximum resident set
reported for the process.

| Operation | Original C++ | Rust port |
| --- | ---: | ---: |
| Load and decode `graph.bin` | about 7.3–8.3 s | about 1.46 s |
| Peak resident memory (max RSS) | about 4.31 GiB | about 1.21 GiB |
| `IJLOSTZ IJLOSTZ`, normal, `-m 1` | about 4.139 s | about 1.716 s |
| `OTJLISO 1`, normal, `-m 1` | about 3.530 s | about 1.933 s |
| `IJLOSTZ IJLOSTZ`, boolean, `-m 24` | about 263 ms | about 152 ms |
| `IJLOSTZ IJLOSTZ`, weighted, `-m 24` | about 509 ms | about 344 ms |
| `SIZTLOJ IJLOSTZ`, `-d --optimal -m 1` | n/a | about 9.6 s solve + 0.77 s tree write |
| Same optimal query, `-m 24` | n/a | about 9.4 s solve + 0.78 s tree write |
| Same optimal query, peak RSS | n/a | about 1.45 GiB (`-m 1`) to 1.59 GiB (`-m 24`) |

CPU topology, allocator behavior, storage, page-cache state, toolchain, and the particular query all
affect these numbers. Run the same inputs on your target machine before drawing conclusions from
them.

## Tests

Unit tests do not require the external graph:

```bash
cargo test --locked
```

After building the release binary and downloading `graph.bin`, run the small compatibility suite:

```bash
scripts/oracle_smoke.sh ./graph.bin ./target/release/zxcl-optimal-solver
```

The smoke script compares known normal, boolean, weighted, two-line, and `see 11` results, checks
legacy decision mode against an exact known `tree_data.js` payload, rejects invalid optimal-mode
combinations, checks score-only optimal mode and its classic survival result, and runs a fast
late-layer V*-optimal tree oracle. The solver checks invoke only the
Rust binary, read the graph, weights, and V* table without modifying them, write decision output in a
temporary directory, and do not need the C++ repository or executable. When `node` is available,
it also checks that the viewer can render a zero-step V* terminal; that optional check is skipped
otherwise.

Set `ZXCL_FULL_ORACLE=1` to additionally solve the empty-field exact oracle and compare its entire
33,123,379-byte tree by SHA-256. `ZXCL_FULL_THREADS` selects its thread count:

```bash
ZXCL_FULL_ORACLE=1 ZXCL_FULL_THREADS=1 \
  scripts/oracle_smoke.sh ./graph.bin ./target/release/zxcl-optimal-solver
```

The expected root is `4353.563114239728`; the tree SHA-256 is
`39775e599fa28d78499ebfce8854645bc3160a6daeb3c5fdbd60a7e620f0b6cf`.

## License

This project is licensed under the GNU General Public License version 3 only (`GPL-3.0-only`). See
[`LICENSE`](LICENSE) and [`NOTICE.md`](NOTICE.md). The original solver design, `weights.txt`, and
browser viewer originate from the upstream GPL-3.0 project linked above. The V* table is the
project-generated data described above. `graph.bin` remains an external download from the upstream
project.
