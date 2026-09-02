//! Empty-field exact see-7 DAG search, adapted from the companion value-analysis engine.
//!
//! The original implementation is GPL-3.0-compatible and is retained structurally here so the
//! public solver can use its validated layered DAG, terminal pruning, and grouped CondFull backup.
//! Field hashes are resolved through this crate's compact graph representation.
//!
//! Terminal placements carry `1 + V*(reset)` from the bundled keyed f32 boundary table.  Values
//! are backed up over reveal-sequence vectors, so maximization happens with exactly the pieces
//! visible at each decision.  Both the five-placement 2-line PC and ten-placement 4-line PC are
//! terminal outcomes.  The tree streamer retains only reachable policy actions and restricts its
//! conditional backups to descendants of the selected depth-4 state.

// `Val` is f64 natively and f32 on wasm; explicit `as f64` conversions are intentionally shared
// by both builds so the algorithm stays one sFource path.
#![allow(clippy::unnecessary_cast)]

use anyhow::{bail, Result};
use hashbrown::{HashMap, HashSet};
#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;
use std::io::{self, Write};

// `maybe_par_collect!(par, slice, iter_method, body)` = `slice.par_iter().method(body).collect()`
// when `par` (native only), else the serial `slice.iter()` form. On wasm32 rayon is neither
// referenced nor linked (par is always false there), so this collapses to the serial path and the
// whole build/prune stays single-threaded — identical results, just no threads.
macro_rules! maybe_par_collect {
    ($par:expr, $slice:expr, $m:ident, $body:expr) => {{
        #[cfg(not(target_arch = "wasm32"))]
        {
            if $par {
                $slice.par_iter().$m($body).collect()
            } else {
                $slice.iter().$m($body).collect()
            }
        }
        #[cfg(target_arch = "wasm32")]
        {
            let _ = &$par;
            $slice.iter().$m($body).collect()
        }
    }};
}

// std::time::Instant panics on wasm32-unknown-unknown; the timings are diagnostics only.
#[cfg(not(target_arch = "wasm32"))]
use std::time::Instant;
#[cfg(target_arch = "wasm32")]
#[derive(Clone, Copy)]
struct Instant;
#[cfg(target_arch = "wasm32")]
impl Instant {
    fn now() -> Self {
        Instant
    }
    fn elapsed(&self) -> std::time::Duration {
        std::time::Duration::ZERO
    }
}

use crate::graph::{Graph as FieldGraph, MAX_HASH, TWO_LINE_HASH};
use crate::optimal::{boundary_key, VStarTable};
use crate::score::piece_char;

// Per-depth ns spent inside cond_backup, summed across leaves (index 10 = the depth-10 seed).
// Diagnostics only, dumped by build_cond_full under VS_CONDBENCH; native builds only.
#[cfg(not(target_arch = "wasm32"))]
thread_local! {
    static COND_DEPTH_NS: std::cell::RefCell<[u64; 11]> = const { std::cell::RefCell::new([0; 11]) };
}

pub type Piece = u8;
const PIECE_COUNT: usize = 7;
const FULL_BAG: u8 = 0b111_1111;

#[inline]
fn pieces(mask: u8) -> impl Iterator<Item = Piece> {
    (0u8..7).filter(move |&piece| mask & (1 << piece) != 0)
}

#[inline]
fn after_reveal(mask: u8, piece: Piece) -> u8 {
    debug_assert!(mask != 0);
    debug_assert!(piece < PIECE_COUNT as u8);
    let remaining = mask & !(1 << piece);
    if remaining == 0 {
        FULL_BAG
    } else {
        remaining
    }
}

/// Terminal boundary evaluator backed by the exact keyed layer-0 V* table.
pub struct ResetEval<'a> {
    table: Option<&'a VStarTable>,
    memo_w4: HashMap<(Piece, u8), f64>,
    pub missing_keys: u64,
}

impl<'a> ResetEval<'a> {
    pub fn new(table: &'a VStarTable) -> Self {
        Self {
            table: Some(table),
            memo_w4: HashMap::new(),
            missing_keys: 0,
        }
    }

    #[inline]
    fn reset_value(&mut self, hold: Piece, queue: [Piece; 6], bag: u8) -> f64 {
        let Some(table) = self.table else {
            return 1.0;
        };
        match table.get(boundary_key(hold, queue, bag)) {
            Some(value) => 1.0 + f64::from(value),
            None => {
                self.missing_keys += 1;
                0.0
            }
        }
    }

    pub fn w4(&mut self, hold: Piece, mask: u8) -> f64 {
        if self.table.is_none() {
            return 1.0;
        }
        if let Some(&value) = self.memo_w4.get(&(hold, mask)) {
            return value;
        }
        let mut queue = [0u8; 6];
        let (sum, count) = self.w4_rec(hold, mask, 0, &mut queue);
        let value = if count == 0 { 0.0 } else { sum / count as f64 };
        self.memo_w4.insert((hold, mask), value);
        value
    }

    fn w4_rec(
        &mut self,
        hold: Piece,
        mask: u8,
        depth: usize,
        queue: &mut [Piece; 6],
    ) -> (f64, u64) {
        if depth == 6 {
            return (self.reset_value(hold, *queue, mask), 1);
        }
        let mut sum = 0.0;
        let mut count = 0;
        for piece in pieces(mask) {
            queue[depth] = piece;
            let (part, n) = self.w4_rec(hold, after_reveal(mask, piece), depth + 1, queue);
            sum += part;
            count += n;
        }
        (sum, count)
    }

    pub fn w_partial(&mut self, hold: Piece, known: &[Piece], mask_after: u8) -> f64 {
        if self.table.is_none() {
            return 1.0;
        }
        debug_assert!(known.len() <= 6);
        if known.len() >= 6 {
            let mut queue = [0u8; 6];
            queue.copy_from_slice(&known[..6]);
            return self.reset_value(hold, queue, mask_after);
        }
        let mut queue = [0u8; 6];
        queue[..known.len()].copy_from_slice(known);
        let (sum, count) = self.w4_rec_from(hold, mask_after, known.len(), &mut queue);
        if count == 0 {
            0.0
        } else {
            sum / count as f64
        }
    }

    fn w4_rec_from(
        &mut self,
        hold: Piece,
        mask: u8,
        depth: usize,
        queue: &mut [Piece; 6],
    ) -> (f64, u64) {
        if depth == 6 {
            return (self.reset_value(hold, *queue, mask), 1);
        }
        let mut sum = 0.0;
        let mut count = 0;
        for piece in pieces(mask) {
            queue[depth] = piece;
            let (part, n) = self.w4_rec_from(hold, after_reveal(mask, piece), depth + 1, queue);
            sum += part;
            count += n;
        }
        (sum, count)
    }

    pub fn w2(&mut self, hold: Piece, q5: Piece, hidden: [Piece; 4], mask: u8) -> f64 {
        let mut sum = 0.0;
        let mut count = 0;
        for piece in pieces(mask) {
            let queue = [q5, hidden[0], hidden[1], hidden[2], hidden[3], piece];
            sum += self.reset_value(hold, queue, after_reveal(mask, piece));
            count += 1;
        }
        if count == 0 {
            0.0
        } else {
            sum / count as f64
        }
    }
}

const FULL_MASK: u8 = 0b111_1111;

/// The DAG is keyed by the 40-bit field HASH directly (no graph FieldId index), so the search is
/// graph.bin-free: the empty box is hash 0 (root), a completed 4LPC is the full box `MAX_HASH`
/// (place() sinks 4 full lines -> all-ones), and a 2LPC is `TWO_LINE_HASH`.
const TERMINAL_HASH: u64 = MAX_HASH;
const ROOT_HASH: u64 = 0;

type NodeId = usize;
/// Element type of the big per-node value vectors. Native keeps f64 (research-exact root
/// values); wasm uses f32 to halve the dominant memory (folds/outputs stay f64 either way).
#[cfg(not(target_arch = "wasm32"))]
type Val = f64;
#[cfg(target_arch = "wasm32")]
type Val = f32;
type ValVec = Vec<Val>;
type ValTable = HashMap<NodeId, ValVec>;
type FoldTable = HashMap<u16, HashMap<NodeId, f64>>;
type CondTables = Vec<HashMap<NodeId, ValVec>>;
type CondBackupOutput = (u8, Vec<(NodeId, Val)>, CondTables, usize);
type EdgeSource<'a> = dyn Fn(u64, u8, &mut Vec<u64>) + 'a;
type SyncEdgeSource<'a> = dyn Fn(u64, u8, &mut Vec<u64>) + Sync + 'a;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct NodeKey {
    depth: u8,  // 0..10
    field: u64, // 40-bit field hash (graph convention); 0=empty box, MAX=4LPC done
    hold: Piece,
    /// Remaining-bag mask (canonical, nonzero). Depths 0..=6: the initial mask (no reveals
    /// consumed). Depths 7..10: the bag after the reveals consumed so far. Keying by MASK instead
    /// of the reveal SEQUENCE merges transpositions: the subtree below (field, hold) depends only
    /// on the DOMAIN of the remaining reveals, so prefixes with equal multiset share one node
    /// whose value vector is indexed by suffix-rank (see SuffixTables).
    mask: u8,
}

#[derive(Clone, Debug)]
struct Node {
    key: NodeKey,
    edges: Vec<NodeId>,
}

#[derive(Clone, Debug)]
struct Dag {
    nodes: Vec<Node>,
    layers: Vec<Vec<NodeId>>, // 0..10
    // Keyed by pack_key(NodeKey): field(40b) | depth(4b) | hold(3b) | mask(7b) = 54 bits.
    // A single-word key hashes much faster than the 12-byte struct (~10^8 lookups per boundary).
    index: HashMap<u64, NodeId>,
    root: NodeId,
}

#[inline]
fn pack_key(k: &NodeKey) -> u64 {
    k.field | ((k.depth as u64) << 40) | ((k.hold as u64) << 44) | ((k.mask as u64) << 47)
}

/// Per-(mask, remaining-length) suffix-enumeration sizes and per-reveal block offsets, matching
/// build_ranges' DFS order exactly (ascending piece index; after_reveal auto-refills the bag).
/// A merged node's value vector is indexed by suffix-rank over (its mask, 10-depth); the child
/// block for reveal q inside a parent's vector starts at off[parent_mask][parent_len][q].
struct SuffixTables {
    cnt: [[u32; 5]; 128], // cnt[m][len] = #reveal sequences of `len` from canonical mask m
    off: [[[u32; 7]; 5]; 128], // off[m][len][q] = block offset of reveal q within (m, len)
}

fn build_suffix_tables() -> Box<SuffixTables> {
    let mut t = Box::new(SuffixTables {
        cnt: [[0; 5]; 128],
        off: [[[0; 7]; 5]; 128],
    });
    for m in 0..128 {
        t.cnt[m][0] = 1;
    }
    for len in 1..=4usize {
        for m in 1..=127usize {
            let mut acc = 0u32;
            for q in 0..7usize {
                if m & (1 << q) != 0 {
                    t.off[m][len][q] = acc;
                    acc += t.cnt[after_reveal(m as u8, q as u8) as usize][len - 1];
                }
            }
            t.cnt[m][len] = acc;
        }
    }
    t
}

#[derive(Clone, Copy, Debug)]
pub struct SeqRange {
    pub start: u32,
    pub len: u32,
}

pub struct VsResult {
    pub root_value: f64,
    pub missing_keys: u64,
    pub nodes_total: usize,
    pub nodes_pruned: usize,
    pub leaf_count: usize,
    /// (depth-1 node field, hold after move, placed piece, score)
    pub first_moves: Vec<FirstMove>,
    retained: Retained,
}

#[derive(Clone, Debug)]
pub struct FirstMove {
    pub field: u64,
    pub hold: Piece,
    pub placed: Piece,
    pub score: f64,
}

struct Retained {
    dag: Dag,
    full_hidden_packs: Vec<u16>,
    ranges: HashMap<(u8, u16), SeqRange>, // (prefix_len 0..=4, pack) -> leaf range
    vals: Vec<ValTable>,                  // depth 4..=10 at index depth
    folds: Vec<FoldTable>,                // index 1..=3: value_d keyed by len-d prefix pack
    initial_mask: u8,
    visible: [Piece; 6],
    two_line_field: u64,
}

pub struct SearchInput<'a> {
    /// Only used by the reference edge path (`edge_ids: None`). A graph-free run passes `None`
    /// here and supplies `edge_ids`, so the whole search touches no graph.bin.
    pub graph: Option<&'a FieldGraph>,
    pub hold: Piece,
    pub visible: [Piece; 6],
    pub mask: u8,
    pub reset: ResetEval<'a>,
    /// Edge source keyed by field HASH (the WASM bot's movegen+ProjFilter): given
    /// (field_hash, piece), APPEND the child field HASHES into the provided buffer (which the
    /// caller clears first — filling a reused buffer avoids per-edge allocation). None = graph.edges.
    pub edge_ids: Option<&'a EdgeSource<'a>>,
    /// Optional SYNC edge source for the PARALLEL build (rayon fans the per-node movegen across
    /// threads). When Some, the build runs in parallel and this replaces `edge_ids`/`graph`; the
    /// result is bit-identical to the serial build. Thread count = the ambient rayon pool.
    pub par_edge: Option<&'a SyncEdgeSource<'a>>,
    /// Skip the BLIND solve (seed/backup/folds): build+prune only. For the see7-exact batch path,
    /// which rescores everything from CondEval/CondFull anyway — the blind tables would be dead
    /// weight (root_value/first_moves come back 0/empty; sample/optimal walks are unusable).
    pub skip_solve: bool,
}

pub fn value_search(mut input: SearchInput<'_>) -> VsResult {
    let verbose = std::env::var("VS_VERBOSE")
        .map(|s| s != "0")
        .unwrap_or(false);
    let initial_mask = canonical_mask(input.mask);
    let two_line_field = TWO_LINE_HASH;

    // Hidden-sequence leaf ranges (DFS order, matching build_hidden_prefixes).
    let mut ranges: HashMap<(u8, u16), SeqRange> = HashMap::new();
    let mut next_leaf = 0u32;
    build_ranges(initial_mask, 0, 0, &mut next_leaf, &mut ranges);
    let leaf_count = next_leaf as usize;

    let mut full_hidden_packs = vec![0u16; leaf_count];
    for (&(len, pack), &r) in &ranges {
        if len == 4 {
            full_hidden_packs[r.start as usize] = pack;
        }
    }

    // Suffix-rank bookkeeping for transposition-merged nodes (depths 7..10).
    let suffix = build_suffix_tables();
    debug_assert_eq!(suffix.cnt[initial_mask as usize][4] as usize, leaf_count);

    let _t = Instant::now();
    let (dag, nodes_total, t_build) = {
        // Scope the UNPRUNED dag so it's freed right after pruning (shadowing alone would keep
        // both DAGs alive through the whole solve), and drop its build-only node index first.
        let mut full = build_dag(
            input.graph,
            input.edge_ids,
            input.par_edge,
            input.hold,
            input.visible,
            initial_mask,
            two_line_field,
        );
        let nodes_total = full.nodes.len();
        let t_build = _t.elapsed();
        full.index = HashMap::new();
        (
            prune_to_terminal_reachable(&full, two_line_field, input.par_edge.is_some()),
            nodes_total,
            t_build,
        )
    };
    let nodes_pruned = dag.nodes.len();
    let t_prune = _t.elapsed() - t_build;
    if verbose {
        eprintln!(
        "value-search: nodes {} -> {} (terminal-reachable), hidden_leaves={} | build {:.2}ms prune {:.2}ms",
        nodes_total, nodes_pruned, leaf_count, t_build.as_secs_f64() * 1e3, t_prune.as_secs_f64() * 1e3
    );
    }
    let _t = Instant::now();
    // Parallel solve rides the same switch as the parallel build.
    let par = input.par_edge.is_some();

    let vec_len = |key: &NodeKey| -> usize {
        if key.depth <= 6 {
            leaf_count
        } else {
            suffix.cnt[key.mask as usize][(10 - key.depth) as usize] as usize
        }
    };

    // ---- seed depth 10 (4LPC terminals) ----
    let mut vals: Vec<ValTable> = vec![ValTable::new(); 11];
    if input.skip_solve {
        // Build+prune only: the caller rescoring via CondEval/CondFull never reads the blind
        // tables. Return the structural result with empty values.
        return VsResult {
            root_value: 0.0,
            missing_keys: input.reset.missing_keys,
            nodes_total,
            nodes_pruned,
            leaf_count,
            first_moves: Vec::new(),
            retained: Retained {
                dag,
                full_hidden_packs,
                ranges,
                vals,
                folds: vec![FoldTable::new(); 4],
                initial_mask,
                visible: input.visible,
                two_line_field,
            },
        };
    }
    {
        let table = &mut vals[10];
        for &id in &dag.layers[10] {
            let key = dag.nodes[id].key;
            if key.field != TERMINAL_HASH {
                continue;
            }
            // key.mask at depth 10 IS the bag after all four reveals (mask4).
            let v = input.reset.w4(key.hold, key.mask);
            if v > 0.0 {
                table.insert(id, vec![v as Val]); // suffix-len 0 -> vector of 1
            }
        }
        if verbose {
            eprintln!("value-search: depth10 live={}", table.len());
        }
    }

    // ---- elementwise-max backup depths 9..4, with 2LPC injection at depth 5 ----
    // FORWARD-edge, per-parent: each parent folds its own children's vectors into a private
    // vector, so the depth is embarrassingly parallel (no reverse index, no write contention).
    // f64 max is exact (no rounding), so serial/parallel/any-order produce IDENTICAL values.
    for depth in (4..10u8).rev() {
        let prev = &vals[(depth + 1) as usize];
        let compute = |&parent_id: &NodeId| -> Option<(NodeId, ValVec)> {
            let parent_key = dag.nodes[parent_id].key;
            let mut dst: ValVec = Vec::new();
            for &child_id in &dag.nodes[parent_id].edges {
                let Some(child_vec) = prev.get(&child_id) else {
                    continue;
                };
                // Reveal-consuming transitions (child depth >= 7): the child's block sits at the
                // parent's suffix offset of the revealed piece q. q is unique from the mask diff
                // (pm\cm = {q}; empty diff means the singleton bag refilled, so q = that piece).
                let offset = if depth >= 6 {
                    let pm = parent_key.mask;
                    let d = pm & !dag.nodes[child_id].key.mask;
                    let q = if d != 0 {
                        d.trailing_zeros()
                    } else {
                        pm.trailing_zeros()
                    };
                    suffix.off[pm as usize][(10 - depth) as usize][q as usize] as usize
                } else {
                    0
                };
                if dst.is_empty() {
                    dst = vec![0.0; vec_len(&parent_key)];
                }
                // branchless elementwise max over the aligned slice -> auto-vectorizes (AVX).
                let d = &mut dst[offset..offset + child_vec.len()];
                for (dd, &cv) in d.iter_mut().zip(child_vec.iter()) {
                    *dd = if cv > *dd { cv } else { *dd };
                }
            }
            if dst.is_empty() {
                None
            } else {
                Some((parent_id, dst))
            }
        };
        let layer = &dag.layers[depth as usize];
        let pairs: Vec<(NodeId, ValVec)> = maybe_par_collect!(par, layer, filter_map, compute);
        let mut next: ValTable = ValTable::with_capacity(pairs.len());
        for (pid, v) in pairs {
            next.insert(pid, v);
        }
        if depth == 5 {
            // 2LPC terminals: childless two-line nodes get their reset value directly.
            let q5 = input.visible[5];
            let mut memo: HashMap<(Piece, u16), f64> = HashMap::new();
            for &id in &dag.layers[5] {
                let key = dag.nodes[id].key;
                if key.field != two_line_field {
                    continue;
                }
                let vec = next.entry(id).or_insert_with(|| vec![0.0; leaf_count]);
                for leaf in 0..leaf_count {
                    let pack = full_hidden_packs[leaf];
                    let v = *memo.entry((key.hold, pack)).or_insert_with(|| {
                        let h = [
                            get_hidden(pack, 0),
                            get_hidden(pack, 1),
                            get_hidden(pack, 2),
                            get_hidden(pack, 3),
                        ];
                        let mask4 = mask_after_hidden_prefix(initial_mask, pack, 4);
                        input.reset.w2(key.hold, q5, h, mask4)
                    });
                    if (v as Val) > vec[leaf] {
                        vec[leaf] = v as Val;
                    }
                }
            }
        }
        if verbose {
            eprintln!("value-search: depth{} live={}", depth, next.len());
        }
        vals[depth as usize] = next;
    }

    // ---- folds: average h4..h1 at depths 3..0 (see7 information timing) ----
    // value_d[prefix_pack(h1..h_d)][node] = max_child avg_{h_{d+1}} value_{d+1}[..][child]
    let folds = build_folds(
        &dag,
        &vals[4],
        &full_hidden_packs,
        leaf_count,
        initial_mask,
        par,
        verbose,
    );

    let root_value = folds[0]
        .get(&0u16)
        .and_then(|t| t.get(&dag.root))
        .copied()
        .unwrap_or(0.0);

    // Best first moves from value1 (average over the unrevealed h1).
    let h1_count = pieces(initial_mask).count() as f64;
    let mut first_moves: Vec<FirstMove> = Vec::new();
    for &child_id in &dag.nodes[dag.root].edges {
        let child_key = dag.nodes[child_id].key;
        let mut sum = 0.0;
        for h1 in pieces(initial_mask) {
            let pack = set_hidden(0, 0, h1);
            if let Some(v) = folds[1].get(&pack).and_then(|t| t.get(&child_id)) {
                sum += v;
            }
        }
        let root_key = dag.nodes[dag.root].key;
        let placed = if child_key.hold == root_key.hold {
            input.visible[0]
        } else {
            root_key.hold
        };
        first_moves.push(FirstMove {
            field: child_key.field,
            hold: child_key.hold,
            placed,
            score: sum / h1_count,
        });
    }
    first_moves.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
    if verbose {
        eprintln!(
            "value-search: solve(rev+backup+fold) {:.2}ms",
            _t.elapsed().as_secs_f64() * 1e3
        );
    }

    let missing_keys = input.reset.missing_keys;
    VsResult {
        root_value,
        missing_keys,
        nodes_total,
        nodes_pruned,
        leaf_count,
        first_moves,
        retained: Retained {
            dag,
            full_hidden_packs,
            ranges,
            vals,
            folds,
            initial_mask,
            visible: input.visible,
            two_line_field,
        },
    }
}

/// Folds 3..0 from a depth-4 per-leaf value table (extracted from value_search so the
/// see7-exact conditional path can rebuild EXACT shallow folds by swapping in conditional
/// depth-4 columns — identical code path, identical f64 accumulation order).
fn build_folds(
    dag: &Dag,
    vals4: &ValTable,
    full_hidden_packs: &[u16],
    leaf_count: usize,
    initial_mask: u8,
    par: bool,
    verbose: bool,
) -> Vec<FoldTable> {
    let mut folds: Vec<FoldTable> = vec![FoldTable::new(); 4];

    // depth 3: consume depth-4 range vectors, average h4. FORWARD per-parent (parallel-safe):
    // per (pack3, child) the sum runs over the child's vector in ascending leaf order — the same
    // accumulation order as the old reverse-index version, so values are identical.
    {
        // Per-leaf depth-3 prefix pack and per-pack h4 branch counts, precomputed once.
        let mut pack3_of = vec![0u16; leaf_count];
        for (i, &fp) in full_hidden_packs.iter().enumerate() {
            pack3_of[i] = prefix_pack(fp, 3);
        }
        let mut branches3: HashMap<u16, f64> = HashMap::new();
        for &p3 in &pack3_of {
            branches3.entry(p3).or_insert_with(|| {
                pieces_in_mask(mask_after_hidden_prefix(initial_mask, p3, 3)).len() as f64
            });
        }
        let compute = |&parent_id: &NodeId| -> (NodeId, Vec<(u16, f64)>) {
            let mut per: HashMap<(u16, NodeId), f64> = HashMap::new();
            for &child_id in &dag.nodes[parent_id].edges {
                if let Some(child_vec) = vals4.get(&child_id) {
                    for (i, &cv) in child_vec.iter().enumerate() {
                        if cv <= 0.0 {
                            continue;
                        }
                        *per.entry((pack3_of[i], child_id)).or_insert(0.0) += cv as f64;
                    }
                }
            }
            // average over h4, then max over children (exact; order-free).
            let mut best: HashMap<u16, f64> = HashMap::new();
            for ((p3, _child), sum) in per {
                let avg = sum / branches3[&p3];
                match best.get_mut(&p3) {
                    Some(b) => {
                        if avg > *b {
                            *b = avg;
                        }
                    }
                    None => {
                        best.insert(p3, avg);
                    }
                }
            }
            (parent_id, best.into_iter().collect())
        };
        let layer = &dag.layers[3];
        let results: Vec<(NodeId, Vec<(u16, f64)>)> = maybe_par_collect!(par, layer, map, compute);
        let out = &mut folds[3];
        for (pid, list) in results {
            for (p3, val) in list {
                out.entry(p3).or_default().insert(pid, val);
            }
        }
        if verbose {
            eprintln!("value-search: fold depth3 tables={}", folds[3].len());
        }
    }

    // Mini reverse index for the shallow folds (children at depths 1..=3, parents at 0..=2);
    // within a layer parents are id-ascending, matching the old full reverse index's order.
    let mut mini_rev: Vec<Vec<NodeId>> = vec![Vec::new(); dag.nodes.len()];
    for d in 0..=2usize {
        for &pid in &dag.layers[d] {
            for &c in &dag.nodes[pid].edges {
                mini_rev[c].push(pid);
            }
        }
    }

    // depths 2..0. Child tables are iterated in SORTED pack order so the f64 accumulation order
    // is canonical (deterministic regardless of hash-map insertion history).
    for depth in (0..3u8).rev() {
        let child_tables = std::mem::take(&mut folds[(depth + 1) as usize]);
        let mut sums: HashMap<(u16, NodeId, NodeId), f64> = HashMap::new();
        let mut packs: Vec<u16> = child_tables.keys().copied().collect();
        packs.sort_unstable();
        for &child_pack in &packs {
            let table = &child_tables[&child_pack];
            let parent_pack = prefix_pack(child_pack, depth);
            for (&child_id, &cv) in table {
                if cv <= 0.0 {
                    continue;
                }
                for &parent_id in &mini_rev[child_id] {
                    if dag.nodes[parent_id].key.depth != depth {
                        continue;
                    }
                    *sums
                        .entry((parent_pack, parent_id, child_id))
                        .or_insert(0.0) += cv;
                }
            }
        }
        let out = fold_from_sums(sums, initial_mask, depth);
        if verbose {
            eprintln!("value-search: fold depth{} tables={}", depth, out.len());
        }
        if depth < 3 {
            folds[(depth + 1) as usize] = child_tables;
        }
        folds[depth as usize] = out;
    }
    folds
}

fn fold_from_sums(
    sums: HashMap<(u16, NodeId, NodeId), f64>,
    initial_mask: u8,
    depth: u8,
) -> FoldTable {
    // Denominator: number of h_{depth+1} choices after the known prefix = mask size at
    // that point (deterministic per level given the initial mask).
    let mut out: FoldTable = FoldTable::new();
    for ((parent_pack, parent_id, _child_id), sum) in sums {
        let mask = mask_after_hidden_prefix(initial_mask, parent_pack, depth);
        let branches = pieces_in_mask(mask).len() as f64;
        let avg = sum / branches;
        let table = out.entry(parent_pack).or_default();
        match table.get_mut(&parent_id) {
            Some(existing) => {
                if avg > *existing {
                    *existing = avg;
                }
            }
            None => {
                table.insert(parent_id, avg);
            }
        }
    }
    out
}

/* -------------------------------------------------------------------------- */
/* Sample-play walk                                                            */
/* -------------------------------------------------------------------------- */

#[derive(Clone)]
pub struct WalkStep {
    pub depth: u8,
    pub placed: Piece,
    pub hold_after: Piece,
    pub field_before: u64,
    pub field_after: u64,
    pub score: f64,
    pub revealed: Option<Piece>, // the piece revealed after this placement (h_{depth+1})
}

pub enum WalkOutcome {
    Pc4L { reset_hold: Piece },
    Pc2L { reset_hold: Piece },
    Death { at_depth: u8 },
}

pub struct Walk {
    pub steps: Vec<WalkStep>,
    pub outcome: WalkOutcome,
    pub hidden: [Piece; 4],
}

/// One candidate placement (a DAG edge) at an analysis node, with its expected value.
#[derive(Clone, Debug)]
pub struct AnalysisCand {
    pub edge: usize, // index into the parent's edge list (stable selector for navigation)
    pub placed: Piece, // the tetromino this placement drops
    pub hold_after: Piece, // hold after the move
    pub field_before: u64,
    pub field_after: u64,
    pub score: f64, // expected consecutive PCs from here (0 = dead line for this reveal)
    pub best: bool, // the policy's argmax move
    /// DAG NodeId of the child (0 when the cand was built from oracle JSON, not a local search);
    /// lets CondEval re-score candidates without re-walking edges.
    pub child_id: usize,
}

/// A navigable analysis position: the current node plus its ranked candidate moves, the line
/// taken to reach it, and the valid reveal options (for reveal what-if).
pub struct AnalysisNode {
    pub depth: u8,
    pub field: u64,
    pub hold: Piece,
    pub active: Piece, // piece placed if NOT swapping hold, at this depth
    pub terminal: u8,  // 0=in-progress 1=4LPC 2=2LPC 3=dead (no positive line for this reveal)
    pub best_score: f64,
    pub root_value: f64,
    pub path_steps: Vec<AnalysisCand>, // the chosen line root->here (board reconstruction/breadcrumb)
    pub cands: Vec<AnalysisCand>,      // candidates at the current node, sorted by score desc
    pub reveal_options: [Vec<Piece>; 4], // valid pieces for h1..h4 (bag process)
    pub visible: [Piece; 6],
}

impl VsResult {
    pub fn two_line_field(&self) -> u64 {
        self.retained.two_line_field
    }

    /// Sample h1..h4 from the bag process using the given RNG state, then follow the
    /// optimal policy (argmax over children of the expected value GIVEN the information
    /// revealed so far — exactly the semantics the solver optimized).
    pub fn sample_walk(&self, rng: &mut u64) -> Walk {
        let r = &self.retained;
        let mut mask = r.initial_mask;
        let mut hidden = [0u8; 4];
        for h in hidden.iter_mut() {
            let count = mask.count_ones() as usize;
            let index = (next_rand(rng) as usize) % count;
            *h = pieces(mask)
                .nth(index)
                .expect("a non-empty bag must yield a piece");
            mask = after_reveal(mask, *h);
        }
        self.optimal_walk(hidden)
    }

    /// Follow the optimal policy for a GIVEN in-loop reveal sequence h1..h4 (supplied by
    /// the environment's piece stream). Each decision uses only the reveals visible before
    /// that placement (depth d sees h1..h_min(d,4)), matching the solver's information
    /// timing exactly. h1..h4 are the pieces the loop actually places at depth 6..9; the
    /// reset queue (h5..) is beyond the policy and is filled by the caller.
    pub fn optimal_walk(&self, hidden: [Piece; 4]) -> Walk {
        let r = &self.retained;
        let full_pack = {
            let mut p = 0u16;
            for (i, &h) in hidden.iter().enumerate() {
                p = set_hidden(p, i as u8, h);
            }
            p
        };
        let leaf = r.ranges[&(4, full_pack)].start as usize;

        let mut steps = Vec::new();
        let mut node = r.dag.root;
        for depth in 0..10u8 {
            let parent_key = r.dag.nodes[node].key;
            let mut best: Option<(NodeId, f64)> = None;
            for &child_id in &r.dag.nodes[node].edges {
                let score = self.child_score(child_id, depth + 1, &hidden, leaf);
                if score > best.map_or(0.0, |b| b.1) {
                    best = Some((child_id, score));
                }
            }
            let Some((child_id, score)) = best else {
                return Walk {
                    steps,
                    outcome: WalkOutcome::Death { at_depth: depth },
                    hidden,
                };
            };
            let child_key = r.dag.nodes[child_id].key;
            let placed = if child_key.hold == parent_key.hold {
                active_piece(&r.visible, &hidden, depth)
            } else {
                parent_key.hold
            };
            let revealed = if depth < 4 {
                Some(hidden[depth as usize])
            } else {
                None
            };
            steps.push(WalkStep {
                depth,
                placed,
                hold_after: child_key.hold,
                field_before: parent_key.field,
                field_after: child_key.field,
                score,
                revealed,
            });
            if child_key.field == r.two_line_field {
                return Walk {
                    steps,
                    outcome: WalkOutcome::Pc2L {
                        reset_hold: child_key.hold,
                    },
                    hidden,
                };
            }
            node = child_id;
        }
        let final_key = r.dag.nodes[node].key;
        Walk {
            steps,
            outcome: WalkOutcome::Pc4L {
                reset_hold: final_key.hold,
            },
            hidden,
        }
    }

    /// Expected value of a depth-d child under the information available before its
    /// placement (h1..h_{d-1} revealed; averages over the rest).
    fn child_score(
        &self,
        child_id: NodeId,
        child_depth: u8,
        hidden: &[Piece; 4],
        leaf: usize,
    ) -> f64 {
        let r = &self.retained;
        match child_depth {
            1..=3 => {
                // folded tables keyed by len-child_depth prefix; the last prefix piece
                // h_{child_depth} is not yet revealed at decision time -> average it.
                let known = child_depth - 1;
                let mut prefix = 0u16;
                for i in 0..known {
                    prefix = set_hidden(prefix, i, hidden[i as usize]);
                }
                let mut mask = r.initial_mask;
                for i in 0..known {
                    mask = after_reveal(mask, hidden[i as usize]);
                }
                let mut sum = 0.0;
                let mut n = 0.0;
                for h in pieces(mask) {
                    let pack = set_hidden(prefix, known, h);
                    if let Some(v) = r.folds[child_depth as usize]
                        .get(&pack)
                        .and_then(|t| t.get(&child_id))
                    {
                        sum += v;
                    }
                    n += 1.0;
                }
                if n == 0.0 {
                    0.0
                } else {
                    sum / n
                }
            }
            4..=10 => {
                let key = r.dag.nodes[child_id].key;
                if key.depth <= 6 {
                    return r.vals[key.depth as usize]
                        .get(&child_id)
                        .map(|v| v[leaf] as f64)
                        .unwrap_or(0.0);
                }
                // Merged node: its vector is indexed by suffix-rank. Rebuild the reveal prefix
                // from `hidden`; a mask mismatch means this child is on a different reveal branch.
                let plen = key.depth - 6;
                let mut pack = 0u16;
                let mut m = r.initial_mask;
                for i in 0..plen {
                    pack = set_hidden(pack, i, hidden[i as usize]);
                    m = after_reveal(m, hidden[i as usize]);
                }
                if m != key.mask {
                    return 0.0;
                }
                let block = r.ranges[&(plen, pack)];
                r.vals[key.depth as usize]
                    .get(&child_id)
                    .map(|v| v[leaf - block.start as usize] as f64)
                    .unwrap_or(0.0)
            }
            _ => 0.0,
        }
    }

    /// Decision-time expected value of a child, respecting reveal timing. For a depth-3 node
    /// (child_depth 4) the piece h4 opens only AS A RESULT of that placement, so it is NOT known
    /// when the move is chosen — average over it (like the folded tables do for h1..h3), rather
    /// than scoring against the one realized h4. Otherwise (depths 0..2 fold h1..h3; depths 4+
    /// have h4 already revealed) use the ordinary child_score.
    fn decision_score(&self, child: NodeId, child_depth: u8, hidden: &[Piece; 4]) -> f64 {
        let r = &self.retained;
        if child_depth == 4 {
            let mut prefix = 0u16;
            for i in 0..3 {
                prefix = set_hidden(prefix, i, hidden[i as usize]);
            }
            let mask = mask_after_hidden_prefix(r.initial_mask, prefix, 3);
            let mut sum = 0.0;
            let mut n = 0.0;
            for h4 in pieces(mask) {
                let mut hid = *hidden;
                hid[3] = h4;
                if let Some(lf) = self.full_leaf(hid) {
                    sum += self.child_score(child, 4, &hid, lf);
                    n += 1.0;
                }
            }
            if n > 0.0 {
                sum / n
            } else {
                0.0
            }
        } else {
            let leaf = self.full_leaf(*hidden).unwrap_or(0);
            self.child_score(child, child_depth, hidden, leaf)
        }
    }

    /// Leaf index of the full reveal sequence `hidden` (None if not a valid bag sequence).
    pub fn full_leaf(&self, hidden: [Piece; 4]) -> Option<usize> {
        let mut pack = 0u16;
        for (i, &h) in hidden.iter().enumerate() {
            pack = set_hidden(pack, i as u8, h);
        }
        self.retained
            .ranges
            .get(&(4, pack))
            .map(|r| r.start as usize)
    }

    /// Navigate the loop DAG: replay `path` (edge indices from the root) under the reveal
    /// sequence `hidden`, and report the resulting node with its ranked candidate moves. This
    /// is the analysis primitive — every alternative placement is a candidate, and changing
    /// `hidden` is the reveal what-if. All scores are the decision-time expected value.
    pub fn analyze(&self, path: &[usize], hidden: [Piece; 4]) -> AnalysisNode {
        let r = &self.retained;
        let leaf = self.full_leaf(hidden).unwrap_or(0);

        let _ = leaf;
        let make_cand =
            |edge: usize, parent: NodeId, child: NodeId, child_depth: u8| -> AnalysisCand {
                let pk = r.dag.nodes[parent].key;
                let ck = r.dag.nodes[child].key;
                let active = active_piece(&r.visible, &hidden, pk.depth);
                let placed = if ck.hold == pk.hold { active } else { pk.hold };
                let score = self.decision_score(child, child_depth, &hidden);
                AnalysisCand {
                    edge,
                    placed,
                    hold_after: ck.hold,
                    field_before: pk.field,
                    field_after: ck.field,
                    score,
                    best: false,
                    child_id: child,
                }
            };

        // Replay the chosen line, recording each step.
        let mut node = r.dag.root;
        let mut path_steps = Vec::new();
        for &e in path {
            let edges = r.dag.nodes[node].edges.clone();
            if e >= edges.len() {
                break;
            }
            let child = edges[e];
            let cd = r.dag.nodes[node].key.depth + 1;
            path_steps.push(make_cand(e, node, child, cd));
            node = child;
        }

        let key = r.dag.nodes[node].key;
        let depth = key.depth;
        let active = active_piece(&r.visible, &hidden, depth);

        let edges = r.dag.nodes[node].edges.clone();
        let mut cands = Vec::new();
        for (ei, &child) in edges.iter().enumerate() {
            // At reveal depths (>=6) the DAG branches over EVERY piece that COULD be revealed here;
            // once the reveal is known (the passed `hidden`), the other branches are impossible, so
            // prune every edge whose reveal disagrees with the actual hidden. Under transposition
            // merging the reveal is identified by the child's mask (unique per reveal from a node).
            if depth >= 6 {
                let idx = (depth - 6) as usize;
                if r.dag.nodes[child].key.mask != after_reveal(key.mask, hidden[idx]) {
                    continue;
                }
            }
            cands.push(make_cand(ei, node, child, depth + 1));
        }
        let mut best_pos = usize::MAX;
        let mut best_s = 0.0f64;
        for (i, c) in cands.iter().enumerate() {
            if c.score > best_s {
                best_s = c.score;
                best_pos = i;
            }
        }
        if best_pos != usize::MAX {
            cands[best_pos].best = true;
        }
        cands.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let terminal = if key.field == r.two_line_field && depth == 5 {
            2
        } else if depth == 10 {
            1
        } else if cands.is_empty() || best_s <= 0.0 {
            3
        } else {
            0
        };

        let opts = |k: u8| -> Vec<Piece> {
            let mut p = 0u16;
            for i in 0..k {
                p = set_hidden(p, i, hidden[i as usize]);
            }
            pieces_in_mask(mask_after_hidden_prefix(r.initial_mask, p, k))
        };
        let reveal_options = [opts(0), opts(1), opts(2), opts(3)];

        AnalysisNode {
            depth,
            field: key.field,
            hold: key.hold,
            active,
            terminal,
            best_score: best_s,
            root_value: self.root_value,
            path_steps,
            cands,
            reveal_options,
            visible: r.visible,
        }
    }
}

/* -------------------------------------------------------------------------- */
/* See7-exact conditional evaluation (depths >= 4)                              */
/*                                                                             */
/* The plain backup averages the WHOLE reset queue at the terminal (w4), but   */
/* physically one piece is revealed after EVERY placement, so decisions at     */
/* depth d >= 5 have already seen the next PC's first r4..r_{d-1} pieces and   */
/* the optimal policy adapts to them (E[max] >= max[E]) — this is the fullV    */
/* MDP's information timing. Since r4.. never touches an in-loop FIELD         */
/* transition (only the reset lookup and when it becomes known), the exact fix */
/* is cheap: once h1..h4 are realized the reveal-branching collapses to a      */
/* single-line field slice, and we back up per-node VECTORS indexed by the     */
/* r4..r_{d-1} sequence rank (same DFS order as SuffixTables, extended to      */
/* length 6). Terminal seed = the fully-known reset lookup; 2LPC's h5 IS r4 so */
/* it conditions too. Query at decision depth D uses the REALIZED prefix       */
/* next_known[0..D-4] and averages the not-yet-revealed remainder (uniform:    */
/* the |bag| chain is fixed by popcount, so all completions are equiprobable). */
/* -------------------------------------------------------------------------- */

/// Suffix count/offset tables like SuffixTables but for lengths 0..=6 (r4..r9).
struct CondSuffix {
    cnt: [[u32; 7]; 128],
    off: [[[u32; 7]; 7]; 128],
}

fn build_cond_suffix() -> Box<CondSuffix> {
    let mut t = Box::new(CondSuffix {
        cnt: [[0; 7]; 128],
        off: [[[0; 7]; 7]; 128],
    });
    for m in 0..128 {
        t.cnt[m][0] = 1;
    }
    for len in 1..=6usize {
        for m in 1..=127usize {
            let mut acc = 0u32;
            for q in 0..7usize {
                if m & (1 << q) != 0 {
                    t.off[m][len][q] = acc;
                    acc += t.cnt[after_reveal(m as u8, q as u8) as usize][len - 1];
                }
            }
            t.cnt[m][len] = acc;
        }
    }
    t
}

/// Conditional value tables for one realized h1..h4: vals[d][node] = vector over the
/// r4..r_{d-1} sequence rank (length cnt[mask4][d-4]), for d in 4..=10.
pub struct CondEval {
    mask4: u8,
    suffix: Box<CondSuffix>,
    vals: Vec<HashMap<NodeId, ValVec>>, // indexed by depth; 0..4 unused
    slice_nodes: usize,
}

/// Reusable scratch for repeated conditional backups (one per hidden completion): dense per-node
/// vector slots (no HashMap on the hot path; cleared via the touched lists) and a
/// (hold, mask4)-keyed depth-10 seed cache (the fully-known reset vector depends on nothing
/// else, and mask4 repeats across leaves).
struct CondScratch {
    cur: Vec<ValVec>,
    nxt: Vec<ValVec>,
    cur_used: Vec<NodeId>,
    nxt_used: Vec<NodeId>,
    seeds: HashMap<(Piece, u8), ValVec>,
    /// Retired dst vectors, reused across nodes/leaves — the leaf loop would otherwise
    /// malloc/free one vector per live node per leaf (hundreds of millions per PC).
    pool: Vec<ValVec>,
    pool_bytes: usize,
}

impl CondScratch {
    fn new(n_nodes: usize) -> Self {
        CondScratch {
            cur: vec![Vec::new(); n_nodes],
            nxt: vec![Vec::new(); n_nodes],
            cur_used: Vec::new(),
            nxt_used: Vec::new(),
            seeds: HashMap::new(),
            pool: Vec::new(),
            pool_bytes: 0,
        }
    }
}

const COND_POOL_MAX_BYTES: usize = 64 * 1024 * 1024;

fn cond_pool_take(pool: &mut Vec<ValVec>, pool_bytes: &mut usize) -> ValVec {
    let value = pool.pop().unwrap_or_default();
    *pool_bytes = pool_bytes.saturating_sub(value.capacity() * std::mem::size_of::<Val>());
    value
}

fn cond_pool_put(pool: &mut Vec<ValVec>, pool_bytes: &mut usize, value: ValVec) {
    let bytes = value.capacity() * std::mem::size_of::<Val>();
    if bytes <= COND_POOL_MAX_BYTES.saturating_sub(*pool_bytes) {
        *pool_bytes += bytes;
        pool.push(value);
    }
}

/// Exact shallow tables for depths 0..3: conditional depth-4 columns for EVERY hidden
/// completion, folded by the ordinary see7 folds — so d0..3 candidate scores (and the root
/// value) carry the future next-queue-exploiting value exactly like the fullV MDP.
pub struct CondFull {
    pub root_value: f64,
    vals4: ValTable, // node -> per-leaf conditional d4 value (same shape as vals[4])
    folds: Vec<FoldTable>, // exact folds 0..=3
}

impl VsResult {
    /// Build the conditional tables for a realized reveal sequence. `reset` must wrap the same
    /// boundary value table the search was built with.
    pub fn build_cond(&self, hidden: [Piece; 4], reset: &mut ResetEval) -> CondEval {
        let suffix = build_cond_suffix();
        let mut scratch = CondScratch::new(self.retained.dag.nodes.len());
        let buckets = self.reveal_layer_buckets();
        let blocks = self.reveal_edge_blocks();
        let (mask4, _d4, vals, slice_nodes) = self.cond_backup(
            hidden,
            reset,
            &suffix,
            &mut scratch,
            &buckets,
            &blocks,
            true,
            4,
            None,
        );
        CondEval {
            mask4,
            suffix,
            vals,
            slice_nodes,
        }
    }

    /// Per-node reveal-block offsets for depths 6..=9: children are emitted (and prune-remapped)
    /// as CONTIGUOUS blocks per revealed piece in ascending order, so [u32;8] start offsets
    /// (offs[p]..offs[p+1], offs[7]=len... see build) let each leaf slice exactly its realized
    /// block — no per-child mask loads. Dense-indexed by NodeId (untouched nodes stay zeroed).
    fn reveal_edge_blocks(&self) -> Vec<[u32; 8]> {
        let r = &self.retained;
        let mut out = vec![[0u32; 8]; r.dag.nodes.len()];
        for depth in 6..=9usize {
            for &id in &r.dag.layers[depth] {
                let key = r.dag.nodes[id].key;
                let edges = &r.dag.nodes[id].edges;
                let mut offs = [edges.len() as u32; 8];
                let mut prev_p = 0usize;
                let mut started = false;
                for (i, &c) in edges.iter().enumerate() {
                    let cm = r.dag.nodes[c].key.mask;
                    // consumed piece: the bit removed from key.mask (refill: single-bit mask).
                    let d = key.mask & !cm;
                    let p = if d != 0 {
                        d.trailing_zeros() as usize
                    } else {
                        key.mask.trailing_zeros() as usize
                    };
                    if !started || p != prev_p {
                        debug_assert!(!started || p > prev_p, "reveal blocks must be ascending");
                        let first = if started { prev_p + 1 } else { 0 };
                        for offset in offs.iter_mut().take(p + 1).skip(first) {
                            *offset = i as u32;
                        }
                        prev_p = p;
                        started = true;
                    }
                }
                // pieces after the last block keep edges.len() (empty ranges).
                let first = if started { prev_p + 1 } else { 0 };
                for offset in offs.iter_mut().skip(first) {
                    *offset = edges.len() as u32;
                }
                out[id] = offs;
            }
        }
        out
    }

    /// Depths 7..=10 node ids bucketed by node mask (one pass): each leaf's backup then walks
    /// exactly its realized bucket instead of scanning whole layers with a mask filter.
    fn reveal_layer_buckets(&self) -> Vec<HashMap<u8, Vec<NodeId>>> {
        let r = &self.retained;
        let mut out: Vec<HashMap<u8, Vec<NodeId>>> = vec![HashMap::new(); 11];
        for (depth, bucket) in out.iter_mut().enumerate().take(11).skip(7) {
            for &id in &r.dag.layers[depth] {
                bucket.entry(r.dag.nodes[id].key.mask).or_default().push(id);
            }
        }
        out
    }

    /// Exact d0..3 tables: run the conditional backup for EVERY hidden completion (leaf), keep
    /// each run's depth-4 scalars as one column, then rebuild the ordinary folds on those exact
    /// columns. One backward pass per leaf — same total work as a joint (h×r) vector backup but
    /// flat memory (each leaf's deeper vectors are dropped as soon as its column is extracted).
    pub fn build_cond_full(&self, reset: &mut ResetEval) -> CondFull {
        let r = &self.retained;
        let suffix = build_cond_suffix();
        let mut scratch = CondScratch::new(r.dag.nodes.len());
        let leaf_count = self.leaf_count;
        let buckets = self.reveal_layer_buckets();
        let blocks = self.reveal_edge_blocks();
        #[cfg(not(target_arch = "wasm32"))]
        COND_DEPTH_NS.with(|t| *t.borrow_mut() = [0; 11]);
        let mut vals4: ValTable = HashMap::new();
        #[cfg(not(target_arch = "wasm32"))]
        let nogroup = std::env::var("VS_NOGROUP").is_ok();
        #[cfg(target_arch = "wasm32")]
        let nogroup = false;
        if nogroup {
            // Reference per-leaf path (A/B verification of the grouped pass below).
            for leaf in 0..leaf_count {
                let pack = r.full_hidden_packs[leaf];
                let hidden = [
                    get_hidden(pack, 0),
                    get_hidden(pack, 1),
                    get_hidden(pack, 2),
                    get_hidden(pack, 3),
                ];
                let (_m4, d4, _tables, _n) = self.cond_backup(
                    hidden,
                    reset,
                    &suffix,
                    &mut scratch,
                    &buckets,
                    &blocks,
                    false,
                    4,
                    None,
                );
                for (id, v) in d4 {
                    vals4
                        .entry(id)
                        .or_insert_with(|| vec![0.0 as Val; leaf_count])[leaf] = v;
                }
            }
        } else {
            self.cond_full_grouped(reset, &suffix, &mut scratch, &buckets, &blocks, &mut vals4);
        }
        #[cfg(not(target_arch = "wasm32"))]
        if std::env::var("VS_CONDBENCH").is_ok() {
            COND_DEPTH_NS.with(|t| {
                let t = t.borrow();
                let total: u64 = t.iter().sum();
                eprintln!(
                    "cond_backup depth breakdown (ms): seed={} d9={} d8={} d7={} d6={} d5={} d4={}  total={}",
                    t[10] / 1_000_000, t[9] / 1_000_000, t[8] / 1_000_000, t[7] / 1_000_000,
                    t[6] / 1_000_000, t[5] / 1_000_000, t[4] / 1_000_000, total / 1_000_000
                );
            });
        }
        let folds = build_folds(
            &r.dag,
            &vals4,
            &r.full_hidden_packs,
            leaf_count,
            r.initial_mask,
            false,
            false,
        );
        let root_value = folds[0]
            .get(&0u16)
            .and_then(|t| t.get(&r.dag.root))
            .copied()
            .unwrap_or(0.0);
        CondFull {
            root_value,
            vals4,
            folds,
        }
    }

    /// Grouped CondFull leaf loop: leaves sharing the realized end-mask `m4` traverse IDENTICAL
    /// depth 4-6 structure (full layers; the depth-6 child block only depends on hidden[0]), so
    /// run depths 9..7 per leaf (realized chains differ) and then ONE joint 6..4 pass per group
    /// with per-node CONCAT vectors (leaf-major segments). Bit-exact vs the per-leaf path: each
    /// leaf's segment sees the same children in the same edge order, and an absent-for-this-leaf
    /// child contributes an all-zero segment whose chunk-avg (0.0) never wins the max — identical
    /// to the per-leaf `cv.is_empty()` skip. The d5/d4 folds run ONE full-width fold_flat per
    /// child (leaf segments are whole multiples of the branch size, so chunks never straddle).
    fn cond_full_grouped(
        &self,
        reset: &mut ResetEval,
        suffix: &CondSuffix,
        scratch: &mut CondScratch,
        buckets: &[HashMap<u8, Vec<NodeId>>],
        blocks: &[[u32; 8]],
        vals4: &mut ValTable,
    ) {
        let r = &self.retained;
        let leaf_count = self.leaf_count;
        // Group leaves by realized m4 (first-seen order keeps determinism; column order in vals4
        // is by absolute leaf index anyway).
        let mut group_of: HashMap<u8, usize> = HashMap::new();
        let mut groups: Vec<(u8, Vec<usize>)> = Vec::new();
        for leaf in 0..leaf_count {
            let pack = r.full_hidden_packs[leaf];
            let mut m = r.initial_mask;
            for i in 0..4 {
                m = after_reveal(m, get_hidden(pack, i));
            }
            let gi = *group_of.entry(m).or_insert_with(|| {
                groups.push((m, Vec::new()));
                groups.len() - 1
            });
            groups[gi].1.push(leaf);
        }

        // Concat scratch (dense slots + touched lists, vectors recycled through a pool).
        let n_nodes = r.dag.nodes.len();
        let mut gcur: Vec<ValVec> = vec![Vec::new(); n_nodes];
        let mut gnxt: Vec<ValVec> = vec![Vec::new(); n_nodes];
        let mut gcur_used: Vec<NodeId> = Vec::new();
        let mut gnxt_used: Vec<NodeId> = Vec::new();
        let mut gpool: Vec<ValVec> = Vec::new();

        for (m4, group) in &groups {
            let m4 = *m4;
            let l_n = group.len();
            let sizes = r_branch_sizes(m4);
            let nk3 = suffix.cnt[m4 as usize][3] as usize;
            let hs: Vec<[Piece; 4]> = group
                .iter()
                .map(|&leaf| {
                    let pack = r.full_hidden_packs[leaf];
                    [
                        get_hidden(pack, 0),
                        get_hidden(pack, 1),
                        get_hidden(pack, 2),
                        get_hidden(pack, 3),
                    ]
                })
                .collect();
            // Participants of each depth-6 reveal block = group leaves whose first hidden is p.
            let mut parts: [Vec<usize>; 7] = Default::default();
            for (gi, h) in hs.iter().enumerate() {
                parts[h[0] as usize].push(gi);
            }

            // ---- per-leaf top phase (seed + depths 9..7) -> concat depth-7 vectors ----
            for (gi, _leaf) in group.iter().enumerate() {
                let hidden = hs[gi];
                let _ = self.cond_backup(
                    hidden, reset, suffix, scratch, buckets, blocks, false, 7, None,
                );
                let CondScratch {
                    cur,
                    cur_used,
                    pool,
                    pool_bytes,
                    ..
                } = scratch;
                for &id in cur_used.iter() {
                    let v = &cur[id];
                    if v.is_empty() {
                        continue;
                    }
                    if gcur[id].is_empty() {
                        let mut d = gpool.pop().unwrap_or_default();
                        d.clear();
                        d.resize(l_n * nk3, 0.0);
                        gcur[id] = d;
                        gcur_used.push(id);
                    }
                    gcur[id][gi * nk3..(gi + 1) * nk3].copy_from_slice(v);
                }
                for &id in cur_used.iter() {
                    cond_pool_put(pool, pool_bytes, std::mem::take(&mut cur[id]));
                }
                cur_used.clear();
            }

            // ---- joint depths 6..4 over the whole group ----
            for depth in (4..7u8).rev() {
                #[cfg(not(target_arch = "wasm32"))]
                let _t0 = std::time::Instant::now();
                let k = (depth - 4) as usize;
                let nkp = suffix.cnt[m4 as usize][k] as usize;
                let b = sizes[k];
                let width = l_n * nkp;
                for &id in &r.dag.layers[depth as usize] {
                    let key = r.dag.nodes[id].key;
                    let edges = &r.dag.nodes[id].edges;
                    let mut dst: ValVec = Vec::new();
                    if depth == 6 {
                        let o = &blocks[id];
                        for p in 0..7usize {
                            if parts[p].is_empty() {
                                continue;
                            }
                            let end = if p + 1 < 8 {
                                o[p + 1] as usize
                            } else {
                                edges.len()
                            };
                            for &c in &edges[o[p] as usize..end] {
                                let cv = &gcur[c];
                                if cv.is_empty() {
                                    continue;
                                }
                                if dst.is_empty() {
                                    dst = gpool.pop().unwrap_or_default();
                                    dst.clear();
                                    dst.resize(width, 0.0);
                                }
                                for &gi in &parts[p] {
                                    fold_flat(
                                        &mut dst[gi * nkp..(gi + 1) * nkp],
                                        &cv[gi * nk3..(gi + 1) * nk3],
                                        b,
                                    );
                                }
                            }
                        }
                    } else {
                        for &c in edges.iter() {
                            let cv = &gcur[c];
                            if cv.is_empty() {
                                continue;
                            }
                            if dst.is_empty() {
                                dst = gpool.pop().unwrap_or_default();
                                dst.clear();
                                dst.resize(width, 0.0);
                            }
                            fold_flat(&mut dst, cv, b);
                        }
                        if depth == 5 && key.field == r.two_line_field {
                            let q5 = r.visible[5];
                            if dst.is_empty() {
                                dst = gpool.pop().unwrap_or_default();
                                dst.clear();
                                dst.resize(width, 0.0);
                            }
                            for (gi, h) in hs.iter().enumerate() {
                                let mut i = 0usize;
                                for q in pieces(m4) {
                                    let queue = [q5, h[0], h[1], h[2], h[3], q];
                                    let v = reset.w_partial(key.hold, &queue, after_reveal(m4, q));
                                    let d = &mut dst[gi * nkp + i];
                                    if (v as Val) > *d {
                                        *d = v as Val;
                                    }
                                    i += 1;
                                }
                                debug_assert_eq!(i, nkp);
                            }
                        }
                    }
                    if !dst.is_empty() {
                        if depth == 4 {
                            debug_assert_eq!(dst.len(), l_n);
                            let ent = vals4
                                .entry(id)
                                .or_insert_with(|| vec![0.0 as Val; leaf_count]);
                            for (gi, &leaf) in group.iter().enumerate() {
                                ent[leaf] = dst[gi];
                            }
                            gpool.push(dst);
                        } else {
                            gnxt[id] = dst;
                            gnxt_used.push(id);
                        }
                    }
                }
                #[cfg(not(target_arch = "wasm32"))]
                COND_DEPTH_NS
                    .with(|t| t.borrow_mut()[depth as usize] += _t0.elapsed().as_nanos() as u64);
                for &id in gcur_used.iter() {
                    gpool.push(std::mem::take(&mut gcur[id]));
                }
                gcur_used.clear();
                std::mem::swap(&mut gcur, &mut gnxt);
                std::mem::swap(&mut gcur_used, &mut gnxt_used);
            }
            // drain (depth-4 pushes nothing forward, but stay defensive)
            for &id in gcur_used.iter() {
                gpool.push(std::mem::take(&mut gcur[id]));
            }
            gcur_used.clear();
        }
    }

    /// `min_depth` = deepest layer to stop AFTER (4 = full run to the depth-4 scalars; 7 = top
    /// phase only, leaving the depth-7 vectors in scratch.cur/cur_used for the caller to consume —
    /// the caller must then drain them back into scratch.pool).
    #[allow(clippy::too_many_arguments)]
    fn cond_backup(
        &self,
        hidden: [Piece; 4],
        reset: &mut ResetEval,
        suffix: &CondSuffix,
        scratch: &mut CondScratch,
        buckets: &[HashMap<u8, Vec<NodeId>>],
        blocks: &[[u32; 8]],
        keep_tables: bool,
        min_depth: u8,
        restricted_layers: Option<&[Vec<NodeId>]>,
    ) -> CondBackupOutput {
        let r = &self.retained;
        // Realized mask chain: m[k] = bag after hidden[0..k]. A depth-d node (d>=7) lies on the
        // realized reveal line iff key.mask == m[d-6] — the mask buckets give those directly
        // (off-line nodes are additionally blocked by the parent-side child filter below).
        let mut m = [r.initial_mask; 5];
        for k in 0..4 {
            m[k + 1] = after_reveal(m[k], hidden[k]);
        }
        let mask4 = m[4];
        let sizes = r_branch_sizes(mask4);

        let CondScratch {
            cur,
            nxt,
            cur_used,
            nxt_used,
            seeds,
            pool,
            pool_bytes,
        } = scratch;
        debug_assert!(cur_used.is_empty() && nxt_used.is_empty());

        let mut tables: Vec<HashMap<NodeId, ValVec>> = if keep_tables {
            vec![HashMap::new(); 11]
        } else {
            Vec::new()
        };

        // ---- depth-10 seed: fully-known reset queue r4..r9 (rank order = suffix DFS) ----
        {
            #[cfg(not(target_arch = "wasm32"))]
            let _t0 = std::time::Instant::now();
            let n6 = suffix.cnt[mask4 as usize][6] as usize;
            let ids: &[NodeId] = if let Some(layers) = restricted_layers {
                &layers[10]
            } else {
                buckets[10].get(&mask4).map(|v| v.as_slice()).unwrap_or(&[])
            };
            for &id in ids {
                let key = r.dag.nodes[id].key;
                if key.field != TERMINAL_HASH {
                    continue;
                }
                let v = seeds.entry((key.hold, mask4)).or_insert_with(|| {
                    let mut v = vec![0.0 as Val; n6];
                    let mut i = 0usize;
                    let mut seq = [0u8; 6];
                    seed_rec(&mut v, &mut i, &mut seq, 0, mask4, key.hold, reset);
                    debug_assert_eq!(i, n6);
                    v
                });
                cur[id] = v.clone();
                cur_used.push(id);
                if keep_tables {
                    tables[10].insert(id, v.clone());
                }
            }
            #[cfg(not(target_arch = "wasm32"))]
            COND_DEPTH_NS.with(|t| t.borrow_mut()[10] += _t0.elapsed().as_nanos() as u64);
        }

        // ---- backup depths 9..4 (+2LPC conditional injection at depth 5) ----
        let mut d4: Vec<(NodeId, Val)> = Vec::new();
        let mut touched = 0usize;
        for depth in (min_depth..10u8).rev() {
            #[cfg(not(target_arch = "wasm32"))]
            let _t0 = std::time::Instant::now();
            let k = (depth - 4) as usize;
            let nk = suffix.cnt[mask4 as usize][k] as usize;
            let ids: &[NodeId] = if let Some(layers) = restricted_layers {
                &layers[depth as usize]
            } else if depth >= 7 {
                buckets[depth as usize]
                    .get(&m[(depth - 6) as usize])
                    .map(|v| v.as_slice())
                    .unwrap_or(&[])
            } else {
                &r.dag.layers[depth as usize]
            };
            for &id in ids {
                let key = r.dag.nodes[id].key;
                let mut dst: ValVec = Vec::new();
                // depth>=6: slice exactly the realized reveal's contiguous child block.
                let edges = &r.dag.nodes[id].edges;
                let child_slice: &[NodeId] = if depth >= 6 {
                    let p = hidden[(depth - 6) as usize] as usize;
                    let o = &blocks[id];
                    let end = if p + 1 < 8 {
                        o[p + 1] as usize
                    } else {
                        edges.len()
                    };
                    &edges[o[p] as usize..end]
                } else {
                    edges
                };
                for &c in child_slice {
                    let cv = &cur[c];
                    if cv.is_empty() {
                        continue;
                    }
                    if dst.is_empty() {
                        dst = cond_pool_take(pool, pool_bytes);
                        dst.clear();
                        dst.resize(nk, 0.0);
                    }
                    // avg over the revealed q then max into dst — uniform-chunk fold (the branch
                    // size at each level is position-independent).
                    fold_flat(&mut dst, cv, sizes[k]);
                }
                if depth == 5 && key.field == r.two_line_field {
                    // 2LPC terminal: reset queue = [q5, h1..h4, h5] and h5 IS r4 — known at rank
                    // index. Conditional per-r4 value replaces the old h5-average (w2).
                    let q5 = r.visible[5];
                    if dst.is_empty() {
                        dst = cond_pool_take(pool, pool_bytes);
                        dst.clear();
                        dst.resize(nk, 0.0);
                    }
                    let mut i = 0usize;
                    for q in pieces(mask4) {
                        let queue = [q5, hidden[0], hidden[1], hidden[2], hidden[3], q];
                        let v = reset.w_partial(key.hold, &queue, after_reveal(mask4, q));
                        if (v as Val) > dst[i] {
                            dst[i] = v as Val;
                        }
                        i += 1;
                    }
                    debug_assert_eq!(i, nk);
                }
                if !dst.is_empty() {
                    if depth == 4 {
                        debug_assert_eq!(dst.len(), 1);
                        d4.push((id, dst[0]));
                        if keep_tables {
                            tables[4].insert(id, dst);
                        } else {
                            cond_pool_put(pool, pool_bytes, dst);
                        }
                    } else {
                        if keep_tables {
                            tables[depth as usize].insert(id, dst.clone());
                        }
                        nxt[id] = dst;
                        nxt_used.push(id);
                    }
                }
            }
            touched += nxt_used.len();
            #[cfg(not(target_arch = "wasm32"))]
            COND_DEPTH_NS
                .with(|t| t.borrow_mut()[depth as usize] += _t0.elapsed().as_nanos() as u64);
            // rotate: the just-written depth becomes the child layer for the next (shallower)
            // one; retired vectors go back to the pool.
            for &id in cur_used.iter() {
                cond_pool_put(pool, pool_bytes, std::mem::take(&mut cur[id]));
            }
            cur_used.clear();
            std::mem::swap(cur, nxt);
            std::mem::swap(cur_used, nxt_used);
        }
        // leave the scratch clean for the next leaf (min_depth > 4: the caller consumes cur first)
        if min_depth == 4 {
            for &id in cur_used.iter() {
                cond_pool_put(pool, pool_bytes, std::mem::take(&mut cur[id]));
            }
            cur_used.clear();
        }

        (mask4, d4, tables, touched)
    }

    /// analyze() with FULLY see7-exact candidate scores: depths >= 4 via the realized-prefix
    /// conditional tables (CondEval), depths 0..3 via the exact folds of CondFull (built on the
    /// per-leaf conditional depth-4 columns). root_value is the exact fold-0 root — matches the
    /// fullV MDP end to end.
    pub fn analyze_cond_full(
        &self,
        path: &[usize],
        hidden: [Piece; 4],
        ce: &CondEval,
        cf: &CondFull,
        next_known: &[Piece],
    ) -> AnalysisNode {
        let plain = self.analyze(path, hidden);
        if plain.depth >= 4 {
            let mut node = self.analyze_cond(path, hidden, ce, next_known);
            node.root_value = cf.root_value;
            return node;
        }
        let mut node = plain;
        let d = node.depth;
        let mut best_pos = usize::MAX;
        let mut best_s = 0.0f64;
        for (i, c) in node.cands.iter_mut().enumerate() {
            c.best = false;
            c.score = self.cond_full_score(cf, c.child_id, d + 1, &hidden);
            if c.score > best_s {
                best_s = c.score;
                best_pos = i;
            }
        }
        if best_pos != usize::MAX {
            node.cands[best_pos].best = true;
        }
        node.cands.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        node.best_score = best_s;
        node.root_value = cf.root_value;
        // Re-judge the score-based terminal from the COND scores (the plain ones may be absent
        // entirely under skip_solve); structural terminals (1=4LPC, 2=2LPC) stand.
        if node.terminal == 0 || node.terminal == 3 {
            node.terminal = if node.cands.is_empty() || best_s <= 0.0 {
                3
            } else {
                0
            };
        }
        node
    }

    /// decision_score against the EXACT tables: folds for child depths 1..3, per-leaf conditional
    /// depth-4 columns (h4 averaged) for child depth 4. Mirrors decision_score's reveal timing.
    fn cond_full_score(
        &self,
        cf: &CondFull,
        child: NodeId,
        child_depth: u8,
        hidden: &[Piece; 4],
    ) -> f64 {
        let r = &self.retained;
        match child_depth {
            1..=3 => {
                let known = child_depth - 1;
                let mut prefix = 0u16;
                for i in 0..known {
                    prefix = set_hidden(prefix, i, hidden[i as usize]);
                }
                let mut mask = r.initial_mask;
                for i in 0..known {
                    mask = after_reveal(mask, hidden[i as usize]);
                }
                let mut sum = 0.0;
                let mut n = 0.0;
                for h in pieces(mask) {
                    let pack = set_hidden(prefix, known, h);
                    if let Some(v) = cf.folds[child_depth as usize]
                        .get(&pack)
                        .and_then(|t| t.get(&child))
                    {
                        sum += v;
                    }
                    n += 1.0;
                }
                if n == 0.0 {
                    0.0
                } else {
                    sum / n
                }
            }
            4 => {
                let Some(col) = cf.vals4.get(&child) else {
                    return 0.0;
                };
                let mut prefix = 0u16;
                for i in 0..3 {
                    prefix = set_hidden(prefix, i, hidden[i as usize]);
                }
                let mask = mask_after_hidden_prefix(r.initial_mask, prefix, 3);
                let mut sum = 0.0;
                let mut n = 0.0;
                for h4 in pieces(mask) {
                    let pack = set_hidden(prefix, 3, h4);
                    if let Some(rg) = r.ranges.get(&(4, pack)) {
                        sum += col[rg.start as usize] as f64;
                    }
                    n += 1.0;
                }
                if n > 0.0 {
                    sum / n
                } else {
                    0.0
                }
            }
            _ => 0.0,
        }
    }

    /// analyze() with see7-exact candidate scores at decision depths >= 4: each candidate is the
    /// conditional expectation given the REALIZED next-PC reveals seen so far (`next_known`,
    /// r4..; may be shorter than D-4 — the un-revealed tail is averaged). Depths 0..3 fall back
    /// to the plain decision_score (no r is revealed there; residual model gap only via folds).
    pub fn analyze_cond(
        &self,
        path: &[usize],
        hidden: [Piece; 4],
        ce: &CondEval,
        next_known: &[Piece],
    ) -> AnalysisNode {
        let mut node = self.analyze(path, hidden);
        if node.depth < 4 {
            return node;
        }
        let d = node.depth;
        let known = &next_known[..next_known.len().min((d - 4) as usize)];
        let mut best_pos = usize::MAX;
        let mut best_s = 0.0f64;
        for (i, c) in node.cands.iter_mut().enumerate() {
            c.best = false;
            c.score = ce.cand_score(c, d, known);
            if c.score > best_s {
                best_s = c.score;
                best_pos = i;
            }
        }
        if best_pos != usize::MAX {
            node.cands[best_pos].best = true;
        }
        node.cands.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        node.best_score = best_s;
        // Re-judge the score-based terminal from the COND scores (the plain ones may be absent
        // entirely under skip_solve); structural terminals (1=4LPC, 2=2LPC) stand.
        if node.terminal == 0 || node.terminal == 3 {
            node.terminal = if node.cands.is_empty() || best_s <= 0.0 {
                3
            } else {
                0
            };
        }
        // path_steps scores are display-only breadcrumbs; leave them at the plain scores.
        node
    }
}

impl CondEval {
    pub fn slice_nodes(&self) -> usize {
        self.slice_nodes
    }

    /// Score one candidate (child at depth d+1) for a decision at depth d, conditioning on the
    /// realized prefix `known` (r4..r_{d-1}, possibly short) and averaging everything after.
    fn cand_score(&self, c: &AnalysisCand, d: u8, known: &[Piece]) -> f64 {
        // Child vectors exist for depths 5..=10; a depth-4 decision (d=4) reads depth-5 vectors.
        let Some(cv) = self.vals[(d + 1) as usize].get(&c.child_id) else {
            return 0.0;
        };
        avg_known(cv, &self.suffix, self.mask4, known, (d + 1 - 4) as usize).unwrap_or(0.0)
    }
}

/// Seed recursion: enumerate r4..r9 in suffix-DFS order, writing the fully-known reset value.
fn seed_rec(
    v: &mut [Val],
    i: &mut usize,
    seq: &mut [u8; 6],
    depth: usize,
    mask: u8,
    hold: Piece,
    reset: &mut ResetEval,
) {
    if depth == 6 {
        v[*i] = reset.w_partial(hold, &seq[..], mask) as Val;
        *i += 1;
        return;
    }
    for p in pieces(mask) {
        seq[depth] = p;
        seed_rec(v, i, seq, depth + 1, after_reveal(mask, p), hold, reset);
    }
}

/// r-branch sizes: |bag| after k next-queue reveals from mask4. The chain depends only on the
/// popcount (refill to 7 at empty), so it is POSITION-INDEPENDENT — every length-k sequence has
/// the same number of continuations.
fn r_branch_sizes(mask4: u8) -> [usize; 7] {
    let mut n = mask4.count_ones() as usize;
    let mut out = [0usize; 7];
    for k in out.iter_mut() {
        *k = n;
        n = if n == 1 { 7 } else { n - 1 };
    }
    out
}

/// Backup fold, flattened: because the branch size at each level is position-independent, the
/// DFS "avg over q then max into dst" collapses to uniform fixed-size chunking (vectorizes).
fn fold_flat(dst: &mut [Val], cv: &[Val], b: usize) {
    debug_assert_eq!(dst.len() * b, cv.len());
    let inv = 1.0 / b as f64;
    for (d, chunk) in dst.iter_mut().zip(cv.chunks_exact(b)) {
        let mut s = 0.0f64;
        for &x in chunk {
            s += x as f64;
        }
        let a = (s * inv) as Val;
        if a > *d {
            *d = a;
        }
    }
}

/// Average a child vector over all completions of a (possibly short) known prefix: walk the
/// off-chain for the known part, then uniformly average the remaining block (the |bag| chain is
/// determined by popcount alone, so completions are equiprobable).
fn avg_known(cv: &[Val], sfx: &CondSuffix, mask4: u8, known: &[Piece], klen: usize) -> Option<f64> {
    let mut b = 0usize;
    let mut m = mask4;
    for (idx, &p) in known.iter().enumerate() {
        if m & (1 << p) == 0 {
            return None;
        }
        b += sfx.off[m as usize][klen - idx][p as usize] as usize;
        m = after_reveal(m, p);
    }
    let rem = klen - known.len();
    let cnt = sfx.cnt[m as usize][rem] as usize;
    if cnt == 0 || b + cnt > cv.len() {
        return None;
    }
    let mut sum = 0.0;
    for x in &cv[b..b + cnt] {
        sum += *x as f64;
    }
    Some(sum / cnt as f64)
}

fn active_piece(visible: &[Piece; 6], hidden: &[Piece; 4], depth: u8) -> Piece {
    if depth < 6 {
        visible[depth as usize]
    } else {
        // depth 6..9 place h1..h4; clamp so a terminal (depth 10) node can't index out of bounds.
        hidden[((depth - 6) as usize).min(3)]
    }
}

#[inline]
fn next_rand(state: &mut u64) -> u64 {
    // xorshift64*
    let mut x = *state;
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    *state = x;
    x.wrapping_mul(0x2545F4914F6CDD1D)
}

/* -------------------------------------------------------------------------- */
/* Build / prune (adapted from fh_search)                                      */
/* -------------------------------------------------------------------------- */

fn build_dag(
    graph: Option<&FieldGraph>,
    edge_ids: Option<&EdgeSource<'_>>,
    par_edge: Option<&SyncEdgeSource<'_>>,
    hold: Piece,
    visible: [Piece; 6],
    initial_mask: u8,
    two_line_field: u64,
) -> Dag {
    let mut dag = Dag {
        nodes: Vec::new(),
        layers: vec![Vec::new(); 11],
        index: HashMap::new(),
        root: 0,
    };
    let root = get_or_add_node(
        &mut dag,
        NodeKey {
            depth: 0,
            field: ROOT_HASH,
            hold,
            mask: initial_mask,
        },
    );
    dag.root = root;

    // Serial fetch (graph-free closure OR graph.edges); parallel uses the Sync `par_edge` instead.
    let serial_fetch = |field: u64, piece: u8, buf: &mut Vec<u64>| match edge_ids {
        Some(f) => f(field, piece, buf),
        None => {
            let g = graph.expect("graph required when edge_ids is None");
            let fid = g.find_hash(field).expect("node field hash not in graph");
            for &c in g.edges(fid, piece) {
                buf.push(g.hash(c));
            }
        }
    };

    // Reused scratch (cleared per use); no edge sort/dedup (duplicates structurally impossible).
    let mut kids: Vec<u64> = Vec::new();
    let mut hold_kids: Vec<u64> = Vec::new();
    let mut triples: Vec<(u64, Piece, u8)> = Vec::new();
    for depth in 0..10u8 {
        let child_depth = depth + 1;
        let frontier = dag.layers[depth as usize].clone();
        match par_edge {
            // PARALLEL BUILD: Phase A (rayon) computes each frontier node's child triples via the
            // Sync edge source (movegen is the cost, read-only). Phase B inserts them SERIALLY in
            // the identical frontier/child order, so NodeIds — and thus the whole DAG — are exactly
            // as the serial build would produce (bit-identical result, just faster).
            Some(pe) => {
                // This arm only runs natively (par_edge = None on wasm), but must still compile
                // there — maybe_par_collect!(true, ...) picks par_iter natively, iter on wasm.
                let all: Vec<Vec<(u64, Piece, u8)>> =
                    maybe_par_collect!(true, frontier, map, |&id| {
                        let key = dag.nodes[id].key;
                        let mut k = Vec::new();
                        let mut h = Vec::new();
                        let mut o = Vec::new();
                        node_child_triples(
                            key,
                            &visible,
                            two_line_field,
                            &|f, p, b| pe(f, p, b),
                            &mut k,
                            &mut h,
                            &mut o,
                        );
                        o
                    });
                for (&id, tr) in frontier.iter().zip(all.iter()) {
                    let mut edges = Vec::with_capacity(tr.len());
                    for &(nf, nh, nm) in tr {
                        edges.push(get_or_add_node(
                            &mut dag,
                            NodeKey {
                                depth: child_depth,
                                field: nf,
                                hold: nh,
                                mask: nm,
                            },
                        ));
                    }
                    dag.nodes[id].edges = edges;
                }
            }
            None => {
                for id in frontier {
                    let key = dag.nodes[id].key;
                    node_child_triples(
                        key,
                        &visible,
                        two_line_field,
                        &serial_fetch,
                        &mut kids,
                        &mut hold_kids,
                        &mut triples,
                    );
                    let mut edges = Vec::with_capacity(triples.len());
                    for &(nf, nh, nm) in &triples {
                        edges.push(get_or_add_node(
                            &mut dag,
                            NodeKey {
                                depth: child_depth,
                                field: nf,
                                hold: nh,
                                mask: nm,
                            },
                        ));
                    }
                    dag.nodes[id].edges = edges;
                }
            }
        }
    }
    dag
}

/// Compute a node's ordered child triples (child field hash, new hold, child remaining-mask) by
/// placing the active/hold pieces (depth<6) or every reveal from the remaining bag (depth>=6).
/// `fetch(field, piece, buf)` APPENDS child field hashes into `buf` (caller-cleared). Order is
/// fixed and matches the serial build so parallel and serial produce identical DAGs.
fn node_child_triples(
    key: NodeKey,
    visible: &[Piece; 6],
    two_line_field: u64,
    fetch: &dyn Fn(u64, u8, &mut Vec<u64>),
    kids: &mut Vec<u64>,
    hold_kids: &mut Vec<u64>,
    out: &mut Vec<(u64, Piece, u8)>,
) {
    out.clear();
    let depth = key.depth;
    if depth == 5 && key.field == two_line_field {
        return; // 2LPC terminal: no children.
    }
    if depth < 6 {
        let active = visible[depth as usize];
        kids.clear();
        fetch(key.field, active, kids);
        for &nf in kids.iter() {
            out.push((nf, key.hold, key.mask));
        }
        if active != key.hold {
            hold_kids.clear();
            fetch(key.field, key.hold, hold_kids);
            for &nf in hold_kids.iter() {
                out.push((nf, active, key.mask));
            }
        }
    } else {
        // Hoist the hold-placement children: (field,hold) is IDENTICAL across this node's reveal
        // branches — fetch once, rewire per branch (only the mask differs).
        hold_kids.clear();
        fetch(key.field, key.hold, hold_kids);
        for p in pieces_in_mask(key.mask) {
            let child_mask = after_reveal(key.mask, p);
            kids.clear();
            fetch(key.field, p, kids);
            for &nf in kids.iter() {
                out.push((nf, key.hold, child_mask));
            }
            if p != key.hold {
                for &nf in hold_kids.iter() {
                    out.push((nf, p, child_mask));
                }
            }
        }
    }
}

fn get_or_add_node(dag: &mut Dag, key: NodeKey) -> NodeId {
    let pk = pack_key(&key);
    if let Some(&id) = dag.index.get(&pk) {
        return id;
    }
    let id = dag.nodes.len();
    dag.nodes.push(Node {
        key,
        edges: Vec::new(),
    });
    dag.index.insert(pk, id);
    dag.layers[key.depth as usize].push(id);
    id
}

fn prune_to_terminal_reachable(dag: &Dag, two_line_field: u64, par: bool) -> Dag {
    // Layered backward sweep (edges go depth -> depth+1, so children are finalized before their
    // parents): a node survives iff it IS a terminal or ANY child survives. Same set as the old
    // reverse-index BFS, but with no reverse index to build, and each layer is parallel-safe.
    let mut marked = vec![false; dag.nodes.len()];
    for depth in (0..=10usize).rev() {
        let layer = &dag.layers[depth];
        let mark_of = |id: NodeId, marked: &[bool]| -> bool {
            let n = &dag.nodes[id];
            if depth == 10 {
                n.key.field == TERMINAL_HASH
            } else if depth == 5 && n.key.field == two_line_field {
                true
            } else {
                n.edges.iter().any(|&c| marked[c])
            }
        };
        #[cfg(not(target_arch = "wasm32"))]
        let done_par = if par {
            let res: Vec<bool> = layer.par_iter().map(|&id| mark_of(id, &marked)).collect();
            for (&id, m) in layer.iter().zip(res) {
                marked[id] = m;
            }
            true
        } else {
            false
        };
        #[cfg(target_arch = "wasm32")]
        let done_par = {
            let _ = par;
            false
        };
        if !done_par {
            for &id in layer {
                let m = mark_of(id, &marked);
                marked[id] = m;
            }
        }
    }

    let mut old_to_new = vec![usize::MAX; dag.nodes.len()];
    let mut nodes = Vec::new();
    let mut layers = vec![Vec::new(); 11];
    // NOTE: the pruned DAG's `index` is never read again (get_or_add_node runs only during build),
    // so we skip rebuilding it — saves ~1.1M hashmap inserts per boundary.
    let index = HashMap::new();
    let mut survivors: Vec<NodeId> = Vec::new();
    for old_id in 0..dag.nodes.len() {
        if !marked[old_id] {
            continue;
        }
        let new_id = nodes.len();
        old_to_new[old_id] = new_id;
        let key = dag.nodes[old_id].key;
        nodes.push(Node {
            key,
            edges: Vec::new(),
        });
        layers[key.depth as usize].push(new_id);
        survivors.push(old_id);
    }
    // Remap edges per surviving node (independent -> parallel-safe); dup-free, original order.
    let remap = |&old_id: &NodeId| -> Vec<NodeId> {
        dag.nodes[old_id]
            .edges
            .iter()
            .filter(|&&c| marked[c])
            .map(|&c| old_to_new[c])
            .collect()
    };
    let new_edges: Vec<Vec<NodeId>> = maybe_par_collect!(par, survivors, map, remap);
    for (new_id, edges) in new_edges.into_iter().enumerate() {
        nodes[new_id].edges = edges;
    }
    let root = old_to_new[dag.root];
    Dag {
        nodes,
        layers,
        index,
        root,
    }
}

/* -------------------------------------------------------------------------- */
/* Hidden-sequence utilities (same semantics as fh_search)                     */
/* -------------------------------------------------------------------------- */

fn build_ranges(
    mask: u8,
    idx: u8,
    pack: u16,
    next_leaf: &mut u32,
    out: &mut HashMap<(u8, u16), SeqRange>,
) -> u32 {
    if idx == 4 {
        out.insert(
            (4, pack),
            SeqRange {
                start: *next_leaf,
                len: 1,
            },
        );
        *next_leaf += 1;
        return 1;
    }
    let start = *next_leaf;
    let mut total = 0u32;
    for p in pieces_in_mask(canonical_mask(mask)) {
        total += build_ranges(
            after_reveal(mask, p),
            idx + 1,
            set_hidden(pack, idx, p),
            next_leaf,
            out,
        );
    }
    out.insert((idx, pack), SeqRange { start, len: total });
    total
}

fn canonical_mask(mask: u8) -> u8 {
    let m = mask & FULL_MASK;
    if m == 0 {
        FULL_MASK
    } else {
        m
    }
}

fn mask_after_hidden_prefix(initial_mask: u8, pack: u16, len: u8) -> u8 {
    let mut mask = canonical_mask(initial_mask);
    for i in 0..len {
        mask = after_reveal(mask, get_hidden(pack, i));
    }
    mask
}

fn pieces_in_mask(mask: u8) -> Vec<Piece> {
    let m = canonical_mask(mask);
    let mut out = Vec::new();
    for p in 0..PIECE_COUNT as u8 {
        if (m & (1u8 << p)) != 0 {
            out.push(p);
        }
    }
    out
}

fn set_hidden(pack: u16, idx: u8, p: Piece) -> u16 {
    debug_assert!(idx < 4);
    let shift = (idx as u16) * 3;
    (pack & !(0b111u16 << shift)) | ((p as u16) << shift)
}

fn get_hidden(pack: u16, idx: u8) -> Piece {
    let shift = (idx as u16) * 3;
    ((pack >> shift) & 0b111) as Piece
}

fn prefix_pack(pack: u16, len: u8) -> u16 {
    if len == 0 {
        return 0;
    }
    pack & ((1u16 << (3 * len)) - 1)
}

/// Build statistics for one exact empty-field round.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct ExactDagStats {
    pub nodes_before_prune: usize,
    pub nodes_after_prune: usize,
    pub reveal_sequences: usize,
}

/// Exact see-7 solution for a complete round starting from the empty field.
///
/// The pruned structural DAG and CondFull tables are retained so the reveal tree can later be
/// streamed without re-solving the round.
pub struct ExactDagSolution<'a> {
    vstar: &'a VStarTable,
    search: VsResult,
    root_value: f64,
    shallow_policy: HashMap<ShallowKey, ShallowAction>,
    stats: ExactDagStats,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
struct ShallowKey {
    node: NodeId,
    depth: u8,
    hidden_pack: u16,
}

#[derive(Clone, Copy)]
struct ShallowAction {
    child: NodeId,
    score: f64,
    placed: Piece,
}

struct TreeWorkspace<'a> {
    suffix: Box<CondSuffix>,
    scratch: CondScratch,
    buckets: Vec<HashMap<u8, Vec<NodeId>>>,
    blocks: Vec<[u32; 8]>,
    reset: ResetEval<'a>,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
struct DeepKey {
    node: NodeId,
    next_pack: u32,
}

#[derive(Clone, Copy)]
struct DeepAction {
    child: NodeId,
    score: f64,
    placed: Piece,
}

struct DeepPolicy {
    actions: HashMap<DeepKey, DeepAction>,
}

#[derive(Clone)]
struct DeepFrontier {
    node: NodeId,
    next_reveals: [Piece; 6],
    next_len: usize,
    bag: u8,
}

fn build_shallow_policy(search: &VsResult, exact: &CondFull) -> HashMap<ShallowKey, ShallowAction> {
    fn visit(
        search: &VsResult,
        exact: &CondFull,
        node: NodeId,
        depth: u8,
        hidden_pack: u16,
        output: &mut HashMap<ShallowKey, ShallowAction>,
    ) {
        debug_assert!(depth < 4);
        let retained = &search.retained;
        let parent = &retained.dag.nodes[node];
        let mut best_child = None;
        let mut best_score = 0.0;
        for &child in &parent.edges {
            let score = shallow_child_score(search, exact, child, depth, hidden_pack);
            if score > best_score {
                best_score = score;
                best_child = Some(child);
            }
        }
        let Some(child) = best_child else {
            return;
        };
        let child_key = retained.dag.nodes[child].key;
        let active = retained.visible[depth as usize];
        let placed = if child_key.hold == parent.key.hold {
            active
        } else {
            parent.key.hold
        };
        output.insert(
            ShallowKey {
                node,
                depth,
                hidden_pack,
            },
            ShallowAction {
                child,
                score: best_score,
                placed,
            },
        );

        if depth < 3 {
            let mask = mask_after_hidden_prefix(retained.initial_mask, hidden_pack, depth);
            for revealed in pieces(mask) {
                visit(
                    search,
                    exact,
                    child,
                    depth + 1,
                    set_hidden(hidden_pack, depth, revealed),
                    output,
                );
            }
        }
    }

    let mut output = HashMap::new();
    visit(search, exact, search.retained.dag.root, 0, 0, &mut output);
    output
}

fn shallow_child_score(
    search: &VsResult,
    exact: &CondFull,
    child: NodeId,
    depth: u8,
    hidden_pack: u16,
) -> f64 {
    let retained = &search.retained;
    let mask = mask_after_hidden_prefix(retained.initial_mask, hidden_pack, depth);
    let mut sum = 0.0;
    let mut count = 0u32;
    for revealed in pieces(mask) {
        let pack = set_hidden(hidden_pack, depth, revealed);
        let value = if depth < 3 {
            exact.folds[(depth + 1) as usize]
                .get(&pack)
                .and_then(|table| table.get(&child))
                .copied()
                .unwrap_or(0.0)
        } else {
            retained
                .ranges
                .get(&(4, pack))
                .and_then(|range| {
                    exact
                        .vals4
                        .get(&child)
                        .map(|column| column[range.start as usize] as f64)
                })
                .unwrap_or(0.0)
        };
        sum += value;
        count += 1;
    }
    if count == 0 {
        0.0
    } else {
        sum / f64::from(count)
    }
}

impl<'a> ExactDagSolution<'a> {
    /// Solve an empty-field ten-placement round with exact see-7 information timing.
    pub fn solve(
        graph: &FieldGraph,
        vstar: &'a VStarTable,
        hold: Piece,
        visible: [Piece; 6],
        bag: u8,
        threads: usize,
    ) -> Result<Self> {
        if hold >= PIECE_COUNT as u8 || visible.iter().any(|&piece| piece >= PIECE_COUNT as u8) {
            bail!("invalid piece in exact DAG boundary");
        }
        if bag == 0 || bag & !FULL_BAG != 0 {
            bail!("exact DAG bag must be a nonempty 7-bit mask");
        }
        let initial_key = boundary_key(hold, visible, bag);
        if vstar.get(initial_key).is_none() {
            bail!("initial see-7 chain is not present in the V* table (key {initial_key})");
        }

        let parallel_edges = |field_hash: u64, piece: u8, output: &mut Vec<u64>| {
            let field = graph
                .find_hash(field_hash)
                .expect("exact DAG field hash must be present in graph.bin");
            output.extend(
                graph
                    .edges(field, piece)
                    .iter()
                    .map(|&child| graph.hash(child)),
            );
        };
        let build = |parallel: bool| {
            value_search(SearchInput {
                graph: Some(graph),
                hold,
                visible,
                mask: bag,
                reset: ResetEval::new(vstar),
                edge_ids: None,
                par_edge: parallel
                    .then_some(&parallel_edges as &(dyn Fn(u64, u8, &mut Vec<u64>) + Sync)),
                skip_solve: true,
            })
        };

        let threads = threads.max(1);
        let search = if threads == 1 {
            build(false)
        } else {
            rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()?
                .install(|| build(true))
        };
        let stats = ExactDagStats {
            nodes_before_prune: search.nodes_total,
            nodes_after_prune: search.nodes_pruned,
            reveal_sequences: search.leaf_count,
        };

        let mut reset = ResetEval::new(vstar);
        let exact = search.build_cond_full(&mut reset);
        if reset.missing_keys != 0 {
            bail!(
                "V* table was missing {} terminal boundary keys",
                reset.missing_keys
            );
        }

        let root_value = exact.root_value;
        let shallow_policy = build_shallow_policy(&search, &exact);
        // CondFull is intentionally dropped here.  The complete shallow policy contains only a
        // few hundred reveal-prefix states, while retaining its wide value columns alongside one
        // deep CondEval can otherwise push tree streaming over the practical memory ceiling.
        drop(exact);
        // CondFull allocates many differently sized vectors across rayon worker arenas.  glibc may
        // otherwise keep those now-free pages resident until process exit, defeating the scoped
        // lifetime above before tree streaming begins.
        #[cfg(all(target_os = "linux", target_env = "gnu"))]
        unsafe {
            libc::malloc_trim(0);
        }

        Ok(Self {
            vstar,
            search,
            root_value,
            shallow_policy,
            stats,
        })
    }

    #[inline]
    pub fn root_value(&self) -> f64 {
        self.root_value
    }

    #[inline]
    pub fn stats(&self) -> ExactDagStats {
        self.stats
    }

    /// The `init_hash` header of the empty-DAG policy, which is always 0.
    #[inline]
    pub fn init_hash(&self) -> u64 {
        0
    }

    /// Stream the complete reveal-conditioned policy as a piece-keyed JSON root node subtree.
    ///
    /// CondFull directly selects depths 0..3.  Once the first four reveals are known, one
    /// conditional deep backup is built, streamed, and dropped before the next reveal leaf.  The
    /// scratch arrays and structural indexes are reused across leaves to keep peak memory bounded.
    pub fn write_root_json(&self, mut output: impl Write) -> io::Result<()> {
        let retained = &self.search.retained;
        let mut workspace = TreeWorkspace {
            suffix: build_cond_suffix(),
            scratch: CondScratch::new(retained.dag.nodes.len()),
            buckets: self.search.reveal_layer_buckets(),
            blocks: self.search.reveal_edge_blocks(),
            reset: ResetEval::new(self.vstar),
        };
        self.write_tree_state_json(
            retained.dag.root,
            0,
            0,
            [0; 4],
            retained.visible,
            retained.initial_mask,
            [0; 6],
            0,
            None,
            &mut workspace,
            &mut output,
        )?;
        if workspace.reset.missing_keys != 0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "V* table was missing {} terminal boundary keys while writing the tree",
                    workspace.reset.missing_keys
                ),
            ));
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn write_tree_state_json(
        &self,
        node: NodeId,
        depth: u8,
        hidden_pack: u16,
        hidden: [Piece; 4],
        queue: [Piece; 6],
        bag: u8,
        next_reveals: [Piece; 6],
        next_len: usize,
        deep: Option<&DeepPolicy>,
        workspace: &mut TreeWorkspace<'a>,
        output: &mut impl Write,
    ) -> io::Result<()> {
        let Some((child, score, placed)) = self.select_action(
            node,
            depth,
            hidden_pack,
            hidden,
            &next_reveals[..next_len],
            deep,
        ) else {
            return output.write_all(b"0");
        };
        let child_key = self.search.retained.dag.nodes[child].key;
        write!(
            output,
            "{{\"hash\":{},\"piece\":\"{}\",\"value\":{},",
            child_key.field,
            piece_char(placed),
            score
        )?;
        write!(output, "\"children\":{{")?;
        let mut first = true;
        for revealed in 0u8..PIECE_COUNT as u8 {
            if bag & (1 << revealed) == 0 {
                continue;
            }
            if !first {
                output.write_all(b",")?;
            }
            first = false;
            write!(output, "\"{}\":", piece_char(revealed))?;

            let next_queue = [queue[1], queue[2], queue[3], queue[4], queue[5], revealed];
            let next_bag = after_reveal(bag, revealed);
            if child_key.field == TERMINAL_HASH || child_key.field == self.search.two_line_field() {
                let value = self
                    .vstar
                    .get(boundary_key(child_key.hold, next_queue, next_bag))
                    .ok_or_else(|| {
                        io::Error::new(
                            io::ErrorKind::InvalidData,
                            "V* table is missing a terminal boundary while writing the tree",
                        )
                    })?;
                write!(output, "{value}", value = 1.0 + f64::from(value))?;
                continue;
            }

            if depth < 3 {
                let mut next_hidden = hidden;
                next_hidden[depth as usize] = revealed;
                self.write_tree_state_json(
                    child,
                    depth + 1,
                    set_hidden(hidden_pack, depth, revealed),
                    next_hidden,
                    next_queue,
                    next_bag,
                    next_reveals,
                    next_len,
                    None,
                    workspace,
                    output,
                )?;
            } else if depth == 3 {
                let mut next_hidden = hidden;
                next_hidden[3] = revealed;
                let full_pack = set_hidden(hidden_pack, 3, revealed);
                debug_assert!(self.search.retained.ranges.contains_key(&(4, full_pack)));
                let branch = self.build_deep_policy(child, next_hidden, workspace)?;
                self.write_tree_state_json(
                    child,
                    4,
                    full_pack,
                    next_hidden,
                    next_queue,
                    next_bag,
                    next_reveals,
                    next_len,
                    Some(&branch),
                    workspace,
                    output,
                )?;
            } else {
                let mut reveals = next_reveals;
                reveals[next_len] = revealed;
                self.write_tree_state_json(
                    child,
                    depth + 1,
                    hidden_pack,
                    hidden,
                    next_queue,
                    next_bag,
                    reveals,
                    next_len + 1,
                    deep,
                    workspace,
                    output,
                )?;
            }
        }
        write!(output, "}}}}")
    }

    fn select_action(
        &self,
        node: NodeId,
        depth: u8,
        hidden_pack: u16,
        hidden: [Piece; 4],
        next_known: &[Piece],
        deep: Option<&DeepPolicy>,
    ) -> Option<(NodeId, f64, Piece)> {
        if depth < 4 {
            let action = self.shallow_policy.get(&ShallowKey {
                node,
                depth,
                hidden_pack,
            })?;
            return Some((action.child, action.score, action.placed));
        }
        let _ = (hidden, depth);
        let action = deep?.actions.get(&DeepKey {
            node,
            next_pack: encode_piece_prefix(next_known),
        })?;
        Some((action.child, action.score, action.placed))
    }

    /// Extract just the reachable deep policy for one realized h1..h4 sequence.
    ///
    /// A general `CondEval` retains value vectors for every live node at depths 4..10.  That is
    /// useful for an interactive arbitrary-path analyzer but unnecessarily large for a single
    /// policy tree.  Here each child layer is backed up independently, its reachable decisions are
    /// copied into a compact map, and the layer vectors are immediately recycled.
    fn build_deep_policy(
        &self,
        start_node: NodeId,
        hidden: [Piece; 4],
        workspace: &mut TreeWorkspace<'a>,
    ) -> io::Result<DeepPolicy> {
        let retained = &self.search.retained;
        let mut full_pack = 0u16;
        for (index, &piece) in hidden.iter().enumerate() {
            full_pack = set_hidden(full_pack, index as u8, piece);
        }
        let initial_bag = mask_after_hidden_prefix(retained.initial_mask, full_pack, 4);
        let restricted_layers = self.deep_reachable_layers(start_node, hidden, &workspace.blocks);
        let mut frontier = vec![DeepFrontier {
            node: start_node,
            next_reveals: [0; 6],
            next_len: 0,
            bag: initial_bag,
        }];
        let mut actions = HashMap::new();

        for depth in 4u8..10 {
            if frontier.is_empty() {
                break;
            }
            let (mask4, _depth4, _tables, _touched) = self.search.cond_backup(
                hidden,
                &mut workspace.reset,
                &workspace.suffix,
                &mut workspace.scratch,
                &workspace.buckets,
                &workspace.blocks,
                false,
                depth + 1,
                Some(&restricted_layers),
            );
            debug_assert_eq!(mask4, initial_bag);

            let mut next: HashMap<DeepKey, DeepFrontier> = HashMap::new();
            for state in &frontier {
                let Some(action) = self.select_deep_layer_action(
                    state,
                    depth,
                    hidden,
                    mask4,
                    &workspace.suffix,
                    &workspace.blocks,
                    &workspace.scratch.cur,
                ) else {
                    continue;
                };
                let key = DeepKey {
                    node: state.node,
                    next_pack: encode_piece_prefix(&state.next_reveals[..state.next_len]),
                };
                actions.insert(key, action);

                let child_key = retained.dag.nodes[action.child].key;
                if child_key.field == TERMINAL_HASH || child_key.field == retained.two_line_field {
                    continue;
                }
                for revealed in pieces(state.bag) {
                    let mut next_reveals = state.next_reveals;
                    next_reveals[state.next_len] = revealed;
                    let next_state = DeepFrontier {
                        node: action.child,
                        next_reveals,
                        next_len: state.next_len + 1,
                        bag: after_reveal(state.bag, revealed),
                    };
                    next.entry(DeepKey {
                        node: next_state.node,
                        next_pack: encode_piece_prefix(
                            &next_state.next_reveals[..next_state.next_len],
                        ),
                    })
                    .or_insert(next_state);
                }
            }
            recycle_cond_current(&mut workspace.scratch);
            frontier = next.into_values().collect();
        }

        if workspace.reset.missing_keys != 0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "V* table was missing {} boundaries during a conditional backup",
                    workspace.reset.missing_keys
                ),
            ));
        }
        Ok(DeepPolicy { actions })
    }

    fn deep_reachable_layers(
        &self,
        start_node: NodeId,
        hidden: [Piece; 4],
        blocks: &[[u32; 8]],
    ) -> Vec<Vec<NodeId>> {
        let dag = &self.search.retained.dag;
        let mut layers = vec![Vec::new(); 11];
        layers[4].push(start_node);
        for depth in 4usize..10 {
            let mut seen = HashSet::new();
            let (parents, children) = layers.split_at_mut(depth + 1);
            for &node in &parents[depth] {
                let edges = if depth < 6 {
                    dag.nodes[node].edges.as_slice()
                } else {
                    let active = hidden[depth - 6] as usize;
                    let start = blocks[node][active] as usize;
                    let end = blocks[node][active + 1] as usize;
                    &dag.nodes[node].edges[start..end]
                };
                for &child in edges {
                    if seen.insert(child) {
                        children[0].push(child);
                    }
                }
            }
        }
        layers
    }

    #[allow(clippy::too_many_arguments)]
    fn select_deep_layer_action(
        &self,
        state: &DeepFrontier,
        depth: u8,
        hidden: [Piece; 4],
        mask4: u8,
        suffix: &CondSuffix,
        blocks: &[[u32; 8]],
        child_values: &[ValVec],
    ) -> Option<DeepAction> {
        let retained = &self.search.retained;
        let parent = &retained.dag.nodes[state.node];
        let edges = if depth < 6 {
            parent.edges.as_slice()
        } else {
            let active = hidden[(depth - 6) as usize] as usize;
            let start = blocks[state.node][active] as usize;
            let end = blocks[state.node][active + 1] as usize;
            &parent.edges[start..end]
        };

        let mut best_child = None;
        let mut best_score = 0.0;
        for &child in edges {
            let values = &child_values[child];
            if values.is_empty() {
                continue;
            }
            let score = avg_known(
                values,
                suffix,
                mask4,
                &state.next_reveals[..state.next_len],
                (depth + 1 - 4) as usize,
            )
            .unwrap_or(0.0);
            if score > best_score {
                best_score = score;
                best_child = Some(child);
            }
        }
        let child = best_child?;
        let active = if depth < 6 {
            retained.visible[depth as usize]
        } else {
            hidden[(depth - 6) as usize]
        };
        let child_key = retained.dag.nodes[child].key;
        let placed = if child_key.hold == parent.key.hold {
            active
        } else {
            parent.key.hold
        };
        Some(DeepAction {
            child,
            score: best_score,
            placed,
        })
    }
}

fn encode_piece_prefix(pieces: &[Piece]) -> u32 {
    pieces
        .iter()
        .enumerate()
        .fold(0u32, |packed, (index, &piece)| {
            packed | (u32::from(piece) << (index * 3))
        })
}

fn recycle_cond_current(scratch: &mut CondScratch) {
    debug_assert!(scratch.nxt_used.is_empty());
    for node in scratch.cur_used.drain(..) {
        cond_pool_put(
            &mut scratch.pool,
            &mut scratch.pool_bytes,
            std::mem::take(&mut scratch.cur[node]),
        );
    }
}

#[cfg(test)]
mod local_oracle_check {
    use super::*;

    #[test]
    #[ignore = "requires the external graph.bin and bundled V* asset"]
    fn known_empty_root_matches_reference() {
        let graph = FieldGraph::load(std::env::var("ZXCL_OPTIMAL_TEST_GRAPH").unwrap()).unwrap();
        let vstar = VStarTable::load(std::env::var("ZXCL_OPTIMAL_TEST_VSTAR").unwrap()).unwrap();
        let solution =
            ExactDagSolution::solve(&graph, &vstar, 4, [0, 6, 5, 2, 3, 1], FULL_BAG, 24).unwrap();
        eprintln!(
            "stats={:?} root={:.12}",
            solution.stats(),
            solution.root_value()
        );
        assert!((solution.root_value() - 4_353.563_114_240).abs() < 1e-9);

        let (child, score, placed) = solution
            .select_action(solution.search.retained.dag.root, 0, 0, [0; 4], &[], None)
            .unwrap();
        assert_eq!(
            solution.search.retained.dag.nodes[child].key.field,
            0x30_180
        );
        assert_eq!(placed, 4);
        assert!((score - solution.root_value()).abs() < 1e-12);

        let started = std::time::Instant::now();
        solution.write_root_json(io::sink()).unwrap();
        eprintln!("tree streamed in {:.3}s", started.elapsed().as_secs_f64());
    }
}
