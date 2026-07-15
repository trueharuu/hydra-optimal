//! Exact see-7, V*-seeded decision-tree search.
//!
//! Unlike the immediate-PC search, a V*-optimal search must reveal a piece after every
//! placement.  Pieces revealed late in the current PC form the next boundary queue, so they can
//! change the best placement before the current PC finishes.  This module therefore keeps the
//! complete `(field, hold, queue6, bag)` state and applies the Bellman backup one reveal at a time.

use crate::graph::{Graph, NUM_FIELDS, PC_COMPLETE_ID};
use crate::score::FULL_BAG;
use anyhow::{bail, Context, Result};
use dashmap::DashMap;
use std::fs;
use std::hash::{BuildHasherDefault, Hasher};
use std::io::{self, Write};
use std::path::Path;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::mpsc;
use std::thread;

pub const VSTAR_RECORDS: usize = 1_120_140;
const VSTAR_MAGIC: &[u8; 8] = b"L0F32V1\0";
const VSTAR_HEADER_BYTES: usize = 16;
const VSTAR_RECORD_BYTES: usize = 8;
const QUEUE_RADIX: u32 = 7;
const QUEUE_STATES: u32 = QUEUE_RADIX.pow(6);
const NO_ACTION: u32 = u32::MAX;
const HOLD_ACTION: u32 = 1 << 24;
const FIELD_MASK: u32 = HOLD_ACTION - 1;

/// A keyed layer-0 V* table.
///
/// Records are sorted by the ordered boundary key
/// `hold << 24 | bag_mask << 17 | queue6`, where `queue6` is little-endian base 7.
pub struct VStarTable {
    keys: Vec<u32>,
    values: Vec<f32>,
}

impl VStarTable {
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let bytes = fs::read(path)
            .with_context(|| format!("failed to read V* table {}", path.display()))?;
        Self::parse(&bytes, VSTAR_RECORDS)
            .with_context(|| format!("failed to parse V* table {}", path.display()))
    }

    fn parse(bytes: &[u8], expected_records: usize) -> Result<Self> {
        if bytes.len() < VSTAR_HEADER_BYTES {
            bail!("V* table is too short: {} bytes", bytes.len());
        }
        if &bytes[..8] != VSTAR_MAGIC {
            bail!("invalid V* table magic; expected L0F32V1");
        }

        let count_u64 = u64::from_le_bytes(bytes[8..16].try_into().expect("8-byte count"));
        let count = usize::try_from(count_u64).context("V* record count does not fit usize")?;
        if count != expected_records {
            bail!("V* table has {count} records; expected {expected_records}");
        }
        let expected_bytes = count
            .checked_mul(VSTAR_RECORD_BYTES)
            .and_then(|body| body.checked_add(VSTAR_HEADER_BYTES))
            .context("V* table size overflow")?;
        if bytes.len() != expected_bytes {
            bail!(
                "V* table size mismatch: got {} bytes, expected {expected_bytes}",
                bytes.len()
            );
        }

        let mut keys = Vec::with_capacity(count);
        let mut values = Vec::with_capacity(count);
        let mut previous = None;
        for (index, record) in bytes[VSTAR_HEADER_BYTES..]
            .chunks_exact(VSTAR_RECORD_BYTES)
            .enumerate()
        {
            let key = u32::from_le_bytes(record[..4].try_into().expect("4-byte key"));
            if !valid_boundary_key(key) {
                bail!("invalid boundary key {key} at V* record {index}");
            }
            if previous.is_some_and(|old| key <= old) {
                bail!("V* keys are not strictly increasing at record {index}");
            }
            previous = Some(key);

            let value = f32::from_le_bytes(record[4..].try_into().expect("4-byte value"));
            if !value.is_finite() || value < 0.0 {
                bail!("invalid V* value {value} at record {index}");
            }
            keys.push(key);
            values.push(value);
        }
        Ok(Self { keys, values })
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.keys.len()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.keys.is_empty()
    }

    #[inline]
    pub fn get(&self, key: u32) -> Option<f32> {
        self.keys
            .binary_search(&key)
            .ok()
            .map(|index| self.values[index])
    }

    #[inline]
    pub fn boundary_value(&self, hold: u8, queue: [u8; 6], bag: u8) -> Option<f32> {
        self.get(boundary_key(hold, queue, bag))
    }
}

/// Search counters for one query.  Serialization does not change these counters.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct OptimalStats {
    pub states_evaluated: u64,
    pub memo_hits: u64,
    pub memo_entries: usize,
    pub actions_evaluated: u64,
    pub reveals_evaluated: u64,
    pub terminal_lookups: u64,
    pub dead_states: u64,
    pub max_depth: usize,
}

#[derive(Clone, Copy)]
struct SearchState {
    field: u32,
    hold: u8,
    queue: [u8; 6],
    bag: u8,
}

#[derive(Clone, Copy)]
struct RootJob {
    next_field: u32,
    used_hold: bool,
}

/// Fast deterministic hashing for the already-packed internal `u64` state keys.
///
/// The keys are trusted solver state, not attacker-controlled input, so SipHash's keyed hashing
/// is unnecessary overhead.  A SplitMix64 finalizer still disperses the structured bit fields
/// across both the bucket index and the hash-table fingerprint.
#[derive(Default)]
struct PackedHasher(u64);

impl Hasher for PackedHasher {
    #[inline]
    fn finish(&self) -> u64 {
        self.0
    }

    #[inline]
    fn write_u64(&mut self, value: u64) {
        let mut mixed = value;
        mixed ^= mixed >> 30;
        mixed = mixed.wrapping_mul(0xbf58_476d_1ce4_e5b9);
        mixed ^= mixed >> 27;
        mixed = mixed.wrapping_mul(0x94d0_49bb_1331_11eb);
        self.0 = mixed ^ (mixed >> 31);
    }

    fn write(&mut self, bytes: &[u8]) {
        // MemoMap only hashes u64 keys (and therefore calls write_u64), but keep the hasher a
        // complete implementation for diagnostics or future wrappers.
        let mut value = 0xcbf2_9ce4_8422_2325u64;
        for &byte in bytes {
            value ^= u64::from(byte);
            value = value.wrapping_mul(0x0000_0100_0000_01b3);
        }
        self.write_u64(value);
    }
}

type PackedBuildHasher = BuildHasherDefault<PackedHasher>;
/// The solve memo intentionally stores values only.
///
/// An action would make the naturally aligned payload 16 bytes instead of 8.  The exact action
/// can be reconstructed from child values while streaming the tree, using the same ordered strict
/// comparison as the solve.  With tens of millions of states this halves the dominant payload
/// without changing values, policy ties, or the serialized tree.
type MemoMap = DashMap<u64, f64, PackedBuildHasher>;

fn new_memo(shards: usize) -> MemoMap {
    MemoMap::with_hasher_and_shard_amount(PackedBuildHasher::default(), shards)
}

fn select_root_action(jobs: &[RootJob], values: &[f64]) -> (f64, u32) {
    debug_assert_eq!(jobs.len(), values.len());
    let mut best_value = 0.0;
    let mut best_action = NO_ACTION;
    for (&job, &value) in jobs.iter().zip(values) {
        // Strict comparison deliberately retains the first equal action.
        if value > best_value {
            best_value = value;
            best_action = encode_action(job.next_field, job.used_hold);
        }
    }
    (best_value, best_action)
}

#[derive(Default)]
struct AtomicOptimalStats {
    states_evaluated: AtomicU64,
    memo_hits: AtomicU64,
    actions_evaluated: AtomicU64,
    reveals_evaluated: AtomicU64,
    terminal_lookups: AtomicU64,
    dead_states: AtomicU64,
    max_depth: AtomicUsize,
}

impl AtomicOptimalStats {
    fn reset(&self) {
        self.states_evaluated.store(0, Ordering::Relaxed);
        self.memo_hits.store(0, Ordering::Relaxed);
        self.actions_evaluated.store(0, Ordering::Relaxed);
        self.reveals_evaluated.store(0, Ordering::Relaxed);
        self.terminal_lookups.store(0, Ordering::Relaxed);
        self.dead_states.store(0, Ordering::Relaxed);
        self.max_depth.store(0, Ordering::Relaxed);
    }

    fn snapshot(&self, memo_entries: usize) -> OptimalStats {
        OptimalStats {
            states_evaluated: self.states_evaluated.load(Ordering::Relaxed),
            memo_hits: self.memo_hits.load(Ordering::Relaxed),
            memo_entries,
            actions_evaluated: self.actions_evaluated.load(Ordering::Relaxed),
            reveals_evaluated: self.reveals_evaluated.load(Ordering::Relaxed),
            terminal_lookups: self.terminal_lookups.load(Ordering::Relaxed),
            dead_states: self.dead_states.load(Ordering::Relaxed),
            max_depth: self.max_depth.load(Ordering::Relaxed),
        }
    }

    /// Publish one worker's counters in a handful of atomics after its root job has finished.
    ///
    /// Updating these counters in the recursive hot path made every worker contend on the same
    /// cache lines once per action and reveal.  Keeping them local preserves the diagnostics while
    /// leaving the Bellman search itself free of global counter traffic.
    fn add(&self, stats: OptimalStats) {
        self.states_evaluated
            .fetch_add(stats.states_evaluated, Ordering::Relaxed);
        self.memo_hits.fetch_add(stats.memo_hits, Ordering::Relaxed);
        self.actions_evaluated
            .fetch_add(stats.actions_evaluated, Ordering::Relaxed);
        self.reveals_evaluated
            .fetch_add(stats.reveals_evaluated, Ordering::Relaxed);
        self.terminal_lookups
            .fetch_add(stats.terminal_lookups, Ordering::Relaxed);
        self.dead_states
            .fetch_add(stats.dead_states, Ordering::Relaxed);
        self.max_depth.fetch_max(stats.max_depth, Ordering::Relaxed);
    }
}

/// A memoized Bellman search with deterministic root-action parallelism.
///
/// The memo table stores one `u64` state key and one compact value/action record per visited
/// state.  Root workers share the same sharded table; recursion within a root job stays serial.
pub struct OptimalSearch<'a> {
    graph: &'a Graph,
    vstar: &'a VStarTable,
    threads: usize,
    memo_shards: usize,
    memo: MemoMap,
    stats: AtomicOptimalStats,
}

impl<'a> OptimalSearch<'a> {
    pub fn new(graph: &'a Graph, vstar: &'a VStarTable, threads: usize) -> Self {
        let threads = threads.max(1);
        // DashMap requires a power-of-two shard count.  Several shards per worker reduce lock
        // collisions when different root jobs converge on nearby packed states.
        let memo_shards = threads.next_power_of_two().saturating_mul(4).max(4);
        Self {
            graph,
            vstar,
            threads,
            memo_shards,
            memo: new_memo(memo_shards),
            stats: AtomicOptimalStats::default(),
        }
    }

    /// Solve from an arbitrary graph field and a complete see-7 chain state.
    ///
    /// `queue[0]` is the active piece.  `bag` is the non-empty set from which the piece revealed
    /// after the next placement is sampled uniformly.  The returned solution owns the memo table
    /// needed to stream the decision tree, leaving this search object ready for another query.
    pub fn solve(
        &mut self,
        field: u32,
        hold: u8,
        queue: [u8; 6],
        bag: u8,
    ) -> Result<OptimalSolution<'a>> {
        validate_chain(hold, &queue, bag)?;
        let initial_chain_key = boundary_key(hold, queue, bag);
        if self.vstar.get(initial_chain_key).is_none() {
            bail!("initial see-7 chain is not present in the V* table (key {initial_chain_key})");
        }
        if field as usize >= NUM_FIELDS {
            bail!("field {field} is out of range");
        }

        self.memo.clear();
        self.stats.reset();
        let root = SearchState {
            field,
            hold,
            queue,
            bag,
        };
        let root_value = if self.is_terminal(root.field) {
            let mut stats = OptimalStats::default();
            let value = self.state_value(root, 0, &mut stats)?;
            self.stats.add(stats);
            value
        } else {
            self.root_value_parallel(root)?
        };
        let memo = std::mem::replace(&mut self.memo, new_memo(self.memo_shards));
        let stats = self.stats.snapshot(memo.len());

        Ok(OptimalSolution {
            graph: self.graph,
            vstar: self.vstar,
            root,
            root_value,
            memo,
            stats,
        })
    }

    fn root_value_parallel(&self, state: SearchState) -> Result<f64> {
        let mut root_stats = OptimalStats {
            states_evaluated: 1,
            ..OptimalStats::default()
        };

        let active = state.queue[0];
        let mut jobs: Vec<RootJob> = self
            .graph
            .edges(state.field, active)
            .iter()
            .copied()
            .map(|next_field| RootJob {
                next_field,
                used_hold: false,
            })
            .collect();
        if active != state.hold {
            jobs.extend(
                self.graph
                    .edges(state.field, state.hold)
                    .iter()
                    .copied()
                    .map(|next_field| RootJob {
                        next_field,
                        used_hold: true,
                    }),
            );
        }

        let mut results: Vec<Option<Result<(f64, OptimalStats)>>> =
            std::iter::repeat_with(|| None).take(jobs.len()).collect();
        if self.threads == 1 || jobs.len() <= 1 {
            for (index, job) in jobs.iter().copied().enumerate() {
                let mut stats = OptimalStats::default();
                results[index] = Some(
                    self.action_value(state, job.next_field, job.used_hold, 0, &mut stats)
                        .map(|value| (value, stats)),
                );
            }
        } else {
            let next_job = AtomicUsize::new(0);
            let workers = self.threads.min(jobs.len());
            let (sender, receiver) = mpsc::channel();

            thread::scope(|scope| -> Result<()> {
                for _ in 0..workers {
                    let sender = sender.clone();
                    let jobs = &jobs;
                    let next_job = &next_job;
                    scope.spawn(move || loop {
                        let index = next_job.fetch_add(1, Ordering::Relaxed);
                        let Some(job) = jobs.get(index).copied() else {
                            break;
                        };
                        let mut stats = OptimalStats::default();
                        let result = self
                            .action_value(state, job.next_field, job.used_hold, 0, &mut stats)
                            .map(|value| (value, stats));
                        if sender.send((index, result)).is_err() {
                            break;
                        }
                    });
                }
                drop(sender);
                for _ in 0..jobs.len() {
                    let (index, result) = receiver
                        .recv()
                        .context("optimal root worker stopped before returning every action")?;
                    results[index] = Some(result);
                }
                Ok(())
            })?;
        }

        // Consume results in original graph/action order.  Completion order can therefore never
        // change a strict-first tie, regardless of worker scheduling.
        let values: Vec<f64> = results
            .into_iter()
            .map(|result| {
                let (value, stats) =
                    result.expect("every root job produced a result in its original slot")?;
                accumulate_stats(&mut root_stats, stats);
                Ok(value)
            })
            .collect::<Result<_>>()?;
        let (best_value, best_action) = select_root_action(&jobs, &values);
        if best_action == NO_ACTION {
            root_stats.dead_states += 1;
        }
        self.memo.insert(state_key(state), best_value);
        self.stats.add(root_stats);
        Ok(best_value)
    }

    fn state_value(
        &self,
        state: SearchState,
        depth: usize,
        stats: &mut OptimalStats,
    ) -> Result<f64> {
        stats.max_depth = stats.max_depth.max(depth);
        if self.is_terminal(state.field) {
            return self.terminal_value(state.hold, state.queue, state.bag, stats);
        }

        let key = state_key(state);
        if let Some(entry) = self.memo.get(&key) {
            let value = *entry;
            drop(entry);
            stats.memo_hits += 1;
            return Ok(value);
        }
        stats.states_evaluated += 1;

        let active = state.queue[0];
        let mut best_value = 0.0;
        let mut best_action = NO_ACTION;

        // Strict `>` keeps the first equal action.  The loop order is part of the public policy:
        // active placements in graph order, followed by hold placements in graph order.
        for &next_field in self.graph.edges(state.field, active) {
            let value = self.action_value(state, next_field, false, depth, stats)?;
            if value > best_value {
                best_value = value;
                best_action = encode_action(next_field, false);
            }
        }
        if active != state.hold {
            for &next_field in self.graph.edges(state.field, state.hold) {
                let value = self.action_value(state, next_field, true, depth, stats)?;
                if value > best_value {
                    best_value = value;
                    best_action = encode_action(next_field, true);
                }
            }
        }

        if best_action == NO_ACTION {
            stats.dead_states += 1;
        }
        self.memo.insert(key, best_value);
        Ok(best_value)
    }

    fn action_value(
        &self,
        state: SearchState,
        next_field: u32,
        used_hold: bool,
        depth: usize,
        stats: &mut OptimalStats,
    ) -> Result<f64> {
        stats.actions_evaluated += 1;
        let active = state.queue[0];
        let next_hold = if used_hold { active } else { state.hold };
        let terminal = self.is_terminal(next_field);
        let mut sum = 0.0;
        let mut count = 0u32;

        for revealed in mask_pieces(state.bag) {
            stats.reveals_evaluated += 1;
            let (next_queue, next_bag) = reveal(state.queue, state.bag, revealed);
            let value = if terminal {
                self.terminal_value(next_hold, next_queue, next_bag, stats)?
            } else {
                self.state_value(
                    SearchState {
                        field: next_field,
                        hold: next_hold,
                        queue: next_queue,
                        bag: next_bag,
                    },
                    depth + 1,
                    stats,
                )?
            };
            sum += value;
            count += 1;
        }
        debug_assert!(count > 0, "canonical bag is nonempty");
        Ok(sum / f64::from(count))
    }

    #[inline]
    fn is_terminal(&self, field: u32) -> bool {
        field == PC_COMPLETE_ID || field == self.graph.two_line_field()
    }

    fn terminal_value(
        &self,
        hold: u8,
        queue: [u8; 6],
        bag: u8,
        stats: &mut OptimalStats,
    ) -> Result<f64> {
        stats.terminal_lookups += 1;
        lookup_terminal(self.vstar, hold, queue, bag)
    }
}

fn accumulate_stats(total: &mut OptimalStats, part: OptimalStats) {
    total.states_evaluated += part.states_evaluated;
    total.memo_hits += part.memo_hits;
    total.actions_evaluated += part.actions_evaluated;
    total.reveals_evaluated += part.reveals_evaluated;
    total.terminal_lookups += part.terminal_lookups;
    total.dead_states += part.dead_states;
    total.max_depth = total.max_depth.max(part.max_depth);
}

/// A solved policy and the compact memo table used to stream its decision tree.
pub struct OptimalSolution<'a> {
    graph: &'a Graph,
    vstar: &'a VStarTable,
    root: SearchState,
    root_value: f64,
    memo: MemoMap,
    stats: OptimalStats,
}

impl OptimalSolution<'_> {
    #[inline]
    pub fn root_value(&self) -> f64 {
        self.root_value
    }

    #[inline]
    pub fn stats(&self) -> OptimalStats {
        self.stats
    }

    /// Stream a `tree_data.js` payload compatible with the bundled seven-child tree viewer.
    /// Scores are positive expected-PC values; the objective header lets the updated viewer avoid
    /// applying the legacy failure-cost negation.
    pub fn write_tree_data(&self, mut output: impl Write) -> io::Result<()> {
        writeln!(output, "init_hash={}", self.graph.hash(self.root.field))?;
        writeln!(output, "objective=\"expected_pc\"")?;
        write!(output, "data=")?;
        self.write_state(self.root, &mut output)
    }

    fn write_state(&self, state: SearchState, output: &mut impl Write) -> io::Result<()> {
        if self.is_terminal(state.field) {
            return self.write_terminal(state.hold, state.queue, state.bag, output);
        }

        let state_value = *self.memo.get(&state_key(state)).ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                "optimal policy memo is missing a serialized state",
            )
        })?;
        let action = self.policy_action(state)?;
        if action == NO_ACTION {
            return output.write_all(b"[-1,-1,0]");
        }

        let (next_field, used_hold) = decode_action(action);
        let active = state.queue[0];
        let placed = if used_hold { state.hold } else { active };
        let next_hold = if used_hold { active } else { state.hold };
        write!(
            output,
            "[{},{},{},[",
            self.graph.hash(next_field),
            placed,
            state_value
        )?;

        for revealed in 0u8..7 {
            if revealed != 0 {
                output.write_all(b",")?;
            }
            if state.bag & (1 << revealed) == 0 {
                output.write_all(b"null")?;
                continue;
            }
            let (next_queue, next_bag) = reveal(state.queue, state.bag, revealed);
            if self.is_terminal(next_field) {
                self.write_terminal(next_hold, next_queue, next_bag, output)?;
            } else {
                self.write_state(
                    SearchState {
                        field: next_field,
                        hold: next_hold,
                        queue: next_queue,
                        bag: next_bag,
                    },
                    output,
                )?;
            }
        }
        output.write_all(b"]]")
    }

    /// Recover the strict-first maximizing action from the solved value memo.
    ///
    /// This repeats only the cheap Bellman reads during serialization; it does not recurse into
    /// the solver or add memo entries.  Iteration and summation order exactly match `state_value`.
    fn policy_action(&self, state: SearchState) -> io::Result<u32> {
        let active = state.queue[0];
        let mut best_value = 0.0;
        let mut best_action = NO_ACTION;

        for &next_field in self.graph.edges(state.field, active) {
            let value = self.policy_action_value(state, next_field, false)?;
            if value > best_value {
                best_value = value;
                best_action = encode_action(next_field, false);
            }
        }
        if active != state.hold {
            for &next_field in self.graph.edges(state.field, state.hold) {
                let value = self.policy_action_value(state, next_field, true)?;
                if value > best_value {
                    best_value = value;
                    best_action = encode_action(next_field, true);
                }
            }
        }
        Ok(best_action)
    }

    fn policy_action_value(
        &self,
        state: SearchState,
        next_field: u32,
        used_hold: bool,
    ) -> io::Result<f64> {
        let active = state.queue[0];
        let next_hold = if used_hold { active } else { state.hold };
        let terminal = self.is_terminal(next_field);
        let mut sum = 0.0;
        let mut count = 0u32;

        for revealed in mask_pieces(state.bag) {
            let (next_queue, next_bag) = reveal(state.queue, state.bag, revealed);
            let value = if terminal {
                lookup_terminal(self.vstar, next_hold, next_queue, next_bag).map_err(|error| {
                    io::Error::new(
                        io::ErrorKind::InvalidData,
                        format!("optimal terminal disappeared during serialization: {error:#}"),
                    )
                })?
            } else {
                let child = SearchState {
                    field: next_field,
                    hold: next_hold,
                    queue: next_queue,
                    bag: next_bag,
                };
                *self.memo.get(&state_key(child)).ok_or_else(|| {
                    io::Error::new(
                        io::ErrorKind::InvalidData,
                        "optimal policy memo is missing a serialized child state",
                    )
                })?
            };
            sum += value;
            count += 1;
        }
        debug_assert!(count > 0, "canonical bag is nonempty");
        Ok(sum / f64::from(count))
    }

    fn write_terminal(
        &self,
        hold: u8,
        queue: [u8; 6],
        bag: u8,
        output: &mut impl Write,
    ) -> io::Result<()> {
        let value = lookup_terminal(self.vstar, hold, queue, bag).map_err(|error| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!("optimal terminal disappeared during serialization: {error:#}"),
            )
        })?;
        write!(output, "[[{value}]]")
    }

    #[inline]
    fn is_terminal(&self, field: u32) -> bool {
        field == PC_COMPLETE_ID || field == self.graph.two_line_field()
    }
}

#[inline]
pub fn encode_queue6(queue: [u8; 6]) -> u32 {
    let mut encoded = 0u32;
    let mut place = 1u32;
    for piece in queue {
        encoded += u32::from(piece) * place;
        place *= QUEUE_RADIX;
    }
    encoded
}

#[inline]
pub fn boundary_key(hold: u8, queue: [u8; 6], bag: u8) -> u32 {
    (u32::from(hold) << 24) | (u32::from(bag) << 17) | encode_queue6(queue)
}

fn valid_boundary_key(key: u32) -> bool {
    if key >> 27 != 0 {
        return false;
    }
    let hold = (key >> 24) as u8;
    let bag = ((key >> 17) & 0x7f) as u8;
    let mut queue_code = key & 0x1ffff;
    if hold >= 7 || bag == 0 || queue_code >= QUEUE_STATES {
        return false;
    }

    let mut queue = [0u8; 6];
    for piece in &mut queue {
        *piece = (queue_code % QUEUE_RADIX) as u8;
        queue_code /= QUEUE_RADIX;
    }

    // This is the canonical chain-state validity rule used by the MDP builder.  The prefix is
    // made of already revealed pieces from the current bag and must be duplicate-free.  The
    // suffix is exactly the previous bag's pieces, i.e. the complement of the current bag.
    let prefix_len = bag.count_ones() as usize - 1;
    let mut prefix_seen = 0u8;
    for &piece in &queue[..prefix_len] {
        let bit = 1 << piece;
        if prefix_seen & bit != 0 {
            return false;
        }
        prefix_seen |= bit;
    }

    let mut suffix_seen = 0u8;
    for &piece in &queue[prefix_len..] {
        let bit = 1 << piece;
        if suffix_seen & bit != 0 {
            return false;
        }
        suffix_seen |= bit;
    }
    suffix_seen == (FULL_BAG ^ bag)
}

fn validate_chain(hold: u8, queue: &[u8; 6], bag: u8) -> Result<()> {
    if hold >= 7 {
        bail!("invalid hold piece {hold}");
    }
    if queue.iter().any(|&piece| piece >= 7) {
        bail!("queue contains an invalid piece");
    }
    if bag == 0 || bag & !FULL_BAG != 0 {
        bail!("bag must be a nonempty 7-bit mask");
    }
    Ok(())
}

#[inline]
fn state_key(state: SearchState) -> u64 {
    // 24 field bits + 3 hold bits + 17 queue bits + 7 bag bits = 51 bits.
    u64::from(state.field)
        | (u64::from(state.hold) << 24)
        | (u64::from(encode_queue6(state.queue)) << 27)
        | (u64::from(state.bag) << 44)
}

#[inline]
fn encode_action(field: u32, used_hold: bool) -> u32 {
    debug_assert!(field <= FIELD_MASK);
    field | if used_hold { HOLD_ACTION } else { 0 }
}

#[inline]
fn decode_action(action: u32) -> (u32, bool) {
    (action & FIELD_MASK, action & HOLD_ACTION != 0)
}

#[inline]
fn reveal(queue: [u8; 6], bag: u8, piece: u8) -> ([u8; 6], u8) {
    debug_assert!(bag & (1 << piece) != 0);
    let queue = [queue[1], queue[2], queue[3], queue[4], queue[5], piece];
    let remaining = bag & !(1 << piece);
    let bag = if remaining == 0 { FULL_BAG } else { remaining };
    (queue, bag)
}

#[inline]
fn lookup_terminal(vstar: &VStarTable, hold: u8, queue: [u8; 6], bag: u8) -> Result<f64> {
    let key = boundary_key(hold, queue, bag);
    let value = vstar
        .get(key)
        .with_context(|| format!("V* table is missing boundary key {key}"))?;
    Ok(1.0 + f64::from(value))
}

struct MaskPieces {
    mask: u8,
}

impl Iterator for MaskPieces {
    type Item = u8;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.mask == 0 {
            return None;
        }
        let piece = self.mask.trailing_zeros() as u8;
        self.mask &= self.mask - 1;
        Some(piece)
    }
}

#[inline]
fn mask_pieces(mask: u8) -> MaskPieces {
    MaskPieces { mask }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn table_bytes(records: &[(u32, f32)]) -> Vec<u8> {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(VSTAR_MAGIC);
        bytes.extend_from_slice(&(records.len() as u64).to_le_bytes());
        for &(key, value) in records {
            bytes.extend_from_slice(&key.to_le_bytes());
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        bytes
    }

    fn valid_key(hold: u8) -> u32 {
        // Full bag => five-piece prefix plus an empty suffix.  Any five distinct pieces are valid.
        boundary_key(hold, [0, 1, 2, 3, 4, 5], FULL_BAG)
    }

    #[test]
    fn queue_encoding_is_little_endian_base_seven() {
        assert_eq!(
            encode_queue6([1, 2, 3, 4, 5, 6]),
            1 + 2 * 7 + 3 * 49 + 4 * 343 + 5 * 2401 + 6 * 16807
        );
        assert_eq!(
            boundary_key(3, [1, 2, 3, 4, 5, 6], 0x55),
            (3 << 24) | (0x55 << 17) | encode_queue6([1, 2, 3, 4, 5, 6])
        );
    }

    #[test]
    fn keyed_f32_table_roundtrips() {
        let records = [(valid_key(0), 4.25), (valid_key(1), 9.5)];
        let table = VStarTable::parse(&table_bytes(&records), records.len()).unwrap();
        assert_eq!(table.len(), 2);
        assert_eq!(table.get(records[0].0), Some(4.25));
        assert_eq!(table.get(records[1].0), Some(9.5));
        assert_eq!(table.get(valid_key(2)), None);
    }

    #[test]
    fn table_rejects_bad_shape_order_keys_and_values() {
        let key0 = valid_key(0);
        let key1 = valid_key(1);

        let mut bad_magic = table_bytes(&[(key0, 1.0)]);
        bad_magic[0] = b'X';
        assert!(VStarTable::parse(&bad_magic, 1).is_err());

        let mut truncated = table_bytes(&[(key0, 1.0)]);
        truncated.pop();
        assert!(VStarTable::parse(&truncated, 1).is_err());

        assert!(VStarTable::parse(&table_bytes(&[(key1, 1.0), (key0, 2.0)]), 2).is_err());
        assert!(VStarTable::parse(&table_bytes(&[(key0, 1.0), (key0, 2.0)]), 2).is_err());
        assert!(VStarTable::parse(&table_bytes(&[(key0, f32::NAN)]), 1).is_err());
        assert!(VStarTable::parse(&table_bytes(&[(key0, -1.0)]), 1).is_err());

        let structurally_invalid = boundary_key(0, [0; 6], FULL_BAG);
        assert!(VStarTable::parse(&table_bytes(&[(structurally_invalid, 1.0)]), 1).is_err());
    }

    #[test]
    fn reveal_shifts_the_queue_and_refills_an_empty_bag() {
        let (queue, bag) = reveal([0, 1, 2, 3, 4, 5], 1 << 6, 6);
        assert_eq!(queue, [1, 2, 3, 4, 5, 6]);
        assert_eq!(bag, FULL_BAG);

        let (queue, bag) = reveal(queue, (1 << 1) | (1 << 4), 1);
        assert_eq!(queue, [2, 3, 4, 5, 6, 1]);
        assert_eq!(bag, 1 << 4);
    }

    #[test]
    fn state_and_action_encodings_roundtrip_without_overlap() {
        let state = SearchState {
            field: 15_000_000,
            hold: 6,
            queue: [6, 5, 4, 3, 2, 1],
            bag: 0x55,
        };
        let other = SearchState { bag: 0x54, ..state };
        assert_ne!(state_key(state), state_key(other));

        for used_hold in [false, true] {
            let encoded = encode_action(state.field, used_hold);
            assert_eq!(decode_action(encoded), (state.field, used_hold));
        }
    }

    #[test]
    fn root_selection_is_strict_first_independent_of_completion_order() {
        let jobs = [
            RootJob {
                next_field: 10,
                used_hold: false,
            },
            RootJob {
                next_field: 20,
                used_hold: false,
            },
            RootJob {
                next_field: 30,
                used_hold: true,
            },
        ];

        // A parallel collector may receive jobs as 2, 0, 1, but it stores each score in its
        // original slot.  Selection over those slots must match the one-worker order.
        let mut parallel_slots = [0.0; 3];
        for (index, value) in [(2, 9.0), (0, 9.0), (1, 9.0)] {
            parallel_slots[index] = value;
        }
        let serial = select_root_action(&jobs, &[9.0, 9.0, 9.0]);
        let parallel = select_root_action(&jobs, &parallel_slots);
        assert_eq!(serial, parallel);
        assert_eq!(decode_action(serial.1), (10, false));

        let later_strictly_better = select_root_action(&jobs, &[9.0, 10.0, 10.0]);
        assert_eq!(decode_action(later_strictly_better.1), (20, false));
    }
}
