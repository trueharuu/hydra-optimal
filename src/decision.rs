use crate::graph::{Graph, PC_COMPLETE_ID};
use crate::score::{pieces, Cost, MinArray, Weights, FULL_BAG};
use std::array;
use std::collections::VecDeque;
use std::fmt;
use std::io::{self, Write};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;
use std::thread;

/// One placement in the fully known suffix of a decision tree.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DecisionStep {
    pub field: u32,
    pub piece: u8,
}

/// An optimal decision tree and its accumulated failure/weight cost.
///
/// The representation intentionally follows the original solver's output model: uncertain
/// reveals form seven-way branches, while the suffix after the last reveal is a compact chain.
pub struct Decision<T: Cost> {
    cost: T,
    cap: T,
    kind: DecisionKind<T>,
}

enum DecisionKind<T: Cost> {
    Branch {
        field: u32,
        piece: u8,
        children: [Option<Box<Decision<T>>>; 7],
    },
    Solve {
        steps: Vec<DecisionStep>,
    },
    Capped,
}

impl<T: Cost> Decision<T> {
    #[inline]
    pub fn cost(&self) -> T {
        self.cost
    }

    #[inline]
    pub fn cap(&self) -> T {
        self.cap
    }

    /// Format the tree using the JavaScript array syntax consumed by `tree_viewer.html`.
    pub fn display<'a>(&'a self, graph: &'a Graph) -> DecisionDisplay<'a, T> {
        DecisionDisplay {
            decision: self,
            graph,
        }
    }

    /// Write a complete `tree_data.js` payload without allocating the serialized tree string.
    pub fn write_tree_data(
        &self,
        graph: &Graph,
        initial_field: u32,
        mut output: impl Write,
    ) -> io::Result<()> {
        writeln!(output, "init_hash={}", graph.hash(initial_field))?;
        write!(output, "data={}", self.display(graph))
    }

    fn capped(cost: T, cap: T) -> Self {
        Self {
            cost,
            cap,
            kind: DecisionKind::Capped,
        }
    }

    fn branch(field: u32, piece: u8, cap: T) -> Self {
        Self {
            cost: T::zero(),
            cap,
            kind: DecisionKind::Branch {
                field,
                piece,
                children: array::from_fn(|_| None),
            },
        }
    }

    fn solve(cost: T, cap: T, steps: Vec<DecisionStep>) -> Self {
        Self {
            cost,
            cap,
            kind: DecisionKind::Solve { steps },
        }
    }

    fn set_child(&mut self, piece: u8, child: Decision<T>) {
        let DecisionKind::Branch { children, .. } = &mut self.kind else {
            unreachable!("children are only valid on branch decisions");
        };
        children[piece as usize] = Some(Box::new(child));
    }

    fn fmt_with_hash(
        &self,
        output: &mut fmt::Formatter<'_>,
        hash: &impl Fn(u32) -> u64,
    ) -> fmt::Result {
        match &self.kind {
            DecisionKind::Branch {
                field,
                piece,
                children,
            } if self.cost < self.cap => {
                write!(
                    output,
                    "[{},{},{},[",
                    hash(*field),
                    piece,
                    self.cost.scaled_output()
                )?;
                for (index, child) in children.iter().enumerate() {
                    if index != 0 {
                        output.write_str(",")?;
                    }
                    match child {
                        Some(child) => child.fmt_with_hash(output, hash)?,
                        None => output.write_str("null")?,
                    }
                }
                output.write_str("]]")
            }
            DecisionKind::Solve { steps } => {
                write!(output, "[[{}]", self.cost.scaled_output())?;
                for step in steps {
                    write!(output, ",[{},{}]", hash(step.field), step.piece)?;
                }
                output.write_str("]")
            }
            DecisionKind::Branch { .. } | DecisionKind::Capped => {
                write!(output, "[-1,-1,{}]", self.cap.scaled_output())
            }
        }
    }
}

pub struct DecisionDisplay<'a, T: Cost> {
    decision: &'a Decision<T>,
    graph: &'a Graph,
}

impl<T: Cost> fmt::Display for DecisionDisplay<'_, T> {
    fn fmt(&self, output: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.decision
            .fmt_with_hash(output, &|field| self.graph.hash(field))
    }
}

/// Decision-tree search with deterministic first-placement tie breaking.
///
/// Only root placements are parallel. A zero-cost root job may cancel later jobs, but jobs with
/// smaller indices continue so an earlier equal strategy always wins, independent of scheduling.
pub struct DecisionSearch<'a, T: Cost> {
    graph: &'a Graph,
    weights: &'a Weights<T>,
    threads: usize,
    first_zero_job: AtomicUsize,
}

impl<'a, T: Cost> DecisionSearch<'a, T> {
    pub fn new(graph: &'a Graph, weights: &'a Weights<T>, threads: usize) -> Self {
        Self {
            graph,
            weights,
            threads: threads.max(1),
            first_zero_job: AtomicUsize::new(usize::MAX),
        }
    }

    /// Find the optimal tree. The queue is restored before this method returns.
    ///
    /// The exclusive receiver prevents root-cancellation state from being shared by concurrent
    /// queries. Worker threads created for this query still share it internally.
    ///
    /// As in the original solver, `cutoffs` describes the number of hidden-piece outcomes at each
    /// reveal depth and `initial_limit` is normally `cutoffs[0]`.
    pub fn solve(
        &mut self,
        field: u32,
        hold: u8,
        queue: &mut VecDeque<u8>,
        bag: u8,
        cutoffs: &[T],
        initial_limit: T,
    ) -> Decision<T> {
        assert!(
            !cutoffs.is_empty(),
            "decision search needs at least one cutoff"
        );
        assert!(
            cutoffs.len() == 1 || !queue.is_empty(),
            "decision search needs a visible queue piece before a cutoff"
        );
        self.first_zero_job.store(usize::MAX, Ordering::Release);
        self.tree_imperfect(field, hold, queue, 0, initial_limit, bag, cutoffs, None)
            .expect("top-level decision job cannot be cancelled")
    }

    #[allow(clippy::too_many_arguments)]
    fn tree_imperfect(
        &self,
        field: u32,
        hold: u8,
        queue: &mut VecDeque<u8>,
        depth: usize,
        mut limit: T,
        mut bag: u8,
        cutoffs: &[T],
        root_job: Option<usize>,
    ) -> Option<Decision<T>> {
        if self.cancelled(root_job) {
            return None;
        }
        if depth >= cutoffs.len() - 1 {
            let row = self.weights.get(bag);
            let known = self.tree_perfect_single(field, hold, queue, row, limit, root_job)?;
            return Some(match known.steps {
                Some(steps) => Decision::solve(known.cost, T::one() - row.min, steps),
                None => Decision::capped(known.cost, T::one() - row.min),
            });
        }
        if bag == 0 {
            bag = FULL_BAG;
        }

        let front = queue
            .pop_front()
            .expect("cutoff requires another visible piece");
        limit = limit.min(cutoffs[depth]);

        let result = if depth == 0 {
            Some(self.tree_root_parallel(field, front, hold, queue, bag, cutoffs, limit))
        } else {
            self.tree_after_front(
                field, front, hold, queue, depth, limit, bag, cutoffs, root_job,
            )
        };

        queue.push_front(front);
        result
    }

    #[allow(clippy::too_many_arguments)]
    fn tree_after_front(
        &self,
        field: u32,
        front: u8,
        hold: u8,
        queue: &mut VecDeque<u8>,
        depth: usize,
        limit: T,
        bag: u8,
        cutoffs: &[T],
        root_job: Option<usize>,
    ) -> Option<Decision<T>> {
        let mut best = Decision::capped(limit, cutoffs[depth]);

        for &next in self.graph.edges(field, front) {
            let candidate = self.tree_action(
                next, front, hold, queue, depth, bag, cutoffs, best.cost, root_job,
            )?;
            if candidate.cost < best.cost {
                best = candidate;
                if best.cost.is_zero() {
                    return Some(best);
                }
            }
        }

        if front != hold {
            for &next in self.graph.edges(field, hold) {
                let candidate = self.tree_action(
                    next, hold, front, queue, depth, bag, cutoffs, best.cost, root_job,
                )?;
                if candidate.cost < best.cost {
                    best = candidate;
                    if best.cost.is_zero() {
                        break;
                    }
                }
            }
        }
        Some(best)
    }

    #[allow(clippy::too_many_arguments)]
    fn tree_action(
        &self,
        next_field: u32,
        placed_piece: u8,
        next_hold: u8,
        queue: &mut VecDeque<u8>,
        depth: usize,
        bag: u8,
        cutoffs: &[T],
        limit: T,
        root_job: Option<usize>,
    ) -> Option<Decision<T>> {
        let mut decision = Decision::branch(next_field, placed_piece, cutoffs[depth]);
        for piece in pieces(bag) {
            if self.cancelled(root_job) {
                return None;
            }
            if decision.cost >= limit {
                break;
            }
            queue.push_back(piece);
            let child = self.tree_imperfect(
                next_field,
                next_hold,
                queue,
                depth + 1,
                limit - decision.cost,
                bag & !(1 << piece),
                cutoffs,
                root_job,
            );
            queue.pop_back();
            let child = child?;
            decision.cost = decision.cost + child.cost;
            decision.set_child(piece, child);
        }
        Some(decision)
    }

    fn tree_perfect_single(
        &self,
        field: u32,
        hold: u8,
        queue: &mut VecDeque<u8>,
        weights: &MinArray<T>,
        limit: T,
        root_job: Option<usize>,
    ) -> Option<KnownResult<T>> {
        if self.cancelled(root_job) {
            return None;
        }
        if field == PC_COMPLETE_ID {
            return Some(KnownResult {
                cost: weights.values[hold as usize],
                steps: Some(Vec::new()),
            });
        }
        let failure = T::one() - weights.min;
        if queue.is_empty() {
            return Some(KnownResult {
                cost: failure,
                steps: None,
            });
        }

        let front = queue.pop_front().expect("queue checked nonempty");
        let mut best = KnownResult {
            cost: limit.min(failure),
            steps: None,
        };

        for &next in self.graph.edges(field, front) {
            let candidate =
                self.tree_perfect_single(next, hold, queue, weights, best.cost, root_job);
            let Some(mut candidate) = candidate else {
                queue.push_front(front);
                return None;
            };
            if candidate.cost < best.cost {
                if let Some(mut suffix) = candidate.steps.take() {
                    let mut steps = Vec::with_capacity(suffix.len() + 1);
                    steps.push(DecisionStep {
                        field: next,
                        piece: front,
                    });
                    steps.append(&mut suffix);
                    candidate.steps = Some(steps);
                    best = candidate;
                    if best.cost.is_zero() {
                        break;
                    }
                }
            }
        }

        if front != hold && !best.cost.is_zero() {
            for &next in self.graph.edges(field, hold) {
                let candidate =
                    self.tree_perfect_single(next, front, queue, weights, best.cost, root_job);
                let Some(mut candidate) = candidate else {
                    queue.push_front(front);
                    return None;
                };
                if candidate.cost < best.cost {
                    if let Some(mut suffix) = candidate.steps.take() {
                        let mut steps = Vec::with_capacity(suffix.len() + 1);
                        steps.push(DecisionStep {
                            field: next,
                            piece: hold,
                        });
                        steps.append(&mut suffix);
                        candidate.steps = Some(steps);
                        best = candidate;
                        if best.cost.is_zero() {
                            break;
                        }
                    }
                }
            }
        }

        queue.push_front(front);
        Some(best)
    }

    #[allow(clippy::too_many_arguments)]
    fn tree_root_parallel(
        &self,
        field: u32,
        front: u8,
        hold: u8,
        queue: &VecDeque<u8>,
        bag: u8,
        cutoffs: &[T],
        initial_limit: T,
    ) -> Decision<T> {
        let mut jobs: Vec<RootJob> = self
            .graph
            .edges(field, front)
            .iter()
            .copied()
            .map(|field| RootJob {
                field,
                piece: front,
                hold,
            })
            .collect();
        if front != hold {
            jobs.extend(
                self.graph
                    .edges(field, hold)
                    .iter()
                    .copied()
                    .map(|field| RootJob {
                        field,
                        piece: hold,
                        hold: front,
                    }),
            );
        }

        if jobs.is_empty() {
            return Decision::capped(initial_limit, cutoffs[0]);
        }

        let next_job = AtomicUsize::new(0);
        let best = Mutex::new(RootBest {
            decision: Decision::capped(initial_limit, cutoffs[0]),
            job: usize::MAX,
        });
        let workers = self.threads.min(jobs.len());

        thread::scope(|scope| {
            for _ in 0..workers {
                let best = &best;
                let next_job = &next_job;
                let jobs = &jobs;
                scope.spawn(move || loop {
                    let job_index = next_job.fetch_add(1, Ordering::Relaxed);
                    let Some(job) = jobs.get(job_index).copied() else {
                        break;
                    };
                    if self.cancelled(Some(job_index)) {
                        break;
                    }

                    let Some(candidate) =
                        self.tree_root_job(job_index, job, queue, bag, cutoffs, best)
                    else {
                        continue;
                    };

                    if candidate.cost.is_zero() {
                        self.first_zero_job.fetch_min(job_index, Ordering::AcqRel);
                    }
                    let mut current = best.lock().expect("root decision lock poisoned");
                    if root_candidate_wins(
                        candidate.cost,
                        job_index,
                        current.decision.cost,
                        current.job,
                    ) {
                        current.decision = candidate;
                        current.job = job_index;
                    }
                });
            }
        });

        best.into_inner()
            .expect("root decision lock poisoned")
            .decision
    }

    fn tree_root_job(
        &self,
        job_index: usize,
        job: RootJob,
        queue: &VecDeque<u8>,
        bag: u8,
        cutoffs: &[T],
        best: &Mutex<RootBest<T>>,
    ) -> Option<Decision<T>> {
        let mut queue = queue.clone();
        let mut decision = Decision::branch(job.field, job.piece, cutoffs[0]);

        for piece in pieces(bag) {
            if self.cancelled(Some(job_index)) {
                return None;
            }
            let (mut allowed, leader) = {
                let current = best.lock().expect("root decision lock poisoned");
                (current.decision.cost, current.job)
            };
            if job_index < leader {
                allowed = allowed + T::epsilon();
            }
            if decision.cost >= allowed {
                break;
            }

            queue.push_back(piece);
            let child = self.tree_imperfect(
                job.field,
                job.hold,
                &mut queue,
                1,
                allowed - decision.cost,
                bag & !(1 << piece),
                cutoffs,
                Some(job_index),
            );
            queue.pop_back();
            let child = child?;
            decision.cost = decision.cost + child.cost;
            decision.set_child(piece, child);
        }
        Some(decision)
    }

    #[inline]
    fn cancelled(&self, root_job: Option<usize>) -> bool {
        root_job.is_some_and(|job| job > self.first_zero_job.load(Ordering::Acquire))
    }
}

struct KnownResult<T: Cost> {
    cost: T,
    steps: Option<Vec<DecisionStep>>,
}

#[derive(Clone, Copy)]
struct RootJob {
    field: u32,
    piece: u8,
    hold: u8,
}

struct RootBest<T: Cost> {
    decision: Decision<T>,
    job: usize,
}

#[inline]
fn root_candidate_wins<T: Cost>(
    candidate_cost: T,
    candidate_job: usize,
    best_cost: T,
    best_job: usize,
) -> bool {
    candidate_cost < best_cost || (candidate_cost == best_cost && candidate_job < best_job)
}

#[cfg(test)]
mod tests {
    use super::*;

    struct TestDisplay<'a, T: Cost>(&'a Decision<T>);

    impl<T: Cost> fmt::Display for TestDisplay<'_, T> {
        fn fmt(&self, output: &mut fmt::Formatter<'_>) -> fmt::Result {
            self.0.fmt_with_hash(output, &|field| 100 + field as u64)
        }
    }

    #[test]
    fn serialization_matches_the_original_tree_shape() {
        let solve = Decision::solve(
            0u64,
            1,
            vec![
                DecisionStep {
                    field: 10,
                    piece: 3,
                },
                DecisionStep {
                    field: 20,
                    piece: 2,
                },
            ],
        );
        let mut branch = Decision::branch(3, 5, 4u64);
        branch.cost = 2;
        branch.set_child(0, solve);

        assert_eq!(
            TestDisplay(&branch).to_string(),
            "[103,5,2,[[[0],[110,3],[120,2]],null,null,null,null,null,null]]"
        );
    }

    #[test]
    fn capped_nodes_use_the_original_sentinel() {
        let capped = Decision::capped(2u64, 7);
        assert_eq!(TestDisplay(&capped).to_string(), "[-1,-1,7]");
    }

    #[test]
    fn an_earlier_root_job_wins_an_equal_score() {
        assert!(root_candidate_wins(4u64, 2, 4, 3));
        assert!(!root_candidate_wins(4u64, 4, 4, 3));
        assert!(root_candidate_wins(3u64, 9, 4, 3));
    }
}
