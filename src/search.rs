use crate::graph::{Graph, NUM_FIELDS};
use crate::score::{pieces, Cost, MinArray, Weights, FULL_BAG};
use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Mutex;
use std::thread;

pub struct Search<'a, T: Cost> {
    graph: &'a Graph,
    weights: &'a Weights<T>,
    two_line: bool,
    threads: usize,
    killed: AtomicBool,
}

impl<'a, T: Cost> Search<'a, T> {
    pub fn new(graph: &'a Graph, weights: &'a Weights<T>, two_line: bool, threads: usize) -> Self {
        Self {
            graph,
            weights,
            two_line,
            threads: threads.max(1),
            killed: AtomicBool::new(false),
        }
    }

    /// Score one query.
    ///
    /// The exclusive receiver prevents cancellation state from being shared by concurrent
    /// queries. Worker threads created for this query still share it internally.
    pub fn score(
        &mut self,
        field: u32,
        hold: u8,
        queue: &mut VecDeque<u8>,
        bag: u8,
        cutoffs: &[T],
        initial_limit: T,
    ) -> T {
        assert!(
            !cutoffs.is_empty(),
            "score search needs at least one cutoff"
        );
        assert!(
            cutoffs.len() == 1 || !queue.is_empty(),
            "score search needs a visible queue piece before a cutoff"
        );
        self.killed.store(false, Ordering::Relaxed);
        self.score_imperfect(field, hold, queue, 0, initial_limit, bag, cutoffs)
    }

    fn score_perfect_single(
        &self,
        field: u32,
        hold: u8,
        queue: &mut VecDeque<u8>,
        weights: &MinArray<T>,
    ) -> T {
        if field as usize == NUM_FIELDS - 1 {
            return weights.values[hold as usize];
        }
        if self.two_line && field == self.graph.two_line_field() {
            return T::zero();
        }
        if queue.is_empty() {
            return T::one() - weights.min;
        }

        let mut best_cost = T::one() - weights.min;
        let front = queue.pop_front().expect("queue checked nonempty");

        for &next in self.graph.edges(field, front) {
            if self.killed.load(Ordering::Relaxed) {
                break;
            }
            let candidate = self.score_perfect_single(next, hold, queue, weights);
            if candidate < best_cost {
                best_cost = candidate;
                if best_cost.is_zero() {
                    break;
                }
            }
        }

        if front != hold && !best_cost.is_zero() {
            for &next in self.graph.edges(field, hold) {
                if self.killed.load(Ordering::Relaxed) {
                    break;
                }
                let candidate = self.score_perfect_single(next, front, queue, weights);
                if candidate < best_cost {
                    best_cost = candidate;
                    if best_cost.is_zero() {
                        break;
                    }
                }
            }
        }

        queue.push_front(front);
        best_cost
    }

    fn score_perfect_multiple(
        &self,
        field: u32,
        hold: u8,
        queue: &mut VecDeque<u8>,
        weight_map: &[Option<MinArray<T>>; 7],
        options: &mut [Option<T>; 7],
    ) -> T {
        if self.two_line && field == self.graph.two_line_field() {
            options.fill(None);
        }
        if options.iter().all(Option::is_none) {
            return T::zero();
        }

        if queue.is_empty() {
            if self.graph.has_edges(field, hold) {
                for piece in 0..7 {
                    if let Some(value) = options[piece] {
                        let row = weight_map[piece].as_ref().expect("weight row for option");
                        let next = value.min(row.values[piece]);
                        options[piece] = (!next.is_zero()).then_some(next);
                    }
                }
            }

            for piece in 0..7u8 {
                if let Some(value) = options[piece as usize] {
                    if self.graph.has_edges(field, piece) {
                        let row = weight_map[piece as usize]
                            .as_ref()
                            .expect("weight row for option");
                        let next = value.min(row.values[hold as usize]);
                        options[piece as usize] = (!next.is_zero()).then_some(next);
                    }
                }
            }
            return sum_options(options);
        }

        let front = queue.pop_front().expect("queue checked nonempty");
        for &next in self.graph.edges(field, front) {
            if self.killed.load(Ordering::Relaxed)
                || self
                    .score_perfect_multiple(next, hold, queue, weight_map, options)
                    .is_zero()
            {
                break;
            }
        }

        if front != hold && !sum_options(options).is_zero() {
            for &next in self.graph.edges(field, hold) {
                if self.killed.load(Ordering::Relaxed)
                    || self
                        .score_perfect_multiple(next, front, queue, weight_map, options)
                        .is_zero()
                {
                    break;
                }
            }
        }
        queue.push_front(front);
        sum_options(options)
    }

    #[allow(clippy::too_many_arguments)]
    fn score_imperfect(
        &self,
        field: u32,
        hold: u8,
        queue: &mut VecDeque<u8>,
        placed: usize,
        mut limit: T,
        mut bag: u8,
        cutoffs: &[T],
    ) -> T {
        if placed >= cutoffs.len() - 1 {
            return self.score_perfect_single(field, hold, queue, self.weights.get(bag));
        }
        if self.two_line && field == self.graph.two_line_field() {
            return T::zero();
        }
        if bag == 0 {
            bag = FULL_BAG;
        }

        let front = queue
            .pop_front()
            .expect("cutoff requires another visible piece");
        limit = limit.min(cutoffs[placed]);

        if placed == cutoffs.len() - 2 && bag.count_ones() > 1 {
            let mut weight_map: [Option<MinArray<T>>; 7] = [None; 7];
            let mut initial_options: [Option<T>; 7] = [None; 7];
            for piece in pieces(bag) {
                let row = *self.weights.get(bag & !(1 << piece));
                weight_map[piece as usize] = Some(row);
                initial_options[piece as usize] = Some(T::one() - row.min);
            }

            for &next in self.graph.edges(field, front) {
                if self.killed.load(Ordering::Relaxed) {
                    break;
                }
                let mut options = initial_options;
                let candidate =
                    self.score_perfect_multiple(next, hold, queue, &weight_map, &mut options);
                limit = limit.min(candidate);
                if limit.is_zero() {
                    break;
                }
            }
            if front != hold && !limit.is_zero() {
                for &next in self.graph.edges(field, hold) {
                    if self.killed.load(Ordering::Relaxed) {
                        break;
                    }
                    let mut options = initial_options;
                    let candidate =
                        self.score_perfect_multiple(next, front, queue, &weight_map, &mut options);
                    limit = limit.min(candidate);
                    if limit.is_zero() {
                        break;
                    }
                }
            }
            queue.push_front(front);
            return limit;
        }

        if placed == 0 && self.threads > 1 {
            let result = self.score_root_parallel(field, front, hold, queue, bag, cutoffs, limit);
            queue.push_front(front);
            return result;
        }

        for &next in self.graph.edges(field, front) {
            let mut total = T::zero();
            for piece in pieces(bag) {
                if self.killed.load(Ordering::Relaxed) {
                    queue.push_front(front);
                    return limit;
                }
                queue.push_back(piece);
                let candidate = self.score_imperfect(
                    next,
                    hold,
                    queue,
                    placed + 1,
                    limit - total,
                    bag & !(1 << piece),
                    cutoffs,
                );
                queue.pop_back();
                total = total + candidate;
                if total >= limit {
                    break;
                }
            }
            limit = limit.min(total);
            if limit.is_zero() {
                break;
            }
        }

        if front != hold && !limit.is_zero() {
            for &next in self.graph.edges(field, hold) {
                let mut total = T::zero();
                for piece in pieces(bag) {
                    if self.killed.load(Ordering::Relaxed) {
                        queue.push_front(front);
                        return limit;
                    }
                    queue.push_back(piece);
                    let candidate = self.score_imperfect(
                        next,
                        front,
                        queue,
                        placed + 1,
                        limit - total,
                        bag & !(1 << piece),
                        cutoffs,
                    );
                    queue.pop_back();
                    total = total + candidate;
                    if total >= limit {
                        break;
                    }
                }
                limit = limit.min(total);
                if limit.is_zero() {
                    break;
                }
            }
        }

        queue.push_front(front);
        limit
    }

    #[allow(clippy::too_many_arguments)]
    fn score_root_parallel(
        &self,
        field: u32,
        front: u8,
        hold: u8,
        queue: &VecDeque<u8>,
        bag: u8,
        cutoffs: &[T],
        initial: T,
    ) -> T {
        let mut jobs: Vec<(u32, u8)> = self
            .graph
            .edges(field, front)
            .iter()
            .copied()
            .map(|next| (next, hold))
            .collect();
        if front != hold {
            jobs.extend(
                self.graph
                    .edges(field, hold)
                    .iter()
                    .copied()
                    .map(|next| (next, front)),
            );
        }
        if jobs.is_empty() {
            return initial;
        }

        let next_job = AtomicUsize::new(0);
        let best = Mutex::new(initial);
        let workers = self.threads.min(jobs.len());

        thread::scope(|scope| {
            for _ in 0..workers {
                scope.spawn(|| loop {
                    let job = next_job.fetch_add(1, Ordering::Relaxed);
                    let Some(&(next, next_hold)) = jobs.get(job) else {
                        break;
                    };
                    if self.killed.load(Ordering::Relaxed) {
                        break;
                    }

                    let mut local_queue = queue.clone();
                    let mut total = T::zero();
                    for piece in pieces(bag) {
                        if self.killed.load(Ordering::Relaxed) {
                            break;
                        }
                        let current = *best.lock().expect("root score lock poisoned");
                        if total >= current {
                            break;
                        }
                        local_queue.push_back(piece);
                        let candidate = self.score_imperfect(
                            next,
                            next_hold,
                            &mut local_queue,
                            1,
                            current - total,
                            bag & !(1 << piece),
                            cutoffs,
                        );
                        local_queue.pop_back();
                        total = total + candidate;
                    }

                    let mut current = best.lock().expect("root score lock poisoned");
                    if total < *current {
                        *current = total;
                    }
                    if total.is_zero() {
                        self.killed.store(true, Ordering::Relaxed);
                    }
                });
            }
        });

        best.into_inner().expect("root score lock poisoned")
    }
}

#[inline]
fn sum_options<T: Cost>(options: &[Option<T>; 7]) -> T {
    options
        .iter()
        .flatten()
        .copied()
        .fold(T::zero(), |sum, value| sum + value)
}
