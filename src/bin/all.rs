use anyhow::Result;
use rayon::prelude::*;
use std::collections::VecDeque;
use std::fs;
use std::io::{self, BufWriter, IsTerminal, Write};
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use zxcl_optimal_solver::decision::DecisionSearch;
use zxcl_optimal_solver::graph::Graph;
use zxcl_optimal_solver::helpers::{make_cutoffs, parse_query_bag, parse_queue};
use zxcl_optimal_solver::score::{WeightedCost, Weights};
use zxcl_optimal_solver::search::Search;

const ORDER: &str = "TIJLOSZ";

/// Residual "first bag" size for each PC group 1..=7.
const PC_RESIDUAL: [usize; 7] = [7, 4, 1, 5, 2, 6, 3];

#[derive(Clone)]
struct QueueState {
    pc: usize,
    name: String,
    major_queues: Vec<String>,
}

/// Shared solver resources, loaded once per run.
struct Ctx {
    graph: Graph,
    weights_u64: Weights<u64>,
    weights_wc: Weights<WeightedCost>,
    out_dir: PathBuf,
}

/// One solver invocation: a full see-7 queue and its residual first-bag size.
struct Task {
    pc: usize,
    queue: String,
    r: usize,
}

/// Progress shared between the worker pool and the dashboard thread.
struct Shared {
    total: usize,
    processed: AtomicUsize,
    start: Instant,
    /// Which PC group is currently being processed (0..7); -1 means finishing.
    current_pc: AtomicUsize,
    /// Start of the current PC group.
    current_pc_start: std::sync::Mutex<Instant>,
}

impl Shared {
    fn new(total: usize) -> Self {
        Self {
            total,
            processed: AtomicUsize::new(0),
            start: Instant::now(),
            current_pc: AtomicUsize::new(usize::MAX),
            current_pc_start: std::sync::Mutex::new(Instant::now()),
        }
    }
}

/// Refresh an updating one-line progress indicator on stderr every ~400ms.
fn run_dashboard(shared: Arc<Shared>, stop: Arc<AtomicUsize>) {
    let mut last = Instant::now();
    while stop.load(Ordering::Relaxed) == 0 {
        let now = Instant::now();
        if now.duration_since(last) >= Duration::from_millis(400) {
            render_progress(&shared);
            last = now;
        }
        std::thread::sleep(Duration::from_millis(100));
    }
    // final clearing line
    eprint!("\r\x1b[K");
    io::stderr().flush().ok();
}

fn render_progress(shared: &Shared) {
    let total = shared.total;
    let done = shared.processed.load(Ordering::Relaxed);
    let pct = if total == 0 {
        0.0
    } else {
        done as f64 / total as f64 * 100.0
    };
    let elapsed = shared.start.elapsed().as_secs_f64();
    let rate = if elapsed > 0.0 { done as f64 / elapsed } else { 0.0 };
    let eta = if rate > 0.0 {
        (total - done) as f64 / rate
    } else {
        0.0
    };
    let pc = shared.current_pc.load(Ordering::Relaxed);
    let pc_elapsed = shared
        .current_pc_start
        .lock()
        .expect("current_pc_start lock poisoned")
        .elapsed()
        .as_secs_f64();
    let pc_label = if (1..=7).contains(&pc) {
        format!("PC {pc} ({pc_elapsed:.0}s)")
    } else {
        String::from("finishing")
    };

    eprint!(
        "\r{pc_label} | {done}/{total} ({pct:.1}%) {rate:.1}/s ETA {:>7.0}s",
        eta
    );
    io::stderr().flush().ok();
}

impl Ctx {
    /// Run both passes for one queue, writing `output/see7/{pc}/{queue}.js`.
    ///
    /// Called from rayon worker threads; each queue search runs single-threaded internally so the
    /// outer worker pool supplies the parallelism without oversubscribing cores.
    fn entry(&self, task: &Task) -> Result<()> {
        let pieces = parse_queue(&task.queue)?;
        let hold = pieces[0];
        let bag = parse_query_bag(&task.r.to_string(), &pieces, 7)?;

        let dir = self.out_dir.join(task.pc.to_string());
        let path = dir.join(format!("{}.js", task.queue));

        if path.exists() {
            return Ok(());
        }

        fs::create_dir_all(&dir)?;

        // Pass A — solve rate (plain u64)
        let cutoffs_u64 = make_cutoffs::<u64>(0, 7, bag);
        let mut queue: VecDeque<u8> = pieces[1..].iter().copied().collect();
        let mut search = Search::new(&self.graph, &self.weights_u64, false, 1);
        let cost = search.score(0, hold, &mut queue, bag, &cutoffs_u64, cutoffs_u64[0]);
        let numerator = cutoffs_u64[0] - cost;
        let denominator = cutoffs_u64[0];

        // Pass B — weighted decision tree (single-threaded; rayon outer pool parallelizes queues)
        let cutoffs_wc = make_cutoffs::<WeightedCost>(0, 7, bag);
        let mut dsearch = DecisionSearch::new(&self.graph, &self.weights_wc, 1);
        let tree = dsearch.solve(0, hold, &mut queue, bag, &cutoffs_wc, cutoffs_wc[0]);

        // Write combined file
        let mut buf = BufWriter::new(std::fs::File::create(&path)?);
        write!(buf, "data={{solve:[{numerator},{denominator}],tree:")?;
        tree.write_root_json(&self.graph, &mut buf)?;
        writeln!(buf, "}}")?;
        buf.flush()?;

        Ok(())
    }
}

fn main() {
    let config = parse_args();
    if let Err(error) = run(&config) {
        eprintln!("{error:#}");
        std::process::exit(1);
    }
}

fn run(config: &Config) -> Result<()> {
    let graph = Graph::load(&config.graph)?;
    let weights_u64 = Weights::<u64>::flat();
    let weights_wc = Weights::<WeightedCost>::load(&config.weights)?;
    let ctx = Arc::new(Ctx {
        graph,
        weights_u64,
        weights_wc,
        out_dir: config.out_dir.clone(),
    });

    let all = enumerate_queues();
    let states: Vec<&QueueState> = all
        .iter()
        .filter(|s| config.pc.is_none_or(|pc| s.pc == pc))
        .filter(|s| config.name.as_deref().is_none_or(|n| s.name == n))
        .collect();

    let mut tasks: Vec<Task> = Vec::new();
    for state in &states {
        let r = PC_RESIDUAL[state.pc - 1];
        let minors = &permutations(ORDER, 7 - r);
        for major in &state.major_queues {
            for minor in minors {
                tasks.push(Task {
                    pc: state.pc,
                    queue: format!("{major}{minor}"),
                    r,
                });
            }
        }
    }

    if tasks.is_empty() {
        return Ok(());
    }

    let shared = Arc::new(Shared::new(tasks.len()));

    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(config.threads)
        .build()
        .expect("failed to build rayon pool");

    // A live updating indicator only makes sense on a real terminal; otherwise emit one line per
    // finished PC group so redirected/background logs stay clean.
    let is_tty = io::stderr().is_terminal();
    let tty_dashboard = if is_tty {
        let stop = Arc::new(AtomicUsize::new(0));
        let dashboard_shared = Arc::clone(&shared);
        let dashboard_stop = Arc::clone(&stop);
        std::thread::spawn(move || run_dashboard(dashboard_shared, dashboard_stop));
        Some(stop)
    } else {
        None
    };

    let start = shared.start;
    let processed = &shared.processed;
    let mut result: Result<()> = Ok(());

    // tasks are grouped by pc since enumerate_queues emits states per pc in order
    let mut task_start = 0usize;
    for pc in 1..=7 {
        let pc_tasks = tasks[task_start..]
            .iter()
            .take_while(|t| t.pc == pc)
            .count();
        if pc_tasks == 0 {
            continue;
        }
        if !is_tty {
            println!("PC {pc} ({pc_tasks}):");
        }
        shared.current_pc.store(pc, Ordering::Relaxed);
        *shared.current_pc_start.lock().expect("poisoned pc start") = Instant::now();
        let pc_processed_before = processed.load(Ordering::Relaxed);

        result = pool.install(|| {
            tasks[task_start..task_start + pc_tasks]
                .par_iter()
                .map(|task| {
                    let r = ctx.entry(task);
                    processed.fetch_add(1, Ordering::Relaxed);
                    r
                })
                .collect::<Result<Vec<_>>>()
                .map(|_| ())
        });

        if !is_tty {
            let total = shared.total;
            let done = processed.load(Ordering::Relaxed) - pc_processed_before;
            let elapsed = start.elapsed().as_secs_f64();
            let rate = done as f64 / elapsed.max(1e-9);
            let eta = (total - processed.load(Ordering::Relaxed)) as f64 / rate.max(1e-9);
            println!(
                "\rPC {pc}: {done}/{pc_tasks} queues ({rate:.1}/s) — total {}/{} ({:.1}%), ETA {:.0}s",
                processed.load(Ordering::Relaxed),
                total,
                processed.load(Ordering::Relaxed) as f64 / total as f64 * 100.0,
                eta,
            );
        }

        task_start += pc_tasks;
        if result.is_err() {
            break;
        }
    }

    shared.current_pc.store(usize::MAX, Ordering::Relaxed);

    if let Some(stop) = tty_dashboard {
        stop.store(1, Ordering::Relaxed);
    }
    let total = shared.total;
    let elapsed = start.elapsed().as_secs_f64();
    if is_tty {
        eprintln!(
            "\rDone: {}/{total} queues ({elapsed:.1}s)",
            processed.load(Ordering::Relaxed)
        );
    } else {
        println!(
            "Done: {}/{total} queues ({elapsed:.1}s)",
            processed.load(Ordering::Relaxed)
        );
    }
    result
}

fn enumerate_queues() -> Vec<QueueState> {
    let mut states = Vec::new();
    for pc in 1..=7 {
        let size = PC_RESIDUAL[pc - 1];

        for c in combinations(ORDER, size) {
            let major_queues = permutations(&c, size);
            states.push(QueueState {
                pc,
                name: c,
                major_queues,
            });
        }

        if size > 1 {
            for d in ORDER.chars() {
                for rest in combinations(ORDER, size - 1) {
                    if !rest.contains(d) {
                        continue;
                    }
                    let name = sort(&format!("{d}{rest}"));
                    let perms = permutations(&rest, size - 1)
                        .iter()
                        .map(|x| format!("{d}{x}"))
                        .collect();
                    states.push(QueueState {
                        pc,
                        name,
                        major_queues: perms,
                    });
                }
            }
        }
    }
    states
}

struct Config {
    pc: Option<usize>,
    name: Option<String>,
    graph: PathBuf,
    weights: PathBuf,
    threads: usize,
    out_dir: PathBuf,
}

fn parse_args() -> Config {
    let mut pc = None;
    let mut name = None;
    let mut graph = PathBuf::from("graph.bin");
    let mut weights = PathBuf::from("weights.txt");
    let mut threads = std::thread::available_parallelism().map_or(1, usize::from);
    let mut out_dir = PathBuf::from("output/see7");
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--pc" => pc = args.next().and_then(|s| s.parse().ok()),
            "--name" => name = args.next(),
            "--graph" => graph = args.next().map(PathBuf::from).unwrap_or(graph),
            "--weights" => weights = args.next().map(PathBuf::from).unwrap_or(weights),
            "--threads" => threads = args.next().and_then(|s| s.parse().ok()).unwrap_or(threads),
            "--out" => out_dir = args.next().map(PathBuf::from).unwrap_or(out_dir),
            other => eprintln!("ignoring unknown argument: {other}"),
        }
    }
    Config {
        pc,
        name,
        graph,
        weights,
        threads,
        out_dir,
    }
}

fn sort(c: &str) -> String {
    let mut freq = [0usize; ORDER.len()];

    for ch in c.bytes() {
        freq[match ch {
            b'T' => 0,
            b'I' => 1,
            b'J' => 2,
            b'L' => 3,
            b'O' => 4,
            b'S' => 5,
            b'Z' => 6,
            _ => unreachable!(),
        }] += 1;
    }

    let mut indices = [0, 1, 2, 3, 4, 5, 6];

    indices.sort_unstable_by(|&a, &b| freq[b].cmp(&freq[a]).then(a.cmp(&b)));

    let mut result = String::with_capacity(c.len());

    for i in indices {
        result.extend(std::iter::repeat_n(ORDER.as_bytes()[i] as char, freq[i]));
    }

    result
}

fn permutations(items: &str, k: usize) -> Vec<String> {
    if k == 0 {
        return vec![String::new()];
    }
    if items.len() < k {
        return vec![];
    }

    let mut result = Vec::new();
    for (i, item) in items.chars().enumerate() {
        let remaining: String = items
            .chars()
            .enumerate()
            .filter_map(|(j, c)| if j != i { Some(c) } else { None })
            .collect();
        for perm in permutations(&remaining, k - 1) {
            let mut new_perm = String::new();
            new_perm.push(item);
            new_perm.push_str(&perm);
            result.push(new_perm);
        }
    }
    result
}

fn combinations(items: &str, k: usize) -> Vec<String> {
    if k == 0 {
        return vec![String::new()];
    }
    if items.len() < k {
        return vec![];
    }

    let mut result = Vec::new();
    for (i, item) in items.chars().enumerate() {
        let remaining: String = items
            .chars()
            .enumerate()
            .filter_map(|(j, c)| if j > i { Some(c) } else { None })
            .collect();
        for comb in combinations(&remaining, k - 1) {
            let mut new_comb = String::new();
            new_comb.push(item);
            new_comb.push_str(&comb);
            result.push(new_comb);
        }
    }
    result
}
