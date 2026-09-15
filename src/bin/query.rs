use std::collections::VecDeque;
use std::io::Read;
use std::path::PathBuf;

use anyhow::{bail, Context, Result};
use clap::Parser;
use solver::graph::Graph;
use solver::helpers::{make_cutoffs, parse_query_bag, parse_queue};
use solver::pattern::{Pattern, Piece};
use solver::score::Weights;
use solver::search::Search;

#[derive(clap::Parser)]
pub struct Program {
    /// The pattern to evaluate, e.g. `TI[JL]!OSZ` or `T[IJL]!*p3`, or `-` to
    /// read it from stdin (newlines separate segments).
    #[clap(value_name = "PATTERN")]
    pattern: String,
    /// The bag, as a piece set (`OSZ`) or a residual size (`4`).
    bag: String,
    #[clap(long, default_value = "graph.bin")]
    graph: PathBuf,
    /// See value (2..=11); must equal the expanded queue length.
    #[clap(short, long, default_value_t = 7)]
    see: usize,
    /// How many solver threads each candidate query may use.
    #[clap(long, default_value_t = hardware_threads())]
    threads: usize,
}

struct Row {
    queue: String,
    wins: u64,
    total: u64,
}

pub fn main() -> Result<()> {
    let program = Program::parse();
    let threads = program.threads.clamp(1, hardware_threads());

    if !(2..=11).contains(&program.see) {
        bail!("invalid see {}", program.see);
    }

    let graph = Graph::load(&program.graph)
        .with_context(|| format!("failed to load graph from {}", program.graph.display()))?;
    let weights = Weights::<u64>::flat();

    let source = if program.pattern.trim() == "-" {
        let mut stdin = String::new();
        std::io::stdin().read_to_string(&mut stdin)?;
        stdin
    } else {
        program.pattern
    };
    let pattern = Pattern::parse(&source)?;

    // each top-level ; segment is an independent query: every expanded sequence is a queue whose
    // PC solvability is scored; rows stream to stdout as each search returns, then the segment
    // collapses to a final aggregate line
    for segment in pattern.segments() {
        let candidates = segment.expand();
        if candidates.is_empty() {
            continue;
        }

        let mut total: u64 = 0;
        let mut wins: u64 = 0;
        let n = candidates.len();
        for (i, candidate) in candidates.iter().enumerate() {
            let row = run_candidate(
                &graph,
                &weights,
                candidate,
                program.see,
                &program.bag,
                threads,
            )?;
            total += row.total;
            wins += row.wins;
            eprint!(
                "\r[{}/{}] {:>6.2}% {}",
                i + 1,
                n,
                win_percent(row.wins, row.total),
                row.queue,
            );
        }
        eprintln!();
        println!(
            "{:>6.2}% {segment} ({wins}/{total})",
            win_percent(wins, total)
        );
    }
    Ok(())
}

fn hardware_threads() -> usize {
    std::thread::available_parallelism().map_or(1, usize::from)
}

/// Score one expanded sequence as a queue from the empty field.
fn run_candidate(
    graph: &Graph,
    weights: &Weights<u64>,
    candidate: &[Piece],
    see: usize,
    bag_text: &str,
    threads: usize,
) -> Result<Row> {
    if candidate.len() != see {
        bail!(
            "expanded queue length {} does not match see {see}",
            candidate.len()
        );
    }
    if candidate.len() < 2 {
        bail!("segments must expand to at least a hold piece plus one queue piece");
    }

    let queue_text: String = candidate.iter().map(ToString::to_string).collect();
    let pieces = parse_queue(&queue_text)?;
    let hold = pieces[0];
    let mut queue: VecDeque<u8> = pieces[1..].iter().copied().collect();
    let bag = parse_query_bag(bag_text, &pieces, see)?;
    let cutoffs = make_cutoffs::<u64>(0, see, bag);

    let mut search = Search::new(graph, weights, false, threads);
    let cost = search.score(0, hold, &mut queue, bag, &cutoffs, cutoffs[0]);

    Ok(Row {
        queue: queue_text,
        wins: cutoffs[0] - cost,
        total: cutoffs[0],
    })
}

fn win_percent(wins: u64, total: u64) -> f64 {
    if total == 0 {
        0.0
    } else {
        wins as f64 / total as f64 * 100.0
    }
}
