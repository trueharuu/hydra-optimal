mod query;
mod repl;

use anyhow::{bail, Context, Result};
use clap::Parser;
use query::SearchConfig;
use repl::{run as run_repl, run_single_query, SessionCtx};
use std::path::PathBuf;
use zxcl_optimal_solver::graph::{Graph, MAX_HASH};
use zxcl_optimal_solver::optimal::VStarTable;
use zxcl_optimal_solver::score::{Cost, WeightedCost, Weights};

const VERSION: &str = "0.4.20240203";

fn hardware_threads() -> usize {
    std::thread::available_parallelism().map_or(1, usize::from)
}

#[derive(Parser, Debug)]
#[command(name = "zxcl-optimal-solver", version = VERSION, about = "Standalone zxcl perfect-clear decision-tree solver with optional V* scoring", disable_version_flag = true)]
struct Cli {
    /// Optional starting field hash.
    #[arg(short = 'f', value_name = "HASH")]
    requested_hash: Option<u64>,

    /// Maximum number of worker threads.
    #[arg(short = 'm', default_value_t = hardware_threads())]
    threads: usize,

    /// See value (2..=11).
    #[arg(short = 's', default_value_t = 7)]
    see: usize,

    /// Boolean mode.
    #[arg(short = 'b')]
    boolean: bool,

    /// Decision mode (writes tree_data.js).
    #[arg(short = 'd')]
    decision: bool,

    /// Optimal V* scoring mode.
    #[arg(long)]
    optimal: bool,

    /// Print the result to stdout.
    #[arg(short = 'o')]
    stdout: bool,

    /// Evaluate a single queue and bag, then exit, e.g. `IJLOSTZ IJLOSTZ`.
    #[arg(value_name = "QUEUE BAG")]
    queue_bag: Option<Vec<String>>,

    /// 2-line mode.
    #[arg(short = 't')]
    two_line: bool,

    /// Weighted mode.
    #[arg(short = 'w')]
    weighted: bool,

    /// Show version and exit.
    #[arg(short = 'v', long)]
    version: bool,

    /// Path to the graph file.
    #[arg(long, default_value = "graph.bin")]
    graph: PathBuf,

    /// Path to the V* table file (optimal mode).
    #[arg(long, default_value = "vstar_l0_f32.bin")]
    vstar: PathBuf,

    /// Path to the weights file (weighted mode).
    #[arg(long, default_value = "weights.txt")]
    weights: PathBuf,
}

fn main() {
    if let Err(error) = real_main() {
        eprintln!("{error:#}");
        std::process::exit(1);
    }
}

fn real_main() -> Result<()> {
    let cli = Cli::parse();
    if cli.version {
        eprintln!("zxcl optimal solver v{VERSION}");
        return Ok(());
    }
    validate(&cli)?;

    if cli.weighted {
        run::<WeightedCost>(cli)
    } else {
        run::<u64>(cli)
    }
}

fn validate(cli: &Cli) -> Result<()> {
    if cli.boolean && cli.decision {
        bail!("Cannot combine -b and -d modes");
    }
    if cli.boolean && cli.weighted {
        bail!("Cannot combine -b and -w modes");
    }
    if cli.decision && cli.two_line {
        bail!("Cannot combine -d and -t modes");
    }
    if cli.optimal && cli.weighted {
        bail!("Cannot combine --optimal and -w modes");
    }
    if cli.optimal && cli.boolean {
        bail!("Cannot combine --optimal and -b modes");
    }
    if cli.optimal && cli.two_line {
        bail!("Cannot combine --optimal and -t modes");
    }
    if cli.optimal && cli.see != 7 {
        bail!("--optimal requires see 7");
    }
    if !(2..=11).contains(&cli.see) {
        bail!("Invalid see");
    }
    if let Some(tokens) = &cli.queue_bag {
        if tokens.len() != 2 {
            bail!("expected a queue followed by a bag, e.g. `IJLOSTZ IJLOSTZ`");
        }
    }
    if let Some(hash) = cli.requested_hash {
        if hash >= MAX_HASH {
            bail!("Invalid hash");
        }
    }
    Ok(())
}

fn run<T: Cost>(cli: Cli) -> Result<()> {
    let threads = cli.threads.clamp(1, hardware_threads());
    if threads != cli.threads {
        eprintln!(
            "Max threads clamped to {threads} (hardware limit {})",
            hardware_threads()
        );
    }

    let weights = if cli.weighted {
        Weights::<T>::load(&cli.weights)?
    } else {
        Weights::<T>::flat()
    };

    let vstar = if cli.optimal {
        Some(VStarTable::load(&cli.vstar)?)
    } else {
        None
    };

    let graph = Graph::load(&cli.graph)?;

    let field = match cli.requested_hash {
        Some(hash) => graph.find_hash(hash).context("Invalid hash")?,
        None => 0,
    };

    let placed = zxcl_optimal_solver::helpers::placed_count(graph.hash(field))?;

    let config = SearchConfig {
        boolean: cli.boolean,
        decision: cli.decision,
        optimal: cli.optimal,
        stdout: cli.stdout,
        two_line: cli.two_line,
        weighted: cli.weighted,
    };

    let ctx = SessionCtx {
        graph: &graph,
        weights: &weights,
        vstar: vstar.as_ref(),
    };

    if let Some(tokens) = &cli.queue_bag {
        return run_single_query(
            ctx, config, threads, cli.see, field, placed, &tokens[0], &tokens[1],
        );
    }

    run_repl(ctx, config, threads, cli.see, field, placed);
    Ok(())
}
