use anyhow::{bail, Context, Result};
use std::collections::VecDeque;
use std::env;
use std::io::{self, BufRead, BufWriter, Write};
use std::path::PathBuf;
use std::time::Instant;
use zxcl_optimal_solver::decision::DecisionSearch;
use zxcl_optimal_solver::graph::{Graph, MAX_HASH};
use zxcl_optimal_solver::optimal::{OptimalSearch, OptimalSolution, VStarTable};
use zxcl_optimal_solver::optimal_dag::ExactDagSolution;
use zxcl_optimal_solver::score::{
    bag_mask, bag_string, piece_id, Cost, WeightedCost, Weights, FULL_BAG,
};
use zxcl_optimal_solver::search::Search;

const VERSION: &str = "0.4.20240203";

enum SolvedOptimal<'a> {
    EmptyDag(Box<ExactDagSolution<'a>>),
    CustomField(OptimalSolution<'a>),
}

impl SolvedOptimal<'_> {
    fn root_value(&self) -> f64 {
        match self {
            Self::EmptyDag(solution) => solution.root_value(),
            Self::CustomField(solution) => solution.root_value(),
        }
    }

    fn stats_text(&self) -> String {
        match self {
            Self::EmptyDag(solution) => {
                let stats = solution.stats();
                format!(
                    "Optimal DAG: {} -> {} states, {} reveal sequences",
                    stats.nodes_before_prune, stats.nodes_after_prune, stats.reveal_sequences
                )
            }
            Self::CustomField(solution) => {
                let stats = solution.stats();
                format!(
                    "Optimal states: {} ({} memo hits), actions: {}",
                    stats.memo_entries, stats.memo_hits, stats.actions_evaluated
                )
            }
        }
    }

    fn write_tree_data(&self, output: &mut impl Write) -> io::Result<()> {
        match self {
            Self::EmptyDag(solution) => solution.write_tree_data(output),
            Self::CustomField(solution) => solution.write_tree_data(output),
        }
    }
}

#[derive(Debug, Clone)]
struct Options {
    requested_hash: Option<u64>,
    threads: usize,
    see: usize,
    boolean: bool,
    decision: bool,
    optimal: bool,
    stdout: bool,
    two_line: bool,
    version: bool,
    weighted: bool,
    graph: PathBuf,
    vstar: PathBuf,
    weights: PathBuf,
}

impl Options {
    fn parse() -> Result<Self> {
        let args: Vec<String> = env::args().skip(1).collect();
        let hardware_threads = std::thread::available_parallelism().map_or(1, usize::from);
        let mut out = Self {
            requested_hash: None,
            threads: hardware_threads,
            see: 7,
            boolean: false,
            decision: false,
            optimal: false,
            stdout: false,
            two_line: false,
            version: false,
            weighted: false,
            graph: PathBuf::from("graph.bin"),
            vstar: PathBuf::from("vstar_l0_f32.bin"),
            weights: PathBuf::from("weights.txt"),
        };

        let mut i = 0;
        while i < args.len() {
            match args[i].as_str() {
                "-b" => out.boolean = true,
                "-d" => out.decision = true,
                "--optimal" => out.optimal = true,
                "-o" => out.stdout = true,
                "-t" => out.two_line = true,
                "-v" | "--version" => out.version = true,
                "-w" => out.weighted = true,
                "-f" => {
                    i += 1;
                    let value = args.get(i).context("-f requires a field hash")?;
                    let hash = value
                        .parse::<u64>()
                        .map_err(|_| anyhow::anyhow!("Invalid hash"))?;
                    if hash >= MAX_HASH {
                        bail!("Invalid hash");
                    }
                    out.requested_hash = Some(hash);
                }
                "-m" => {
                    i += 1;
                    let value = args.get(i).context("-m requires a thread count")?;
                    let threads = value
                        .parse::<isize>()
                        .map_err(|_| anyhow::anyhow!("Invalid number of threads"))?;
                    out.threads = threads.clamp(1, hardware_threads as isize) as usize;
                }
                "-s" => {
                    i += 1;
                    let value = args.get(i).context("-s requires a value")?;
                    out.see = value
                        .parse::<usize>()
                        .map_err(|_| anyhow::anyhow!("Invalid see"))?;
                    if !(2..=11).contains(&out.see) {
                        bail!("Invalid see");
                    }
                }
                "--graph" => {
                    i += 1;
                    out.graph = PathBuf::from(args.get(i).context("--graph requires a path")?);
                }
                "--vstar" => {
                    i += 1;
                    out.vstar = PathBuf::from(args.get(i).context("--vstar requires a path")?);
                }
                "--weights" => {
                    i += 1;
                    out.weights = PathBuf::from(args.get(i).context("--weights requires a path")?);
                }
                "-h" | "--help" => {
                    print_help();
                    std::process::exit(0);
                }
                _ => {
                    // The C++ program ignores unknown standalone arguments. Preserve that behavior
                    // so old wrapper scripts do not unexpectedly stop working.
                }
            }
            i += 1;
        }

        if out.boolean && out.decision {
            bail!("Cannot combine -b and -d modes");
        }
        if out.boolean && out.weighted {
            bail!("Cannot combine -b and -w modes");
        }
        if out.decision && out.two_line {
            bail!("Cannot combine -d and -t modes");
        }
        if out.optimal && out.weighted {
            bail!("Cannot combine --optimal and -w modes");
        }
        if out.optimal && out.boolean {
            bail!("Cannot combine --optimal and -b modes");
        }
        if out.optimal && out.two_line {
            bail!("Cannot combine --optimal and -t modes");
        }
        if out.optimal && out.see != 7 {
            bail!("--optimal requires see 7");
        }
        Ok(out)
    }
}

fn print_help() {
    println!("zxcl optimal solver v{VERSION} (Rust port)");
    println!("usage: zxcl-optimal-solver [-b] [-d] [--optimal] [-f HASH] [-m THREADS] [-o]");
    println!("                    [-s SEE] [-t] [-v] [-w] [--graph PATH]");
    println!("                    [--vstar PATH] [--weights PATH]");
    println!("       --optimal requires -s 7; add -d to write tree_data.js");
}

fn main() {
    if let Err(error) = real_main() {
        eprintln!("{error:#}");
        std::process::exit(1);
    }
}

fn real_main() -> Result<()> {
    let options = Options::parse()?;
    eprintln!("zxcl optimal solver v{VERSION}");
    if options.version {
        return Ok(());
    }
    if options.weighted {
        run::<WeightedCost>(options)
    } else {
        run::<u64>(options)
    }
}

fn run<T: Cost>(mut options: Options) -> Result<()> {
    eprint!("Loading weights... ");
    io::stderr().flush()?;
    let weights = if options.weighted {
        Weights::<T>::load(&options.weights)?
    } else {
        Weights::<T>::flat()
    };
    eprintln!();

    let vstar = if options.optimal {
        eprint!("Loading V*... ");
        io::stderr().flush()?;
        let start = Instant::now();
        let table = VStarTable::load(&options.vstar)?;
        eprintln!(
            "{} states (took {} ms)",
            table.len(),
            start.elapsed().as_millis()
        );
        Some(table)
    } else {
        None
    };

    eprint!("Loading graph... ");
    io::stderr().flush()?;
    let load_start = Instant::now();
    let graph = Graph::load(&options.graph)?;
    eprintln!(" (took {} ms)", load_start.elapsed().as_millis());

    let mut field = match options.requested_hash {
        Some(hash) => graph.find_hash(hash).context("Invalid hash")?,
        None => 0,
    };
    eprintln!("Starting from field {field} (hash {})", graph.hash(field));
    eprintln!("Max threads is {}", options.threads);

    let mut placed = placed_count(graph.hash(field))?;
    print_modes(&options);

    let stdin = io::stdin();
    let mut input = TokenReader::new(stdin.lock());
    let mut query_count = 0usize;

    loop {
        eprint!("> ");
        io::stderr().flush()?;
        let Some(queue_text) = input.next()? else {
            break;
        };

        match queue_text.as_str() {
            "-f" => {
                let hash = input
                    .next()?
                    .context("-f requires a field hash")?
                    .parse::<u64>()
                    .context("Invalid hash")?;
                if hash >= MAX_HASH {
                    bail!("Invalid hash");
                }
                field = graph.find_hash(hash).context("Invalid hash")?;
                placed = placed_count(graph.hash(field))?;
                eprintln!(
                    "Changed starting field to {field} (hash {})\n",
                    graph.hash(field)
                );
                continue;
            }
            "-m" => {
                let requested = input
                    .next()?
                    .context("-m requires a thread count")?
                    .parse::<isize>()
                    .context("Invalid number of threads")?;
                let hardware = std::thread::available_parallelism().map_or(1, usize::from);
                options.threads = requested.clamp(1, hardware as isize) as usize;
                eprintln!("Changed max threads to {}\n", options.threads);
                continue;
            }
            "-s" => {
                let see = input
                    .next()?
                    .context("-s requires a value")?
                    .parse::<usize>()
                    .context("Invalid see")?;
                if !(2..=11).contains(&see) {
                    bail!("Invalid see");
                }
                if options.optimal && see != 7 {
                    bail!("--optimal requires see 7");
                }
                options.see = see;
                eprintln!("Changed see to {see}\n");
                continue;
            }
            _ => {}
        }

        if queue_text.len() != options.see {
            break;
        }
        let queue_pieces = parse_queue(&queue_text)?;
        let hold = queue_pieces[0];
        let mut queue: VecDeque<u8> = queue_pieces[1..].iter().copied().collect();

        let bag_text = input.next()?.context("missing bag")?;
        let bag = parse_query_bag(&bag_text, &queue_pieces, options.see)?;
        let cutoffs = make_cutoffs::<T>(placed, options.see, bag);

        eprintln!(
            "[{query_count}] Testing queue {queue_text} with bag {}",
            bag_string(bag)
        );

        if options.optimal {
            let visible: [u8; 6] = queue_pieces[1..]
                .try_into()
                .expect("see 7 always has six visible pieces");

            // First report the baseline objective: the independently optimal probability
            // of completing the current four-line PC.  This is separate from the V* policy,
            // which maximizes the expected number of future PCs and accepts 2L terminals.
            let survival_denominator = cutoffs[0];
            let (survival, survival_elapsed) = {
                let survival_start = Instant::now();
                let mut survival_queue = queue.clone();
                let mut survival_search = Search::new(&graph, &weights, false, options.threads);
                let survival_cost = survival_search.score(
                    field,
                    hold,
                    &mut survival_queue,
                    bag,
                    &cutoffs,
                    survival_denominator,
                );
                (
                    survival_denominator - survival_cost,
                    survival_start.elapsed().as_millis(),
                )
            };
            #[cfg(all(target_os = "linux", target_env = "gnu"))]
            unsafe {
                libc::malloc_trim(0);
            }

            let start = Instant::now();
            let vstar = vstar.as_ref().expect("optimal mode loaded V*");
            let solution = if field == 0 {
                SolvedOptimal::EmptyDag(Box::new(ExactDagSolution::solve(
                    &graph,
                    vstar,
                    hold,
                    visible,
                    bag,
                    options.threads,
                )?))
            } else {
                let mut search = OptimalSearch::new(&graph, vstar, options.threads);
                SolvedOptimal::CustomField(search.solve(field, hold, visible, bag)?)
            };
            let optimal_elapsed = start.elapsed().as_millis();
            let result_text = solution.root_value().to_string();

            eprintln!("Result: {result_text}");
            eprintln!("Survival: {survival}/{survival_denominator}");
            eprintln!("{}", solution.stats_text());
            eprintln!("Time: optimal {optimal_elapsed} ms, survival {survival_elapsed} ms");
            io::stderr().flush()?;

            if options.decision {
                let write_start = Instant::now();
                let file = std::fs::File::create("tree_data.js")
                    .context("failed to create tree_data.js")?;
                let mut output = BufWriter::new(file);
                solution
                    .write_tree_data(&mut output)
                    .context("failed to write tree_data.js")?;
                writeln!(output)?;
                writeln!(output, "survival_success={survival}")?;
                write!(output, "survival_total={survival_denominator}")?;
                output.flush().context("failed to write tree_data.js")?;
                let write_elapsed = write_start.elapsed().as_millis();
                let tree_bytes = std::fs::metadata("tree_data.js")
                    .context("failed to stat tree_data.js")?
                    .len();
                eprintln!("Tree: {tree_bytes} bytes (wrote in {write_elapsed} ms)\n");
            } else {
                eprintln!();
            }
            drop(solution);
            // Repeated exact-DAG queries allocate differently sized vectors.  glibc otherwise
            // keeps many released pages in its arenas, so return them before the next query.
            #[cfg(all(target_os = "linux", target_env = "gnu"))]
            unsafe {
                libc::malloc_trim(0);
            }
            if options.stdout {
                println!("{result_text}");
            }
            query_count += 1;
            continue;
        }

        let initial_limit = if options.boolean {
            T::one()
        } else {
            cutoffs[0]
        };
        let start = Instant::now();
        let (cost, elapsed) = if options.decision {
            let mut search = DecisionSearch::new(&graph, &weights, options.threads);
            let tree = search.solve(field, hold, &mut queue, bag, &cutoffs, initial_limit);
            let elapsed = start.elapsed().as_millis();

            let file =
                std::fs::File::create("tree_data.js").context("failed to create tree_data.js")?;
            let mut output = BufWriter::new(file);
            tree.write_tree_data(&graph, field, &mut output)
                .context("failed to write tree_data.js")?;
            output.flush().context("failed to write tree_data.js")?;
            (tree.cost(), elapsed)
        } else {
            let mut search = Search::new(&graph, &weights, options.two_line, options.threads);
            let cost = search.score(field, hold, &mut queue, bag, &cutoffs, initial_limit);
            (cost, start.elapsed().as_millis())
        };

        let (result_text, denominator) = if options.weighted {
            (cost.scaled_output(), None)
        } else {
            let denominator = if options.boolean {
                T::one()
            } else {
                cutoffs[0]
            };
            (
                (denominator - cost).to_string(),
                Some(denominator.to_string()),
            )
        };

        match denominator {
            Some(denominator) => eprintln!("Result: {result_text}/{denominator}"),
            None => eprintln!("Result: {result_text}"),
        }
        eprintln!("Time: {elapsed} ms\n");
        if options.stdout {
            println!("{result_text}");
        }
        query_count += 1;
    }
    Ok(())
}

fn placed_count(hash: u64) -> Result<usize> {
    let minos = hash.count_ones() as usize;
    if minos & 3 != 0 {
        bail!("Somehow not a multiple of 4 minos");
    }
    Ok(minos / 4)
}

fn print_modes(options: &Options) {
    eprintln!("Running see {}", options.see);
    if options.boolean {
        eprintln!("Running boolean mode");
    }
    if options.decision {
        eprintln!("Running decision mode");
    }
    if options.optimal {
        eprintln!("Running optimal V* mode");
    }
    if options.stdout {
        eprintln!("Running stdout mode");
    }
    if options.two_line {
        eprintln!("Running 2L mode");
    }
    if options.weighted {
        eprintln!("Running weighted mode");
    }
    eprintln!();
}

fn parse_queue(text: &str) -> Result<Vec<u8>> {
    text.bytes()
        .map(|byte| piece_id(byte).with_context(|| format!("Invalid piece {}", byte as char)))
        .collect()
}

fn parse_query_bag(text: &str, queue: &[u8], see: usize) -> Result<u8> {
    if text.len() == 1 && matches!(text.as_bytes()[0], b'1'..=b'7') {
        let size = (text.as_bytes()[0] - b'0') as usize;
        if size + see < 7 {
            bail!("Too few pieces to infer bag");
        }
        let consumed = 7 - size;
        let mut bag = FULL_BAG;
        for &piece in queue[see - consumed..see].iter().rev() {
            let bit = 1 << piece;
            if bag & bit == 0 {
                bail!("Cannot infer bag from duplicate pieces");
            }
            bag &= !bit;
        }
        Ok(bag)
    } else {
        bag_mask(text).map_err(|_| anyhow::anyhow!("Invalid piece"))
    }
}

fn make_cutoffs<T: Cost>(placed: usize, see: usize, bag: u8) -> Vec<T> {
    let count = (12isize - placed as isize - see as isize).max(1) as usize;
    let mut cutoffs = vec![T::zero(); count];
    *cutoffs.last_mut().expect("at least one cutoff") = T::one();
    for i in (0..count - 1).rev() {
        let n = (bag.count_ones() as usize + 7 - i) % 7;
        cutoffs[i] = cutoffs[i + 1].mul_small(if n == 0 { 7 } else { n });
    }
    cutoffs
}

struct TokenReader<R> {
    reader: R,
    queued: VecDeque<String>,
    line: String,
}

impl<R: BufRead> TokenReader<R> {
    fn new(reader: R) -> Self {
        Self {
            reader,
            queued: VecDeque::new(),
            line: String::new(),
        }
    }

    fn next(&mut self) -> io::Result<Option<String>> {
        loop {
            if let Some(token) = self.queued.pop_front() {
                return Ok(Some(token));
            }
            self.line.clear();
            if self.reader.read_line(&mut self.line)? == 0 {
                return Ok(None);
            }
            self.queued
                .extend(self.line.split_whitespace().map(str::to_owned));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn infers_the_remaining_bag() {
        let q = parse_queue("OTJLISO").unwrap();
        assert_eq!(bag_string(parse_query_bag("1", &q, 7).unwrap()), "Z");
    }

    #[test]
    fn cutoff_counts_match_known_example() {
        let bag = bag_mask("IJLOSTZ").unwrap();
        assert_eq!(make_cutoffs::<u64>(0, 7, bag), vec![840, 120, 20, 4, 1]);
    }
}
