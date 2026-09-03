use solver::helpers::{make_cutoffs, parse_query_bag, parse_queue, placed_count};
use crate::query::{run_optimal_query, run_search_query, Query, SearchConfig};
use anyhow::{bail, Context, Result};
use std::cell::Cell;
use std::collections::VecDeque;
use std::io::{self, BufRead, IsTerminal, Write};
use solver::graph::{Graph, MAX_HASH};
use solver::optimal::VStarTable;
use solver::score::{Cost, Weights};

/// Shared immutable resources that outlive every query.
pub struct SessionCtx<'a, T: Cost> {
    pub graph: &'a Graph,
    pub weights: &'a Weights<T>,
    pub vstar: Option<&'a VStarTable>,
}

/// Mutable state that carries across interactive queries.
pub struct Session<'a, T: Cost> {
    pub ctx: SessionCtx<'a, T>,
    pub config: SearchConfig,
    pub threads: usize,
    pub see: usize,
    pub field: u32,
    pub placed: usize,
    pub failed: Cell<bool>,
}

impl<T: Cost> Session<'_, T> {
    fn change_field(&mut self, hash: u64) -> Result<()> {
        if hash >= MAX_HASH {
            bail!("Invalid hash");
        }
        self.field = self.ctx.graph.find_hash(hash).context("Invalid hash")?;
        self.placed = placed_count(self.ctx.graph.hash(self.field))?;
        Ok(())
    }

    fn change_threads(&mut self, requested: isize) {
        let hardware = std::thread::available_parallelism().map_or(1, usize::from);
        self.threads = requested.clamp(1, hardware as isize) as usize;
    }

    fn change_see(&mut self, see: usize) -> Result<()> {
        if !(2..=11).contains(&see) {
            bail!("Invalid see");
        }
        if self.config.optimal && see != 7 {
            bail!("--optimal requires see 7");
        }
        self.see = see;
        Ok(())
    }

    fn run_query(&mut self, queue_text: &str, bag_text: &str) -> Result<()> {
        if queue_text.len() != self.see {
            bail!(
                "queue length does not match see {} (got {})",
                self.see,
                queue_text.len()
            );
        }
        let queue_pieces = parse_queue(queue_text)?;
        let hold = queue_pieces[0];
        let queue: VecDeque<u8> = queue_pieces[1..].iter().copied().collect();
        let bag = parse_query_bag(bag_text, &queue_pieces, self.see)?;
        let cutoffs = make_cutoffs::<T>(self.placed, self.see, bag);
        let visible: [u8; 6] = if self.config.optimal {
            queue_pieces[1..]
                .try_into()
                .expect("see 7 always has six visible pieces")
        } else {
            [0; 6]
        };

        if self.config.optimal {
            let vstar = self.ctx.vstar.expect("optimal mode loaded V*");
            let query = Query {
                field: self.field,
                hold,
                queue,
                bag,
                cutoffs: &cutoffs,
                visible,
            };
            run_optimal_query(
                self.ctx.graph,
                self.ctx.weights,
                vstar,
                self.config,
                self.threads,
                &query,
            )?;
        } else {
            let mut query = Query {
                field: self.field,
                hold,
                queue,
                bag,
                cutoffs: &cutoffs,
                visible,
            };
            run_search_query(
                self.ctx.graph,
                self.ctx.weights,
                self.config,
                self.threads,
                &mut query,
            )?;
        }
        Ok(())
    }
}

enum ReplCommand {
    /// Change starting field hash.
    Field { hash: u64 },
    /// Change maximum thread count.
    Threads { count: isize },
    /// Change see value.
    See { value: usize },
    /// A bare queue and its bag, e.g. `IJLOSTZ IJLOSTZ`.
    Query(Vec<String>),
}

/// Parse one REPL input line into a command.
fn parse_command(line: &str) -> Result<ReplCommand> {
    let fields: Vec<&str> = line.split_whitespace().collect();
    if fields.is_empty() {
        bail!("nothing to run");
    }
    match fields[0] {
        "-f" | "-m" | "-s" if fields.len() != 2 => {
            bail!("{} expects a single argument", fields[0])
        }
        "-f" => Ok(ReplCommand::Field {
            hash: fields[1].parse()?,
        }),
        "-m" => Ok(ReplCommand::Threads {
            count: fields[1].parse()?,
        }),
        "-s" => Ok(ReplCommand::See {
            value: fields[1].parse()?,
        }),
        _ if fields.len() != 2 => bail!(
            "expected a queue followed by a bag, e.g. `IJLOSTZ IJLOSTZ`; got {}",
            fields.join(" ")
        ),
        _ => Ok(ReplCommand::Query(
            fields.into_iter().map(str::to_owned).collect(),
        )),
    }
}

impl ReplCommand {
    fn run<T: Cost>(self, session: &mut Session<'_, T>) -> Result<()> {
        match self {
            Self::Field { hash } => session.change_field(hash),
            Self::Threads { count } => {
                session.change_threads(count);
                Ok(())
            }
            Self::See { value } => session.change_see(value),
            Self::Query(tokens) => {
                let [queue, bag] = tokens.as_slice() else {
                    bail!(
                        "expected a queue followed by a bag, e.g. `IJLOSTZ IJLOSTZ`; got {}",
                        tokens.join(" ")
                    );
                };
                session.run_query(queue, bag)
            }
        }
    }
}

fn handle_command<T: Cost>(command: ReplCommand, session: &mut Session<'_, T>) {
    if let Err(err) = command.run(session) {
        eprintln!("{err:#}");
        session.failed.set(true);
    }
}

/// Run a single query without entering the REPL, then exit.
#[allow(clippy::too_many_arguments)]
pub fn run_single_query<T: Cost>(
    ctx: SessionCtx<'_, T>,
    config: SearchConfig,
    threads: usize,
    see: usize,
    field: u32,
    placed: usize,
    queue_text: &str,
    bag_text: &str,
) -> Result<()> {
    let mut session = Session {
        ctx,
        config,
        threads,
        see,
        field,
        placed,
        failed: Cell::new(false),
    };
    session.run_query(queue_text, bag_text)
}

/// Enter the interactive REPL loop, reading commands from stdin (a pipe or a terminal).
pub fn run<T: Cost>(
    ctx: SessionCtx<'_, T>,
    config: SearchConfig,
    threads: usize,
    see: usize,
    field: u32,
    placed: usize,
) {
    let mut session = Session {
        ctx,
        config,
        threads,
        see,
        field,
        placed,
        failed: Cell::new(false),
    };

    let input = io::stdin();
    let mut reader = input.lock();
    let mut line = String::new();
    loop {
        if input.is_terminal() {
            print!("> ");
            io::stdout().flush().expect("failed to flush prompt");
        }
        line.clear();
        match reader.read_line(&mut line) {
            Ok(0) => break,
            Ok(_) => {}
            Err(error) => {
                eprintln!("failed to read input: {error}");
                break;
            }
        }
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        match parse_command(trimmed) {
            Ok(command) => handle_command(command, &mut session),
            Err(error) => {
                eprintln!("{error:#}");
                session.failed.set(true);
            }
        }
    }
}
