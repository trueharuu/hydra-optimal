use anyhow::{Context, Result};
use std::collections::VecDeque;
use std::io::{self, BufWriter, Write};
use solver::decision::DecisionSearch;
use solver::graph::Graph;
use solver::optimal::{OptimalSearch, OptimalSolution, VStarTable};
use solver::optimal_dag::ExactDagSolution;
use solver::score::{Cost, Weights};
use solver::search::Search;

/// The mode flags that select which search variant and output behavior to use.
#[derive(Clone, Copy)]
pub struct SearchConfig {
    pub boolean: bool,
    pub decision: bool,
    pub optimal: bool,
    pub stdout: bool,
    pub two_line: bool,
    pub weighted: bool,
}

pub enum SolvedOptimal<'a> {
    EmptyDag(Box<ExactDagSolution<'a>>),
    CustomField(OptimalSolution<'a>),
}

impl SolvedOptimal<'_> {
    pub fn root_value(&self) -> f64 {
        match self {
            Self::EmptyDag(solution) => solution.root_value(),
            Self::CustomField(solution) => solution.root_value(),
        }
    }

    pub fn init_hash(&self) -> u64 {
        match self {
            Self::EmptyDag(solution) => solution.init_hash(),
            Self::CustomField(solution) => solution.init_hash(),
        }
    }

    /// Stream the policy as a piece-keyed JSON root node subtree.
    pub fn write_root_json(&self, output: &mut impl Write) -> io::Result<()> {
        match self {
            Self::EmptyDag(solution) => solution.write_root_json(output),
            Self::CustomField(solution) => solution.write_root_json(output),
        }
    }

    /// Stream a complete `tree_data.js` top-level payload for optimal mode.
    ///
    /// The header carries the mode and initial field hash; the reveal-conditioned tree follows.
    /// Terminal and no-action leaves are bare numbers; the survival report stays on stderr.
    pub fn write_tree_json(&self, mut output: impl Write) -> io::Result<()> {
        write!(
            output,
            "{{\"mode\":\"optimal\",\"init_hash\":{},\"root\":",
            self.init_hash()
        )?;
        self.write_root_json(&mut output)?;
        write!(output, "}}")
    }
}

/// A single query to evaluate: the current field, chain state, bag, and cutoffs.
pub struct Query<'a, T: Cost> {
    pub field: u32,
    pub hold: u8,
    pub queue: VecDeque<u8>,
    pub bag: u8,
    pub cutoffs: &'a [T],
    pub visible: [u8; 6],
}

/// Evaluate a query in optimal V* mode from the current field.
///
/// Reports the baseline survival probability (independently optimal four-line PC completion),
/// then the V* policy result which maximizes the expected number of future PCs.
pub fn run_optimal_query<T: Cost>(
    graph: &Graph,
    weights: &Weights<T>,
    vstar: &VStarTable,
    config: SearchConfig,
    threads: usize,
    query: &Query<'_, T>,
) -> Result<()> {
    let survival_denominator = query.cutoffs[0];
    let survival = {
        let mut survival_queue = query.queue.clone();
        let mut survival_search = Search::new(graph, weights, false, threads);
        let survival_cost = survival_search.score(
            query.field,
            query.hold,
            &mut survival_queue,
            query.bag,
            query.cutoffs,
            survival_denominator,
        );
        survival_denominator - survival_cost
    };
    #[cfg(all(target_os = "linux", target_env = "gnu"))]
    unsafe {
        libc::malloc_trim(0);
    }

    let solution = if query.field == 0 {
        SolvedOptimal::EmptyDag(Box::new(ExactDagSolution::solve(
            graph,
            vstar,
            query.hold,
            query.visible,
            query.bag,
            threads,
        )?))
    } else {
        let mut search = OptimalSearch::new(graph, vstar, threads);
        SolvedOptimal::CustomField(search.solve(
            query.field,
            query.hold,
            query.visible,
            query.bag,
        )?)
    };
    let result_text = solution.root_value().to_string();

    eprintln!("Result: {result_text}");
    eprintln!("Survival: {survival}/{survival_denominator}");

    if config.decision {
        let file =
            std::fs::File::create("tree_data.js").context("failed to create tree_data.js")?;
        let mut output = BufWriter::new(file);
        write!(output, "tree_data=")?;
        solution
            .write_tree_json(&mut output)
            .context("failed to write tree_data.js")?;
        output.flush().context("failed to write tree_data.js")?;
    }
    drop(solution);
    #[cfg(all(target_os = "linux", target_env = "gnu"))]
    unsafe {
        libc::malloc_trim(0);
    }
    if config.stdout {
        println!("{result_text}");
    }
    Ok(())
}

/// Evaluate a query in boolean / decision / plain / weighted search mode.
pub fn run_search_query<T: Cost>(
    graph: &Graph,
    weights: &Weights<T>,
    config: SearchConfig,
    threads: usize,
    query: &mut Query<'_, T>,
) -> Result<()> {
    let initial_limit = if config.boolean {
        T::one()
    } else {
        query.cutoffs[0]
    };
    let cost = if config.decision {
        let mut search = DecisionSearch::new(graph, weights, threads);
        let tree = search.solve(
            query.field,
            query.hold,
            &mut query.queue,
            query.bag,
            query.cutoffs,
            initial_limit,
        );

        let file =
            std::fs::File::create("tree_data.js").context("failed to create tree_data.js")?;
        let mut output = BufWriter::new(file);
        write!(
            output,
            "tree_data={{\"mode\":\"decision\",\"init_hash\":{},\"root\":",
            graph.hash(query.field)
        )?;
        tree.write_root_json(graph, &mut output)
            .context("failed to write tree_data.js")?;
        write!(output, "}}")?;
        output.flush().context("failed to write tree_data.js")?;
        tree.cost()
    } else {
        let mut search = Search::new(graph, weights, config.two_line, threads);
        search.score(
            query.field,
            query.hold,
            &mut query.queue,
            query.bag,
            query.cutoffs,
            initial_limit,
        )
    };

    let (result_text, denominator) = if config.weighted {
        (cost.scaled_output(), None)
    } else {
        let denominator = if config.boolean {
            T::one()
        } else {
            query.cutoffs[0]
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
    if config.stdout {
        println!("{result_text}");
    }
    Ok(())
}
