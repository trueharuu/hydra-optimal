use clap::Parser;
use solver::pattern::Pattern;
use std::io::Read;

#[derive(clap::Parser)]
pub struct Program {
    /// The pattern to evaluate, e.g. `TI[JL]!OSZ` or `T[IJL]!*p3`, or `-` to read
    /// it from stdin (newlines separate segments).
    #[clap(value_name = "PATTERN")]
    pattern: String,
    /// Whether to write the count to stderr.
    #[clap(long)]
    count: bool,
}

pub fn main() -> anyhow::Result<()> {
    let program = Program::parse();
    let source = if program.pattern.trim() == "-" {
        let mut stdin = String::new();
        std::io::stdin().read_to_string(&mut stdin)?;
        stdin
    } else {
        program.pattern
    };
    let pattern = Pattern::parse(&source)?;
    let mut count = 0;
    for segment in pattern.segments() {
        let candidates = segment.expand();
        if candidates.is_empty() {
            continue;
        }

        for candidate in &candidates {
            let queue = candidate
                .iter()
                .map(|piece| piece.to_string())
                .collect::<Vec<_>>()
                .join("");
            println!("{queue}");
        }
        count += candidates.len();
    }

    if program.count {
        eprintln!("{count}");
    }
    Ok(())
}