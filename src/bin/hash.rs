//! # Field hash
//! To find the hash of a field, first place all cleared lines at the bottom of the stack. Then the hash is simply the field read as a 40-bit binary number from left to right from top to bottom, where an empty mino represents 0 and a filled mino represents 1. Example:
//!
//! ```
//! xxxxxx....
//! xxxxx.....
//! xxxxxxxxxx
//! xxxxxx...x
//!
//! ```
//! 
//! Bring the cleared line to the bottom of the stack:
//!
//! ```
//! xxxxxx....
//! xxxxx.....
//! xxxxxx...x
//! xxxxxxxxxx
//!
//! ```
//!
//! Now read it as a binary number:
//!
//! ```
//! 1111110000
//! 1111100000
//! 1111110001
//! 1111111111
//! ```
//!
//! ```rs
//! 0b1111110000_1111100000_1111110001_1111111111 = 1083372980223
//! ```

use clap::Parser;

#[derive(clap::Parser)]
pub struct Program {
    pub fumen: String,
}
fn filled(cell: fumen::CellColor) -> bool {
    cell != fumen::CellColor::Empty
}

fn row_bits(row: &[fumen::CellColor; 10]) -> u64 {
    let mut bits = 0;
    for cell in row {
        bits = (bits << 1) | u64::from(filled(*cell));
    }
    bits
}

pub fn main() -> anyhow::Result<()> {
    let program = Program::parse();
    let fumen = fumen::Fumen::decode(&program.fumen)?;
    // y-up
    let field: [[fumen::CellColor; 10]; 23] = fumen.pages[0].field;

    // height = topmost filled row + 1
    let height = (0..23)
        .rev()
        .find(|&y| field[y].iter().any(|&c| filled(c)))
        .map_or(0, |y| y + 1);

    let cleared = |y: usize| field[y].iter().all(|&c| filled(c));

    // push all cleared lines to the bottom of the stack, then read the field
    // top-to-bottom (most significant bits first) as a 10-bit-per-row number.
    let mut hash = 0;
    for y in (0..height).rev() {
        if !cleared(y) {
            hash = (hash << 10) | row_bits(&field[y]);
        }
    }
    for y in (0..height).rev() {
        if cleared(y) {
            hash = (hash << 10) | row_bits(&field[y]);
        }
    }

    println!("{hash}");

    Ok(())
}