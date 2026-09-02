use anyhow::{bail, Context, Result};
use crate::score::{bag_mask, piece_id, Cost, FULL_BAG};

pub fn placed_count(hash: u64) -> Result<usize> {
    let minos = hash.count_ones() as usize;
    if minos & 3 != 0 {
        bail!("Somehow not a multiple of 4 minos");
    }
    Ok(minos / 4)
}

pub fn parse_queue(text: &str) -> Result<Vec<u8>> {
    text.bytes()
        .map(|byte| piece_id(byte).with_context(|| format!("Invalid piece {}", byte as char)))
        .collect()
}

pub fn parse_query_bag(text: &str, queue: &[u8], see: usize) -> Result<u8> {
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

pub fn make_cutoffs<T: Cost>(placed: usize, see: usize, bag: u8) -> Vec<T> {
    let count = (12isize - placed as isize - see as isize).max(1) as usize;
    let mut cutoffs = vec![T::zero(); count];
    *cutoffs.last_mut().expect("at least one cutoff") = T::one();
    for i in (0..count - 1).rev() {
        let n = (bag.count_ones() as usize + 7 - i) % 7;
        cutoffs[i] = cutoffs[i + 1].mul_small(if n == 0 { 7 } else { n });
    }
    cutoffs
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::score::bag_string;

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
