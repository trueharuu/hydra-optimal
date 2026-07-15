use anyhow::{bail, Context, Result};
use std::array;
use std::fmt::Display;
use std::fs;
use std::ops::{Add, Sub};
use std::path::Path;

pub const PIECE_ORDER: &[u8; 7] = b"IJLOSTZ";
pub const FULL_BAG: u8 = 0x7f;
pub const WEIGHT_SCALE: f64 = 1.0 / 4_294_967_296.0;

pub trait Cost:
    Copy
    + Send
    + Sync
    + PartialEq
    + PartialOrd
    + Add<Output = Self>
    + Sub<Output = Self>
    + Default
    + Display
    + 'static
{
    fn zero() -> Self;
    fn one() -> Self;
    fn epsilon() -> Self;
    fn from_weight(value: u64) -> Self;
    fn mul_small(self, value: usize) -> Self;
    fn min(self, other: Self) -> Self {
        if self < other {
            self
        } else {
            other
        }
    }
    fn is_zero(self) -> bool {
        self == Self::zero()
    }
    fn scaled_output(self) -> String;
}

impl Cost for u64 {
    #[inline]
    fn zero() -> Self {
        0
    }
    #[inline]
    fn one() -> Self {
        1
    }
    #[inline]
    fn epsilon() -> Self {
        1
    }
    #[inline]
    fn from_weight(value: u64) -> Self {
        value
    }
    #[inline]
    fn mul_small(self, value: usize) -> Self {
        self * value as u64
    }
    fn scaled_output(self) -> String {
        self.to_string()
    }
}

impl Cost for f64 {
    #[inline]
    fn zero() -> Self {
        0.0
    }
    #[inline]
    fn one() -> Self {
        1.0
    }
    #[inline]
    fn epsilon() -> Self {
        WEIGHT_SCALE
    }
    #[inline]
    fn from_weight(value: u64) -> Self {
        value as f64 * WEIGHT_SCALE
    }
    #[inline]
    fn mul_small(self, value: usize) -> Self {
        self * value as f64
    }
    fn scaled_output(self) -> String {
        let value = self / WEIGHT_SCALE;
        if value.fract() == 0.0 {
            format!("{value:.0}")
        } else {
            value.to_string()
        }
    }
}

/// Weighted-mode cost in the original integer units from `weights.txt`.
///
/// The C++ implementation divides every weight by 2^32, performs the search in f64, then
/// multiplies the result by 2^32 for output. All supported totals fit in u64, so retaining the
/// integer units is both faster and exactly deterministic.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord)]
pub struct WeightedCost(pub u64);

impl Add for WeightedCost {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self::Output {
        Self(self.0 + rhs.0)
    }
}

impl Sub for WeightedCost {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self::Output {
        Self(self.0 - rhs.0)
    }
}

impl Display for WeightedCost {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.0.fmt(f)
    }
}

impl Cost for WeightedCost {
    #[inline]
    fn zero() -> Self {
        Self(0)
    }
    #[inline]
    fn one() -> Self {
        Self(1u64 << 32)
    }
    #[inline]
    fn epsilon() -> Self {
        Self(1)
    }
    #[inline]
    fn from_weight(value: u64) -> Self {
        Self(value)
    }
    #[inline]
    fn mul_small(self, value: usize) -> Self {
        Self(self.0 * value as u64)
    }
    fn scaled_output(self) -> String {
        self.0.to_string()
    }
}

#[derive(Clone, Copy, Debug)]
pub struct MinArray<T: Cost> {
    pub values: [T; 7],
    pub min: T,
}

impl<T: Cost> MinArray<T> {
    pub fn new(mut values: [T; 7]) -> Self {
        let mut min = values[0];
        for &value in &values[1..] {
            min = min.min(value);
        }
        for value in &mut values {
            *value = *value - min;
        }
        Self { values, min }
    }
}

impl<T: Cost> Default for MinArray<T> {
    fn default() -> Self {
        Self::new([T::zero(); 7])
    }
}

#[derive(Clone)]
pub struct Weights<T: Cost> {
    by_mask: [Option<MinArray<T>>; 128],
    default: MinArray<T>,
}

impl<T: Cost> Weights<T> {
    pub fn flat() -> Self {
        Self {
            by_mask: array::from_fn(|_| None),
            default: MinArray::default(),
        }
    }

    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let input = fs::read_to_string(path)
            .with_context(|| format!("failed to read {}", path.display()))?;
        Self::parse(&input).with_context(|| format!("failed to parse {}", path.display()))
    }

    fn parse(input: &str) -> Result<Self> {
        let mut tokens = input.split_whitespace();
        let mut by_mask: [Option<MinArray<T>>; 128] = array::from_fn(|_| None);

        for row in 0..127 {
            let name = tokens
                .next()
                .with_context(|| format!("missing name in weight row {}", row + 1))?;
            let mask = if name == "null" {
                0
            } else {
                bag_mask(name).with_context(|| format!("invalid weight row name {name:?}"))?
            };
            if mask == FULL_BAG {
                bail!("weights file must not contain a full-bag row");
            }
            if by_mask[mask as usize].is_some() {
                bail!("duplicate weight row for {name:?}");
            }
            let mut values = [T::zero(); 7];
            for (piece, slot) in values.iter_mut().enumerate() {
                let raw = tokens
                    .next()
                    .with_context(|| format!("missing piece {piece} in weight row {}", row + 1))?
                    .parse::<u64>()
                    .with_context(|| format!("invalid number in weight row {}", row + 1))?;
                if raw > (1u64 << 32) {
                    bail!("weights must be in the range [0, 2^32]");
                }
                *slot = T::from_weight(raw);
            }
            by_mask[mask as usize] = Some(MinArray::new(values));
        }

        if tokens.next().is_some() {
            bail!("weights file contains data after the expected 127 rows");
        }

        Ok(Self {
            by_mask,
            default: MinArray::default(),
        })
    }

    #[inline]
    pub fn get(&self, mask: u8) -> &MinArray<T> {
        self.by_mask[mask as usize]
            .as_ref()
            .unwrap_or(&self.default)
    }
}

pub fn piece_id(piece: u8) -> Option<u8> {
    PIECE_ORDER
        .iter()
        .position(|&p| p == piece)
        .map(|p| p as u8)
}

pub fn piece_char(piece: u8) -> char {
    PIECE_ORDER[piece as usize] as char
}

pub fn bag_mask(text: &str) -> Result<u8> {
    let mut mask = 0u8;
    for byte in text.bytes() {
        let piece = piece_id(byte).with_context(|| format!("invalid piece {}", byte as char))?;
        mask |= 1 << piece;
    }
    Ok(mask)
}

#[inline]
pub fn pieces(mask: u8) -> impl Iterator<Item = u8> {
    (0u8..7).filter(move |&piece| mask & (1 << piece) != 0)
}

pub fn bag_string(mask: u8) -> String {
    pieces(mask).map(piece_char).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bags_are_ordered_and_deduplicated() {
        assert_eq!(bag_mask("ZZIJS").unwrap(), 0b1010011);
        assert_eq!(bag_string(bag_mask("ZZIJS").unwrap()), "IJSZ");
    }

    #[test]
    fn min_array_subtracts_the_row_minimum() {
        let row = MinArray::new([8u64, 3, 5, 9, 4, 7, 6]);
        assert_eq!(row.min, 3);
        assert_eq!(row.values, [5, 0, 2, 6, 1, 4, 3]);
    }

    #[test]
    fn rejects_duplicate_weight_compositions() {
        use std::fmt::Write as _;

        let mut input = String::new();
        for mask in 0..FULL_BAG {
            let mask = if mask == 1 { 0 } else { mask };
            let name = if mask == 0 {
                "null".to_owned()
            } else {
                bag_string(mask)
            };
            writeln!(input, "{name} 0 0 0 0 0 0 0").unwrap();
        }

        let error = match Weights::<u64>::parse(&input) {
            Ok(_) => panic!("duplicate weight composition was accepted"),
            Err(error) => error,
        };
        assert!(error.to_string().contains("duplicate weight row"));
    }
}
