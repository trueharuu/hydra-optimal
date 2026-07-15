use anyhow::{bail, Context, Result};
use memmap2::MmapOptions;
use std::fs::File;
use std::path::Path;

pub const NUM_FIELDS: usize = 15_185_706;
pub const NUM_EDGES: usize = 109_562_993;
pub const PIECE_COUNT: usize = 7;
pub const PC_COMPLETE_ID: u32 = NUM_FIELDS as u32 - 1;
pub const MAX_HASH: u64 = 0xFF_FF_FF_FF_FF;
pub const TWO_LINE_HASH: u64 = 0xF_FF_FF;

const HASH_BYTES: usize = 5;
const TARGET_BYTES: usize = 3;

/// Compact adjacency storage for `graph.bin`.
///
/// Each field stores one base target offset and seven cumulative u8 degrees. All targets live in
/// one flat u32 array. This avoids the original implementation's 106 million `std::vector`
/// objects while keeping the search hot path free of u24 decoding and record scans.
pub struct Graph {
    hashes: Vec<u64>,
    bases: Vec<u32>,
    cumulative_degrees: Vec<[u8; PIECE_COUNT]>,
    targets: Vec<u32>,
    two_line_field: u32,
}

impl Graph {
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let file =
            File::open(path).with_context(|| format!("failed to open {}", path.display()))?;
        let len = file
            .metadata()
            .with_context(|| format!("failed to stat {}", path.display()))?
            .len();
        if len > u32::MAX as u64 {
            bail!("graph is too large for 32-bit offsets: {len} bytes");
        }

        // SAFETY: this is a read-only mapping. graph.bin must not be replaced or truncated while
        // loading. The mapping is dropped as soon as the compact arrays have been built.
        let bytes = unsafe { MmapOptions::new().map(&file) }
            .with_context(|| format!("failed to map {}", path.display()))?;
        #[cfg(unix)]
        let _ = bytes.advise(memmap2::Advice::Sequential);

        parse_graph(&bytes, NUM_FIELDS, NUM_EDGES)
            .with_context(|| format!("failed to parse {}", path.display()))
    }

    #[inline]
    pub fn hash(&self, field: u32) -> u64 {
        self.hashes[field as usize]
    }

    pub fn find_hash(&self, hash: u64) -> Option<u32> {
        self.hashes
            .binary_search(&hash)
            .ok()
            .map(|field| field as u32)
    }

    #[inline]
    pub fn two_line_field(&self) -> u32 {
        self.two_line_field
    }

    #[inline]
    pub fn edges(&self, field: u32, piece: u8) -> &[u32] {
        debug_assert!((field as usize) < self.bases.len());
        debug_assert!((piece as usize) < PIECE_COUNT);
        let field = field as usize;
        let piece = piece as usize;
        let base = self.bases[field] as usize;
        let start = base
            + if piece == 0 {
                0
            } else {
                self.cumulative_degrees[field][piece - 1] as usize
            };
        let end = base + self.cumulative_degrees[field][piece] as usize;
        // SAFETY: parse_graph constructs every base/end from the same targets vector and validates
        // cumulative degrees. This accessor is the innermost search hot path.
        unsafe { self.targets.get_unchecked(start..end) }
    }

    #[inline]
    pub fn has_edges(&self, field: u32, piece: u8) -> bool {
        !self.edges(field, piece).is_empty()
    }

    pub fn edge_count(&self) -> usize {
        self.targets.len()
    }
}

fn parse_graph(bytes: &[u8], fields: usize, edge_capacity: usize) -> Result<Graph> {
    let mut hashes = Vec::with_capacity(fields);
    let mut bases = Vec::with_capacity(fields);
    let mut cumulative_degrees = Vec::with_capacity(fields);
    let mut targets = Vec::with_capacity(edge_capacity);
    let mut pos = 0usize;
    let mut previous_hash = None;

    for field in 0..fields {
        if pos + HASH_BYTES > bytes.len() {
            bail!("truncated hash at field {field}");
        }
        let hash = read_u40_be(&bytes[pos..pos + HASH_BYTES]);
        if let Some(previous) = previous_hash {
            if hash <= previous {
                bail!("field hashes are not strictly increasing at field {field}");
            }
        }
        previous_hash = Some(hash);
        hashes.push(hash);
        pos += HASH_BYTES;

        if targets.len() > u32::MAX as usize {
            bail!("graph contains too many edges for u32 offsets");
        }
        bases.push(targets.len() as u32);
        let mut ends = [0u8; PIECE_COUNT];
        let mut cumulative = 0usize;

        for (piece, end) in ends.iter_mut().enumerate() {
            let Some(&degree) = bytes.get(pos) else {
                bail!("truncated degree at field {field}, piece {piece}");
            };
            pos += 1;
            cumulative += degree as usize;
            if cumulative > u8::MAX as usize {
                bail!("field {field} has more than 255 total outgoing edges");
            }
            *end = cumulative as u8;

            let target_len = degree as usize * TARGET_BYTES;
            if pos + target_len > bytes.len() {
                bail!("truncated targets at field {field}, piece {piece}");
            }
            for target in bytes[pos..pos + target_len].chunks_exact(TARGET_BYTES) {
                let target = read_u24_le(target);
                if target as usize >= fields {
                    bail!("target {target} is out of range at field {field}, piece {piece}");
                }
                targets.push(target);
            }
            pos += target_len;
        }
        cumulative_degrees.push(ends);
    }

    if pos != bytes.len() {
        bail!("graph has {} trailing bytes", bytes.len() - pos);
    }
    if fields == NUM_FIELDS && targets.len() != NUM_EDGES {
        bail!("graph has {} edges; expected {NUM_EDGES}", targets.len());
    }
    if hashes.first().copied() != Some(0) {
        bail!("field 0 is not the empty field");
    }
    if fields == NUM_FIELDS && hashes.last().copied() != Some(MAX_HASH) {
        bail!("last field is not the perfect-clear terminal");
    }
    let two_line_field = hashes
        .binary_search(&TWO_LINE_HASH)
        .ok()
        .context("graph is missing the two-line perfect-clear field")?
        as u32;

    Ok(Graph {
        hashes,
        bases,
        cumulative_degrees,
        targets,
        two_line_field,
    })
}

#[inline]
fn read_u40_be(bytes: &[u8]) -> u64 {
    ((bytes[0] as u64) << 32)
        | ((bytes[1] as u64) << 24)
        | ((bytes[2] as u64) << 16)
        | ((bytes[3] as u64) << 8)
        | bytes[4] as u64
}

#[inline]
fn read_u24_le(bytes: &[u8]) -> u32 {
    (bytes[0] as u32) | ((bytes[1] as u32) << 8) | ((bytes[2] as u32) << 16)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_and_indexes_small_graph() {
        let mut bytes = Vec::new();
        // field 0: hash 0, piece I has targets 1 and 0; every other piece is empty.
        bytes.extend_from_slice(&[0, 0, 0, 0, 0]);
        bytes.push(2);
        bytes.extend_from_slice(&[1, 0, 0, 0, 0, 0]);
        bytes.extend_from_slice(&[0; 6]);
        // field 1 uses the 2-line terminal hash so the generic parser can resolve it.
        bytes.extend_from_slice(&[0, 0, 0x0f, 0xff, 0xff]);
        bytes.extend_from_slice(&[0; 7]);

        let graph = parse_graph(&bytes, 2, 2).unwrap();
        assert_eq!(graph.hash(0), 0);
        assert_eq!(graph.hash(1), TWO_LINE_HASH);
        assert_eq!(graph.find_hash(TWO_LINE_HASH), Some(1));
        assert_eq!(graph.edges(0, 0), &[1, 0]);
        assert!(graph.edges(0, 1).is_empty());
        assert_eq!(graph.edge_count(), 2);
    }

    #[test]
    fn rejects_unsorted_hashes_and_bad_targets() {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&[0, 0, 0, 0, 1]);
        bytes.push(1);
        bytes.extend_from_slice(&[2, 0, 0]);
        bytes.extend_from_slice(&[0; 6]);
        bytes.extend_from_slice(&[0, 0, 0, 0, 0]);
        bytes.extend_from_slice(&[0; 7]);
        assert!(parse_graph(&bytes, 2, 1).is_err());
    }
}
