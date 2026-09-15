use itertools::Itertools;
use std::fmt::Display;

// X;Y
#[derive(Clone, PartialEq, Debug)]
pub struct Pattern(pub Vec<Segment>);

#[derive(Clone, PartialEq, Debug)]
pub enum Segment {
    Single(Piece),                // X
    Sequence(Vec<Self>),          // XY or X,Y
    Group(Box<Self>),             // (X)
    Bag(Vec<Self>),               // [XYZ], the set of pieces X, Y, and Z
    Except(Vec<Self>),            // [^XYZ], which is [TIJLOSZ] \ {X, Y, Z}
    Wildcard,                     // *, is exactly [TIJLOSZ]
    Permute(Box<Self>, usize),    // XpN, returns all permutations of N elements from X
    Choose(Box<Self>, usize),     // XcN, returns all combinations of N elements from X
    All(Box<Self>),               // X! returns all permutations of all elements in X
    Filter(Box<Self>, Condition), // X{C} returns all elements of X that satisfy condition C
}

#[derive(Clone, PartialEq, Debug)]
pub enum Condition {
    And(Vec<Self>),                                // X & Y
    Or(Vec<Self>),                                 // X | Y
    Not(Box<Self>),                                // !X
    Order(Box<Pattern>, Box<Pattern>, Comparator), // X < Y, X > Y, X <= Y, X >= Y, X == Y, X != Y
    Count(Box<Pattern>, usize, Comparator),        // X < N, X > N, X <= N, X >= N, X == N, X != N
    Exists(Box<Pattern>),                              // X exists, i.e. count(X) > 0
}

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Comparator {
    Eq,
    Ne,
    Gt,
    Lt,
    Ge,
    Le,
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum Piece {
    T,
    I,
    J,
    L,
    O,
    S,
    Z,
}

impl Display for Piece {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let c = match self {
            Self::T => 'T',
            Self::I => 'I',
            Self::J => 'J',
            Self::L => 'L',
            Self::O => 'O',
            Self::S => 'S',
            Self::Z => 'Z',
        };
        write!(f, "{c}")
    }
}

impl Pattern {
    pub fn expand(&self) -> Vec<Vec<Piece>> {
        let mut result = Vec::new();

        for segment in &self.0 {
            result.extend(segment.expand());
        }

        result
    }

    /// Parse a pattern from its textual form (see the grammar docs).
    pub fn parse(src: &str) -> anyhow::Result<Self> {
        let mut parser = Parser::new(src);
        let pattern = parser.parse_pattern()?;
        parser.skip_ws();
        if parser.pos != parser.chars.len() {
            return Err(anyhow::anyhow!(
                "unexpected '{}' at character {}",
                parser.chars[parser.pos],
                parser.pos
            ));
        }
        Ok(pattern)
    }

    /// The top-level `X;Y` segments that make up the pattern.
    pub fn segments(&self) -> &[Segment] {
        &self.0
    }
}

impl std::str::FromStr for Pattern {
    type Err = anyhow::Error;

    fn from_str(src: &str) -> std::result::Result<Self, Self::Err> {
        Self::parse(src)
    }
}

impl Segment {
    pub fn expand(&self) -> Vec<Vec<Piece>> {
        match self {
            Self::Single(piece) => vec![vec![*piece]],

            // AB is a product of A and B, [TI][JL] should produce TJ;TL;IJ;IL
            Self::Sequence(segments) => {
                let mut result = vec![vec![]];
                for segment in segments {
                    let expanded = segment.expand();
                    let mut new_result = Vec::new();
                    for prefix in result {
                        for suffix in &expanded {
                            let mut combined = prefix.clone();
                            combined.extend(suffix.clone());
                            new_result.push(combined);
                        }
                    }
                    result = new_result;
                }
                result
            }

            Self::Group(inner) => inner.expand(),
            Self::Bag(inner) => {
                let mut result = Vec::new();
                for segment in inner {
                    result.extend(segment.expand());
                }
                result
            }

            Self::Wildcard => {
                vec![
                    vec![Piece::T],
                    vec![Piece::I],
                    vec![Piece::J],
                    vec![Piece::L],
                    vec![Piece::O],
                    vec![Piece::S],
                    vec![Piece::Z],
                ]
            }

            Self::Except(inner) => {
                let mut excluded = Vec::new();
                for segment in inner {
                    for exp in segment.expand() {
                        excluded.extend(exp);
                    }
                }
                let all_pieces = vec![
                    Piece::T,
                    Piece::I,
                    Piece::J,
                    Piece::L,
                    Piece::O,
                    Piece::S,
                    Piece::Z,
                ];
                let mut result = Vec::new();
                for piece in all_pieces {
                    if !excluded.contains(&piece) {
                        result.push(vec![piece]);
                    }
                }
                result
            }

            Self::Permute(expr, n) => {
                expr.expand()
                    .into_iter()
                    .permutations(*n)
                    .map(|perm| perm.into_iter().flatten().collect())
                    .unique()
                    .collect()
            }

            Self::Choose(expr, n) => {
                expr.expand()
                    .into_iter()
                    .combinations(*n)
                    .map(|comb| comb.into_iter().flatten().collect())
                    .unique()
                    .collect()
            }

            Self::All(expr) => {
                let mut elements = Vec::new();
                for exp in expr.expand() {
                    elements.extend(exp);
                }
                let len = elements.len();
                elements
                    .into_iter()
                    .permutations(len)
                    .unique()
                    .collect()
            }

            Self::Filter(a, c) => {
                let expanded = a.expand();
                let mut result = Vec::new();
                for exp in expanded {
                    if c.evaluate(&exp) {
                        result.push(exp);
                    }
                }
                result
            }
        }
    }
}

impl Condition {
    pub fn evaluate(&self, pieces: &[Piece]) -> bool {
        match self {
            Self::And(conditions) => conditions.iter().all(|c| c.evaluate(pieces)),
            Self::Or(conditions) => conditions.iter().any(|c| c.evaluate(pieces)),
            Self::Not(condition) => !condition.evaluate(pieces),
            Self::Order(a, b, comparator) => {
                let aw = occurrences(a, pieces);
                let bw = occurrences(b, pieces);
                if aw.is_empty() || bw.is_empty() {
                    return false;
                }
                match comparator {
                    Comparator::Eq | Comparator::Ne => {
                        let same = aw
                            .iter()
                            .any(|wa| bw.iter().any(|wb| wa == wb));
                        match comparator {
                            Comparator::Eq => same,
                            _ => !same,
                        }
                    }
                    Comparator::Lt | Comparator::Le => aw
                        .iter()
                        .any(|wa| bw.iter().any(|wb| wa.1 <= wb.0)),
                    Comparator::Gt | Comparator::Ge => bw
                        .iter()
                        .any(|wb| aw.iter().any(|wa| wb.1 <= wa.0)),
                }
            }
            Self::Count(a, n, comparator) => {
                let count = count_occurrences(a, pieces);
                compare_count(count, *n, *comparator)
            }
            Self::Exists(a) => count_occurrences(a, pieces) > 0,
        }
    }
}

/// The end index of the longest expansion of `pattern` matching `pieces[start..]`, if any.
fn matches_at(pattern: &Pattern, pieces: &[Piece], start: usize) -> Option<usize> {
    let mut longest = None;
    for expansion in pattern.expand() {
        let len = expansion.len();
        if len == 0 || start + len > pieces.len() {
            continue;
        }
        if pieces[start..start + len] == expansion
            && longest.is_none_or(|best: usize| len > best - start)
        {
            longest = Some(start + len);
        }
    }
    longest
}

/// Every (start, end) window of `pieces` matched by some expansion of `pattern`.
fn occurrences(pattern: &Pattern, pieces: &[Piece]) -> Vec<(usize, usize)> {
    let mut result = Vec::new();
    for expansion in pattern.expand() {
        let len = expansion.len();
        if len == 0 || len > pieces.len() {
            continue;
        }
        for start in 0..=(pieces.len() - len) {
            if pieces[start..start + len] == expansion {
                result.push((start, start + len));
            }
        }
    }
    result.sort_unstable();
    result.dedup();
    result
}

/// The number of non-overlapping matches of `pattern` in `pieces`, counting greedily
/// left-to-right with the longest match taken at each position.
fn count_occurrences(pattern: &Pattern, pieces: &[Piece]) -> usize {
    let mut count = 0;
    let mut pos = 0;
    while pos < pieces.len() {
        match matches_at(pattern, pieces, pos) {
            Some(end) => {
                count += 1;
                pos = end;
            }
            None => pos += 1,
        }
    }
    count
}

fn compare_count(count: usize, n: usize, comparator: Comparator) -> bool {
    match comparator {
        Comparator::Eq => count == n,
        Comparator::Ne => count != n,
        Comparator::Lt => count < n,
        Comparator::Le => count <= n,
        Comparator::Gt => count > n,
        Comparator::Ge => count >= n,
    }
}

struct Parser {
    chars: Vec<char>,
    pos: usize,
}

fn seq_segment(terms: Vec<Segment>) -> Segment {
    match terms.len() {
        1 => terms.into_iter().next().unwrap(),
        _ => Segment::Sequence(terms),
    }
}

impl Parser {
    fn new(src: &str) -> Self {
        Self {
            chars: src.chars().collect(),
            pos: 0,
        }
    }

    fn error<T>(&self, msg: impl std::fmt::Display) -> anyhow::Result<T> {
        Err(anyhow::anyhow!("{msg} at character {}", self.pos))
    }

    fn peek(&self) -> Option<char> {
        self.chars.get(self.pos).copied()
    }

    fn peek_next(&self) -> Option<char> {
        self.chars.get(self.pos + 1).copied()
    }

    fn eat(&mut self, c: char) -> bool {
        if self.peek() == Some(c) {
            self.pos += 1;
            true
        } else {
            false
        }
    }

    fn expect(&mut self, c: char) -> anyhow::Result<()> {
        if self.eat(c) {
            Ok(())
        } else {
            self.error(format!("expected '{c}'"))
        }
    }

    fn skip_ws(&mut self) {
        while self.peek().is_some_and(|c| c.is_whitespace() && c != '\n') {
            self.pos += 1;
        }
    }

    fn digits(&mut self) -> anyhow::Result<usize> {
        let mut n: usize = 0;
        let mut seen = false;
        while let Some(c) = self.peek() {
            if !c.is_ascii_digit() {
                break;
            }
            seen = true;
            let d = c.to_digit(10).unwrap() as usize;
            n = n
                .checked_mul(10)
                .and_then(|v| v.checked_add(d))
                .ok_or_else(|| anyhow::anyhow!("number too large at character {}", self.pos))?;
            self.pos += 1;
        }
        if seen {
            Ok(n)
        } else {
            self.error("expected digits")
        }
    }

    fn parse_pattern(&mut self) -> anyhow::Result<Pattern> {
        self.skip_ws();
        let mut segments = vec![self.parse_sequence()?];
        loop {
            self.skip_ws();
            if self.eat(';') {
                self.skip_ws_after_newline();
                segments.push(self.parse_sequence()?);
                continue;
            }
            if self.eat('\n') {
                self.skip_ws_after_newline();
                if self.peek().is_none() {
                    break;
                }
                segments.push(self.parse_sequence()?);
                continue;
            }
            break;
        }
        Ok(Pattern(segments))
    }

    /// Skip whitespace including newlines, used after a newline separator so
    /// leading blank lines and runs of newlines don't create empty segments.
    fn skip_ws_after_newline(&mut self) {
        while self.peek().is_some_and(|c| c.is_whitespace()) {
            self.pos += 1;
        }
    }

    fn parse_sequence(&mut self) -> anyhow::Result<Segment> {
        self.skip_ws();
        let mut terms = vec![self.parse_concatenation()?];
        loop {
            self.skip_ws();
            if !self.eat(',') {
                break;
            }
            terms.push(self.parse_concatenation()?);
        }
        Ok(seq_segment(terms))
    }

    /// A run of juxtaposed terms. Stops at any token that cannot continue a term.
    fn parse_concatenation(&mut self) -> anyhow::Result<Segment> {
        self.skip_ws();
        let mut terms = vec![self.parse_term()?];
        while let Some(c) = self.peek() {
            match c {
                ';' | '\n' | ',' | ']' | '}' | ')' | '&' | '|' | '<' | '>' | '=' | '!' => break,
                _ => {}
            }
            terms.push(self.parse_term()?);
        }
        Ok(seq_segment(terms))
    }

    fn parse_term(&mut self) -> anyhow::Result<Segment> {
        self.skip_ws();
        let mut segment = self.parse_atom()?;
        loop {
            self.skip_ws();
            match self.peek() {
                Some('p') => {
                    self.pos += 1;
                    let n = self.digits()?;
                    segment = Segment::Permute(Box::new(segment), n);
                }
                Some('c') => {
                    self.pos += 1;
                    let n = self.digits()?;
                    segment = Segment::Choose(Box::new(segment), n);
                }
                // '!' followed by '=' starts a comparison inside a condition, so it
                // cannot be the All postfix.
                Some('!') if self.peek_next() != Some('=') => {
                    self.pos += 1;
                    segment = Segment::All(Box::new(segment));
                }
                Some('{') => {
                    self.pos += 1;
                    let condition = self.parse_condition()?;
                    self.expect('}')?;
                    segment = Segment::Filter(Box::new(segment), condition);
                }
                _ => break,
            }
        }
        Ok(segment)
    }

    fn parse_atom(&mut self) -> anyhow::Result<Segment> {
        self.skip_ws();
        match self.peek() {
            Some(c) if c.is_alphabetic() => {
                let piece = match c.to_ascii_uppercase() {
                    'T' => Piece::T,
                    'I' => Piece::I,
                    'J' => Piece::J,
                    'L' => Piece::L,
                    'O' => Piece::O,
                    'S' => Piece::S,
                    'Z' => Piece::Z,
                    _ => return self.error(format!("unknown piece '{c}'")),
                };
                self.pos += 1;
                Ok(Segment::Single(piece))
            }
            Some('*') => {
                self.pos += 1;
                Ok(Segment::Wildcard)
            }
            Some('(') => {
                self.pos += 1;
                let inner = self.parse_sequence()?;
                self.expect(')')?;
                Ok(Segment::Group(Box::new(inner)))
            }
            Some('[') => self.parse_bag(),
            Some(c) => self.error(format!("expected an atom, found '{c}'")),
            None => self.error("unexpected end of input, expected an atom"),
        }
    }

    fn parse_bag(&mut self) -> anyhow::Result<Segment> {
        self.expect('[')?;
        let except = self.eat('^');
        let mut items = Vec::new();
        loop {
            self.skip_ws();
            if self.peek() == Some(']') {
                break;
            }
            items.push(self.parse_term()?);
            self.skip_ws();
            if self.peek() == Some(']') {
                break;
            }
            self.eat(',');
        }
        self.expect(']')?;
        if items.is_empty() {
            return self.error("empty bag");
        }
        Ok(if except {
            Segment::Except(items)
        } else {
            Segment::Bag(items)
        })
    }

    fn parse_condition(&mut self) -> anyhow::Result<Condition> {
        self.skip_ws();
        let mut parts = vec![self.parse_and_term()?];
        loop {
            self.skip_ws();
            if !self.eat('|') {
                break;
            }
            parts.push(self.parse_and_term()?);
        }
        Ok(match parts.len() {
            1 => parts.pop().unwrap(),
            _ => Condition::Or(parts),
        })
    }

    fn parse_and_term(&mut self) -> anyhow::Result<Condition> {
        self.skip_ws();
        let mut parts = vec![self.parse_unary()?];
        loop {
            self.skip_ws();
            if !self.eat('&') {
                break;
            }
            parts.push(self.parse_unary()?);
        }
        Ok(match parts.len() {
            1 => parts.pop().unwrap(),
            _ => Condition::And(parts),
        })
    }

    fn parse_unary(&mut self) -> anyhow::Result<Condition> {
        self.skip_ws();
        if self.eat('!') {
            Ok(Condition::Not(Box::new(self.parse_unary()?)))
        } else if self.peek() == Some('(') {
            self.pos += 1;
            let inner = self.parse_condition()?;
            self.expect(')')?;
            Ok(inner)
        } else {
            self.parse_comparison()
        }
    }

    fn parse_comparison(&mut self) -> anyhow::Result<Condition> {
        self.skip_ws();
        let lhs = Box::new(self.parse_pattern()?);
        self.skip_ws();
        let comparator = self.parse_relop()?;
        self.skip_ws();
        if self.peek().is_some_and(|c| c.is_ascii_digit()) {
            let n = self.digits()?;
            Ok(Condition::Count(lhs, n, comparator))
        } else {
            let rhs = self.parse_pattern()?;
            Ok(Condition::Order(lhs, Box::new(rhs), comparator))
        }
    }

    fn parse_relop(&mut self) -> anyhow::Result<Comparator> {
        self.skip_ws();
        let c = self
            .peek()
            .ok_or_else(|| anyhow::anyhow!("expected a comparison, found end of input at character {}", self.pos))?;
        let (comparator, len) = match (c, self.peek_next()) {
            ('=', Some('=')) => (Comparator::Eq, 2),
            ('!', Some('=')) => (Comparator::Ne, 2),
            ('<', Some('=')) => (Comparator::Le, 2),
            ('>', Some('=')) => (Comparator::Ge, 2),
            ('<', _) => (Comparator::Lt, 1),
            ('>', _) => (Comparator::Gt, 1),
            _ => return self.error(format!("expected a comparison operator, found '{c}'")),
        };
        self.pos += len;
        Ok(comparator)
    }
}

impl Display for Pattern {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let segments: Vec<String> = self.0.iter().map(|s| s.to_string()).collect();
        write!(f, "{}", segments.join(";"))
    }
}

impl Display for Segment {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Single(piece) => write!(f, "{piece}"),
            Self::Sequence(segments) => {
                let segments: Vec<String> = segments.iter().map(|s| s.to_string()).collect();
                write!(f, "{}", segments.join(""))
            }
            Self::Group(inner) => write!(f, "({inner})"),
            Self::Bag(inner) => {
                let inner: Vec<String> = inner.iter().map(|s| s.to_string()).collect();
                write!(f, "[{}]", inner.join(""))
            }
            Self::Except(inner) => {
                let inner: Vec<String> = inner.iter().map(|s| s.to_string()).collect();
                write!(f, "[^{}]", inner.join(""))
            }
            Self::Wildcard => write!(f, "*"),
            Self::Permute(inner, n) => write!(f, "{}p{n}", inner),
            Self::Choose(inner, n) => write!(f, "{}c{n}", inner),
            Self::All(inner) => write!(f, "{}!", inner),
            Self::Filter(inner, condition) => write!(f, "{}{{{condition}}}", inner),
        }
    }
}

impl Display for Condition {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::And(conditions) => {
                let conditions: Vec<String> = conditions.iter().map(|c| c.to_string()).collect();
                write!(f, "{}", conditions.join("&"))
            }
            Self::Or(conditions) => {
                let conditions: Vec<String> = conditions.iter().map(|c| c.to_string()).collect();
                write!(f, "{}", conditions.join("|"))
            }
            Self::Not(condition) => write!(f, "!{condition}"),
            Self::Order(lhs, rhs, comparator) => {
                let op = match comparator {
                    Comparator::Eq => "==",
                    Comparator::Ne => "!=",
                    Comparator::Lt => "<",
                    Comparator::Le => "<=",
                    Comparator::Gt => ">",
                    Comparator::Ge => ">=",
                };
                write!(f, "{lhs}{op}{rhs}")
            }
            Self::Count(pattern, n, comparator) => {
                let op = match comparator {
                    Comparator::Eq => "==",
                    Comparator::Ne => "!=",
                    Comparator::Lt => "<",
                    Comparator::Le => "<=",
                    Comparator::Gt => ">",
                    Comparator::Ge => ">=",
                };
                write!(f, "{pattern}{op}{n}")
            }
            Self::Exists(pattern) => write!(f, "{pattern}"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn seq(s: &str) -> Vec<Piece> {
        s.chars()
            .map(|c| match c {
                'T' => Piece::T,
                'I' => Piece::I,
                'J' => Piece::J,
                'L' => Piece::L,
                'O' => Piece::O,
                'S' => Piece::S,
                'Z' => Piece::Z,
                _ => panic!("bad piece {c}"),
            })
            .collect()
    }

    fn single(p: Piece) -> Segment {
        Segment::Single(p)
    }

    fn sequence(segments: Vec<Segment>) -> Segment {
        Segment::Sequence(segments)
    }

    fn bag(segments: Vec<Segment>) -> Segment {
        Segment::Bag(segments)
    }

    fn order(a: &str, b: &str, comparator: Comparator) -> Condition {
        Condition::Order(
            Box::new(Pattern(vec![single(seq(a)[0])])),
            Box::new(Pattern(vec![single(seq(b)[0])])),
            comparator,
        )
    }

    fn count(pieces: Vec<Piece>, n: usize, comparator: Comparator) -> Condition {
        let sequence = sequence(pieces.iter().map(|p| single(*p)).collect());
        Condition::Count(Box::new(Pattern(vec![sequence])), n, comparator)
    }

    #[test]
    fn order_single_occurrences() {
        let candidate = seq("ILJZ");
        assert!(order("I", "J", Comparator::Lt).evaluate(&candidate));
        assert!(!order("J", "I", Comparator::Lt).evaluate(&candidate));
        assert!(order("I", "Z", Comparator::Le).evaluate(&candidate));
        assert!(order("J", "I", Comparator::Gt).evaluate(&candidate));
        assert!(order("Z", "I", Comparator::Ge).evaluate(&candidate));
        assert!(order("O", "O", Comparator::Eq).evaluate(&seq("O")));
        assert!(!order("I", "O", Comparator::Eq).evaluate(&seq("O")));
        assert!(order("I", "J", Comparator::Ne).evaluate(&seq("IOJZ")));
    }

    #[test]
    fn order_exists_loose_with_repeats() {
        let tot = seq("TOT");
        assert!(order("T", "O", Comparator::Lt).evaluate(&tot));
        assert!(order("O", "T", Comparator::Lt).evaluate(&tot));
        let oot = seq("OOT");
        assert!(order("O", "T", Comparator::Lt).evaluate(&oot));
        assert!(!order("T", "O", Comparator::Lt).evaluate(&oot));
    }

    #[test]
    fn order_sequence_args() {
        let il = Pattern(vec![sequence(vec![single(Piece::I), single(Piece::L)])]);
        let j = Pattern(vec![single(Piece::J)]);
        let cond = Condition::Order(Box::new(il), Box::new(j), Comparator::Lt);
        assert!(cond.evaluate(&seq("ILJZ")));
        assert!(!cond.evaluate(&seq("LIJZ")));
    }

    #[test]
    fn order_absent_side_is_false() {
        let ool = seq("OOL");
        assert!(!order("I", "J", Comparator::Lt).evaluate(&ool));
        assert!(!order("I", "J", Comparator::Eq).evaluate(&ool));
        assert!(!order("S", "S", Comparator::Eq).evaluate(&ool));
        assert!(!order("S", "S", Comparator::Ne).evaluate(&ool));
    }

    #[test]
    fn order_nested_bool() {
        let candidate = seq("ILJZ");
        let cond = Condition::And(vec![
            order("I", "L", Comparator::Lt),
            order("L", "J", Comparator::Lt),
        ]);
        assert!(cond.evaluate(&candidate));
        let cond = Condition::And(vec![
            order("I", "L", Comparator::Lt),
            order("J", "L", Comparator::Lt),
        ]);
        assert!(!cond.evaluate(&candidate));
    }

    #[test]
    fn count_single_pieces() {
        assert!(count(seq("O"), 1, Comparator::Eq).evaluate(&seq("O")));
        assert!(!count(seq("O"), 1, Comparator::Eq).evaluate(&seq("OO")));
        assert!(count(seq("O"), 2, Comparator::Eq).evaluate(&seq("OO")));
        assert!(count(seq("O"), 1, Comparator::Ge).evaluate(&seq("OOO")));
        assert!(count(seq("S"), 0, Comparator::Eq).evaluate(&seq("OOL")));
        assert!(count(seq("S"), 0, Comparator::Ne).evaluate(&seq("OSL")));
    }

    #[test]
    fn count_greedy_non_overlap() {
        assert!(count(seq("OO"), 1, Comparator::Eq).evaluate(&seq("OOO")));
        assert!(!count(seq("OO"), 2, Comparator::Ge).evaluate(&seq("OOO")));
        assert!(count(seq("OO"), 1, Comparator::Eq).evaluate(&seq("OOOI")));
        assert!(count(seq("OO"), 2, Comparator::Eq).evaluate(&seq("OOOO")));
    }

    #[test]
    fn count_bag_and_wildcard() {
        let cond = Condition::Count(
            Box::new(Pattern(vec![bag(vec![single(Piece::O), single(Piece::L)])])),
            3,
            Comparator::Eq,
        );
        assert!(cond.evaluate(&seq("OOL")));

        let cond = Condition::Count(
            Box::new(Pattern(vec![Segment::Wildcard])),
            3,
            Comparator::Eq,
        );
        assert!(cond.evaluate(&seq("OOL")));
        assert!(!cond.evaluate(&seq("OOLO")));
    }

    #[test]
    fn count_longest_match_at_position() {
        let mixed = Pattern(vec![single(Piece::O), sequence(vec![single(Piece::O), single(Piece::O)])]);
        let cond = Condition::Count(Box::new(mixed), 2, Comparator::Eq);
        assert!(cond.evaluate(&seq("OOO")));
    }

    #[test]
    fn filter_end_to_end() {
        let inner = sequence(vec![
            bag(vec![single(Piece::I), single(Piece::L)]),
            single(Piece::J),
            bag(vec![single(Piece::O), single(Piece::S)]),
        ]);
        let filter = Segment::Filter(Box::new(inner), order("J", "S", Comparator::Lt));
        let expanded = Pattern(vec![filter]).expand();
        assert_eq!(expanded, vec![seq("IJS"), seq("LJS")]);
    }

    fn parsed(s: &str) -> Pattern {
        Pattern::parse(s).unwrap()
    }

    #[test]
    fn parse_basic_patterns() {
        assert_eq!(parsed("I"), Pattern(vec![single(Piece::I)]));
        assert_eq!(parsed("il"), parsed("IL"));
        assert_eq!(parsed("I,J"), parsed("IJ"));
        assert_eq!(
            parsed("I;J"),
            Pattern(vec![single(Piece::I), single(Piece::J)])
        );
        assert_eq!(parsed("*"), Pattern(vec![Segment::Wildcard]));
        assert_eq!(
            parsed("(IL)"),
            Pattern(vec![Segment::Group(Box::new(sequence(vec![
                single(Piece::I),
                single(Piece::L),
            ])))])
        );
        assert_eq!(parsed("ILJ").expand(), vec![seq("ILJ")]);
        assert_eq!(
            parsed("I\nJ"),
            Pattern(vec![single(Piece::I), single(Piece::J)])
        );
        assert_eq!(
            parsed("I;\nJ"),
            Pattern(vec![single(Piece::I), single(Piece::J)])
        );
        assert_eq!(
            parsed("I\n"),
            Pattern(vec![single(Piece::I)])
        );
        assert_eq!(
            parsed("I\n\nJ"),
            Pattern(vec![single(Piece::I), single(Piece::J)])
        );
    }

    #[test]
    fn parse_postfix() {
        assert_eq!(
            parsed("TIp2"),
            Pattern(vec![sequence(vec![
                single(Piece::T),
                Segment::Permute(Box::new(single(Piece::I)), 2),
            ])])
        );
        assert_eq!(
            parsed("(TI)p2"),
            Pattern(vec![Segment::Permute(
                Box::new(Segment::Group(Box::new(sequence(vec![
                    single(Piece::T),
                    single(Piece::I),
                ])))),
                2
            )])
        );
        assert_eq!(
            parsed("Tc2"),
            Pattern(vec![Segment::Choose(Box::new(single(Piece::T)), 2)])
        );
        let zssz = parsed("[ZSSZ]!");
        assert_eq!(
            zssz,
            Pattern(vec![Segment::All(Box::new(bag(vec![
                single(Piece::Z),
                single(Piece::S),
                single(Piece::S),
                single(Piece::Z),
            ])))])
        );
        let mut got: Vec<String> = zssz
            .expand()
            .iter()
            .map(|p| p.iter().map(ToString::to_string).collect())
            .collect();
        let mut want = ["ZSSZ", "ZSZS", "ZZSS", "SZSZ", "SZZS", "SSZZ"];
        want.sort_unstable();
        got.sort();
        assert_eq!(got, want);
        assert_eq!(
            parsed("[Z,S,S,Z]"),
            Pattern(vec![bag(vec![
                single(Piece::Z),
                single(Piece::S),
                single(Piece::S),
                single(Piece::Z),
            ])])
        );
        assert_eq!(
            parsed("[^I]"),
            Pattern(vec![Segment::Except(vec![single(Piece::I)])])
        );
    }

    #[test]
    fn parse_conditions() {
        let expected = Pattern(vec![Segment::Filter(
            Box::new(Segment::All(Box::new(bag(vec![
                single(Piece::Z),
                single(Piece::S),
                single(Piece::S),
                single(Piece::Z),
            ])))),
            Condition::Count(Box::new(Pattern(vec![single(Piece::Z)])), 2, Comparator::Lt),
        )]);
        assert_eq!(parsed("[ZSSZ]!{Z<2}"), expected);

        let expected = Pattern(vec![Segment::Filter(
            Box::new(single(Piece::T)),
            Condition::Or(vec![
                Condition::And(vec![
                    Condition::Order(
                        Box::new(Pattern(vec![single(Piece::I)])),
                        Box::new(Pattern(vec![single(Piece::J)])),
                        Comparator::Lt,
                    ),
                    Condition::Count(
                        Box::new(Pattern(vec![single(Piece::Z)])),
                        2,
                        Comparator::Ge,
                    ),
                ]),
                Condition::Not(Box::new(Condition::Order(
                    Box::new(Pattern(vec![single(Piece::O)])),
                    Box::new(Pattern(vec![single(Piece::L)])),
                    Comparator::Eq,
                ))),
            ]),
        )]);
        assert_eq!(parsed("T{I<J & Z>=2 | !(O==L)}"), expected);

        assert_eq!(
            parsed("T{J != L}"),
            Pattern(vec![Segment::Filter(
                Box::new(single(Piece::T)),
                Condition::Order(
                    Box::new(Pattern(vec![single(Piece::J)])),
                    Box::new(Pattern(vec![single(Piece::L)])),
                    Comparator::Ne,
                )
            )])
        );

        assert_eq!(
            parsed("T{[ZSSZ]! < Z}"),
            Pattern(vec![Segment::Filter(
                Box::new(single(Piece::T)),
                Condition::Order(
                    Box::new(Pattern(vec![Segment::All(Box::new(bag(vec![
                        single(Piece::Z),
                        single(Piece::S),
                        single(Piece::S),
                        single(Piece::Z),
                    ])))])),
                    Box::new(Pattern(vec![single(Piece::Z)])),
                    Comparator::Lt,
                )
            )])
        );
    }

    #[test]
    fn parse_expands_like_hand_built() {
        assert_eq!(parsed("[ZSO]!").expand().len(), 6);
        assert_eq!(parsed("[ZSO]!{Z<O}").expand().len(), 3);
        assert_eq!(
            parsed("[ZSSZ]").expand(),
            vec![seq("Z"), seq("S"), seq("S"), seq("Z")]
        );
        assert_eq!(parsed("[Z,S,S,Z]").expand(), vec![seq("Z"), seq("S"), seq("S"), seq("Z")]);
    }

    #[test]
    fn bag_piece_sets() {
        assert_eq!(parsed("[TIJ]").expand(), parsed("[T,I,J]").expand());
        assert_eq!(
            parsed("[TI][JL]").expand(),
            vec![seq("TJ"), seq("TL"), seq("IJ"), seq("IL")]
        );
        assert_eq!(parsed("[^TIJ]").expand(), parsed("[LOSZ]").expand());
        assert_eq!(
            parsed("[^TIJ]").expand(),
            vec![seq("L"), seq("O"), seq("S"), seq("Z")]
        );
        assert_eq!(
            parsed("[TI]!").expand(),
            vec![seq("TI"), seq("IT")]
        );
    }

    #[test]
    fn permute_and_choose_dedup() {
        assert_eq!(parsed("[SS]p2").expand(), vec![seq("SS")]);
        assert_eq!(parsed("[TI]p3").expand(), Vec::<Vec<Piece>>::new());
        assert_eq!(parsed("[TI]c3").expand(), Vec::<Vec<Piece>>::new());
    }

    #[test]
    fn parse_whitespace_insensitive() {
        assert_eq!(parsed(" [ZSSZ]! { Z < 2 } "), parsed("[ZSSZ]!{Z<2}"));
    }

    #[test]
    fn parse_errors() {
        for src in [
            "", ";", "T;", "[TI", "Tp", "{", "T{}", "Q", "()", "[]", "[^]",
            "I,,J", "I <> J", "I<",
        ] {
            assert!(
                Pattern::parse(src).is_err(),
                "expected an error for {src:?}"
            );
        }
    }
}
