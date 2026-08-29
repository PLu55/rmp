//! Max selection over frames.
//!
//! The update pattern is "the value at leaf `n` changed", never insert or delete, so a segment tree
//! fits where a heap does not: O(log F) per changed leaf, O(1) to read the maximum, and no
//! lazy-deletion tombstones to reconcile.
//!
//! Ties resolve to the **lowest index**, in both [`SegTree::argmax`] and any linear scan the caller
//! writes. That is not cosmetic: the incremental path and the full-recompute path must select the
//! same atom in the same order for their books to be comparable, and equal energies do occur (an
//! untouched frame keeps its stored value exactly).

/// A max segment tree over `f64` leaves.
#[derive(Clone, Debug)]
pub struct SegTree {
    /// Leaf count padded to a power of two.
    size: usize,
    /// Real leaf count.
    len: usize,
    /// 1-indexed; `nodes[1]` is the root.
    nodes: Vec<f64>,
}

impl SegTree {
    pub fn new(values: &[f64]) -> Self {
        let len = values.len();
        let size = len.next_power_of_two().max(1);
        let mut nodes = vec![f64::NEG_INFINITY; 2 * size];
        nodes[size..size + len].copy_from_slice(values);
        for i in (1..size).rev() {
            nodes[i] = nodes[2 * i].max(nodes[2 * i + 1]);
        }
        Self { size, len, nodes }
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Replace leaf `i` and repair the path to the root.
    pub fn set(&mut self, i: usize, v: f64) {
        debug_assert!(i < self.len);
        let mut j = self.size + i;
        self.nodes[j] = v;
        while j > 1 {
            j /= 2;
            self.nodes[j] = self.nodes[2 * j].max(self.nodes[2 * j + 1]);
        }
    }

    pub fn get(&self, i: usize) -> f64 {
        self.nodes[self.size + i]
    }

    /// The maximum leaf value, or `-inf` when empty.
    pub fn max(&self) -> f64 {
        self.nodes[1]
    }

    /// Index of the maximum leaf, lowest index on ties.
    pub fn argmax(&self) -> usize {
        let mut j = 1;
        while j < self.size {
            // `>=` takes the left subtree on equality, which yields the lowest index.
            j = if self.nodes[2 * j] >= self.nodes[2 * j + 1] {
                2 * j
            } else {
                2 * j + 1
            };
        }
        j - self.size
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn brute_argmax(v: &[f64]) -> usize {
        let mut best = 0;
        for i in 1..v.len() {
            if v[i] > v[best] {
                best = i;
            }
        }
        best
    }

    #[test]
    fn matches_brute_force_after_random_updates() {
        let mut seed = 0x1234_5678u64;
        let mut rnd = || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            (seed >> 40) as f64 / 8_388_608.0
        };

        for len in [1usize, 2, 3, 7, 8, 9, 64, 100] {
            let mut values: Vec<f64> = (0..len).map(|_| rnd()).collect();
            let mut tree = SegTree::new(&values);
            assert_eq!(tree.len(), len);

            for _ in 0..200 {
                let i = (rnd() * len as f64) as usize % len;
                let v = rnd();
                values[i] = v;
                tree.set(i, v);

                let want = values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                assert_eq!(tree.max(), want, "len={len}");
                assert_eq!(tree.argmax(), brute_argmax(&values), "len={len}");
                assert_eq!(tree.get(i), v);
            }
        }
    }

    #[test]
    fn ties_resolve_to_the_lowest_index() {
        // Equal values are routine: an untouched frame keeps its stored energy exactly, so tie
        // handling has to match the linear scan the full-recompute path uses.
        let tree = SegTree::new(&[1.0, 5.0, 5.0, 5.0, 2.0]);
        assert_eq!(tree.argmax(), 1);

        let tree = SegTree::new(&[7.0; 8]);
        assert_eq!(tree.argmax(), 0);

        let mut tree = SegTree::new(&[1.0, 2.0, 3.0]);
        tree.set(0, 3.0);
        assert_eq!(tree.argmax(), 0, "a new leaf equal to the max must win by index");
    }

    #[test]
    fn padded_leaves_never_win() {
        // len=3 pads to 4; the padding must stay -inf so it cannot be selected.
        let tree = SegTree::new(&[-10.0, -20.0, -30.0]);
        assert_eq!(tree.argmax(), 0);
        assert_eq!(tree.max(), -10.0);
    }

    #[test]
    fn single_leaf() {
        let mut tree = SegTree::new(&[42.0]);
        assert_eq!(tree.max(), 42.0);
        assert_eq!(tree.argmax(), 0);
        tree.set(0, -1.0);
        assert_eq!(tree.max(), -1.0);
    }

    #[test]
    fn empty_tree_reports_negative_infinity() {
        let tree = SegTree::new(&[]);
        assert!(tree.is_empty());
        assert_eq!(tree.max(), f64::NEG_INFINITY);
    }
}
