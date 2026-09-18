//! Per-frame spectral peaks of the accumulated map: the candidates ridge tracking links.

use super::accumulation::TfGrid;

/// Cents are counted from this frequency throughout partial tracking, whatever the grid's axis.
pub const CENTS_REF_HZ: f64 = 440.0;

pub fn cents(hz: f64) -> f64 {
    1200.0 * (hz / CENTS_REF_HZ).log2()
}

pub fn hz(cents: f64) -> f64 {
    CENTS_REF_HZ * (cents / 1200.0).exp2()
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Peak {
    pub frame: usize,
    /// Interpolated position, cents re [`CENTS_REF_HZ`].
    pub cents: f64,
    /// The peak cell and its two neighbours, summed: the energy of the ridge in this frame, not
    /// just the height of its crest.
    pub energy: f64,
}

/// Local maxima of every frame at or above `min_relative_level_db` below the map's strongest cell.
///
/// The position is refined by a parabola through the log energies of the peak and its neighbours,
/// which is exact for the Gaussian frequency kernel of a single atom.
pub fn find_peaks(grid: &TfGrid, min_relative_level_db: f64) -> Vec<Vec<Peak>> {
    let max = grid.cells.iter().copied().fold(0.0f64, f64::max);
    if max.is_nan() || max <= 0.0 {
        return vec![Vec::new(); grid.n_frames];
    }
    let floor = max * 10f64.powf(min_relative_level_db / 10.0);
    (0..grid.n_frames)
        .map(|k| {
            let row = grid.frame(k);
            let n = row.len();
            let mut peaks = Vec::new();
            for b in 0..n {
                let c = row[b];
                let l = if b > 0 { row[b - 1] } else { 0.0 };
                let r = if b + 1 < n { row[b + 1] } else { 0.0 };
                if !(c >= floor && c > l && c >= r) {
                    continue;
                }
                let mut delta = 0.0;
                if l > 0.0 && r > 0.0 {
                    let (ll, lc, lr) = (l.ln(), c.ln(), r.ln());
                    let den = ll - 2.0 * lc + lr;
                    if den < 0.0 {
                        delta = (0.5 * (ll - lr) / den).clamp(-0.5, 0.5);
                    }
                }
                let f = grid.axis.frequency(b as f64 + delta);
                peaks.push(Peak { frame: k, cents: cents(f), energy: l + c + r });
            }
            peaks
        })
        .collect()
}
