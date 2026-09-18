//! From a frame-by-frame track to a few breakpoints (§16).
//!
//! The dense series is first filled across missed frames by linear interpolation, then smoothed
//! with a centred moving average, then reduced by Ramer–Douglas–Peucker. RDP here measures the
//! *vertical* deviation from the chord — cents for frequency, dB for level — because the tolerance
//! is stated in those units and a perpendicular distance would mix them with time.

/// Values for every frame `first..=last`, linearly interpolated across missing frames.
/// `points` are `(frame, value)` in increasing frame order.
pub fn fill(points: &[(usize, f64)]) -> Vec<f64> {
    let (first, last) = (points[0].0, points[points.len() - 1].0);
    let mut out = Vec::with_capacity(last - first + 1);
    for w in points.windows(2) {
        let ((ka, a), (kb, b)) = (w[0], w[1]);
        for k in ka..kb {
            out.push(a + (b - a) * (k - ka) as f64 / (kb - ka) as f64);
        }
    }
    out.push(points[points.len() - 1].1);
    out
}

/// Centred moving average of `width` frames, the window shrinking at the ends.
pub fn smooth(x: &[f64], width: usize) -> Vec<f64> {
    if width <= 1 {
        return x.to_vec();
    }
    let (before, after) = (width / 2, (width - 1) / 2);
    (0..x.len())
        .map(|i| {
            let lo = i.saturating_sub(before);
            let hi = (i + after + 1).min(x.len());
            x[lo..hi].iter().sum::<f64>() / (hi - lo) as f64
        })
        .collect()
}

/// Indices of the points RDP keeps, in order. The first and last are always kept.
pub fn rdp(y: &[f64], tolerance: f64) -> Vec<usize> {
    let n = y.len();
    if n <= 2 {
        return (0..n).collect();
    }
    let mut keep = vec![false; n];
    keep[0] = true;
    keep[n - 1] = true;
    let mut stack = vec![(0usize, n - 1)];
    while let Some((a, b)) = stack.pop() {
        let (ya, yb) = (y[a], y[b]);
        let mut worst = (0.0f64, 0usize);
        for (i, &yi) in y.iter().enumerate().take(b).skip(a + 1) {
            let chord = ya + (yb - ya) * (i - a) as f64 / (b - a) as f64;
            let d = (yi - chord).abs();
            if d > worst.0 {
                worst = (d, i);
            }
        }
        if worst.0 > tolerance {
            keep[worst.1] = true;
            stack.push((a, worst.1));
            stack.push((worst.1, b));
        }
    }
    (0..n).filter(|&i| keep[i]).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn filling_interpolates_across_gaps() {
        assert_eq!(fill(&[(3, 0.0), (4, 1.0), (7, 4.0)]), vec![0.0, 1.0, 2.0, 3.0, 4.0]);
        assert_eq!(fill(&[(5, 2.0)]), vec![2.0]);
    }

    #[test]
    fn smoothing_is_centred_and_keeps_a_ramp() {
        let ramp: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let s = smooth(&ramp, 3);
        assert_eq!(&s[1..9], &ramp[1..9], "a centred average leaves a line alone inside");
        assert_eq!(smooth(&ramp, 1), ramp);
    }

    #[test]
    fn rdp_reduces_a_line_to_its_ends_and_keeps_a_corner() {
        let line: Vec<f64> = (0..50).map(|i| 3.0 * i as f64).collect();
        assert_eq!(rdp(&line, 0.1), vec![0, 49]);
        let corner: Vec<f64> = (0..50).map(|i| if i < 20 { 0.0 } else { (i - 20) as f64 }).collect();
        assert_eq!(rdp(&corner, 0.1), vec![0, 20, 49]);
    }

    #[test]
    fn nothing_rdp_drops_is_further_from_the_result_than_the_tolerance() {
        let y: Vec<f64> = (0..300).map(|i| 30.0 * (i as f64 * 0.07).sin() + 0.01 * i as f64).collect();
        for tol in [0.5, 2.0, 5.0] {
            let kept = rdp(&y, tol);
            assert!(kept.len() < y.len() / 3, "tol {tol}: kept {}", kept.len());
            for w in kept.windows(2) {
                let (a, b) = (w[0], w[1]);
                for i in a..=b {
                    let chord = y[a] + (y[b] - y[a]) * (i - a) as f64 / (b - a) as f64;
                    assert!((y[i] - chord).abs() <= tol + 1e-12);
                }
            }
        }
    }
}
