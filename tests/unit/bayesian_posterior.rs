use super::*;

fn flat_prior(n: usize, half_life_days: f64) -> Prior {
    Prior {
        mean: vec![0.0; n],
        variance: vec![1e8; n],
        half_life_days,
        shape: 2.0,
        scale: 0.1,
    }
}

/// Solve a small linear system by Gaussian elimination with partial pivoting.
fn solve(mut a: Vec<Vec<f64>>, mut b: Vec<f64>) -> Vec<f64> {
    let n = b.len();
    for col in 0..n {
        let pivot = (col..n)
            .max_by(|i, j| a[*i][col].abs().total_cmp(&a[*j][col].abs()))
            .unwrap();
        a.swap(col, pivot);
        b.swap(col, pivot);
        for row in col + 1..n {
            let factor = a[row][col] / a[col][col];
            for k in col..n {
                a[row][k] -= factor * a[col][k];
            }
            b[row] -= factor * b[col];
        }
    }
    let mut x = vec![0.0; n];
    for row in (0..n).rev() {
        let rest = (row + 1..n).map(|k| a[row][k] * x[k]).sum::<f64>();
        x[row] = (b[row] - rest) / a[row][row];
    }
    x
}

fn rows() -> Vec<(Vec<f64>, f64)> {
    (0..40)
        .map(|i| {
            let t = f64::from(i);
            let x = vec![1.0, (t * 0.7).sin(), (t * 0.3).cos()];
            let y = 0.5 + 0.2 * x[1] - 0.1 * x[2] + 0.01 * (t * 1.3).sin() + 0.002 * t;
            (x, y)
        })
        .collect()
}

/// Weighted least squares, with observation `i` weighted by `weight(i)`.
fn weighted_least_squares(rows: &[(Vec<f64>, f64)], weight: impl Fn(usize) -> f64) -> Vec<f64> {
    let mut xtx = vec![vec![0.0; 3]; 3];
    let mut xty = vec![0.0; 3];
    for (n, (x, y)) in rows.iter().enumerate() {
        let w = weight(n);
        for i in 0..3 {
            xty[i] += w * x[i] * y;
            for j in 0..3 {
                xtx[i][j] += w * x[i] * x[j];
            }
        }
    }
    solve(xtx, xty)
}

fn assert_weights(prior: &Prior, posterior: &Posterior, expected: &[f64]) {
    for (i, expected) in expected.iter().enumerate() {
        let mut unit = vec![0.0; 3];
        unit[i] = 1.0;
        let actual = posterior.predict(prior, &unit);
        assert!((actual - expected).abs() < 1e-6, "{actual} vs {expected}");
    }
}

#[test]
fn updates_match_batch_least_squares() {
    let prior = flat_prior(3, f64::INFINITY);
    let mut posterior = Posterior::empty(3);
    let rows = rows();
    for (x, y) in &rows {
        posterior.update(x, *y, 1.0);
        posterior.discount(&prior, 1.0);
    }
    assert_weights(&prior, &posterior, &weighted_least_squares(&rows, |_| 1.0));
}

#[test]
fn discounting_weights_evidence_by_age() {
    let prior = flat_prior(3, 10.0);
    let mut posterior = Posterior::empty(3);
    let rows = rows();
    for (x, y) in &rows {
        posterior.discount(&prior, 1.0);
        posterior.update(x, *y, 1.0);
    }
    let last = rows.len() - 1;
    let expected = weighted_least_squares(&rows, |n| 0.5_f64.powf((last - n) as f64 / 10.0));
    assert_weights(&prior, &posterior, &expected);
}

#[test]
fn old_evidence_fades_back_to_the_prior() {
    let mut prior = flat_prior(2, 7.0);
    prior.variance = vec![1.0, 2.0];
    prior.mean = vec![0.3, 0.0];
    let mut posterior = Posterior::empty(2);
    for _ in 0..50 {
        posterior.update(&[1.0, 1.0], 5.0, 1.0);
    }
    assert!(posterior.predict(&prior, &[1.0, 0.0]) > 1.0);
    posterior.discount(&prior, 700.0);
    assert!((posterior.predict(&prior, &[1.0, 0.0]) - 0.3).abs() < 1e-9);
    assert!((posterior.spread(&prior, &[1.0, 0.0]) - 1.0).abs() < 1e-9);
    assert!((posterior.spread(&prior, &[0.0, 1.0]) - 2.0).abs() < 1e-9);
    assert!(posterior.matches(&prior));
}

#[test]
fn estimates_the_noise_level() {
    let prior = flat_prior(1, f64::INFINITY);
    let mut posterior = Posterior::empty(1);
    // Alternating ±0.3 around 1.0 has variance 0.09.
    for i in 0..10_000 {
        let y = if i % 2 == 0 { 1.3 } else { 0.7 };
        posterior.update(&[1.0], y, 1.0);
    }
    assert!((posterior.predict(&prior, &[1.0]) - 1.0).abs() < 1e-9);
    assert!((posterior.noise_variance(&prior) - 0.09).abs() < 1e-3);
}
