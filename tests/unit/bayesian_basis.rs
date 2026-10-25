use super::*;
use chrono::TimeZone;

/// Midpoint-rule integral of the features, in hours.
fn numerical(start: DateTime<Utc>, end: DateTime<Utc>, timezone: Tz) -> Vec<f64> {
    let step = Duration::seconds(10);
    let mut total = vec![0.0; DIM];
    let mut t = start;
    while t < end {
        let x = features(t + step / 2, timezone);
        for (sum, v) in total.iter_mut().zip(x) {
            *sum += v * 10.0 / 3600.0;
        }
        t += step;
    }
    total
}

fn assert_close(analytic: &[f64], numerical: &[f64]) {
    for (i, (a, n)) in analytic.iter().zip(numerical).enumerate() {
        assert!((a - n).abs() < 1e-4, "weight {i}: {a} vs {n}");
    }
}

#[test]
fn features_select_shared_and_one_weekday_block() {
    // 2026-10-19 is a Monday.
    let monday_noon = Utc.with_ymd_and_hms(2026, 10, 19, 12, 0, 0).unwrap();
    let x = features(monday_noon, chrono_tz::UTC);
    assert_eq!(x.len(), DIM);
    assert_eq!(x[0], 1.0);
    assert_eq!(x[SHARED_LEN], 1.0);
    assert!(x[SHARED_LEN + DAY_LEN..].iter().all(|v| *v == 0.0));
    // cos(2π · ½) = −1 at noon for the first harmonic.
    assert!((x[2] + 1.0).abs() < 1e-12);
}

#[test]
fn harmonic_numbers_follow_the_layout() {
    assert_eq!(harmonic(0), 0);
    assert_eq!(harmonic(1), 1);
    assert_eq!(harmonic(2), 1);
    assert_eq!(harmonic(SHARED_LEN - 1), SHARED_HARMONICS);
    assert_eq!(harmonic(SHARED_LEN), 0);
    assert_eq!(harmonic(SHARED_LEN + DAY_LEN - 1), DAY_HARMONICS);
    assert!(is_shared(SHARED_LEN - 1) && !is_shared(SHARED_LEN));
}

#[test]
fn integral_matches_numerical_across_midnight() {
    let start = Utc.with_ymd_and_hms(2026, 10, 19, 21, 7, 0).unwrap();
    let end = start + Duration::hours(5);
    let analytic = integral(start, end, chrono_tz::Europe::London);
    assert_close(&analytic, &numerical(start, end, chrono_tz::Europe::London));
    assert!((analytic[0] - 5.0).abs() < 1e-9);
}

#[test]
fn integral_matches_numerical_across_daylight_saving_changes() {
    let london = chrono_tz::Europe::London;
    // Clocks go back at 01:00 UTC on 25 October 2026 and forward on 29 March.
    for (start, length) in [
        (Utc.with_ymd_and_hms(2026, 10, 24, 22, 0, 0).unwrap(), 6),
        (Utc.with_ymd_and_hms(2026, 3, 28, 22, 0, 0).unwrap(), 6),
    ] {
        let end = start + Duration::hours(length);
        assert_close(
            &integral(start, end, london),
            &numerical(start, end, london),
        );
    }
    // The long autumn day still lasts 25 hours.
    let midnight = Utc.with_ymd_and_hms(2026, 10, 24, 23, 0, 0).unwrap();
    let day = integral(midnight, midnight + Duration::hours(25), london);
    assert!((day[0] - 25.0).abs() < 1e-9);
}

#[test]
fn shared_profile_weights_recover_a_daily_fourier_series() {
    let sample_count = 288;
    let samples_kw: Vec<f64> = (0..sample_count)
        .map(|sample_index| {
            let angle = TAU * sample_index as f64 / sample_count as f64;
            0.4 + 0.1 * angle.sin() - 0.2 * (3.0 * angle).cos()
        })
        .collect();
    let weights = shared_profile_weights(&samples_kw);
    let mut expected = vec![0.0; DIM];
    expected[0] = 0.4;
    // Layout: [constant, sin 1, cos 1, sin 2, cos 2, sin 3, cos 3, ...].
    expected[1] = 0.1;
    expected[6] = -0.2;
    for (weight_index, (weight, expected_weight)) in weights.iter().zip(&expected).enumerate() {
        assert!(
            (weight - expected_weight).abs() < 1e-12,
            "weight {weight_index}: {weight} vs {expected_weight}"
        );
    }
}
