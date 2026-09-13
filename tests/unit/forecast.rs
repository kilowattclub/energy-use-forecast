use super::*;
use chrono::{Duration, TimeZone};

fn point(at: DateTime<Utc>, usage: f64) -> Observation {
    Observation {
        time: at,
        energy_kwh: usage,
    }
}

fn history_for(
    target: DateTime<Utc>,
    days: i64,
    usage: impl Fn(i64, DateTime<Utc>) -> f64,
) -> Vec<Observation> {
    (1..=days)
        .map(|age| {
            let at = target - Duration::days(age);
            point(at, usage(age, at))
        })
        .collect()
}

fn yesterday_average(history: &[Observation], now: DateTime<Utc>) -> f64 {
    let yesterday = now.date_naive().pred_opt().unwrap();
    let mut energy = 0.0;
    let mut weight = 0.0;
    for sample in history.iter().filter(|p| p.time >= now - Duration::days(7)) {
        let w = if sample.time.date_naive() == yesterday {
            4.0
        } else {
            1.0
        };
        energy += sample.energy_kwh * w;
        weight += w;
    }
    energy / weight
}

#[test]
fn weekday_and_weekend_profiles_reduce_causal_next_day_error() {
    let start = Utc.with_ymd_and_hms(2026, 9, 7, 18, 0, 0).unwrap();
    let mut adaptive_error = 0.0;
    let mut previous_error = 0.0;
    for day in 0..7 {
        let target = start + Duration::days(day);
        let now = target - Duration::hours(18);
        let demand = |at: DateTime<Utc>| if weekend(at.date_naive()) { 1.0 } else { 0.2 };
        let history = history_for(target, 28, |_, at| demand(at));
        assert!(history
            .iter()
            .all(|p| p.time + Duration::minutes(30) <= now));
        adaptive_error +=
            (predict(&history, target, now, chrono_tz::UTC).unwrap() - demand(target)).abs();
        previous_error += (yesterday_average(&history, now) - demand(target)).abs();
    }
    assert!(adaptive_error < 1e-10);
    assert!(previous_error > 1.0);
}

#[test]
fn one_off_spike_has_bounded_influence_but_recurring_loads_remain() {
    let target = Utc.with_ymd_and_hms(2026, 9, 9, 18, 0, 0).unwrap();
    let now = target - Duration::hours(18);
    let isolated = history_for(target, 28, |age, _| if age == 1 { 3.5 } else { 0.2 });
    let single = predict(&isolated, target, now, chrono_tz::UTC).unwrap();
    assert!((0.2..0.3).contains(&single), "isolated forecast: {single}");
    assert!(yesterday_average(&isolated, now) > 1.0);

    let recurring = history_for(target, 28, |_, at| {
        if at.weekday() == Weekday::Tue {
            3.5
        } else {
            0.2
        }
    });
    let repeated = predict(&recurring, target, now, chrono_tz::UTC).unwrap();
    assert!(repeated > 0.7, "recurring forecast: {repeated}");
    assert!(repeated > single * 2.0);
}

#[test]
fn recent_sustained_usage_change_outweighs_older_routine() {
    let target = Utc.with_ymd_and_hms(2026, 9, 9, 18, 0, 0).unwrap();
    let now = target - Duration::hours(18);
    let history = history_for(target, 28, |age, _| if age <= 7 { 0.4 } else { 0.2 });
    let forecast = predict(&history, target, now, chrono_tz::UTC).unwrap();
    assert!(forecast > 0.30 && forecast < 0.4, "forecast: {forecast}");
}

#[test]
fn sparse_history_pools_day_types_and_missing_clock_time_returns_none() {
    let target = Utc.with_ymd_and_hms(2026, 9, 7, 18, 0, 0).unwrap();
    let now = target - Duration::hours(18);
    assert_eq!(predict(&[], target, now, chrono_tz::UTC), None);
    let history = vec![point(target - Duration::days(1), 0.8)];
    assert_eq!(predict(&history, target, now, chrono_tz::UTC), Some(0.8));
    assert_eq!(
        predict(
            &history,
            target + Duration::minutes(30),
            now,
            chrono_tz::UTC
        ),
        None
    );

    let history = vec![
        point(target - Duration::days(1), 0.8),
        point(target - Duration::days(3), 0.2),
    ];
    let forecast = predict(&history, target, now, chrono_tz::UTC).unwrap();
    assert!(forecast > 0.2 && forecast < 0.8);
}

#[test]
fn local_clock_matching_survives_dst_and_repeated_hours_count_as_one_day() {
    let timezone = chrono_tz::Europe::London;
    let target = timezone
        .with_ymd_and_hms(2026, 10, 26, 1, 0, 0)
        .unwrap()
        .with_timezone(&Utc);
    let now = target - Duration::hours(1);
    let first = Utc.with_ymd_and_hms(2026, 10, 25, 0, 0, 0).unwrap();
    let repeated = first + Duration::hours(1);
    let history = [point(first, 0.2), point(repeated, 0.6)];
    let forecast = predict(&history, target, now, timezone).unwrap();
    assert!((forecast - 0.4).abs() < 1e-10);
}

fn complete_history(
    now: DateTime<Utc>,
    usage: impl Fn(i64, i64, DateTime<Utc>) -> f64,
) -> Vec<Observation> {
    (1..=28)
        .flat_map(|age| {
            let usage = &usage;
            (0..48).map(move |slot| {
                let at = now - Duration::days(age) + Duration::minutes(slot * 30);
                point(at, usage(age, slot, at))
            })
        })
        .collect()
}

#[test]
fn two_complete_days_confirm_a_sustained_increase_or_decrease() {
    let now = Utc.with_ymd_and_hms(2026, 9, 9, 0, 0, 0).unwrap();
    for level in [0.5, 1.5] {
        let history = complete_history(now, |age, _, _| 0.2 * if age <= 2 { level } else { 1.0 });
        let baseline = predict(&history, now, now, chrono_tz::UTC).unwrap();
        let corrected = baseline * adjustment(&history, now, chrono_tz::UTC);
        let actual = 0.2 * level;
        assert!(
            (corrected - actual).abs() < 0.012,
            "corrected: {corrected}, actual: {actual}"
        );
        assert!((corrected - actual).abs() < (baseline - actual).abs() * 0.2);
    }
}

#[test]
fn level_adjustment_rejects_one_day_changes_appliance_spikes_and_missing_days() {
    let now = Utc.with_ymd_and_hms(2026, 9, 9, 0, 0, 0).unwrap();
    let one_day = complete_history(now, |age, _, _| if age == 1 { 0.4 } else { 0.2 });
    assert_eq!(adjustment(&one_day, now, chrono_tz::UTC), 1.0);
    let spikes = complete_history(
        now,
        |age, slot, _| if age <= 2 && slot < 4 { 3.5 } else { 0.2 },
    );
    assert_eq!(adjustment(&spikes, now, chrono_tz::UTC), 1.0);
    let alternating = complete_history(now, |age, _, _| if age % 2 == 0 { 0.3 } else { 0.15 });
    assert_eq!(adjustment(&alternating, now, chrono_tz::UTC), 1.0);
    let incomplete = one_day
        .into_iter()
        .filter(|p| p.time.hour() < 12)
        .collect::<Vec<_>>();
    assert_eq!(adjustment(&incomplete, now, chrono_tz::UTC), 1.0);
}

#[test]
fn level_adjustment_keeps_an_established_weekend_routine() {
    let now = Utc.with_ymd_and_hms(2026, 9, 7, 0, 0, 0).unwrap();
    let history = complete_history(now, |_, slot, at| {
        if weekend(at.date_naive()) && (18..40).contains(&slot) {
            0.5
        } else {
            0.2
        }
    });
    assert_eq!(adjustment(&history, now, chrono_tz::UTC), 1.0);
}
