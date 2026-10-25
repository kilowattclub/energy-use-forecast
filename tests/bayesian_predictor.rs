use chrono::{DateTime, Datelike, Duration, TimeZone, Timelike, Utc};
use chrono_tz::Tz;
use energy_use_forecast::bayesian_predictor::{BayesianPredictor, BayesianState};
use energy_use_forecast::default_predictor::DefaultPredictor;
use energy_use_forecast::historic_predictor::{HistoricPredictor, History};
use energy_use_forecast::{IntervalMeterReading, Predictor};
use std::f64::consts::TAU;
use uom::si::energy::kilowatt_hour;
use uom::si::f64::{Energy, Power};
use uom::si::power::kilowatt;

const LONDON: Tz = chrono_tz::Europe::London;

fn reading(at: DateTime<Utc>, power_kw: f64) -> IntervalMeterReading {
    IntervalMeterReading::new(at, Power::new::<kilowatt>(power_kw))
}

fn kwh(energy: Energy) -> f64 {
    energy.get::<kilowatt_hour>()
}

/// A Monday at local midnight, clear of daylight-saving changes.
fn start() -> DateTime<Utc> {
    Utc.with_ymd_and_hms(2026, 4, 5, 23, 0, 0).unwrap()
}

/// Smooth household power with a different shape on each weekday.
fn weekly_kw(at: DateTime<Utc>) -> f64 {
    let local = at.with_timezone(&LONDON);
    let u = f64::from(local.num_seconds_from_midnight()) / 86_400.0;
    let day = f64::from(local.weekday().num_days_from_monday());
    0.5 - 0.25 * (TAU * u).cos()
        + 0.05 * day
        + 0.15 * (2.0 * TAU * u + day).cos()
        + 0.1 * (3.0 * TAU * u - 0.5 * day).sin()
}

/// Repeatable noise in roughly [-0.2, 0.2] kW.
fn noise(i: i64) -> f64 {
    let hashed = (i as u64).wrapping_mul(6_364_136_223_846_793_005) >> 33;
    (hashed % 1000) as f64 / 2500.0 - 0.2
}

/// Readings every minute over [from, from + days).
fn readings(
    from: DateTime<Utc>,
    days: i64,
    power: impl Fn(DateTime<Utc>) -> f64,
) -> Vec<IntervalMeterReading> {
    (0..days * 24 * 60)
        .map(|i| {
            let at = from + Duration::minutes(i);
            reading(at, (power(at) + noise(i)).max(0.0))
        })
        .collect()
}

/// True energy over [from, from + 30 min), by one-minute sums.
fn true_kwh(from: DateTime<Utc>, power: &impl Fn(DateTime<Utc>) -> f64) -> f64 {
    (0..30)
        .map(|m| power(from + Duration::seconds(m * 60 + 30)) / 60.0)
        .sum()
}

/// Mean absolute half-hour error over a week from `from`.
fn week_mae(
    predict: impl Fn(DateTime<Utc>) -> f64,
    from: DateTime<Utc>,
    power: impl Fn(DateTime<Utc>) -> f64,
) -> f64 {
    let slots = 7 * 48;
    (0..slots)
        .map(|i| {
            let at = from + Duration::minutes(30 * i);
            (predict(at) - true_kwh(at, &power)).abs()
        })
        .sum::<f64>()
        / slots as f64
}

#[test]
fn untrained_forecast_follows_the_default_profile() {
    let bayesian = BayesianPredictor::new(LONDON);
    let default = DefaultPredictor::new();
    // In winter, London's clock is UTC, the default profile's clock.
    let winter_monday = Utc.with_ymd_and_hms(2026, 1, 5, 0, 0, 0).unwrap();
    let largest_difference_kw = (0..48)
        .map(|half_hour_index| {
            let at = winter_monday + Duration::minutes(30 * half_hour_index + 15);
            (bayesian.predict_at(&at) - default.predict_at(&at))
                .get::<kilowatt>()
                .abs()
        })
        .fold(0.0, f64::max);
    // The model's smooth daily curve can't follow every kink of the profile.
    assert!(
        largest_difference_kw < 0.02,
        "largest difference {largest_difference_kw} kW"
    );

    let day_energy = kwh(bayesian.predict_energy(winter_monday, winter_monday + Duration::days(1)));
    assert!(
        (day_energy - 10.32).abs() < 0.01,
        "daily energy {day_energy} kWh"
    );
    assert_eq!(
        kwh(bayesian.predict_energy(winter_monday, winter_monday)),
        0.0
    );
}

#[test]
fn learns_a_separate_profile_for_each_weekday() {
    let mut predictor = BayesianPredictor::new(LONDON);
    predictor.accept_readings(&readings(start(), 28, weekly_kw));
    let next_week = start() + Duration::days(28);
    let mae = week_mae(
        |at| kwh(predictor.predict_energy(at, at + Duration::minutes(30))),
        next_week,
        weekly_kw,
    );
    assert!(mae < 0.005, "mean half-hour error {mae} kWh");
}

#[test]
fn an_unseen_weekday_uses_the_shared_profile() {
    let same_every_day = |at: DateTime<Utc>| {
        let local = at.with_timezone(&LONDON);
        let u = f64::from(local.num_seconds_from_midnight()) / 86_400.0;
        0.5 - 0.3 * (TAU * u).cos()
    };
    let mut predictor = BayesianPredictor::new(LONDON);
    for week in 0..4 {
        // Monday to Saturday only.
        let monday = start() + Duration::weeks(week);
        predictor.accept_readings(&readings(monday, 6, same_every_day));
    }
    let sunday = start() + Duration::days(27);
    let error = (0..48)
        .map(|i| {
            let at = sunday + Duration::minutes(30 * i);
            (kwh(predictor.predict_energy(at, at + Duration::minutes(30)))
                - true_kwh(at, &same_every_day))
            .abs()
        })
        .sum::<f64>()
        / 48.0;
    assert!(error < 0.02, "mean half-hour error {error} kWh");
}

#[test]
fn adapts_to_a_change_in_household_level() {
    let mut predictor = BayesianPredictor::new(LONDON);
    predictor.accept_readings(&readings(start(), 21, weekly_kw));
    let raised = |at: DateTime<Utc>| 1.5 * weekly_kw(at);
    // Fraction of the change learned after `weeks` of raised use, on the next Monday.
    let mut learned = |weeks: i64| {
        let from = start() + Duration::weeks(2 + weeks);
        predictor.accept_readings(&readings(from, 7, raised));
        let day = from + Duration::days(7);
        let predicted = kwh(predictor.predict_energy(day, day + Duration::days(1)));
        let before = (0..48)
            .map(|i| true_kwh(day + Duration::minutes(30 * i), &weekly_kw))
            .sum::<f64>();
        (predicted - before) / (0.5 * before)
    };
    // Evidence is weighted by age, so a step change is learned steadily and
    // never overshot: about 40% after one week, over 80% after four.
    let first = learned(1);
    assert!((0.3..1.0).contains(&first), "learned {first}");
    let fourth = [learned(2), learned(3), learned(4)][2];
    assert!((0.75..1.0).contains(&fourth), "learned {fourth}");
}

#[test]
fn intervals_add_up_and_state_round_trips() {
    let mut predictor = BayesianPredictor::new(LONDON);
    predictor.accept_readings(&readings(start(), 3, weekly_kw));
    let at = start() + Duration::days(4) + Duration::hours(7);

    let grid = (0..=6)
        .map(|i| at + Duration::minutes(5 * i))
        .collect::<Vec<_>>();
    let parts = predictor.predict_interval_energy(&grid);
    assert_eq!(parts.len(), 6);
    let whole = kwh(predictor.predict_energy(at, at + Duration::minutes(30)));
    assert!((parts.iter().map(|e| kwh(*e)).sum::<f64>() - whole).abs() < 1e-9);
    assert!(kwh(predictor.energy_std_dev(at, at + Duration::minutes(30))) > 0.0);

    let saved = serde_json::to_string(predictor.state()).unwrap();
    let state: BayesianState = serde_json::from_str(&saved).unwrap();
    let mut restored = BayesianPredictor::from_state(state, LONDON).unwrap();
    // JSON may round the last bit of a float.
    assert!((kwh(restored.predict_energy(at, at + Duration::minutes(30))) - whole).abs() < 1e-12);

    // Out-of-order readings change nothing.
    let last = start() + Duration::days(3);
    restored.accept_reading(&reading(last - Duration::hours(6), 9.0));
    restored.accept_reading(&reading(last + Duration::minutes(10), 0.4));
    predictor.accept_reading(&reading(last + Duration::minutes(10), 0.4));
    let difference = kwh(restored.predict_energy(at, at + Duration::minutes(30)))
        - kwh(predictor.predict_energy(at, at + Duration::minutes(30)));
    assert!(difference.abs() < 1e-12);
}

#[test]
fn rejects_state_that_does_not_fit_the_model() {
    let mut value = serde_json::to_value(BayesianPredictor::new(LONDON).state()).unwrap();
    value["posterior"]["targets"] = serde_json::json!([0.32]);
    let state: BayesianState = serde_json::from_value(value).unwrap();
    assert!(BayesianPredictor::from_state(state, LONDON).is_err());
}

#[test]
fn beats_the_historic_predictor_when_weekdays_differ() {
    let dir = std::env::temp_dir().join(format!(
        "energy_use_forecast-bayesian-{}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&dir);
    let training = readings(start(), 21, weekly_kw);
    let mut historic = HistoricPredictor::new(
        History::open(dir.join("history.json"), start()).unwrap(),
        LONDON,
    );
    historic.accept_readings(&training);
    assert_eq!(historic.take_persist_error(), None);

    let mut bayesian = BayesianPredictor::new(LONDON);
    bayesian.accept_readings(&training);

    let next_week = start() + Duration::days(21);
    let bayesian_mae = week_mae(
        |at| kwh(bayesian.predict_energy(at, at + Duration::minutes(30))),
        next_week,
        weekly_kw,
    );
    // Skip the first slot: the historic predictor uses the live reading there.
    let historic_mae = week_mae(
        |at| kwh(historic.predict_at(&(at + Duration::minutes(1)))),
        next_week + Duration::minutes(30),
        weekly_kw,
    );
    println!("half-hour MAE: bayesian {bayesian_mae:.4}, historic {historic_mae:.4} kWh");
    assert!(bayesian_mae < historic_mae);
    std::fs::remove_dir_all(dir).unwrap();
}
