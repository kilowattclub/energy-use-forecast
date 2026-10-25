use chrono::{DateTime, Duration, TimeZone, Utc};
use energy_use_forecast::default_predictor::DefaultPredictor;
use energy_use_forecast::{IntervalMeterReading, Predictor};
use uom::si::f64::Power;
use uom::si::power::kilowatt;

fn utc_time(hour: u32, minute: u32) -> DateTime<Utc> {
    Utc.with_ymd_and_hms(2026, 9, 1, hour, minute, 0).unwrap()
}

fn predicted_kw(predictor: &DefaultPredictor, time: DateTime<Utc>) -> f64 {
    predictor.predict_at(&time).get::<kilowatt>()
}

fn assert_close(actual: f64, expected: f64) {
    assert!((actual - expected).abs() < 1e-9, "{actual} != {expected}");
}

const PROFILE: &str = r#"[
    { "time": "18:00", "kw": 0.5 },
    { "time": "06:00", "kw": 0.1 },
    { "time": "12:00", "kw": 0.3 }
]"#;

#[test]
fn predicts_listed_values_and_interpolates_between_them() {
    let predictor = DefaultPredictor::from_json(PROFILE).unwrap();
    assert_close(predicted_kw(&predictor, utc_time(6, 0)), 0.1);
    assert_close(predicted_kw(&predictor, utc_time(12, 0)), 0.3);
    assert_close(predicted_kw(&predictor, utc_time(18, 0)), 0.5);
    assert_close(predicted_kw(&predictor, utc_time(9, 0)), 0.2);
    assert_close(predicted_kw(&predictor, utc_time(16, 30)), 0.45);
}

#[test]
fn interpolation_wraps_through_midnight() {
    let predictor = DefaultPredictor::from_json(PROFILE).unwrap();
    // 18:00 (0.5) to 06:00 the next day (0.1), 12 hours apart.
    assert_close(predicted_kw(&predictor, utc_time(0, 0)), 0.3);
    assert_close(predicted_kw(&predictor, utc_time(21, 0)), 0.4);
    assert_close(predicted_kw(&predictor, utc_time(3, 0)), 0.2);
}

#[test]
fn single_entry_is_constant_and_readings_change_nothing() {
    let json = r#"[{ "time": "07:30:15", "kw": 0.2 }]"#;
    let mut predictor = DefaultPredictor::from_json(json).unwrap();
    predictor.accept_reading(&IntervalMeterReading::new(
        utc_time(7, 0),
        Power::new::<kilowatt>(9.0),
    ));
    let every_half_hour: Vec<_> = (0..48)
        .map(|half_hour_index| utc_time(0, 0) + Duration::minutes(half_hour_index * 30))
        .collect();
    for power in predictor.predict_range(&every_half_hour) {
        assert_close(power.get::<kilowatt>(), 0.2);
    }
}

#[test]
fn rejects_repeated_times_and_invalid_profiles() {
    for json in [
        r#"[{ "time": "06:00", "kw": 0.1 }, { "time": "06:00", "kw": 0.2 }]"#,
        r#"[{ "time": "06:00", "kw": 0.1 }, { "time": "06:00:00", "kw": 0.1 }]"#,
        r#"[]"#,
        r#"[{ "time": "06:00", "kw": -0.1 }]"#,
        r#"[{ "time": "25:00", "kw": 0.1 }]"#,
        r#"[{ "time": "06:00" }]"#,
        "corrupt",
    ] {
        assert!(
            DefaultPredictor::from_json(json).is_err(),
            "accepted: {json}"
        );
    }
}

#[test]
fn bundled_profile_loads_and_matches_its_file() {
    let predictor = DefaultPredictor::new();
    assert_close(predicted_kw(&predictor, utc_time(18, 0)), 0.604);
    assert_close(predicted_kw(&predictor, utc_time(18, 30)), 0.630);

    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/src/default_predictor/default_profile.json"
    );
    let opened = DefaultPredictor::open(path).unwrap();
    assert_close(
        predicted_kw(&opened, utc_time(3, 15)),
        predicted_kw(&predictor, utc_time(3, 15)),
    );
    assert!(DefaultPredictor::open("missing-profile.json").is_err());
}
