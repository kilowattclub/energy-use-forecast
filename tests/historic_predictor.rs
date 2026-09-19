use chrono::{DateTime, Duration, TimeZone, Utc};
use energy_use_forecast::historic_predictor::{HistoricPredictor, History};
use energy_use_forecast::{IntervalMeterReading, Predictor};
use uom::si::energy::kilowatt_hour;
use uom::si::f64::{Energy, Power};
use uom::si::power::kilowatt;

fn reading(at: DateTime<Utc>, power_kw: f64) -> IntervalMeterReading {
    IntervalMeterReading::new(at, Power::new::<kilowatt>(power_kw))
}

fn kwh(energy: Energy) -> f64 {
    energy.get::<kilowatt_hour>()
}

#[test]
fn learns_completed_slots_from_readings_and_forecasts_them() {
    let dir = std::env::temp_dir().join(format!(
        "energy_use_forecast-predictor-{}",
        std::process::id()
    ));
    let file = dir.join("history.json");
    let _ = std::fs::remove_dir_all(&dir);
    let start = Utc.with_ymd_and_hms(2026, 9, 1, 0, 0, 0).unwrap();
    let history = History::open(&file, start).unwrap();
    let mut predictor = HistoricPredictor::new(history, chrono_tz::UTC);

    // Three days of steady 1 kW, sampled every 10 minutes, then one more reading.
    let readings: Vec<_> = (0..=3 * 24 * 6)
        .map(|i| reading(start + Duration::minutes(i * 10), 1.0))
        .collect();
    let now = readings.last().unwrap().at();
    predictor.accept_readings(&readings);
    assert_eq!(predictor.take_persist_error(), None);

    // 1 kW for half an hour is 0.5 kWh.
    let tomorrow_noon = now + Duration::hours(12);
    assert!((kwh(predictor.predict_at(&tomorrow_noon)) - 0.5).abs() < 1e-9);
    let range = vec![tomorrow_noon, tomorrow_noon + Duration::minutes(30)];
    assert_eq!(predictor.predict_range(&range).len(), 2);

    // A live reading replaces only the current slot, and is not yet history.
    let forecast = predictor.accept_and_predict(&reading(now + Duration::minutes(1), 2.0), &range);
    assert!((kwh(forecast[0]) - 0.5).abs() < 1e-9);
    assert!((kwh(predictor.predict_at(&now)) - 1.0).abs() < 1e-9);

    // Out-of-order readings change nothing.
    predictor.accept_reading(&reading(now - Duration::hours(6), 9.0));
    assert!((kwh(predictor.predict_at(&now)) - 1.0).abs() < 1e-9);

    // Completed slots were persisted; the partial slot was not.
    let persisted = History::open(&file, now).unwrap();
    assert!(!persisted.observations().is_empty());
    assert!(persisted
        .observations()
        .iter()
        .all(|p| p.energy_kwh == 0.5 && p.time + Duration::minutes(30) <= now));
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn predicts_a_default_without_any_history() {
    let dir = std::env::temp_dir().join(format!(
        "energy_use_forecast-predictor-empty-{}",
        std::process::id()
    ));
    let now = Utc.with_ymd_and_hms(2026, 9, 1, 12, 0, 0).unwrap();
    let history = History::open(dir.join("history.json"), now).unwrap();
    let predictor = HistoricPredictor::new(history, chrono_tz::UTC);
    assert_eq!(kwh(predictor.predict_at(&now)), 0.16);
}
