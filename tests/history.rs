use chrono::{Duration, TimeZone, Utc};
use energy_use_forecast::historic_predictor::{History, Record};
use uom::si::energy::kilowatt_hour;
use uom::si::f64::Energy;
#[test]
fn history_survives_restart_replaces_slots_and_rejects_incomplete_data() {
    let dir = std::env::temp_dir().join(format!(
        "energy_use_forecast-history-{}",
        std::process::id()
    ));
    let file = dir.join("history.json");
    let _ = std::fs::remove_dir_all(&dir);
    let now = Utc.with_ymd_and_hms(2026, 9, 1, 12, 0, 0).unwrap();
    let mut history = History::open(&file, now).unwrap();
    let point = Record {
        time: now - Duration::minutes(30),
        energy: Energy::new::<kilowatt_hour>(0.4),
    };
    history.record(point, now).unwrap();
    history
        .record(
            Record {
                energy: Energy::new::<kilowatt_hour>(0.8),
                ..point
            },
            now,
        )
        .unwrap();
    for invalid in [
        Record { time: now, ..point },
        Record {
            energy: Energy::new::<kilowatt_hour>(f64::NAN),
            ..point
        },
        Record {
            time: point.time + Duration::seconds(1),
            ..point
        },
        Record {
            time: now - Duration::days(29),
            ..point
        },
    ] {
        assert!(history.record(invalid, now).is_err());
    }
    let restored = History::open(&file, now).unwrap();
    assert_eq!(restored.observations().len(), 1);
    assert_eq!(restored.observations()[0].energy.get::<kilowatt_hour>(), 0.8);
    assert!(History::open(&file, now + Duration::days(29))
        .unwrap()
        .observations()
        .is_empty());
    std::fs::write(&file, "corrupt").unwrap();
    assert!(History::open(&file, now).is_err());
    assert_eq!(std::fs::read_to_string(&file).unwrap(), "corrupt");
    std::fs::remove_dir_all(dir).unwrap();
}
