use chrono::{Duration, TimeZone, Utc};
use energy_use_forecast::{History, Observation};
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
    let point = Observation {
        time: now - Duration::minutes(30),
        energy_kwh: 0.4,
    };
    history.record(point, now).unwrap();
    history
        .record(
            Observation {
                energy_kwh: 0.8,
                ..point
            },
            now,
        )
        .unwrap();
    for invalid in [
        Observation { time: now, ..point },
        Observation {
            energy_kwh: f64::NAN,
            ..point
        },
        Observation {
            time: point.time + Duration::seconds(1),
            ..point
        },
        Observation {
            time: now - Duration::days(29),
            ..point
        },
    ] {
        assert!(history.record(invalid, now).is_err());
    }
    let restored = History::open(&file, now).unwrap();
    assert_eq!(restored.observations().len(), 1);
    assert_eq!(restored.observations()[0].energy_kwh, 0.8);
    assert!(History::open(&file, now + Duration::days(29))
        .unwrap()
        .observations()
        .is_empty());
    std::fs::write(&file, "corrupt").unwrap();
    assert!(History::open(&file, now).is_err());
    assert_eq!(std::fs::read_to_string(&file).unwrap(), "corrupt");
    std::fs::remove_dir_all(dir).unwrap();
}
