//! Durable completed half-hour measurements for forecasting.
use chrono::{DateTime, Duration, DurationRound, TimeDelta, Utc};
use std::path::{Path, PathBuf};
use serde::{Deserialize, Serialize};
use uom::si::energy::kilowatt_hour;
use uom::si::f64::Energy;

pub const HISTORY_DAYS: i64 = 28;

/// Completed half-hour energies. Observations may arrive in any order.
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct Record {
    pub time: DateTime<Utc>,
    /// Stored as kWh under its original key, so existing history files still load.
    #[serde(rename = "energy_kwh", with = "kwh")]
    pub energy: Energy,
}

/// Serializes an [`Energy`] as a plain number of kilowatt-hours.
mod kwh {
    use serde::{Deserialize, Deserializer, Serialize, Serializer};
    use uom::si::energy::kilowatt_hour;
    use uom::si::f64::Energy;

    pub fn serialize<S: Serializer>(energy: &Energy, serializer: S) -> Result<S::Ok, S::Error> {
        energy.get::<kilowatt_hour>().serialize(serializer)
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(deserializer: D) -> Result<Energy, D::Error> {
        f64::deserialize(deserializer).map(Energy::new::<kilowatt_hour>)
    }
}

pub struct History {
    path: PathBuf,
    observations: Vec<Record>,
}

impl History {
    /// Open a history file. A missing file starts empty; corrupt files return an
    /// error rather than silently overwriting the caller's recorded history.
    pub fn open(path: impl AsRef<Path>, now: DateTime<Utc>) -> Result<Self, String> {
        let observations = match std::fs::read(path.as_ref()) {
            Ok(bytes) => serde_json::from_slice(&bytes)
                .map_err(|e| format!("invalid energy history: {e}"))?,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Vec::new(),
            Err(e) => return Err(e.to_string()),
        };
        Ok(Self {
            path: path.as_ref().to_owned(),
            observations: prepare_history(observations, now),
        })
    }
    pub fn observations(&self) -> &[Record] {
        &self.observations
    }
    /// Persist a measured complete slot. Re-recording its timestamp replaces
    /// that slot. Partial, future, expired, negative and non-finite values fail.
    /// Keep one writer per file; use separate paths for separate meters.
    pub fn record(&mut self, observation: Record, now: DateTime<Utc>) -> Result<(), String> {
        if !valid_record(&observation, now, HISTORY_DAYS) {
            return Err("invalid completed half-hour observation".into());
        }
        let mut next = self.observations.clone();
        next.retain(|p| p.time != observation.time);
        next.push(observation);
        let next = prepare_history(next, now);
        let parent = self
            .path
            .parent()
            .filter(|p| !p.as_os_str().is_empty())
            .unwrap_or_else(|| Path::new("."));
        std::fs::create_dir_all(parent).map_err(|e| e.to_string())?;
        let temporary = self.path.with_extension("json.tmp");
        let bytes = serde_json::to_vec(&next).map_err(|e| e.to_string())?;
        use std::io::Write;
        let mut file = std::fs::File::create(&temporary).map_err(|e| e.to_string())?;
        file.write_all(&bytes)
            .and_then(|_| file.sync_all())
            .map_err(|e| e.to_string())?;
        std::fs::rename(&temporary, &self.path).map_err(|e| e.to_string())?;
        self.observations = next;
        Ok(())
    }
}

/// The start of the UTC half-hour containing `time`.
pub fn slot_start(time: DateTime<Utc>) -> DateTime<Utc> {
    // Truncation only fails for dates beyond chrono's nanosecond range.
    time.duration_trunc(TimeDelta::minutes(30)).unwrap_or(time)
}

/// Keep the collector and direct model inputs consistent. Never learn from a
/// partial slot, an off-grid timestamp or a corrupt energy value.
pub fn valid_record(point: &crate::historic_predictor::Record, now: DateTime<Utc>, days: i64) -> bool {
    point.time >= now - Duration::days(days)
        && point.time + Duration::minutes(30) <= now
        && point.time.timestamp().rem_euclid(1800) == 0
        && point.time.timestamp_subsec_nanos() == 0
        && point.energy.is_finite()
        && (0.0..=500.0).contains(&point.energy.get::<kilowatt_hour>())
}

pub fn prepare_history(
    history: impl IntoIterator<Item =Record>,
    now: DateTime<Utc>,
) -> Vec<Record> {
    let mut history: Vec<_> = history
        .into_iter()
        .filter(|p| valid_record(p, now, HISTORY_DAYS))
        .collect();
    history.sort_by_key(|p| p.time);
    history.dedup_by_key(|p| p.time);
    history
}