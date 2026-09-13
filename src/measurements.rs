//! Shared measured-energy records and validity rules.
use chrono::{DateTime, Duration, Timelike, Utc};
use chrono_tz::Tz;
use serde::{Deserialize, Serialize};
use std::ops::Range;
pub const HISTORY_DAYS: i64 = 28;
/// Completed half-hour energies. Observations may arrive in any order.
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct Observation {
    pub time: DateTime<Utc>,
    pub energy_kwh: f64,
}

#[derive(Clone, Copy)]
pub struct Reading {
    pub at: DateTime<Utc>,
    pub power_kw: f64,
}

pub fn local_slot(time: DateTime<Utc>, timezone: Tz) -> usize {
    let local = time.with_timezone(&timezone);
    (local.hour() * 2 + local.minute() / 30) as usize
}

/// Keep the collector and direct model inputs consistent. Never learn from a
/// partial slot, an off-grid timestamp or a corrupt energy value.
pub fn valid_observation(point: &Observation, now: DateTime<Utc>, days: i64) -> bool {
    point.time >= now - Duration::days(days)
        && point.time + Duration::minutes(30) <= now
        && point.time.timestamp().rem_euclid(1800) == 0
        && point.time.timestamp_subsec_nanos() == 0
        && [point.energy_kwh]
            .iter()
            .all(|v| v.is_finite() && (0.0..=500.0).contains(v))
}
pub fn prepare_history(
    history: impl IntoIterator<Item = Observation>,
    now: DateTime<Utc>,
) -> Vec<Observation> {
    let mut history: Vec<_> = history
        .into_iter()
        .filter(|p| valid_observation(p, now, HISTORY_DAYS))
        .collect();
    history.sort_by_key(|p| p.time);
    history.dedup_by_key(|p| p.time);
    history
}

pub fn live_slot(
    reading: Reading,
    horizon: &Range<DateTime<Utc>>,
    now: DateTime<Utc>,
) -> Option<usize> {
    let valid = reading.at <= now
        && now - reading.at <= Duration::minutes(2)
        && [reading.power_kw]
            .iter()
            .all(|v| v.is_finite() && (0.0..=1000.0).contains(v));
    if valid && horizon.contains(&now) {
        let count = ((horizon.end - horizon.start).num_seconds() / 1800).max(0) as usize;
        let index = ((now - horizon.start).num_seconds() / 1800) as usize;
        if index < count {
            return Some(index);
        }
    }
    None
}

pub const HALF_HOUR_SLOTS: usize = 48;
