#![doc = include_str!("../README.md")]
//! Measured household energy, durable collection, and demand forecasting.
use crate::measurements::HALF_HOUR_SLOTS;
use crate::measurements::{live_slot, local_slot, prepare_history};
pub use crate::measurements::{Observation, Reading};
use chrono::{DateTime, Duration, Utc};
use chrono_tz::Tz;
use std::ops::Range;
mod model;

/// Predict energy in each complete half-hour; current readings replace only the current slot.
pub fn predict(
    history: impl IntoIterator<Item = Observation>,
    horizon: Range<DateTime<Utc>>,
    now: DateTime<Utc>,
    timezone: Tz,
    reading: Option<Reading>,
) -> Vec<f64> {
    let history = prepare_history(history, now);
    let adjustment = model::adjustment(&history, now, timezone);
    let mut buckets: [Vec<Observation>; HALF_HOUR_SLOTS] = std::array::from_fn(|_| Vec::new());
    for point in &history {
        buckets[local_slot(point.time, timezone)].push(*point);
    }
    let count = ((horizon.end - horizon.start).num_seconds() / 1800).max(0) as usize;
    let mut result = vec![0.16; count];
    for (index, value) in result.iter_mut().enumerate() {
        let time = horizon.start + Duration::minutes(index as i64 * 30);
        if let Some(predicted) =
            model::predict(&buckets[local_slot(time, timezone)], time, now, timezone)
        {
            *value = predicted * adjustment;
        }
    }
    if let Some(reading) = reading {
        if let Some(index) = live_slot(reading, &horizon, now) {
            result[index] = reading.power_kw * 0.5;
        }
    }
    result
}

mod history;
mod measurements;
pub use history::History;
