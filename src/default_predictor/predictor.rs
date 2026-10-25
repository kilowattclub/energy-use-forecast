use crate::{IntervalMeterReading, Predictor};
use chrono::{DateTime, NaiveTime, Timelike, Utc};
use serde::Deserialize;
use std::path::Path;
use uom::si::f64::Power;
use uom::si::power::kilowatt;

const SECONDS_PER_DAY: u32 = 86_400;
const BUNDLED_PROFILE: &str = include_str!("default_profile.json");

/// One entry exactly as written in the profile file.
#[derive(Deserialize)]
struct FileEntry {
    time: NaiveTime,
    kw: f64,
}

/// One point of the profile: the power expected at a clock time.
/// Deserialized by reading a [`FileEntry`] and converting it with `try_from`.
#[derive(Clone, Copy, Debug, Deserialize)]
#[serde(try_from = "FileEntry")]
struct Entry {
    /// Counted from midnight.
    seconds: u32,
    power: Power,
}

impl TryFrom<FileEntry> for Entry {
    type Error = String;

    fn try_from(file_entry: FileEntry) -> Result<Self, Self::Error> {
        if !(file_entry.kw.is_finite() && file_entry.kw >= 0.0) {
            return Err(format!("invalid power at {}", file_entry.time));
        }
        Ok(Self {
            seconds: file_entry.time.num_seconds_from_midnight(),
            power: Power::new::<kilowatt>(file_entry.kw),
        })
    }
}

/// Forecasts from a fixed daily profile of UTC clock times and powers. A time
/// between two entries is linearly interpolated, wrapping from the last entry
/// of the day to the first. It never learns.
#[derive(Clone, Debug)]
pub struct DefaultPredictor {
    entries: Vec<Entry>,
}

impl Default for DefaultPredictor {
    fn default() -> Self {
        Self::new()
    }
}

impl DefaultPredictor {
    pub fn new() -> Self {
        Self::from_json(BUNDLED_PROFILE).expect("Profile is valid")
    }

    /// Load a profile file. See [`DefaultPredictor::from_json`] for the format.
    pub fn open(path: impl AsRef<Path>) -> Result<Self, String> {
        let json = std::fs::read_to_string(path).map_err(|error| error.to_string())?;
        Self::from_json(&json)
    }

    /// Parse a profile: a JSON array of `{"time": "HH:MM[:SS]", "kw": n}`
    /// in any order. It needs at least one entry, no repeated time, and finite,
    /// non-negative powers.
    pub fn from_json(json: &str) -> Result<Self, String> {
        let mut entries: Vec<Entry> = serde_json::from_str(json)
            .map_err(|error| format!("invalid default profile: {error}"))?;
        if entries.is_empty() {
            return Err("Profile has no entries".into());
        }
        entries.shrink_to_fit();
        entries.sort_by_key(|entry| entry.seconds);

        if let Some(neighbours) = entries
            .windows(2)
            .find(|neighbours| neighbours[0].seconds == neighbours[1].seconds)
        {
            let repeated_seconds = neighbours[0].seconds;
            let repeated_time = NaiveTime::from_num_seconds_from_midnight_opt(repeated_seconds, 0)
                .map_or_else(
                    || format!("{repeated_seconds}s after midnight"),
                    |time| time.to_string(),
                );
            return Err(format!("default profile repeats the time {repeated_time}"));
        }
        Ok(Self { entries })
    }

    fn interpolate(&self, time: NaiveTime) -> Power {
        let entry_count: usize = self.entries.len();
        let seconds: u32 = time.num_seconds_from_midnight();

        // The first entry after `time`. The entry before it is at or before
        // `time`. Both wrap through midnight past either end of the day.
        let next_index: usize = self
            .entries
            .partition_point(|entry| entry.seconds <= seconds);
        let before: &Entry = &self.entries[(next_index + entry_count - 1) % entry_count];
        let after: &Entry = &self.entries[next_index % entry_count];

        // A u32 cannot go negative, so add a day before subtracting whenever
        // the result would wrap through midnight.
        let gap_seconds: u32 = if after.seconds > before.seconds {
            after.seconds - before.seconds
        } else {
            // Wrapping through midnight; a single entry spans the whole day.
            after.seconds + SECONDS_PER_DAY - before.seconds
        };
        let elapsed_seconds: u32 = (seconds + SECONDS_PER_DAY - before.seconds) % SECONDS_PER_DAY;
        // Convert before dividing, or integer division rounds the fraction down to 0.
        let fraction = f64::from(elapsed_seconds) / f64::from(gap_seconds);
        before.power + (after.power - before.power) * fraction
    }
}

impl Predictor<Power> for DefaultPredictor {
    /// The profile's power at the UTC clock time of `time`.
    fn predict_at(&self, time: &DateTime<Utc>) -> Power {
        self.interpolate(time.time())
    }

    /// The profile is static, so readings are ignored.
    fn accept_reading(&mut self, _: &IntervalMeterReading) {}
}
