//! [`Predictor`] implementation backed by durable half-hour history.
use super::history::{History, prepare_history, slot_start, Record};
use super::model;
use crate::{Predictor, Reading};
use chrono::{DateTime, Duration, Utc};
use chrono_tz::Tz;

/// Forecast used until there is history to learn from (kWh per half-hour).
const DEFAULT_KWH: f64 = 0.16;
const SLOT_HOURS: f64 = 0.5;

/// The readings seen so far in one half-hour.
struct Slot {
    start: DateTime<Utc>,
    power_sum_kw: f64,
    readings: u32,
}

impl Slot {
    fn new(start: DateTime<Utc>) -> Self {
        Self {
            start,
            power_sum_kw: 0.0,
            readings: 0,
        }
    }

    fn add(&mut self, power_kw: f64) {
        self.power_sum_kw += power_kw;
        self.readings += 1;
    }

    fn energy_kwh(&self) -> f64 {
        self.power_sum_kw / f64::from(self.readings) * SLOT_HOURS
    }
}

/// Readings are averaged into half-hour energies. A slot joins the persisted
/// [`History`] once a later reading shows it is complete, so a partial slot is
/// never learned from. The predictor's clock is the latest reading's timestamp.
pub struct HistoricPredictor {
    history: History,
    timezone: Tz,
    slot: Option<Slot>,
    latest: Option<Reading>,
    persist_error: Option<String>,
}

impl HistoricPredictor {
    pub fn new(history: History, timezone: Tz) -> Self {
        Self {
            history,
            timezone,
            slot: None,
            latest: None,
            persist_error: None,
        }
    }

    /// The most recent failure to persist a completed slot, if any. A failed slot
    /// is not learned from. Reading it clears it.
    pub fn take_persist_error(&mut self) -> Option<String> {
        self.persist_error.take()
    }

    /// The latest reading, else the end of the newest recorded slot.
    fn now(&self) -> Option<DateTime<Utc>> {
        self.latest.map(|reading| reading.at()).or_else(|| {
            let newest = self.history.observations().last()?;
            Some(newest.time + Duration::minutes(30))
        })
    }

    fn finish_slot(&mut self, slot: &Slot, now: DateTime<Utc>) {
        let observation = Record {
            time: slot.start,
            energy_kwh: slot.energy_kwh(),
        };
        if let Err(error) = self.history.record(observation, now) {
            self.persist_error = Some(error);
        }
    }
}

impl Predictor for HistoricPredictor {
    /// Energy in the half-hour containing `time`. The slot of the latest reading
    /// is that reading's power; every other slot comes from the learned history.
    fn predict_at(&self, time: DateTime<Utc>) -> f64 {
        let slot = slot_start(time);
        if let Some(reading) = self.latest.filter(|r| slot_start(r.at()) == slot) {
            return reading.power_kw() * SLOT_HOURS;
        }
        let now = self.now().unwrap_or(time);
        let history = prepare_history(self.history.observations().iter().copied(), now);
        let adjustment = model::adjustment(&history, now, self.timezone);
        model::predict(&history, slot, now, self.timezone)
            .map_or(DEFAULT_KWH, |predicted| predicted * adjustment)
    }

    /// Readings older than one already seen are ignored.
    fn accept_measurement(&mut self, reading: Reading) {
        if self.latest.is_some_and(|last| reading.at() < last.at()) {
            return;
        }
        let start = slot_start(reading.at());
        if let Some(slot) = self.slot.as_mut().filter(|slot| slot.start == start) {
            slot.add(reading.power_kw());
        } else {
            let mut next = Slot::new(start);
            next.add(reading.power_kw());
            if let Some(finished) = self.slot.replace(next) {
                self.finish_slot(&finished, reading.at());
            }
        }
        self.latest = Some(reading);
    }
}
