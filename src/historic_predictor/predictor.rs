//! [`Predictor`] implementation backed by durable half-hour history.
use super::history::{prepare_history, slot_start, History, Record};
use super::model;
use crate::{IntervalMeterReading, Predictor};
use chrono::{DateTime, Duration, Utc};
use chrono_tz::Tz;
use uom::si::energy::kilowatt_hour;
use uom::si::f64::{Energy, Power, Time};
use uom::si::power::kilowatt;
use uom::si::time::hour;

/// Forecast used until there is history to learn from (kWh per half-hour).
const DEFAULT_KWH: f64 = 0.16;

/// Length of one settlement period. uom constructors aren't `const`.
fn period_length() -> Time {
    Time::new::<hour>(0.5)
}

/// The readings seen so far in one half-hour.
struct SettlementPeriod {
    start: DateTime<Utc>,
    cumulative_power: Power,
    readings: u32,
}

impl SettlementPeriod {
    fn new(start: DateTime<Utc>) -> Self {
        Self {
            start,
            cumulative_power: Power::new::<kilowatt>(0.0),
            readings: 0,
        }
    }

    fn add(&mut self, power: Power) {
        self.cumulative_power += power;
        self.readings += 1;
    }

    fn energy(&self) -> Energy {
        self.cumulative_power / f64::from(self.readings) * period_length()
    }
}

/// Readings are averaged into half-hour energies. A slot joins the persisted
/// [`History`] once a later reading shows it is complete, so a partial slot is
/// never learned from. The predictor's clock is the latest reading's timestamp.
pub struct HistoricPredictor {
    history: History,
    timezone: Tz,
    slot: Option<SettlementPeriod>,
    latest: Option<IntervalMeterReading>,
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

    fn finish_slot(&mut self, slot: &SettlementPeriod, now: DateTime<Utc>) {
        let observation = Record {
            time: slot.start,
            energy: slot.energy(),
        };
        if let Err(error) = self.history.record(observation, now) {
            self.persist_error = Some(error);
        }
    }
}

impl Predictor<Energy> for HistoricPredictor {
    /// Energy in the half-hour containing `time`. The slot of the latest reading
    /// is that reading's power; every other slot comes from the learned history.
    fn predict_at(&self, time: &DateTime<Utc>) -> Energy {
        let slot = slot_start(*time);
        if let Some(reading) = self.latest.filter(|r| slot_start(r.at()) == slot) {
            return reading.value() * period_length();
        }
        let now = self.now().unwrap_or(*time);
        let history = prepare_history(self.history.observations().iter().copied(), now);
        let adjustment = model::adjustment(&history, now, self.timezone);
        let kwh = model::predict(&history, slot, now, self.timezone)
            .map_or(DEFAULT_KWH, |predicted| predicted * adjustment);
        Energy::new::<kilowatt_hour>(kwh)
    }

    /// Readings older than one already seen are ignored.
    fn accept_reading(&mut self, reading: &IntervalMeterReading) {
        let reading = *reading;
        if self.latest.is_some_and(|last| reading.at() < last.at()) {
            return;
        }
        let start = slot_start(reading.at());
        if let Some(slot) = self.slot.as_mut().filter(|slot| slot.start == start) {
            slot.add(reading.value());
        } else {
            let mut next = SettlementPeriod::new(start);
            next.add(reading.value());
            if let Some(finished) = self.slot.replace(next) {
                self.finish_slot(&finished, reading.at());
            }
        }
        self.latest = Some(reading);
    }
}
