#![doc = include_str!("../README.md")]

use chrono::{DateTime, Utc};
use uom::si::f64::Power;

pub mod historic_predictor;

/// A measured household power draw. Only valid readings can be constructed.
#[derive(Clone, Copy, Debug)]
pub struct IntervalMeterReading {
    at: DateTime<Utc>,
    value: Power,
}

impl IntervalMeterReading {
    pub fn new(at: DateTime<Utc>, value: Power) -> Self {
        Self { at, value }
    }

    pub fn at(&self) -> DateTime<Utc> {
        self.at
    }

    pub fn value(&self) -> Power {
        self.value
    }
}

/// Defines a generic household energy-use predictor that can learn from live power readings
pub trait Predictor<T> {
    fn predict_at(&self, time: &DateTime<Utc>) -> T;
    fn predict_range(&self, range: &[DateTime<Utc>]) -> Vec<T> {
        range.iter().map(|time| self.predict_at(time)).collect()
    }

    fn accept_reading(&mut self, reading: &IntervalMeterReading);
    fn accept_readings(&mut self, readings: &[IntervalMeterReading]) {
        for reading in readings {
            self.accept_reading(reading);
        }
    }

    fn accept_and_predict(
        &mut self,
        reading: &IntervalMeterReading,
        range: &[DateTime<Utc>],
    ) -> Vec<T> {
        self.accept_reading(reading);
        self.predict_range(range)
    }
}
