#![doc = include_str!("../README.md")]

use chrono::{DateTime, Utc};

pub mod historic_predictor;

/// A measured household power draw. Only valid readings can be constructed.
#[derive(Clone, Copy, Debug)]
pub struct Reading {
    at: DateTime<Utc>,
    power_kw: f64,
}

impl Reading {
    /// `None` unless `power_kw` is finite and between 0 and 1000 kW.
    pub fn new(at: DateTime<Utc>, power_kw: f64) -> Option<Self> {
        (power_kw.is_finite() && (0.0..=1000.0).contains(&power_kw))
            .then_some(Self { at, power_kw })
    }

    pub fn at(&self) -> DateTime<Utc> {
        self.at
    }

    pub fn power_kw(&self) -> f64 {
        self.power_kw
    }
}

/// Defines a generic household energy-use predictor that can learn from live power readings
pub trait Predictor {
    fn predict_at(&self, time: DateTime<Utc>) -> f64;
    fn predict_range(&self, range: Vec<DateTime<Utc>>) -> Vec<f64> {
        range
            .into_iter()
            .map(|time| self.predict_at(time))
            .collect()
    }

    fn accept_measurement(&mut self, reading: Reading);
    fn accept_measurements(&mut self, readings: Vec<Reading>) {
        for reading in readings {
            self.accept_measurement(reading);
        }
    }

    fn accept_and_predict(&mut self, reading: Reading, range: Vec<DateTime<Utc>>) -> Vec<f64> {
        self.accept_measurement(reading);
        self.predict_range(range)
    }
}
