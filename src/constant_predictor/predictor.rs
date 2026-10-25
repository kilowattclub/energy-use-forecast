use crate::{IntervalMeterReading, Predictor};
use chrono::{DateTime, Utc};

pub struct ConstantPredictor<T: Copy> {
    value: T,
}

impl<T: Copy> ConstantPredictor<T> {
    pub fn new(value: T) -> Self {
        Self { value }
    }
}

impl<T: Copy> Predictor<T> for ConstantPredictor<T> {
    fn predict_at(&self, _: &DateTime<Utc>) -> T {
        self.value
    }

    fn accept_reading(&mut self, _: &IntervalMeterReading) {}
}
