//! Forecasts household demand from its own measured half-hour history.
mod history;
mod model;
mod predictor;

pub use history::History;
pub use history::Record;
pub use predictor::HistoricPredictor;
