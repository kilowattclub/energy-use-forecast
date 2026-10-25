//! Forecasts household demand from an online Bayesian model of the weekly
//! profile, continuous in time so any price interval length can be priced.
mod basis;
mod posterior;
mod predictor;

pub use predictor::{BayesianPredictor, BayesianState};
