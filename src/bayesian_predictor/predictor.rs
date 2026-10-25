//! [`Predictor`] implementation backed by an online Bayesian weekly profile.
use super::basis::{self, DIM};
use super::posterior::{Posterior, Prior};
use crate::default_predictor::DefaultPredictor;
use crate::{IntervalMeterReading, Predictor};
use chrono::{DateTime, DurationRound, TimeDelta, Utc};
use chrono_tz::Tz;
use serde::{Deserialize, Serialize};
use uom::si::energy::kilowatt_hour;
use uom::si::f64::{Energy, Power};
use uom::si::power::kilowatt;

/// Readings are averaged over windows of this length before the model learns
/// from them, so a meter that reports often doesn't make the model overconfident.
const WINDOW_MINUTES: i64 = 5;
/// How often the default profile is sampled to fit the prior: every 5 minutes.
const PROFILE_SAMPLES_PER_DAY: i64 = 288;
/// Expected noise of a window's mean power, in kW².
const PRIOR_NOISE_KW2: f64 = 0.1;
/// Prior standard deviations, in kW. Harmonic k is scaled by 1/√k.
const BASE_SD_KW: f64 = 0.5;
const SHARED_SD_KW: f64 = 0.3;
const DAY_SD_KW: f64 = 0.15;
/// Age at which evidence counts half. Each weekday is only seen once a week,
/// so this spans a few of them.
const HALF_LIFE_DAYS: f64 = 14.0;

fn window() -> TimeDelta {
    TimeDelta::minutes(WINDOW_MINUTES)
}

fn hours(duration: TimeDelta) -> f64 {
    duration.num_milliseconds() as f64 / 3_600_000.0
}

fn days(duration: TimeDelta) -> f64 {
    hours(duration) / 24.0
}

/// The bundled default profile over one day, sampled evenly from midnight.
/// Its clock times are read as the household's local clock times.
fn default_profile_samples_kw() -> Vec<f64> {
    let default_predictor = DefaultPredictor::new();
    let midnight = DateTime::<Utc>::UNIX_EPOCH;
    (0..PROFILE_SAMPLES_PER_DAY)
        .map(|sample_index| {
            let seconds_after_midnight = 86_400 * sample_index / PROFILE_SAMPLES_PER_DAY;
            let sample_time = midnight + TimeDelta::seconds(seconds_after_midnight);
            default_predictor.predict_at(&sample_time).get::<kilowatt>()
        })
        .collect()
}

/// Before any data, every day follows the default profile. Evidence that
/// fades with age also returns the model towards it.
fn prior() -> Prior {
    let standard_deviation_kw = |weight_index: usize| {
        let harmonic = basis::harmonic(weight_index);
        let unscaled_kw = match (basis::is_shared(weight_index), harmonic) {
            (true, 0) => BASE_SD_KW,
            (true, _) => SHARED_SD_KW,
            (false, _) => DAY_SD_KW,
        };
        unscaled_kw / (harmonic.max(1) as f64).sqrt()
    };
    let mean = basis::shared_profile_weights(&default_profile_samples_kw());
    // A shape of 2 has mean scale / (shape - 1) = PRIOR_NOISE_KW2.
    let shape = 2.0;
    Prior {
        mean,
        variance: (0..DIM)
            .map(|weight_index| standard_deviation_kw(weight_index).powi(2) / PRIOR_NOISE_KW2)
            .collect(),
        half_life_days: HALF_LIFE_DAYS,
        shape,
        scale: PRIOR_NOISE_KW2 * (shape - 1.0),
    }
}

/// The readings seen so far in one window.
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
struct Window {
    start: DateTime<Utc>,
    power_sum_kw: f64,
    readings: u32,
}

/// Everything a [`BayesianPredictor`] has learned. Serialize it to persist the
/// model between runs; it is a fixed size and needs no history to rebuild.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BayesianState {
    posterior: Posterior,
    /// The end of the newest observation learned from.
    learned_until: Option<DateTime<Utc>>,
    window: Option<Window>,
    latest: Option<DateTime<Utc>>,
}

/// Forecasts mean power as a smooth function of local time of week.
///
/// It starts from the [`DefaultPredictor`]'s bundled profile. Each day of the
/// week has its own profile. A weekday with little evidence falls back to the
/// profile shared by all days, and older evidence gradually counts for less. Because the model is continuous, energy over any interval
/// comes straight from [`BayesianPredictor::predict_energy`], whatever the
/// price interval length.
pub struct BayesianPredictor {
    timezone: Tz,
    prior: Prior,
    state: BayesianState,
}

impl BayesianPredictor {
    /// An untrained predictor, forecasting the default profile until it has
    /// readings to learn from. Phase follows the household's local clock.
    pub fn new(timezone: Tz) -> Self {
        let prior = prior();
        let state = BayesianState {
            posterior: Posterior::empty(DIM),
            learned_until: None,
            window: None,
            latest: None,
        };
        Self {
            timezone,
            prior,
            state,
        }
    }

    /// Resume from a saved [`BayesianState`]. Fails if it doesn't fit this model.
    pub fn from_state(state: BayesianState, timezone: Tz) -> Result<Self, String> {
        let prior = prior();
        if !state.posterior.matches(&prior) {
            return Err("saved state doesn't match this model".into());
        }
        Ok(Self {
            timezone,
            prior,
            state,
        })
    }

    pub fn state(&self) -> &BayesianState {
        &self.state
    }

    /// Expected energy used over [start, end). Empty or reversed spans are zero.
    pub fn predict_energy(&self, start: DateTime<Utc>, end: DateTime<Utc>) -> Energy {
        if end <= start {
            return Energy::new::<kilowatt_hour>(0.0);
        }
        let x = basis::integral(start, end, self.timezone);
        Energy::new::<kilowatt_hour>(self.state.posterior.predict(&self.prior, &x).max(0.0))
    }

    /// Expected energy in each interval between consecutive `boundaries`, such as
    /// a grid of price periods. Returns one fewer value than there are boundaries.
    pub fn predict_interval_energy(&self, boundaries: &[DateTime<Utc>]) -> Vec<Energy> {
        boundaries
            .windows(2)
            .map(|pair| self.predict_energy(pair[0], pair[1]))
            .collect()
    }

    /// Standard deviation of the energy used over [start, end). It combines
    /// uncertainty in the profile with noise, treating windows as independent.
    pub fn energy_std_dev(&self, start: DateTime<Utc>, end: DateTime<Utc>) -> Energy {
        if end <= start {
            return Energy::new::<kilowatt_hour>(0.0);
        }
        let length = hours(end - start);
        let x = basis::integral(start, end, self.timezone)
            .into_iter()
            .map(|v| v / length)
            .collect::<Vec<_>>();
        let posterior = &self.state.posterior;
        let windows = length / hours(window());
        let power_variance = posterior.noise_variance(&self.prior)
            * (posterior.spread(&self.prior, &x) + 1.0 / windows);
        Energy::new::<kilowatt_hour>(length * power_variance.sqrt())
    }

    /// Learn that mean power over [start, end) was `mean_kw`. Longer spans count
    /// as several windows' worth of evidence.
    fn observe(&mut self, start: DateTime<Utc>, end: DateTime<Utc>, mean_kw: f64) {
        let length = hours(end - start);
        let x = basis::integral(start, end, self.timezone)
            .into_iter()
            .map(|v| v / length)
            .collect::<Vec<_>>();
        let posterior = &mut self.state.posterior;
        if let Some(until) = self.state.learned_until {
            posterior.discount(&self.prior, days(end - until));
        }
        posterior.update(&x, mean_kw, length / hours(window()));
        self.state.learned_until = self.state.learned_until.max(Some(end));
    }
}

impl Predictor<Power> for BayesianPredictor {
    /// Expected mean power at `time`.
    fn predict_at(&self, time: &DateTime<Utc>) -> Power {
        let x = basis::features(*time, self.timezone);
        Power::new::<kilowatt>(self.state.posterior.predict(&self.prior, &x).max(0.0))
    }

    /// Readings older than one already seen, negative or non-finite are ignored.
    /// A window is learned from once a later reading shows it is complete.
    fn accept_reading(&mut self, reading: &IntervalMeterReading) {
        let kw = reading.value().get::<kilowatt>();
        let at = reading.at();
        if !kw.is_finite() || kw < 0.0 || self.state.latest.is_some_and(|last| at < last) {
            return;
        }
        let start = at.duration_trunc(window()).unwrap_or(at);
        match self.state.window.as_mut().filter(|w| w.start == start) {
            Some(current) => {
                current.power_sum_kw += kw;
                current.readings += 1;
            }
            None => {
                let next = Window {
                    start,
                    power_sum_kw: kw,
                    readings: 1,
                };
                if let Some(done) = self.state.window.replace(next) {
                    let mean_kw = done.power_sum_kw / f64::from(done.readings);
                    self.observe(done.start, done.start + window(), mean_kw);
                }
            }
        }
        self.state.latest = Some(at);
    }
}
