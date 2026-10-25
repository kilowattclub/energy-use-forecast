//! Time-of-week features: a daily Fourier series shared by every day, plus a
//! smaller one for each weekday. Phase is local clock time, so profiles follow
//! the household's clock across daylight-saving changes.
use chrono::{DateTime, Datelike, Duration, NaiveTime, Offset, TimeZone, Timelike, Utc};
use chrono_tz::Tz;
use std::f64::consts::TAU;

/// Shortest shared period is 24 h / 10 = 2.4 h. Finer detail is mostly
/// appliance spikes that don't recur at the same time.
pub(crate) const SHARED_HARMONICS: usize = 10;
pub(crate) const DAY_HARMONICS: usize = 6;
pub(crate) const SHARED_LEN: usize = 1 + 2 * SHARED_HARMONICS;
pub(crate) const DAY_LEN: usize = 1 + 2 * DAY_HARMONICS;
/// Number of model weights: the shared block, then Monday to Sunday blocks.
pub(crate) const DIM: usize = SHARED_LEN + 7 * DAY_LEN;

/// Whether weight `i` belongs to the shared daily profile.
pub(crate) fn is_shared(i: usize) -> bool {
    i < SHARED_LEN
}

/// The harmonic number of weight `i`; 0 for a block's constant term.
pub(crate) fn harmonic(i: usize) -> usize {
    let j = if is_shared(i) {
        i
    } else {
        (i - SHARED_LEN) % DAY_LEN
    };
    j.div_ceil(2)
}

/// Each block is laid out as [constant, sin 1, cos 1, sin 2, cos 2, ...].
/// `terms(k)` gives the (sin, cos) entries for harmonic `k`.
fn add_block(block: &mut [f64], constant: f64, terms: impl Fn(f64) -> (f64, f64)) {
    block[0] += constant;
    for k in 1..=block.len() / 2 {
        let (sin, cos) = terms(k as f64);
        block[2 * k - 1] += sin;
        block[2 * k] += cos;
    }
}

fn blocks(x: &mut [f64], weekday: usize) -> (&mut [f64], &mut [f64]) {
    let (shared, days) = x.split_at_mut(SHARED_LEN);
    let day = &mut days[weekday * DAY_LEN..(weekday + 1) * DAY_LEN];
    (shared, day)
}

/// Fraction of the local day elapsed at `time`, in [0, 1).
fn day_fraction(time: NaiveTime) -> f64 {
    (f64::from(time.num_seconds_from_midnight()) + f64::from(time.nanosecond()) * 1e-9) / 86_400.0
}

/// Features at the instant `time`.
pub(crate) fn features(time: DateTime<Utc>, timezone: Tz) -> Vec<f64> {
    let local = time.with_timezone(&timezone);
    let u = day_fraction(local.time());
    let weekday = local.weekday().num_days_from_monday() as usize;
    let mut x = vec![0.0; DIM];
    let (shared, day) = blocks(&mut x, weekday);
    let terms = |k: f64| ((TAU * k * u).sin(), (TAU * k * u).cos());
    add_block(shared, 1.0, terms);
    add_block(day, 1.0, terms);
    x
}

/// ∫ features dt over [start, end), in hours. Dotted with weights in kW, this is
/// energy in kWh.
pub(crate) fn integral(start: DateTime<Utc>, end: DateTime<Utc>, timezone: Tz) -> Vec<f64> {
    let mut x = vec![0.0; DIM];
    let mut from = start;
    while from < end {
        let to = piece_end(from, end, timezone);
        let local = from.with_timezone(&timezone);
        let weekday = local.weekday().num_days_from_monday() as usize;
        // Within a piece, local clock time advances linearly with real time.
        let u0 = day_fraction(local.time());
        let u1 = u0 + seconds(to - from) / 86_400.0;
        let (shared, day) = blocks(&mut x, weekday);
        // dt in hours is 24 du.
        let terms = |k: f64| {
            let scale = 24.0 / (TAU * k);
            (
                scale * ((TAU * k * u0).cos() - (TAU * k * u1).cos()),
                scale * ((TAU * k * u1).sin() - (TAU * k * u0).sin()),
            )
        };
        add_block(shared, 24.0 * (u1 - u0), terms);
        add_block(day, 24.0 * (u1 - u0), terms);
        from = to;
    }
    x
}

/// Weights whose shared block best fits a daily profile, with every weekday
/// block zero. `samples_kw` must be evenly spaced from local midnight, at least
/// `SHARED_LEN` of them. Even spacing makes the Fourier terms orthogonal, so
/// each weight is a plain weighted average of the samples.
pub(crate) fn shared_profile_weights(samples_kw: &[f64]) -> Vec<f64> {
    let sample_count = samples_kw.len() as f64;
    let mut weights = vec![0.0; DIM];
    for (sample_index, sample_kw) in samples_kw.iter().enumerate() {
        let fraction_of_day = sample_index as f64 / sample_count;
        let terms = |harmonic: f64| {
            let angle = TAU * harmonic * fraction_of_day;
            (
                2.0 * sample_kw * angle.sin() / sample_count,
                2.0 * sample_kw * angle.cos() / sample_count,
            )
        };
        add_block(&mut weights[..SHARED_LEN], sample_kw / sample_count, terms);
    }
    weights
}

fn seconds(duration: Duration) -> f64 {
    duration.num_milliseconds() as f64 / 1000.0
}

fn offset_seconds(time: DateTime<Utc>, timezone: Tz) -> i32 {
    timezone
        .offset_from_utc_datetime(&time.naive_utc())
        .fix()
        .local_minus_utc()
}

/// The end of the stretch from `from` with one weekday and one UTC offset:
/// the earliest of `end`, the next local midnight, and the next offset change.
fn piece_end(from: DateTime<Utc>, end: DateTime<Utc>, timezone: Tz) -> DateTime<Utc> {
    let offset = offset_seconds(from, timezone);
    let local = from.naive_utc() + Duration::seconds(i64::from(offset));
    let midnight = (local.date() + Duration::days(1)).and_time(NaiveTime::MIN);
    let candidate = end.min(from + (midnight - local));
    if offset_seconds(candidate - Duration::nanoseconds(1), timezone) == offset {
        return candidate;
    }
    // Offsets change on whole seconds; bisect to the first second with the new one.
    let (mut same, mut changed) = (from.timestamp(), candidate.timestamp());
    while changed - same > 1 {
        let middle = same + (changed - same) / 2;
        let at = DateTime::from_timestamp(middle, 0).unwrap_or(candidate);
        if offset_seconds(at, timezone) == offset {
            same = middle;
        } else {
            changed = middle;
        }
    }
    DateTime::from_timestamp(changed, 0)
        .unwrap_or(candidate)
        .max(from + Duration::nanoseconds(1))
}

#[cfg(test)]
#[path = "../../tests/unit/bayesian_basis.rs"]
mod tests;
