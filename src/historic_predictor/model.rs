//! Household demand by local clock time, with a gradual day-type preference.

use super::history::Record;
use chrono::{DateTime, Datelike, Duration, NaiveDate, Timelike, Utc, Weekday};
use chrono_tz::Tz;
use std::collections::BTreeMap;

const RECENCY_HALF_LIFE_DAYS: f64 = 7.0;
const MATCHING_DAYS_FOR_FULL_WEIGHT: f64 = 4.0;
const MIN_DAYS_FOR_OUTLIER_CAP: usize = 5;

struct Sample {
    usage: f64,
    weight: f64,
    matching_day_type: bool,
}

fn weekend(date: NaiveDate) -> bool {
    matches!(date.weekday(), Weekday::Sat | Weekday::Sun)
}

fn estimate(samples: &[&Sample]) -> Option<f64> {
    if samples.is_empty() {
        return None;
    }
    // Bound only a solitary extreme high day. Two or more high days remain
    // in the mean, so regularly recurring appliance loads retain their energy.
    let upper = if samples.len() >= MIN_DAYS_FOR_OUTLIER_CAP {
        let mut values = samples.iter().map(|s| s.usage).collect::<Vec<_>>();
        values.sort_by(f64::total_cmp);
        3.0 * values[values.len() - 2]
    } else {
        f64::INFINITY
    };
    let weight = samples.iter().map(|s| s.weight).sum::<f64>();
    Some(
        samples
            .iter()
            .map(|s| s.usage.min(upper) * s.weight)
            .sum::<f64>()
            / weight,
    )
}

fn median(values: &mut [f64]) -> f64 {
    values.sort_by(f64::total_cmp);
    let middle = values.len() / 2;
    if values.len().is_multiple_of(2) {
        (values[middle - 1] + values[middle]) / 2.0
    } else {
        values[middle]
    }
}

/// Correct a sustained household-wide level change once two completed local
/// days agree. Ratios use the current model, so its existing adaptation is not
/// counted again; isolated appliance loads cannot move the median day ratio.
pub(crate) fn adjustment(history: &[Record], now: DateTime<Utc>, timezone: Tz) -> f64 {
    let today = now.with_timezone(&timezone).date_naive();
    let recent = [today - Duration::days(1), today - Duration::days(2)];
    let mut slots: [Vec<Record>; 48] = std::array::from_fn(|_| Vec::new());
    for point in history {
        let local = point.time.with_timezone(&timezone);
        let slot = (local.hour() * 2 + local.minute() / 30) as usize;
        slots[slot].push(*point);
    }
    let mut levels = Vec::with_capacity(2);
    for date in recent {
        let mut ratios = Vec::new();
        for slot in &slots {
            let actuals = slot
                .iter()
                .filter(|p| p.time.with_timezone(&timezone).date_naive() == date)
                .collect::<Vec<_>>();
            let Some(first) = actuals.first() else {
                continue;
            };
            let matching_days = slot
                .iter()
                .map(|p| p.time.with_timezone(&timezone).date_naive())
                .filter(|day| weekend(*day) == weekend(date))
                .collect::<std::collections::BTreeSet<_>>();
            if matching_days.len() < MATCHING_DAYS_FOR_FULL_WEIGHT as usize {
                continue;
            }
            let expected = predict(slot, first.time, now, timezone).unwrap_or(0.0);
            if expected < 0.01 {
                continue;
            }
            let actual = actuals.iter().map(|p| p.energy_kwh).sum::<f64>() / actuals.len() as f64;
            ratios.push(actual / expected);
        }
        // This also permits the 46-slot spring day while rejecting incomplete
        // collection days and evidence concentrated in a few appliance slots.
        if ratios.len() < 40 {
            return 1.0;
        }
        let level = median(&mut ratios);
        let mut deviations = ratios.iter().map(|r| (r - level).abs()).collect::<Vec<_>>();
        let noise = median(&mut deviations);
        let minimum_change = (2.0 * noise / (ratios.len() as f64).sqrt()).max(0.03);
        if (level - 1.0).abs() <= minimum_change {
            return 1.0;
        }
        levels.push(level);
    }
    if (levels[0] - 1.0) * (levels[1] - 1.0) <= 0.0 {
        return 1.0;
    }
    let level = (levels[0] + levels[1]) / 2.0;
    if (levels[0] - levels[1]).abs() > level * 0.15 {
        return 1.0;
    }
    // Keep some historical influence and bound the response to a newly observed
    // regime; further completed days can confirm a larger persistent change.
    1.0 + 0.9 * (level.clamp(0.5, 2.0) - 1.0)
}

/// Predict one complete half-hour from valid, completed historical readings.
pub(crate) fn predict(
    history: &[Record],
    target: DateTime<Utc>,
    now: DateTime<Utc>,
    timezone: Tz,
) -> Option<f64> {
    let local_target = target.with_timezone(&timezone);
    let target_slot = local_target.hour() * 2 + local_target.minute() / 30;
    let target_weekend = weekend(local_target.date_naive());
    let mut days = BTreeMap::<NaiveDate, (f64, usize, DateTime<Utc>)>::new();
    for point in history {
        let local = point.time.with_timezone(&timezone);
        if local.hour() * 2 + local.minute() / 30 != target_slot {
            continue;
        }
        let day = days
            .entry(local.date_naive())
            .or_insert((0.0, 0, point.time));
        day.0 += point.energy_kwh;
        day.1 += 1;
        day.2 = day.2.max(point.time);
    }
    let samples = days
        .into_iter()
        .map(|(date, (usage, count, at))| {
            let age_days = (now - at).num_seconds().max(0) as f64 / 86_400.0;
            Sample {
                // The repeated autumn clock hour is one day's evidence.
                usage: usage / count as f64,
                weight: 0.5_f64.powf(age_days / RECENCY_HALF_LIFE_DAYS),
                matching_day_type: weekend(date) == target_weekend,
            }
        })
        .collect::<Vec<_>>();
    let all = samples.iter().collect::<Vec<_>>();
    let pooled = estimate(&all)?;
    let matching = samples
        .iter()
        .filter(|s| s.matching_day_type)
        .collect::<Vec<_>>();
    let Some(specific) = estimate(&matching) else {
        return Some(pooled);
    };
    // A new installation still learns from every available day; after four
    // matching days, weekdays and weekends have independent load profiles.
    let confidence = (matching.len() as f64 / MATCHING_DAYS_FOR_FULL_WEIGHT).min(1.0);
    Some(specific * confidence + pooled * (1.0 - confidence))
}

#[cfg(test)]
#[path = "../../tests/unit/forecast.rs"]
mod tests;
