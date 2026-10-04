# energy-use-forecast

Household electricity demand forecasts learned from measured half-hour usage.
No Brain configuration, inverter, private repositories, or other Kilowatt Club
packages are required.

Predictors implement the `Predictor` trait and each lives in its own module, so
they can be swapped. `historic_predictor::HistoricPredictor` forecasts from the
household's own measured history. `predict_at` returns the full half-hour energy
in kWh for the slot containing a time. The slot of the most recent `Reading`
uses that reading's power instead. The predictor's clock is the latest reading.
<<<<<<< HEAD
The model distinguishes weekday/weekend usage, local clock time and recent sustained changes while limiting the influence of isolated spikes. Sparse history falls back to a pooled profile.
=======
The model distinguishes weekday/weekend usage, local clock time and recent
sustained changes while limiting the influence of isolated spikes. Sparse history
falls back to a pooled profile.
>>>>>>> 8370141 (Rework the API...)

`History` records **completed measured energy**, not synthetic data or a forecast.
It atomically persists at most 28 days of valid complete slots, replaces duplicate
timestamps, and rejects partial/future/non-finite/negative records. Missing
files begin empty; corrupt files report an error. Use one writer per file.
`HistoricPredictor` averages the readings in each half-hour into one record and
records it once a later reading shows the slot is complete. A failed write is
available from `take_persist_error`.

## Development and release

Run `cargo test --locked`, `cargo clippy --locked --all-targets -- --deny warnings`,
and `cargo package --locked`. CI tests the library and its package independently.
Prepared for crates.io; not yet published. Consumers currently pin a Git revision.
When ready, review the package, publish with `cargo publish --locked`, and tag the
released version. MIT licensed.
