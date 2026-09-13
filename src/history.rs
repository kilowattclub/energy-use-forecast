//! Durable completed half-hour measurements for forecasting.
use crate::{
    measurements::{prepare_history, valid_observation, HISTORY_DAYS},
    Observation,
};
use chrono::{DateTime, Utc};
use std::path::{Path, PathBuf};

pub struct History {
    path: PathBuf,
    observations: Vec<Observation>,
}
impl History {
    /// Open a history file. A missing file starts empty; corrupt files return an
    /// error rather than silently overwriting the caller's recorded history.
    pub fn open(path: impl AsRef<Path>, now: DateTime<Utc>) -> Result<Self, String> {
        let observations = match std::fs::read(path.as_ref()) {
            Ok(bytes) => serde_json::from_slice(&bytes)
                .map_err(|e| format!("invalid energy history: {e}"))?,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Vec::new(),
            Err(e) => return Err(e.to_string()),
        };
        Ok(Self {
            path: path.as_ref().to_owned(),
            observations: prepare_history(observations, now),
        })
    }
    pub fn observations(&self) -> &[Observation] {
        &self.observations
    }
    /// Persist a measured complete slot (kWh). Re-recording its timestamp replaces
    /// that slot. Partial, future, expired, negative and non-finite values fail.
    /// Keep one writer per file; use separate paths for separate meters.
    pub fn record(&mut self, observation: Observation, now: DateTime<Utc>) -> Result<(), String> {
        if !valid_observation(&observation, now, HISTORY_DAYS) {
            return Err("invalid completed half-hour observation".into());
        }
        let mut next = self.observations.clone();
        next.retain(|p| p.time != observation.time);
        next.push(observation);
        let next = prepare_history(next, now);
        let parent = self
            .path
            .parent()
            .filter(|p| !p.as_os_str().is_empty())
            .unwrap_or_else(|| Path::new("."));
        std::fs::create_dir_all(parent).map_err(|e| e.to_string())?;
        let temporary = self.path.with_extension("json.tmp");
        let bytes = serde_json::to_vec(&next).map_err(|e| e.to_string())?;
        use std::io::Write;
        let mut file = std::fs::File::create(&temporary).map_err(|e| e.to_string())?;
        file.write_all(&bytes)
            .and_then(|_| file.sync_all())
            .map_err(|e| e.to_string())?;
        std::fs::rename(&temporary, &self.path).map_err(|e| e.to_string())?;
        self.observations = next;
        Ok(())
    }
}
