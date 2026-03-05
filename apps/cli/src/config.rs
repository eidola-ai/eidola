use std::fs;
use std::path::PathBuf;

use serde::{Deserialize, Serialize};

#[derive(Debug, Default, Serialize, Deserialize)]
pub struct Config {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub base_url: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub account_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub account_secret: Option<String>,
}

impl Config {
    pub fn path() -> Option<PathBuf> {
        dirs::config_dir().map(|d| d.join("eidolons").join("config.toml"))
    }

    pub fn load() -> Config {
        let Some(path) = Self::path() else {
            return Config::default();
        };
        let Ok(contents) = fs::read_to_string(&path) else {
            return Config::default();
        };
        toml::from_str(&contents).unwrap_or_default()
    }

    pub fn save(&self) -> Result<(), String> {
        let path = Self::path().ok_or("could not determine config directory")?;
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)
                .map_err(|e| format!("failed to create config directory: {e}"))?;
        }
        let contents =
            toml::to_string_pretty(self).map_err(|e| format!("failed to serialize config: {e}"))?;
        fs::write(&path, contents).map_err(|e| format!("failed to write config: {e}"))
    }
}
