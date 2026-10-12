//! What dimOS Desktop tells this server at start: the `DIMOS_APP` env var, one JSON object (docs/apps.md in
//! dimos-desktop), the whole interface: Desktop passes no flags or other variables.

use std::sync::OnceLock;

#[derive(serde::Deserialize, Default, Debug, Clone)]
#[serde(rename_all = "camelCase", default)]
pub struct DimosApp {
    pub version: u64,
    pub name: Option<String>,
    pub socket: Option<String>,
    pub url: Option<String>,
    pub path: Option<String>,
    pub data_dir: Option<String>,
    pub desktop_url: Option<String>,
    pub zenoh_gateway_url: Option<String>,
    pub zenoh_connect: Option<String>,
    pub dimos_dir: Option<String>,
    pub dimos_python: Option<String>,
    pub recordings_dir: Option<String>,
    pub zenoh_namespace: Option<String>,
    pub zenoh_prefix: Option<String>,
}

pub fn parse(json: &str) -> Result<DimosApp, serde_json::Error> {
    serde_json::from_str(json)
}

/// DIMOS_APP, parsed once (logged once, so the App Store's log shows what the app was given); None when unset.
pub fn get() -> Option<&'static DimosApp> {
    static APP: OnceLock<Option<DimosApp>> = OnceLock::new();
    APP.get_or_init(|| {
        let json = std::env::var("DIMOS_APP").ok().filter(|json| !json.is_empty())?;
        eprintln!("DIMOS_APP: {json}");
        parse(&json).map_err(|error| eprintln!("DIMOS_APP isn't valid JSON ({error}); ignoring it")).ok()
    })
    .as_ref()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reads_desktops_json() {
        let app = parse(r#"{"version":2,"name":"b","socket":"/s/b.sock","url":"http://127.0.0.1:7341/apps/b/","path":"/apps/b/","dataDir":"/d/b","desktopUrl":"http://127.0.0.1:7341","zenohGatewayUrl":"/zenoh-gateway","zenohConnect":"","later":"ignored"}"#).unwrap();
        assert_eq!(app.version, 2);
        assert_eq!(app.name.as_deref(), Some("b"));
        assert_eq!(app.socket.as_deref(), Some("/s/b.sock"));
        assert_eq!(app.data_dir.as_deref(), Some("/d/b"));
        assert_eq!(app.desktop_url.as_deref(), Some("http://127.0.0.1:7341"));
        assert_eq!(app.zenoh_connect.as_deref(), Some(""));
        assert_eq!(app.zenoh_gateway_url.as_deref(), Some("/zenoh-gateway"));
    }
}
