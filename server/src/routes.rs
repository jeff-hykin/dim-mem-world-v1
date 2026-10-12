//! Every user-facing action is an endpoint registered here, and the same table is the served `agent.json` (Desktop's
//! agent finds the endpoints there; dimos.yaml's `agent:` repeats it and `deno task check-endpoints` keeps the two
//! equal). A route can't be added without its description, so the page, the agent and agent.json never drift apart.
//! Page plumbing that is not an action (the event socket, binary map data, the page's own reports) uses `plumbing`.
use axum::handler::Handler;
use axum::routing::{on, MethodFilter};
use axum::Router;
use serde_json::{json, Map, Value};

pub struct Routes<S> {
    pub router: Router<S>,
    pub endpoints: Vec<Value>,
}

fn filter(method: &str) -> MethodFilter {
    match method {
        "GET" => MethodFilter::GET,
        "POST" => MethodFilter::POST,
        "PUT" => MethodFilter::PUT,
        "PATCH" => MethodFilter::PATCH,
        "DELETE" => MethodFilter::DELETE,
        other => panic!("unsupported method {other}"),
    }
}

impl<S: Clone + Send + Sync + 'static> Routes<S> {
    pub fn new() -> Self {
        Routes { router: Router::new(), endpoints: Vec::new() }
    }

    /// An action: routed and listed. `params` is Desktop's shorthand (`{ name: { type, description, required } }`);
    /// `{name}` path parts are added to it as required strings when it doesn't describe them.
    pub fn endpoint<H, T>(mut self, method: &str, path: &str, description: &str, params: Value, handler: H) -> Self
    where
        H: Handler<T, S>,
        T: 'static,
    {
        self.router = self.router.route(&format!("/{path}"), on(filter(method), handler));
        let mut params = match params {
            Value::Object(map) => map,
            _ => Map::new(),
        };
        for part in path.split('/').filter_map(|part| part.strip_prefix('{')?.strip_suffix('}')) {
            params.entry(part.to_string()).or_insert_with(|| json!({ "type": "string", "required": true }));
        }
        let mut endpoint = json!({ "method": method, "path": path, "description": description });
        if !params.is_empty() {
            endpoint["params"] = Value::Object(params);
        }
        self.endpoints.push(endpoint);
        self
    }

    /// Marks the endpoint just added: "view" (what Desktop's screenshot() calls) or "context" (what desktop_context
    /// adds while the app is focused).
    pub fn role(mut self, role: &str) -> Self {
        if let Some(last) = self.endpoints.last_mut() {
            last["role"] = json!(role);
        }
        self
    }

    /// Routed but not an action (not in agent.json): the page's plumbing.
    pub fn plumbing<H, T>(mut self, method: &str, path: &str, handler: H) -> Self
    where
        H: Handler<T, S>,
        T: 'static,
    {
        self.router = self.router.route(&format!("/{path}"), on(filter(method), handler));
        self
    }

    pub fn manifest(&self, description: &str) -> Value {
        json!({ "description": description, "endpoints": self.endpoints })
    }
}
