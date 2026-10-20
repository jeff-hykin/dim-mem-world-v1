//! Every user-facing action is an endpoint registered here, and the same table is the served `openapi.json` (Desktop's
//! agent finds the endpoints there; dimos.yaml's `provides:` repeats it and a test in api.rs keeps the two equal). A
//! route can't be added without its description, so the page, the agent and openapi.json never drift apart.
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

    /// When to use the endpoint just added: one line that reads after "If " (e.g. "you need to open a recording").
    pub fn when(mut self, when: &str) -> Self {
        if let Some(last) = self.endpoints.last_mut() {
            last["when"] = json!(when);
        }
        self
    }

    /// Routed but not an action (not in openapi.json): the page's plumbing.
    pub fn plumbing<H, T>(mut self, method: &str, path: &str, handler: H) -> Self
    where
        H: Handler<T, S>,
        T: 'static,
    {
        self.router = self.router.route(&format!("/{path}"), on(filter(method), handler));
        self
    }

    /// The table as an OpenAPI 3.1 document (dimos.yaml's `provides:`, served at openapi.json): `{name}` path parts are
    /// path parameters, the other params query parameters (GET, DELETE) or a JSON body's properties (POST, PUT, PATCH).
    pub fn openapi(&self, title: &str, version: &str, when: &str, description: &str) -> Value {
        let mut paths = Map::new();
        for endpoint in &self.endpoints {
            let method = endpoint["method"].as_str().unwrap();
            let path = endpoint["path"].as_str().unwrap();
            let mut properties = endpoint["params"].as_object().cloned().unwrap_or_default();
            let mut required = Vec::new();
            // only a boolean `required` is the shorthand's flag; a list is the param's own object schema's
            for (name, spec) in properties.iter_mut() {
                if let Some(spec) = spec.as_object_mut().filter(|spec| spec.get("required").is_some_and(Value::is_boolean)) {
                    if spec.remove("required") == Some(json!(true)) {
                        required.push(name.clone());
                    }
                }
            }
            let mut parameters = Vec::new();
            let mut parameter = |name: &str, place: &str, required: bool, mut schema: Value| {
                let text = schema.as_object_mut().and_then(|schema| schema.remove("description"));
                if schema.as_object().is_none_or(|schema| schema.is_empty()) {
                    schema = json!({ "type": "string" });
                }
                let mut parameter = json!({ "name": name, "in": place, "schema": schema });
                if required {
                    parameter["required"] = json!(true);
                }
                if let Some(text) = text {
                    parameter["description"] = text;
                }
                parameters.push(parameter);
            };
            let in_path: Vec<&str> = path.split('/').filter_map(|part| part.strip_prefix('{')?.strip_suffix('}')).collect();
            for name in &in_path {
                parameter(name, "path", true, properties.remove(*name).unwrap_or(json!({})));
            }
            required.retain(|name| !in_path.contains(&name.as_str()));
            let mut operation = Map::new();
            let text = endpoint["description"].as_str().unwrap_or_default().split_whitespace().collect::<Vec<_>>().join(" ");
            if !text.is_empty() {
                let key = if text.chars().count() <= 100 { "summary" } else { "description" };
                operation.insert(key.into(), json!(text));
            }
            if endpoint["when"].as_str().is_some_and(|when| !when.is_empty()) {
                operation.insert("x-when".into(), endpoint["when"].clone());
            }
            if let Some(role) = endpoint.get("role") {
                operation.insert("x-dimos-role".into(), role.clone());
            }
            if matches!(method, "POST" | "PUT" | "PATCH") {
                if !properties.is_empty() {
                    let mut schema = json!({ "type": "object", "properties": properties });
                    if !required.is_empty() {
                        schema["required"] = json!(required);
                    }
                    operation.insert("requestBody".into(), json!({ "content": { "application/json": { "schema": schema } } }));
                }
            } else {
                for (name, schema) in properties {
                    parameter(&name, "query", required.contains(&name), schema);
                }
            }
            if !parameters.is_empty() {
                operation.insert("parameters".into(), json!(parameters));
            }
            let item = paths.entry(format!("/{path}")).or_insert_with(|| json!({}));
            item[method.to_lowercase()] = Value::Object(operation);
        }
        let mut info = json!({ "title": title, "version": version, "description": description.split_whitespace().collect::<Vec<_>>().join(" ") });
        info["x-when"] = json!(when);
        json!({ "openapi": "3.1.0", "info": info, "paths": paths })
    }
}
