use crux_http::protocol::{HttpHeader, HttpRequest, HttpResponse, HttpResult};

pub async fn execute_http(request: HttpRequest) -> HttpResult {
    let client = reqwest::Client::new();

    let method = match request.method.as_str() {
        "GET" => reqwest::Method::GET,
        "POST" => reqwest::Method::POST,
        "PUT" => reqwest::Method::PUT,
        "DELETE" => reqwest::Method::DELETE,
        "PATCH" => reqwest::Method::PATCH,
        "HEAD" => reqwest::Method::HEAD,
        "OPTIONS" => reqwest::Method::OPTIONS,
        other => {
            return HttpResult::Err(crux_http::HttpError::Io(format!(
                "unsupported method: {other}"
            )));
        }
    };

    let mut req = client.request(method, &request.url);
    for header in &request.headers {
        req = req.header(&header.name, &header.value);
    }
    if !request.body.is_empty() {
        req = req.body(request.body);
    }

    match req.send().await {
        Ok(resp) => {
            let status = resp.status().as_u16();
            let headers: Vec<HttpHeader> = resp
                .headers()
                .iter()
                .map(|(k, v)| HttpHeader {
                    name: k.to_string(),
                    value: v.to_str().unwrap_or("").to_string(),
                })
                .collect();
            let body = resp.bytes().await.unwrap_or_default().to_vec();
            HttpResult::Ok(HttpResponse {
                status,
                headers,
                body,
            })
        }
        Err(e) => HttpResult::Err(crux_http::HttpError::Io(e.to_string())),
    }
}
