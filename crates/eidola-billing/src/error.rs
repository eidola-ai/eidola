//! Errors contain classifications and fixed descriptions, never provider bodies.

#[derive(Debug)]
pub enum BillingError {
    Invalid(String),
    Conflict(String),
    NotFound(String),
    RequiresReview(String),
    Database(String),
    Network(String),
    Parse(String),
    ServiceUnavailable(String),
    ProviderRejected(String),
}

impl std::fmt::Display for BillingError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let (kind, detail) = match self {
            Self::Invalid(s) => ("invalid billing request", s),
            Self::Conflict(s) => ("billing conflict", s),
            Self::NotFound(s) => ("billing record not found", s),
            Self::RequiresReview(s) => ("billing review required", s),
            Self::Database(s) => ("billing database", s),
            Self::Network(s) => ("billing network", s),
            Self::Parse(s) => ("billing response", s),
            Self::ServiceUnavailable(s) => ("billing unavailable", s),
            Self::ProviderRejected(s) => ("billing provider refused", s),
        };
        write!(f, "{kind}: {detail}")
    }
}

impl std::error::Error for BillingError {}

impl From<tokio_postgres::Error> for BillingError {
    fn from(error: tokio_postgres::Error) -> Self {
        // PostgreSQL's diagnostic detail can contain account IDs and row values.
        Self::Database(
            error
                .code()
                .map_or("connection failure", |c| c.code())
                .into(),
        )
    }
}

impl From<deadpool_postgres::PoolError> for BillingError {
    fn from(_: deadpool_postgres::PoolError) -> Self {
        Self::Database("connection unavailable".into())
    }
}

pub fn parse_error_summary(error: &serde_json::Error) -> String {
    format!(
        "{:?} at line {} column {}",
        error.classify(),
        error.line(),
        error.column()
    )
}
