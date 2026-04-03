uniffi::setup_scaffolding!();

/// A simple greeting as a smoke test for the UniFFI bridge.
#[uniffi::export]
pub fn greet() -> String {
    "Hello, World!".to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_greet() {
        assert_eq!(greet(), "Hello, World!");
    }
}
