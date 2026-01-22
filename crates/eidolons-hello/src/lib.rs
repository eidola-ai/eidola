/// Returns a greeting for the given name.
pub fn hello(name: &str) -> String {
    format!("Hello, {}!", name)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hello_returns_greeting() {
        assert_eq!(hello("World"), "Hello, World!");
    }

    #[test]
    fn hello_handles_empty_name() {
        assert_eq!(hello(""), "Hello, !");
    }
}
