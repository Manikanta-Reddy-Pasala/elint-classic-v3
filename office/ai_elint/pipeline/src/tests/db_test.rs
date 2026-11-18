/// Tests for this module (moved out of the inline #[cfg(test)] block).
mod tests {
    use ai_processor::db::*;

    #[test]
    fn normalize_keeps_pg_prefix() {
        let dsn = "postgres://elint:elint@127.0.0.1:5432/sigint";
        assert_eq!(normalize(dsn), dsn);
    }

    #[test]
    fn normalize_keeps_postgresql_prefix() {
        let dsn = "postgresql://elint:elint@127.0.0.1:5432/sigint";
        assert_eq!(normalize(dsn), dsn);
    }

    #[test]
    fn normalize_adds_missing_prefix() {
        assert_eq!(
            normalize("elint:elint@127.0.0.1:5432/sigint"),
            "postgres://elint:elint@127.0.0.1:5432/sigint"
        );
    }

    #[test]
    fn normalize_trims_whitespace() {
        assert_eq!(normalize("  elint@db:5432/x  "), "postgres://elint@db:5432/x");
    }

}
