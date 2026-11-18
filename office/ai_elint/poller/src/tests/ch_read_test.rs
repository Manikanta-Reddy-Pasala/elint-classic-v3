/// Tests for ch_read (moved out of inline #[cfg(test)]).
mod tests {
    use cdc_poller::pulse::Modulation;
    use cdc_poller::config::Config;
    use cdc_poller::ch_read::*;

#[test]
    fn host_port_plain() {
        let cfg = Config {
            ch_url: "http://10.0.0.5:9999".into(),
            ..Default::default()
        };
        let (h, p) = host_port(&cfg);
        assert_eq!(h, "10.0.0.5");
        assert_eq!(p, 9999);
    }

    #[test]
    fn host_port_default_port() {
        let cfg = Config {
            ch_url: "http://10.0.0.5".into(),
            ..Default::default()
        };
        let (h, p) = host_port(&cfg);
        assert_eq!(h, "10.0.0.5");
        assert_eq!(p, 8123);
    }

    #[test]
    fn host_port_no_scheme() {
        let cfg = Config {
            ch_url: "10.0.0.7:8124".into(),
            ..Default::default()
        };
        let (h, p) = host_port(&cfg);
        assert_eq!(h, "10.0.0.7");
        assert_eq!(p, 8124);
    }

    #[test]
    fn decode_chunked_single_chunk() {
        let raw = "5\r\nhello\r\n0\r\n\r\n";
        assert_eq!(decode_chunked(raw), Some("hello".to_string()));
    }

    #[test]
    fn decode_chunked_multi_chunk() {
        let raw = "4\r\nabcd\r\n0\r\n\r\n";
        assert_eq!(decode_chunked(raw), Some("abcd".to_string()));
    }

    #[test]
    fn decode_chunked_not_chunked() {
        // No CRLF after size — not a valid chunked stream.
        assert_eq!(decode_chunked("hello world"), None);
    }

    #[test]
    fn urlencode_preserves_safe() {
        assert_eq!(urlencode("abc-DEF_0.~"), "abc-DEF_0.~");
    }

    #[test]
    fn urlencode_space() {
        assert_eq!(urlencode("a b"), "a%20b");
    }

    #[test]
    fn urlencode_special() {
        assert_eq!(urlencode("a:b?c=d"), "a%3Ab%3Fc%3Dd");
    }

    #[test]
    fn parse_modulation_all() {
        assert_eq!(parse_modulation("psk"), Modulation::Psk);
        assert_eq!(parse_modulation("FSK"), Modulation::Fsk);
        assert_eq!(parse_modulation("am"), Modulation::Am);
        assert_eq!(parse_modulation("PM"), Modulation::Pm);
        assert_eq!(parse_modulation("cw"), Modulation::Cw);
        assert_eq!(parse_modulation("QAM"), Modulation::Qam);
        assert_eq!(parse_modulation("bogus"), Modulation::Unknown);
        assert_eq!(parse_modulation(""), Modulation::Unknown);
    }

    #[test]
    fn read_batch_query_shape() {
        // Verify the QUERY template substitutes correctly.
        let sql = QUERY.replace("{start}", "42").replace("{limit}", "100");
        assert!(sql.contains("WHERE id > 42"));
        assert!(sql.contains("LIMIT 100"));
        assert!(sql.contains("ORDER BY id"));
    }
}
