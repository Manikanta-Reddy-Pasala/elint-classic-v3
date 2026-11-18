/// Tests for ch_read (moved out of inline #[cfg(test)]).
mod tests {
    use elint_engine::ch_read::*;

#[test]
    fn host_port_plain_url() {
        let (h, p) = host_port("http://127.0.0.1:8123");
        assert_eq!(h, "127.0.0.1");
        assert_eq!(p, 8123);
    }

    #[test]
    fn host_port_strips_scheme_and_trailing_slash() {
        let (h, p) = host_port("https://ch.internal/");
        assert_eq!(h, "ch.internal");
        assert_eq!(p, 8123); // no port in URL → default
    }

    #[test]
    fn host_port_custom_port() {
        let (h, p) = host_port("http://ch.lab:9999/");
        assert_eq!(h, "ch.lab");
        assert_eq!(p, 9999);
    }

    #[test]
    fn host_port_bad_port_falls_back_to_default() {
        let (h, p) = host_port("http://ch.lab:oops");
        assert_eq!(h, "ch.lab");
        assert_eq!(p, 8123);
    }

    #[test]
    fn pulse_emitter_id() {
        let p = Pulse {
            id: 42,
            power_dbm: 0.0,
            center_frequency_hz: 0.0,
            time_of_arrival_ns: 0,
            duration_ns: 0.0,
            bandwidth_hz: 0.0,
            direction_of_arrival_deg: 0.0,
            snr_db: 0.0,
            modulation: String::new(),
            is_cw: false,
            receiver_id: 1,
        };
        assert_eq!(p.emitter_id(), "em-42");
    }

    #[test]
    fn body_contains_crlf_finds_separator() {
        assert!(!body_contains_crlf(b"abc"));
        assert!(!body_contains_crlf(b"\r"));
        assert!(body_contains_crlf(b"HTTP/1.1 200 OK\r\n"));
    }

    #[test]
    fn find_crlf_position() {
        assert_eq!(find_crlf(b"ab\r\ncd"), Some(2));
        assert_eq!(find_crlf(b"abc"), None);
    }

    #[test]
    fn find_header_end_position() {
        let hdr = b"HTTP/1.1 200 OK\r\nContent-Length: 5\r\n\r\nhello";
        let pos = find_header_end(hdr).unwrap();
        assert_eq!(&hdr[pos..], b"hello");
        assert_eq!(find_header_end(b"incomplete"), None);
    }

    #[test]
    fn urlencode_preserves_safe_and_encodes_others() {
        assert_eq!(urlencode("ABCxyz019-_.~"), "ABCxyz019-_.~");
        assert_eq!(urlencode("a b"), "a%20b");
        assert_eq!(urlencode("a/b&c"), "a%2Fb%26c");
        assert_eq!(urlencode(""), "");
    }

    #[test]
    fn fetch_limit_default_is_20000() {
        // Tests may run with a polluted env; the default only applies
        // when CH_ROWS is unset. Re-set it explicitly for the check.
        std::env::remove_var("CH_ROWS");
        assert_eq!(fetch_limit(), 20_000);
    }
}
