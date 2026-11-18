/// Tests for this module (moved out of the inline #[cfg(test)] block).
mod tests {
    use ai_processor::ch_read::*;

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
    fn decode_chunked_two_chunks() {
        let raw = "5\r\nhello\r\n6\r\n world\r\n0\r\n";
        assert_eq!(decode_chunked(raw).as_deref(), Some("hello world"));
    }

    #[test]
    fn decode_chunked_single_chunk() {
        let raw = "3\r\nabc\r\n0\r\n";
        assert_eq!(decode_chunked(raw).as_deref(), Some("abc"));
    }

    #[test]
    fn decode_chunked_bad_size_is_none() {
        assert_eq!(decode_chunked("zz\r\nabc\r\n0\r\n"), None);
    }

    #[test]
    fn urlencode_preserves_safe_and_encodes_others() {
        assert_eq!(urlencode("ABCxyz019-_.~"), "ABCxyz019-_.~");
        assert_eq!(urlencode("a b"), "a%20b");
        assert_eq!(urlencode("a/b&c"), "a%2Fb%26c");
        assert_eq!(urlencode(""), "");
    }

    #[test]
    fn pdw_table_default() {
        std::env::remove_var("PDW_TABLE");
        assert_eq!(pdw_table(), "signals.pdw");
    }

}
