/// Tests for this module (moved out of the inline #[cfg(test)] block).
mod tests {
    use ai_processor::refresh_logger::*;
    use std::fs;

    use std::path::{Path, PathBuf};

    /// Fresh temp dir for one test (unique per call).
    fn tmp_dir(tag: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!(
            "elint_logtest_{tag}_{}", std::process::id()
        ));
        let _ = fs::remove_dir_all(&d);
        fs::create_dir_all(&d).unwrap();
        d
    }

    fn read_all(d: &Path) -> String {
        let mut s = String::new();
        for entry in fs::read_dir(d).unwrap() {
            let name = entry.unwrap().file_name().to_string_lossy().into_owned();
            if name == "elint.log" {
                s.push_str(&fs::read_to_string(d.join(&name)).unwrap());
            }
        }
        s
    }

    #[test]
    fn timestamp_shape() {
        // 19 chars: "YYYY-MM-DD HH:MM:SS"
        assert_eq!(timestamp().len(), 19);
        assert_eq!(&timestamp()[4..5], "-");
        assert_eq!(&timestamp()[10..11], " ");
        assert_eq!(&timestamp()[13..14], ":");
    }

    #[test]
    fn append_writes_line_in_expected_format() {
        let d = tmp_dir("append");
        let sink = RotatingSink::new(d.to_str().unwrap()).unwrap();
        sink.log("INFO", "refresh", "hello world");
        let s = read_all(&d);
        // Level is right-aligned in a 7-char field: "|    INFO | target | msg"
        assert!(
            s.contains("INFO | refresh | hello world"),
            "log file content: {s:?}"
        );
        let _ = fs::remove_dir_all(&d);
    }

    #[test]
    fn rotate_when_file_exceeds_max_bytes() {
        let d = tmp_dir("rotate");
        let sink = RotatingSink::new(d.to_str().unwrap()).unwrap();
        // 12-byte max, 2 backups
        let mut st = ai_processor::refresh_logger::test_lock(&sink).unwrap();
        st.max_bytes = 12;
        st.backups = 2;
        drop(st);
        sink.log("INFO", "t", "aaaaaaaaaaaa"); // exceeds 12 → rotate before write
        sink.log("INFO", "t", "bbbb");
        sink.log("INFO", "t", "cccc"); // forces .1 → .2
        let mut files: Vec<String> = fs::read_dir(&d)
            .unwrap()
            .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
            .collect();
        files.sort();
        // elint.log, elint.log.1, elint.log.2 — .3 would be dropped (backups=2)
        assert!(files.contains(&"elint.log".to_string()), "files: {files:?}");
        assert!(files.contains(&"elint.log.1".to_string()), "files: {files:?}");
        assert!(files.contains(&"elint.log.2".to_string()), "files: {files:?}");
        assert!(!files.contains(&"elint.log.3".to_string()), "files: {files:?}");
        let _ = fs::remove_dir_all(&d);
    }

    #[test]
    fn init_returns_sink_when_dir_is_writable() {
        let d = tmp_dir("init");
        let sink = init(d.to_str().unwrap());
        assert!(sink.is_some());
        // A tracing subscriber may or may not already be installed (other
        // tests); either way the sink must be usable.
        if let Some(s) = sink {
            s.log("INFO", "t", "ok");
            assert!(d.join("elint.log").exists());
        }
        let _ = fs::remove_dir_all(&d);
    }

}
