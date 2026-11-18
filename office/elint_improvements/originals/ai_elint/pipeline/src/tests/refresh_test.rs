/// Tests for this module (moved out of the inline #[cfg(test)] block).
mod tests {
    use ai_processor::refresh::*;

    #[test]
    fn toa_timestamp_converts_to_nanoseconds() {
        // 2026-07-01 is 20,635 days after the Unix epoch; 05:30:26 is
        // 19,826 s into the day.
        let v = time_of_arrival_to_ns("2026-07-01 05:30:26").unwrap();
        let expected = (20635.0 * 86400.0 + 19826.0) * 1e9;
        assert!((v - expected).abs() < 1e3, "got {v}, want {expected}");
        assert!(v > 1.0e18); // ~2026, in the 10^18 ns range
    }

    #[test]
    fn toa_epoch_is_zero() {
        let v = time_of_arrival_to_ns("1970-01-01 00:00:00").unwrap();
        assert_eq!(v, 0.0);
    }

    #[test]
    fn toa_known_date_2021() {
        // 2021-01-01 is 18,628 days after the epoch.
        let v = time_of_arrival_to_ns("2021-01-01 00:00:00").unwrap();
        let expected = (18628.0 * 86400.0) * 1e9;
        assert!((v - expected).abs() < 1e3, "got {v}, want {expected}");
    }

    #[test]
    fn toa_numeric_passthrough() {
        let v = time_of_arrival_to_ns("84992").unwrap();
        assert!((v - 84992.0).abs() < 1e-9);
    }

    #[test]
    fn toa_garbage_is_error() {
        assert!(time_of_arrival_to_ns("not-a-time").is_err());
    }

    #[test]
    fn parse_csv_missing_id_column_fails() {
        let dir = std::env::temp_dir();
        let p = dir.join("refresh_test_no_id.csv");
        std::fs::write(
            &p,
            "power_dbm,center_frequency_hz,time_of_arrival_ns\n-40.0,9e9,0\n",
        )
        .unwrap();
        assert!(parse_pdw_csv(&p).is_err());
        let _ = std::fs::remove_file(&p);
    }

    #[test]
    fn parse_csv_skips_short_lines() {
        let dir = std::env::temp_dir();
        let p = dir.join("refresh_test_short.csv");
        std::fs::write(
            &p,
            "id,power_dbm,center_frequency_hz,time_of_arrival_ns,duration_ns\n\
             1,-40.0,9e9,1000,500\n\
             2,-30.0\n\
             3,-20.0,9e9,2000,600\n",
        )
        .unwrap();
        let rows = parse_pdw_csv(&p).unwrap();
        assert_eq!(rows.len(), 2, "short line must be skipped");
        assert_eq!(rows[0].id, 1);
        assert_eq!(rows[1].id, 3);
        let _ = std::fs::remove_file(&p);
    }

    #[test]
    fn parse_csv_empty_file_fails() {
        let dir = std::env::temp_dir();
        let p = dir.join("refresh_test_empty.csv");
        std::fs::write(&p, "").unwrap();
        assert!(parse_pdw_csv(&p).is_err());
        let _ = std::fs::remove_file(&p);
    }

    #[test]
    fn parse_csv_pdwtable_layout() {
        let dir = std::env::temp_dir();
        let p = dir.join("refresh_test_pdwtable.csv");
        std::fs::write(
            &p,
            "time_of_arrival_ns,power_dbm,center_frequency_hz,bandwidth_hz,duration_ns,\
             direction_of_arrival_deg,snr_db,emitter_id,source,id\n\
             0,-71.5,9100052640.6,1044442.5,996.2,252.6,31.4,1,omni,1\n\
             84992,-84.4,9099936669.4,1127237.7,995.5,252.6,18.5,1,spinning-DF,2\n",
        )
        .unwrap();
        let rows = parse_pdw_csv(&p).unwrap();
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0].id, 1);
        assert_eq!(rows[1].id, 2);
        assert!((rows[0].power_dbm - (-71.5f32)).abs() < 0.01);
        assert!((rows[0].center_frequency_hz - 9100052640.6).abs() < 1.0);
        assert_eq!(rows[0].source, "omni");
        let _ = std::fs::remove_file(&p);
    }

    #[test]
    fn parse_csv_uuid_id_uses_row_number() {
        let dir = std::env::temp_dir();
        let p = dir.join("refresh_test_uuid.csv");
        std::fs::write(
            &p,
            "id,power_dbm,center_frequency_hz,time_of_arrival_ns,duration_ns,source,is_cw\n\
             019f1c28-3edf-7f94-b52a-20f9820648de,-43.75,9393442816,2026-07-01 05:30:26,469,OMNI,False\n\
             019f1c28-3fa8-7bdf-bd4e-b18e1d53a3d1,-42.61,9398670336,2026-07-01 05:30:26,215,OMNI,False\n",
        )
        .unwrap();
        let rows = parse_pdw_csv(&p).unwrap();
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0].id, 1); // UUID → 1-based row number
        assert_eq!(rows[1].id, 2);
        assert!(!rows[0].time_of_arrival_ns.is_empty());
        assert!(!rows[0].modulation.is_empty()); // default UNKNOWN
        assert_eq!(rows[0].is_cw, 0);
        let _ = std::fs::remove_file(&p);
    }

}
