/// Tests for band (moved out of inline #[cfg(test)]).
mod tests {
    use elint_engine::band::*;

#[test]
    fn routes_by_range() {
        let specs = specs_default();
        assert_eq!(for_freq(&specs, 1.5e9), Some(Band::L));
        assert_eq!(for_freq(&specs, 3.0e9), Some(Band::S));
        assert_eq!(for_freq(&specs, 10.0e9), Some(Band::X));
        assert_eq!(for_freq(&specs, 5.0e9), None); // 4-8 GHz gap
    }
}
