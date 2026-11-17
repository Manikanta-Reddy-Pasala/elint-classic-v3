/// Tests for this module (moved out of the inline #[cfg(test)] block).
mod tests {
    use ai_processor::inference::*;

    #[test]
    fn stub_output_size_is_n_by_out_dim() {
        let input = vec![0.5f32; 4 * MODEL_INPUT_DIM];
        let out = OnnxSession::stub_embeddings(&input, 4, MODEL_INPUT_DIM);
        assert_eq!(out.len(), 4 * MODEL_OUTPUT_DIM);
    }

    #[test]
    fn stub_is_deterministic() {
        let input = vec![0.25f32; MODEL_INPUT_DIM];
        let a = OnnxSession::stub_embeddings(&input, 1, MODEL_INPUT_DIM);
        let b = OnnxSession::stub_embeddings(&input, 1, MODEL_INPUT_DIM);
        assert_eq!(a, b);
    }

    #[test]
    fn stub_rows_differ_for_different_inputs() {
        // The stub hashes the row's bit patterns; 0.1 vs 0.2 give
        // different bit patterns, so the embeddings must differ.
        let a = OnnxSession::stub_embeddings(&[0.1f32; MODEL_INPUT_DIM], 1, MODEL_INPUT_DIM);
        let b = OnnxSession::stub_embeddings(&[0.2f32; MODEL_INPUT_DIM], 1, MODEL_INPUT_DIM);
        assert_ne!(a, b);
    }

    #[test]
    fn stub_row_is_unit_norm() {
        let input = vec![0.42f32; MODEL_INPUT_DIM];
        let out = OnnxSession::stub_embeddings(&input, 1, MODEL_INPUT_DIM);
        let norm: f32 = out.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-4, "norm={norm}");
    }

    #[test]
    fn model_dims_are_fixed() {
        assert_eq!(MODEL_INPUT_DIM, 458);
        assert_eq!(MODEL_OUTPUT_DIM, 1024);
    }

}
