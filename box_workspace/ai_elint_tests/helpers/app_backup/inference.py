# inference.py
import os
import numpy as np
import onnxruntime as ort
from create_features import generate_features_per_window
WINDOW_SIZE = 40
# Adjust this import to match wherever 'get_features_per_window' actually lives
# from feature_utils import get_features_per_window

def init_onnx_session(onnx_path: str) -> ort.InferenceSession:
    """Validates the ONNX file path and initializes the inference session on CPU."""
    if not os.path.exists(onnx_path):
        raise FileNotFoundError(f"Aborting: Missing ONNX file at '{onnx_path}'.")

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 8
    opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

    session = ort.InferenceSession(onnx_path, sess_options=opts, providers=['CPUExecutionProvider'])
    print("ONNX session successfully bound to CPU.")
    return session

def run_onnx_inference(onnx_session: ort.InferenceSession, freq, power, raw_pw, raw_toa) -> np.ndarray:


    # Generate features directly using individual inputs
    feat = generate_features_per_window(freq, power, raw_pw, raw_toa, window_size=WINDOW_SIZE)
    X_expanded = feat[None, ...].astype(np.float32)
    input_feed = {'input_features': X_expanded}
    raw_outputs = onnx_session.run(None, input_feed)


    return raw_outputs





def run_onnx_inference__(onnx_session: ort.InferenceSession, freq, power, raw_pw, raw_toa) -> np.ndarray:


    feat = generate_features_per_window(freq, power, raw_pw, raw_toa, window_size=WINDOW_SIZE)
    X_expanded = feat[None, ...].astype(np.float32)

    input_feed = {'input_features': X_expanded}

    # === FIX 1: Extract the raw NumPy array out of the ONNX return list ===
    # Adding [0] immediately extracts the tensor array object from the list wrapper
    raw_outputs_list = onnx_session.run(None, input_feed)

    raw_outputs = raw_outputs_list[0]
    print(raw_outputs.shape)
    # === FIX 2: Squeeze out the 3D batch dimension right now ===
    # Changes shape from (1, 512, 256) down to a flat 2D (512, 256) matrix
    raw_outputs_2d = np.squeeze(raw_outputs, axis=0) if raw_outputs.ndim == 3 else raw_outputs
    print(raw_outputs_2d.shape)
    # === CRITICAL RE-INJECTION: Force L2 Vector Normalization on 2D Matrix ===
    # Calculate the vector length for each pulse along the 256-dimension axis (axis=-1)
    norm = np.linalg.norm(raw_outputs_2d, ord=2, axis=-1, keepdims=True)
    norm = np.where(norm == 0, 1e-12, norm)

    # Divide the raw matrix by the lengths to normalize them perfectly to unit length
    normalized_embeddings = raw_outputs_2d / norm
    print(normalized_embeddings.shape)
    return normalized_embeddings
