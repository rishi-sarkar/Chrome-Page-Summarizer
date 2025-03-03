# summarize_api.py
from flask import Flask, request, jsonify
from transformers import AutoTokenizer
import onnxruntime as ort
import numpy as np

app = Flask(__name__)

# Load tokenizer and ONNX sessions
tokenizer = AutoTokenizer.from_pretrained("onnx_flant5_small")
encoder_session = ort.InferenceSession("onnx_flant5_small/encoder_model.onnx")
decoder_session = ort.InferenceSession("onnx_flant5_small/decoder_model.onnx")


def summarize_text(text):
    # Tokenize the input text
    inputs = tokenizer(text, return_tensors="np", truncation=True)
    encoder_inputs = {k: v for k, v in inputs.items()}

    # Run the encoder
    encoder_outputs = encoder_session.run(None, encoder_inputs)
    # Here, you would need to set up decoder inputs and run a decoding loop.
    # This part depends on how your ONNX model is structured.
    # For illustration, let's assume we have a helper function:
    summary_ids = generate_summary_ids(encoder_outputs)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary


@app.route('/summarize', methods=['POST'])
def summarize():
    data = request.get_json()
    text = data.get("text", "")
    if not text:
        return jsonify({"error": "No text provided"}), 400
    summary = summarize_text(text)
    return jsonify({"summary": summary})


def generate_summary_ids(encoder_outputs):
    # Placeholder: Implement your decoding logic with ONNX decoder.
    # This may involve iterative calls to the decoder until an end token is produced.
    # For now, return a dummy summary token IDs.
    return [tokenizer("Summarized output", return_tensors="np")["input_ids"]]


if __name__ == '__main__':
    app.run(debug=True)
