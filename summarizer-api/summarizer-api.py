# summarize_api.py
from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import AutoTokenizer, AutoConfig, pipeline
import onnxruntime as ort
import numpy as np
import logging

app = Flask(__name__)
CORS(app, resources={r"/summarize": {"origins": "*"}})
logging.basicConfig(level=logging.DEBUG)

# Load tokenizer, model configuration, and ONNX model
MODEL_DIR = "models/onnx_flant5_small"
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True)
config = AutoConfig.from_pretrained(MODEL_DIR, local_files_only=True)

encoder_session = ort.InferenceSession(f"{MODEL_DIR}/encoder_model.onnx")
decoder_session = ort.InferenceSession(f"{MODEL_DIR}/decoder_model.onnx")


def summarize_text(text):
    input_text = "summarize: " + text.strip()
    inputs = tokenizer(input_text, return_tensors="np", truncation=True, max_length=512, padding="max_length")
    
    encoder_inputs = {
        "input_ids": inputs["input_ids"],
        "attention_mask": inputs["attention_mask"],
    }

    # Run encoder
    encoder_outputs = encoder_session.run(None, encoder_inputs)

    # Generate summary
    summary_ids = generate_summary_ids(
        encoder_outputs, encoder_inputs["attention_mask"], max_length=100, min_length=30
    )

    # Ensure proper decoding
    summary = tokenizer.batch_decode(summary_ids, skip_special_tokens=True)[0]
    return summary.strip()


def generate_summary_ids(encoder_outputs, encoder_attention_mask, max_length=100, min_length=30):
    """Greedy decoding loop using ONNX decoder session."""

    decoder_start_token_id = config.decoder_start_token_id or tokenizer.pad_token_id
    decoder_input_ids = np.array([[decoder_start_token_id]], dtype=np.int64)

    generated_tokens = []

    for step in range(max_length):
        decoder_attention_mask = np.ones(decoder_input_ids.shape, dtype=np.int64)
        decoder_inputs = {
            "input_ids": decoder_input_ids,
            "encoder_hidden_states": encoder_outputs[0],
            "encoder_attention_mask": encoder_attention_mask,
        }
        
        outputs = decoder_session.run(None, decoder_inputs)
        logits = outputs[0]
        next_token_id = np.argmax(logits[:, -1, :], axis=-1)

        # Append token to the list
        generated_tokens.append(next_token_id[0])

        # Stop generation if EOS is reached after min_length
        if step >= min_length and next_token_id[0] == tokenizer.eos_token_id:
            break

        # Update decoder input
        decoder_input_ids = np.concatenate([decoder_input_ids, next_token_id[:, None]], axis=1)

    return [generated_tokens]  # Return as list of lists for batch_decode

@app.route('/')
def index():
    return "Summarization API is running."


@app.route('/summarize', methods=['POST'])
def summarize():
    data = request.get_json()
    text = data.get("text", "").strip()
    if not text:
        return jsonify({"error": "No text provided"}), 400
    try:
        # summary = summarize_text(text)
        # use t5 in tf
        summarizer = pipeline("summarization")
        summary = summarizer(text, min_length=50, max_length=100)
        return jsonify({"summary": summary})
    except Exception as e:
        logging.error(f"Summarization error: {str(e)}")
        return jsonify({"error": "Failed to generate summary"}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=2850, debug=True)
