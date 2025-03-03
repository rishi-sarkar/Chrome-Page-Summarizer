# summarize_api.py
from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin
from transformers import AutoTokenizer
import onnxruntime as ort
import numpy as np
import logging

app = Flask(__name__)
# This enables CORS for all routes by default
CORS(app, resources={r"/summarize": {"origins": "*"}})
logging.basicConfig(level=logging.DEBUG)

# Load tokenizer and ONNX sessions
tokenizer = AutoTokenizer.from_pretrained(
    "models/onnx_flant5_small", local_files_only=True)
encoder_session = ort.InferenceSession(
    "models/onnx_flant5_small/encoder_model.onnx")
decoder_session = ort.InferenceSession(
    "models/onnx_flant5_small/decoder_model.onnx")


def summarize_text(text):
    # Prepend the task instruction to guide the model
    input_text = "summarize: " + text.strip()
    # Tokenize input text; ensure both input_ids and attention_mask are returned
    inputs = tokenizer(input_text, return_tensors="np", truncation=True)
    encoder_inputs = {k: v for k, v in inputs.items()}

    # Run the encoder model: using 'input_ids' and 'attention_mask'
    encoder_outputs = encoder_session.run(None, encoder_inputs)
    # encoder_outputs[0] corresponds to 'last_hidden_state'

    # Generate summary token ids using greedy decoding (increased max_length)
    summary_ids = generate_summary_ids(
        encoder_outputs, encoder_inputs["attention_mask"], max_length=150
    )
    # Decode token ids to a string
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary


@app.route('/')
@cross_origin()
def index():
    logging.debug("Welcome to the API!")
    return "Welcome to the API!"


@app.route('/summarize', methods=['POST'])
@cross_origin()
def summarize():
    data = request.get_json()
    text = data.get("text", "")
    if not text:
        return jsonify({"error": "No text provided"}), 400
    summary = summarize_text(text)
    logging.debug("Sending summary response")
    return jsonify({"summary": summary})


def generate_summary_ids(encoder_outputs, encoder_attention_mask, max_length=150):
    """
    Greedy decoding loop using the ONNX decoder session.
    Uses the provided encoder outputs and encoder attention mask.
    """
    # T5 (or FLAN-T5) doesn't define a decoder_start_token_id so use pad_token_id as fallback.
    decoder_start_token_id = getattr(
        tokenizer, "decoder_start_token_id", None) or tokenizer.pad_token_id

    # Initialize decoder input_ids with shape (1, 1)
    decoder_input_ids = np.array([[decoder_start_token_id]], dtype=np.int64)

    for step in range(max_length):
        # Create an attention mask for the decoder; here it's simply ones.
        decoder_attention_mask = np.ones(
            decoder_input_ids.shape, dtype=np.int64)

        # Prepare inputs for the decoder with the exact names expected.
        decoder_inputs = {
            "input_ids": decoder_input_ids,
            "encoder_hidden_states": encoder_outputs[0],
            "encoder_attention_mask": encoder_attention_mask,
        }

        # Run the decoder; the first output is 'logits'
        outputs = decoder_session.run(None, decoder_inputs)
        logits = outputs[0]  # shape: (batch_size, sequence_length, vocab_size)

        # Greedy decoding: select token with highest logit from the last position
        next_token_logits = logits[:, -1, :]
        next_token_id = np.argmax(next_token_logits, axis=-1)

        # Append the predicted token to the sequence
        decoder_input_ids = np.concatenate(
            [decoder_input_ids, next_token_id[:, None]], axis=1)

        # Stop if EOS token is produced
        if next_token_id[0] == tokenizer.eos_token_id:
            break

    return decoder_input_ids


if __name__ == '__main__':
    app.run(host='localhost', port=2850, debug=True)
