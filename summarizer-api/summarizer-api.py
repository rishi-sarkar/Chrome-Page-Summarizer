# summarize_api.py
from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin
from transformers import AutoTokenizer
import onnxruntime as ort
import numpy as np
import logging

app = Flask(__name__)
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
    # Prepend an instruction to help the model focus on summarization.
    input_text = "summarize: " + text.strip()
    # Tokenize input text; let the tokenizer handle truncation if needed.
    inputs = tokenizer(input_text, return_tensors="np", truncation=True)
    encoder_inputs = {k: v for k, v in inputs.items()}

    # Run the encoder using 'input_ids' and 'attention_mask'
    encoder_outputs = encoder_session.run(None, encoder_inputs)
    # encoder_outputs[0] is expected to be the last_hidden_state

    # Generate summary token IDs using greedy decoding (with min_length constraint)
    summary_ids = generate_summary_ids(
        encoder_outputs, encoder_inputs["attention_mask"], max_length=300, min_length=20
    )
    # Decode the token IDs to text
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


def generate_summary_ids(encoder_outputs, encoder_attention_mask, max_length=300, min_length=20):
    """
    Greedy decoding loop using the ONNX decoder session.
    The loop continues until the EOS token is generated AND
    the generated sequence length reaches at least min_length.

    Args:
        encoder_outputs: list from the encoder session (encoder_outputs[0] is the last_hidden_state)
        encoder_attention_mask: the attention mask from the encoder input
        max_length: maximum number of tokens to generate.
        min_length: minimum number of tokens to generate before stopping if EOS token is produced.

    Returns:
        A numpy array containing the generated sequence of token IDs.
    """
    # Use decoder_start_token_id if available; else use pad_token_id (common for T5/FLAN-T5)
    decoder_start_token_id = getattr(
        tokenizer, "decoder_start_token_id", None) or tokenizer.pad_token_id

    # Initialize decoder input_ids with shape (1, 1)
    decoder_input_ids = np.array([[decoder_start_token_id]], dtype=np.int64)

    for step in range(max_length):
        # Create a decoder attention mask (all ones)
        decoder_attention_mask = np.ones(
            decoder_input_ids.shape, dtype=np.int64)

        # Prepare inputs for the decoder with the expected names.
        decoder_inputs = {
            "input_ids": decoder_input_ids,
            "encoder_hidden_states": encoder_outputs[0],
            "encoder_attention_mask": encoder_attention_mask,
        }

        outputs = decoder_session.run(None, decoder_inputs)
        logits = outputs[0]  # shape: (batch_size, sequence_length, vocab_size)

        # Greedy decoding: select token with highest logit from the last position.
        next_token_logits = logits[:, -1, :]  # shape: (1, vocab_size)
        next_token_id = np.argmax(next_token_logits, axis=-1)  # shape: (1,)

        # Append the new token to the sequence.
        decoder_input_ids = np.concatenate(
            [decoder_input_ids, next_token_id[:, None]], axis=1)

        # Check if EOS token is produced and minimum length is reached.
        if step >= min_length and next_token_id[0] == tokenizer.eos_token_id:
            break

    return decoder_input_ids


if __name__ == '__main__':
    app.run(host='localhost', port=2850, debug=True)
