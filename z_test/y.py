import onnxruntime as ort
import numpy as np
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch
import os

# Load model and tokenizer
model_name = "google/flan-t5-small"
tokenizer = T5Tokenizer.from_pretrained(
    model_name, legacy=False)  # Avoids legacy behavior warning
model = T5ForConditionalGeneration.from_pretrained(model_name)
model.eval()

# Load the ONNX model
ort_session = ort.InferenceSession("z_test/flan-t5-small.onnx")

# Example input text
text = """summarize: In ancient Greece and Rome, the tympanon (τύμπανον) or tympanum, was a type of frame drum or tambourine. It was circular, shallow, and beaten with the palm of the hand or a stick. Some representations show decorations or zill-like objects around the rim. The instrument was played by worshippers in the rites of Dionysus, Cybele, and Sabazius.[1] The instrument came to Rome from Greece and the Near East, probably in association with the cult of Cybele.[2] The first depiction in Greek art appears in the 8th century BC, on a bronze votive disc found in a cave on Crete that was a cult site for Zeus.[3][4] Dionysian rites[edit] The tympanum is one of the objects often carried in the thiasos, the retinue of Dionysus. The instrument is typically played by a maenad, while wind instruments such as pipes or the aulos are played by satyrs. The performance of frenzied music contributed to achieving the ecstatic state that Dionysian worshippers desired.[5] The cult of Cybele[edit] The tympanum was the most common of the musical instruments associated with the rites of Cybele in the art and literature of Greece and Rome, but does not appear in representations from Anatolia, where the goddess originated.[6] From the 6th century BC, the iconography of Cybele as Meter ("Mother", or in Latin Magna Mater, "Great Mother") may show her with the tympanum balanced on her left arm, usually seated and with a lion on her lap or in attendance.[7] The Homeric Hymn to the Great Mother says that the goddess loves the sound of the tympanum. The drum continued to feature as an attribute of Cybele into the Roman Imperial era."""
inputs = tokenizer(text, return_tensors="pt")

# Extract required input tensors
input_ids = inputs["input_ids"]
attention_mask = inputs["attention_mask"]
# Prepare initial decoder input (start with decoder_start_token_id)
decoder_start_token_id = model.config.decoder_start_token_id
decoder_input_ids = torch.tensor([[decoder_start_token_id]], dtype=torch.long)

# Set generation parameters
max_length = 400              # maximum tokens to generate
# ensure we generate at least 50 tokens before stopping on EOS
min_generated_tokens = 150
eos_token_id = model.config.eos_token_id

# Initialize the generated sequence with the start token
generated = decoder_input_ids.clone()

# Iteratively generate tokens
for _ in range(max_length - 1):
    # Prepare ONNX inputs (make sure tensors are on CPU and converted to NumPy)
    onnx_input = {
        "input_ids": input_ids.numpy(),
        "attention_mask": attention_mask.numpy(),
        "decoder_input_ids": generated.numpy(),
    }

    # Run inference with ONNX Runtime
    onnx_output = ort_session.run(None, onnx_input)
    # shape: (1, current_seq_len, vocab_size)
    output_logits = np.array(onnx_output[0])

    # Get logits for the last token in the sequence
    next_token_logits = output_logits[:, -1, :]

    # Select the token with the highest logit (greedy decoding)
    next_token_id = int(np.argmax(next_token_logits, axis=-1))

    # Append the predicted token to the generated sequence
    next_token_tensor = torch.tensor([[next_token_id]], dtype=torch.long)
    generated = torch.cat([generated, next_token_tensor], dim=1)

    # If EOS is predicted and we have generated enough tokens, break the loop
    if next_token_id == eos_token_id and generated.size(1) >= min_generated_tokens:
        break

# Convert the generated token IDs to a list and decode into text
output_ids = generated.squeeze().tolist()
summary = tokenizer.decode(output_ids, skip_special_tokens=True)
print("Summary:", summary)
