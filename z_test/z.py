from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch
import os

# Load model and tokenizer
model_name = "google/flan-t5-small"
tokenizer = T5Tokenizer.from_pretrained(model_name, legacy=False)  # Avoids legacy behavior warning
model = T5ForConditionalGeneration.from_pretrained(model_name)
model.eval()

# Example input text
text = "summarize: OpenAI is revolutionizing AI research and deployment."
inputs = tokenizer(text, return_tensors="pt")

# Extract required input tensors
input_ids = inputs["input_ids"]
attention_mask = inputs["attention_mask"]
decoder_input_ids = torch.ones((1, 1), dtype=torch.long) * model.config.decoder_start_token_id

# Define ONNX filename
onnx_filename = "flan-t5-small.onnx"

# Export to ONNX without TorchScript (which was causing errors)
torch.onnx.export(
    model,
    (input_ids, attention_mask, decoder_input_ids),
    onnx_filename,
    input_names=["input_ids", "attention_mask", "decoder_input_ids"],
    output_names=["logits"],
    dynamic_axes={
        "input_ids": {0: "batch_size", 1: "seq_length"},
        "attention_mask": {0: "batch_size", 1: "seq_length"},
        "decoder_input_ids": {0: "batch_size", 1: "seq_length"},
    },
    opset_version=13
)

print(f"Model successfully exported to {onnx_filename}")
