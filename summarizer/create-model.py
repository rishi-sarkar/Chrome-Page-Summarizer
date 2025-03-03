from optimum.onnxruntime.configuration import AutoQuantizationConfig
from optimum.onnxruntime import ORTQuantizer
from optimum.onnxruntime import ORTModelForSeq2SeqLM
from transformers import AutoTokenizer


model_id = "google/flan-t5-small"
save_dir = "onnx_flant5_small"

# 1. Load the HF model & tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id)
# Download and prepare the ONNX model
ort_model = ORTModelForSeq2SeqLM.from_pretrained(
    model_id, from_transformers=True)

# 2. Save the ONNX model (this will produce encoder_model.onnx, decoder_model.onnx, etc.)
ort_model.save_pretrained(save_dir)
tokenizer.save_pretrained(save_dir)

print("Exported multi-file T5 model to:", save_dir)

# # 3. Optional: Quantize
# # Use dynamic quantization instead of static
# qconfig = AutoQuantizationConfig.dynamic()  # This uses dynamic quantization

# quantizer = ORTQuantizer.from_pretrained(
#     save_dir, file_name="encoder_model.onnx")
# qconfig = AutoQuantizationConfig.avx512_vnni(is_static=True)
# quantizer.quantize(save_dir=save_dir, quantization_config=qconfig)

# # Repeat quantization for decoder_model.onnx if desired
# quantizer = ORTQuantizer.from_pretrained(
#     save_dir, file_name="decoder_model.onnx")
# quantizer.quantize(save_dir=save_dir, quantization_config=qconfig)

# print("Quantized multi-file T5 model in:", save_dir)