import onnxruntime as ort

session = ort.InferenceSession("models/onnx_flant5_small/encoder_model.onnx")
print("Inputs:", [inp.name for inp in session.get_inputs()])
print("Outputs:", [out.name for out in session.get_outputs()])

session = ort.InferenceSession("models/onnx_flant5_small/decoder_model.onnx")
print("Inputs:", [inp.name for inp in session.get_inputs()])
print("Outputs:", [out.name for out in session.get_outputs()])
