import numpy as np
import onnxruntime
from transformers import AutoTokenizer, pipeline

# 1. Load the tokenizer (assume you have the original tokenizer directory or name)
tokenizer = AutoTokenizer.from_pretrained("models/onnx_flant5_small", local_files_only=True)

# 2. Create a custom wrapper class for the ONNX model
class OnnxSummarizationModel:
    def __init__(self, onnx_model_path):
        # Create an ONNX Runtime session
        self.session = onnxruntime.InferenceSession(onnx_model_path)
    
    def __call__(self, input_ids, attention_mask, **kwargs):
        # Convert inputs to numpy arrays if needed
        ort_inputs = {
            "input_ids": np.array(input_ids.cpu() if hasattr(input_ids, "cpu") else input_ids),
            "attention_mask": np.array(attention_mask.cpu() if hasattr(attention_mask, "cpu") else attention_mask)
        }
        # Run the session. The output name(s) must match those expected from the ONNX model.
        ort_outs = self.session.run(None, ort_inputs)
        # Here we assume the model returns logits or predictions in the first output.
        # You may need to adjust this according to your model’s output specification.
        return ort_outs[0]

# 3. Initialize your custom ONNX model
onnx_model = OnnxSummarizationModel("z_test/flan-t5-small.onnx")

# 4. Create the summarization pipeline with your custom model and tokenizer
summarizer = pipeline("summarization", model=onnx_model, tokenizer=tokenizer)

# 5. Use the pipeline to summarize text
text = """In ancient Greece and Rome, the tympanon (τύμπανον) or tympanum, was a type of frame drum or tambourine. It was circular, shallow, and beaten with the palm of the hand or a stick. Some representations show decorations or zill-like objects around the rim. The instrument was played by worshippers in the rites of Dionysus, Cybele, and Sabazius.[1] The instrument came to Rome from Greece and the Near East, probably in association with the cult of Cybele.[2] The first depiction in Greek art appears in the 8th century BC, on a bronze votive disc found in a cave on Crete that was a cult site for Zeus.[3][4] Dionysian rites[edit] The tympanum is one of the objects often carried in the thiasos, the retinue of Dionysus. The instrument is typically played by a maenad, while wind instruments such as pipes or the aulos are played by satyrs. The performance of frenzied music contributed to achieving the ecstatic state that Dionysian worshippers desired.[5] The cult of Cybele[edit] The tympanum was the most common of the musical instruments associated with the rites of Cybele in the art and literature of Greece and Rome, but does not appear in representations from Anatolia, where the goddess originated.[6] From the 6th century BC, the iconography of Cybele as Meter ("Mother", or in Latin Magna Mater, "Great Mother") may show her with the tympanum balanced on her left arm, usually seated and with a lion on her lap or in attendance.[7] The Homeric Hymn to the Great Mother says that the goddess loves the sound of the tympanum. The drum continued to feature as an attribute of Cybele into the Roman Imperial era."""
summary = summarizer(text)
print(summary)
