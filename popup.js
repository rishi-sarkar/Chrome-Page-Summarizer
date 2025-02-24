// popup.js

import { loadTokenizer, tokenizeText, decodeTokens } from './tokenizer.js';

// Global variables for models and tokenizer status.
let encoderSession;
let decoderSession;
let tokenizerReady = false;

// 1. Initialize the tokenizer and ONNX models.
(async function initialize() {
  // Load the tokenizer
  try {
    await loadTokenizer();
    tokenizerReady = true;
    console.log("Tokenizer loaded successfully.");
  } catch (err) {
    console.error("Failed to load tokenizer:", err);
  }

  // Load the encoder model
  try {
    console.log("Loading encoder model...");
    encoderSession = await ort.InferenceSession.create(
      chrome.runtime.getURL('models/flan-t5-small/encoder_model.onnx')
    );
    console.log("Encoder model loaded successfully.");
  } catch (err) {
    console.error("Failed to load encoder model:", err);
  }

  // Load the decoder model
  try {
    console.log("Loading decoder model...");
    decoderSession = await ort.InferenceSession.create(
      chrome.runtime.getURL('models/flan-t5-small/decoder_model.onnx')
    );
    console.log("Decoder model loaded successfully.");
  } catch (err) {
    console.error("Failed to load decoder model:", err);
  }
})();

// 2. Summarize Button Event Listener
document.getElementById('summarize').addEventListener('click', async () => {
  let [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
  chrome.scripting.executeScript({
    target: { tabId: tab.id },
    func: getPageText
  }, async (results) => {
    if (results && results[0]) {
      const pageText = results[0].result;
      // Generate a summary from the page text.
      const summary = await generateSummary(pageText);
      document.getElementById('summary').textContent = summary;
      adjustPopupSize(document.body);
    }
  });
});

// 3. Q&A Button Event Listener
document.getElementById('ask').addEventListener('click', async () => {
  let [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
  chrome.scripting.executeScript({
    target: { tabId: tab.id },
    func: getPageText
  }, async (results) => {
    if (results && results[0]) {
      const pageText = results[0].result;
      const question = document.getElementById('question').value;
      // Generate an answer based on the page text and user query.
      const answer = await answerQuestion(pageText, question);
      document.getElementById('answer').textContent = answer;
      adjustPopupSize(document.body);
    }
  });
});

// Helper: This function runs in the active tab to capture page text.
function getPageText() {
  return document.body.innerText;
}

/* ---------------------------------------
   Multi-file Encoder/Decoder Inference
--------------------------------------- */

// Summarization: process page text and generate a summary.
async function generateSummary(text) {
  if (!encoderSession || !decoderSession) return "Model not loaded.";
  if (!tokenizerReady) return "Tokenizer not loaded.";

  try {
    // 1. Tokenize the input text for the encoder.
    //    e.g. for T5, you typically have an input prefix like "summarize: " or similar
    //    depending on your fine-tuning. We'll skip that for simplicity.
    const encoderTokens = tokenizeText(text);

    // 2. Create input tensor for the encoder
    const encoderInputTensor = new ort.Tensor('int32', Int32Array.from(encoderTokens), [1, encoderTokens.length]);
    const encoderAttentionMask = new ort.Tensor('int32', Int32Array.from(new Array(encoderTokens.length).fill(1)), [1, encoderTokens.length]);

    // 3. Run the encoder
    const encoderFeeds = {
      input_ids: encoderInputTensor,
      attention_mask: encoderAttentionMask
    };
    // The key might differ based on your ONNX export. Check the model's actual input names.
    const encoderResults = await encoderSession.run(encoderFeeds);

    // Typically the encoder output is "last_hidden_state" or similar
    const encoderHiddenStates = encoderResults['last_hidden_state'];
    if (!encoderHiddenStates) {
      throw new Error("No encoder hidden state found in results.");
    }

    // 4. Prepare the decoder input. T5 typically uses a <pad> token or <bos> to start decoding.
    //    We'll do a single pass, so let's just feed an empty or minimal input.
    //    In real usage, you'd do iterative decoding or a forced approach. 
    const decoderStartTokens = [0]; // e.g. <pad> token ID, adjust as needed
    const decoderInputTensor = new ort.Tensor('int32', Int32Array.from(decoderStartTokens), [1, decoderStartTokens.length]);

    // 5. Create a feed dictionary for the decoder, including the encoder hidden states
    //    The key names (e.g. "encoder_hidden_states") might differ in your model.
    const decoderFeeds = {
      input_ids: decoderInputTensor,
      encoder_hidden_states: encoderHiddenStates
    };

    // You may also need a decoder_attention_mask, or if you have "decoder_with_past", you'll feed
    // "past_key_values" from step to step. For a single pass, we skip that.

    // 6. Run the decoder
    const decoderResults = await decoderSession.run(decoderFeeds);
    // The decoder output might be "logits" or "last_hidden_state" or "output_ids"
    // depending on how the model was exported.
    // If you have "logits", you'd do argmax to pick a token, etc.
    // If you have "output_ids", you can directly decode them.
    const decoderOutput = decoderResults['logits'] || decoderResults['output_ids'];
    if (!decoderOutput) {
      throw new Error("No decoder output found in results.");
    }

    // 7. Convert the output to token IDs (if needed) and decode
    //    - If the output is raw logits, you'd do: nextToken = argmax(logits).
    //    - If it's already "output_ids", you can decode them.
    //    For a single pass, let's pretend we got final IDs (some models do that).
    let outputTokenIds;
    if (decoderOutput.dims.length === 3) {
      // shape: [batch_size, seq_len, vocab_size], so we might need an argmax
      const data = decoderOutput.data; // Float32Array
      // Argmax last dimension, etc. (This is an advanced step. We'll show a naive approach.)
      // In a real scenario, you'd do iterative generation.
      outputTokenIds = naiveArgmaxDecoder(decoderOutput);
    } else {
      // shape: [batch_size, seq_len], might already be token IDs
      outputTokenIds = Array.from(decoderOutput.data);
    }

    // 8. Decode tokens to text
    const summary = decodeTokens(outputTokenIds);
    return summary || "(No output)";
  } catch (error) {
    console.error("Error during summarization inference:", error);
    return "Inference error during summarization.";
  }
}

// Q&A: combine page text and query to produce an answer.
async function answerQuestion(text, query) {
  if (!encoderSession || !decoderSession) return "Model not loaded.";
  if (!tokenizerReady) return "Tokenizer not loaded.";

  try {
    // 1. Construct the prompt (for T5 Q&A, you might do "question: <question>  context: <text>")
    const prompt = `question: ${query}  context: ${text}`;
    const encoderTokens = tokenizeText(prompt);

    // 2. Create encoder input
    const encoderInputTensor = new ort.Tensor('int32', Int32Array.from(encoderTokens), [1, encoderTokens.length]);
    const encoderAttentionMask = new ort.Tensor('int32', Int32Array.from(new Array(encoderTokens.length).fill(1)), [1, encoderTokens.length]);

    // 3. Run the encoder
    const encoderFeeds = {
      input_ids: encoderInputTensor,
      attention_mask: encoderAttentionMask
    };
    const encoderResults = await encoderSession.run(encoderFeeds);
    const encoderHiddenStates = encoderResults['last_hidden_state'];
    if (!encoderHiddenStates) {
      throw new Error("No encoder hidden state found.");
    }

    // 4. Decoder input: typically start with <pad> or <bos>
    const decoderStartTokens = [0]; // or the ID for <pad> or <bos>
    const decoderInputTensor = new ort.Tensor('int32', Int32Array.from(decoderStartTokens), [1, decoderStartTokens.length]);

    // 5. Prepare decoder feeds
    const decoderFeeds = {
      input_ids: decoderInputTensor,
      encoder_hidden_states: encoderHiddenStates
    };

    // 6. Run the decoder
    const decoderResults = await decoderSession.run(decoderFeeds);
    const decoderOutput = decoderResults['logits'] || decoderResults['output_ids'];
    if (!decoderOutput) {
      throw new Error("No decoder output found.");
    }

    // 7. Convert or decode tokens
    let outputTokenIds;
    if (decoderOutput.dims.length === 3) {
      // Possibly need to do an argmax if we have [batch, seq, vocab_size]
      outputTokenIds = naiveArgmaxDecoder(decoderOutput);
    } else {
      outputTokenIds = Array.from(decoderOutput.data);
    }

    // 8. Decode tokens to text
    const answer = decodeTokens(outputTokenIds);
    return answer || "(No output)";
  } catch (error) {
    console.error("Error during Q&A inference:", error);
    return "Inference error during Q&A.";
  }
}

// (Optional) Example naive argmax function if we get logits [1, seq_len, vocab_size].
function naiveArgmaxDecoder(tensor) {
  // shape: [1, seq_len, vocab_size]
  const [batchSize, seqLen, vocabSize] = tensor.dims;
  const data = tensor.data; // Float32Array of length batchSize*seqLen*vocabSize
  const tokenIds = [];
  for (let i = 0; i < seqLen; i++) {
    let maxVal = -Infinity;
    let maxIdx = 0;
    for (let j = 0; j < vocabSize; j++) {
      const val = data[i * vocabSize + j];
      if (val > maxVal) {
        maxVal = val;
        maxIdx = j;
      }
    }
    tokenIds.push(maxIdx);
  }
  return tokenIds;
}

// Dynamically adjust the popup size based on content.
function adjustPopupSize(container) {
  const contentHeight = container.scrollHeight + 100;
  const maxHeight = 600;
  const newHeight = Math.min(contentHeight, maxHeight);
  document.body.style.height = newHeight + 'px';
}
