let session;

// Load the quantized, fine-tuned model on startup.
async function loadModel() {
  try {
    console.log("Trying to load model...");
    // Load the model file from the extension bundle.
    session = await ort.InferenceSession.create(chrome.runtime.getURL('flan-t5.onnx'));
    console.log("Model loaded successfully.");
  } catch (err) {
    console.error("Failed to load model:", err);
  }
}
loadModel();

// Summarize button event listener.
document.getElementById('summarize').addEventListener('click', async () => {
  let [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
  chrome.scripting.executeScript({
    target: { tabId: tab.id },
    func: getPageText
  }, async (results) => {
    if (results && results[0]) {
      const pageText = results[0].result;
      // Run the summarization function on the page text.
      const summary = await generateSummary(pageText);
      document.getElementById('summary').textContent = summary;
      adjustPopupSize(document.body);
    }
  });
});

// Q&A button event listener.
document.getElementById('ask').addEventListener('click', async () => {
  let [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
  chrome.scripting.executeScript({
    target: { tabId: tab.id },
    func: getPageText
  }, async (results) => {
    if (results && results[0]) {
      const pageText = results[0].result;
      const question = document.getElementById('question').value;
      // Run the Q&A function using the page text and the user-provided query.
      const answer = await answerQuestion(pageText, question);
      document.getElementById('answer').textContent = answer;
      adjustPopupSize(document.body);
    }
  });
});

// This function runs in the context of the active tab to capture page text.
function getPageText() {
  return document.body.innerText;
}

/* ----------------------------
   Production-level Inference
   ----------------------------

In a production system, the following steps are required:

1. **Tokenization:**  
   Replace the dummy `tokenize()` with a tokenizer that converts input text into the correct sequence of token IDs.  
   - Use a tokenizer library (e.g., Hugging Face Tokenizers compiled to WASM) that supports your model.  
   - Ensure proper handling of special tokens (e.g., <pad>, <eos>, etc.) and input truncation/padding.

2. **Input Tensor Creation:**  
   Convert the tokenized output into tensors (e.g., using Int32Array) with the proper shape that your model expects (typically `[batch_size, sequence_length]`).  
   - Create additional tensors if your model requires an attention mask or other inputs.

3. **Model Inference:**  
   Call `session.run(feeds)` with a feed dictionary mapping input names (e.g., `"input_ids"`, `"attention_mask"`) to their corresponding tensors.  
   - Handle errors and asynchronous processing gracefully.

4. **Decoding:**  
   Replace the dummy `decode()` with a proper decoding function that converts output token IDs back into text.  
   - Implement or use beam search/greedy decoding as needed.
   - Ensure your vocabulary mapping is correct.

5. **Performance Considerations:**  
   - Offload heavy computation to Web Workers if necessary.
   - Optimize memory usage for large inputs or long pages.

Below are the updated inference functions with these steps in mind.
*/

// Summarization: process the page text and generate a summary.
async function generateSummary(text) {
  if (!session) return "Model not loaded.";

  // 1. Tokenize the input text.
  const tokens = tokenize(text); // Replace with a production tokenizer.
  
  // 2. Create input tensors (adjust types/dimensions as needed).
  const inputTensor = new ort.Tensor('int32', Int32Array.from(tokens), [1, tokens.length]);
  const attentionMask = new ort.Tensor('int32', Int32Array.from(new Array(tokens.length).fill(1)), [1, tokens.length]);

  // 3. Run the model inference.
  const feeds = { 'input_ids': inputTensor, 'attention_mask': attentionMask };
  let results;
  try {
    results = await session.run(feeds);
  } catch (error) {
    console.error("Error during inference:", error);
    return "Inference error.";
  }

  // 4. Decode the output tokens to text.
  const outputTokens = results['output_ids'].data; // Adjust key if needed.
  const summary = decode(outputTokens); // Replace with a production decoder.
  return summary;
}

// Q&A: combine the page text and query to produce an answer.
async function answerQuestion(text, query) {
  if (!session) return "Model not loaded.";

  // 1. Construct a prompt that combines context and question.
  const prompt = `Context: ${text}\nQuestion: ${query}\nAnswer:`;
  
  // 2. Tokenize the prompt.
  const tokens = tokenize(prompt); // Replace with a production tokenizer.
  
  // 3. Create input tensors.
  const inputTensor = new ort.Tensor('int32', Int32Array.from(tokens), [1, tokens.length]);
  const attentionMask = new ort.Tensor('int32', Int32Array.from(new Array(tokens.length).fill(1)), [1, tokens.length]);
  
  // 4. Run model inference.
  const feeds = { 'input_ids': inputTensor, 'attention_mask': attentionMask };
  let results;
  try {
    results = await session.run(feeds);
  } catch (error) {
    console.error("Error during inference:", error);
    return "Inference error.";
  }

  // 5. Decode the output tokens to text.
  const outputTokens = results['output_ids'].data; // Adjust key if needed.
  const answer = decode(outputTokens); // Replace with a production decoder.
  return answer;
}

// Optional: Adjust the popup height dynamically based on content.
function adjustPopupSize(container) {
  const contentHeight = container.scrollHeight + 100;
  const maxHeight = 600;
  const newHeight = Math.min(contentHeight, maxHeight);
  document.body.style.height = newHeight + 'px';
}

/* 
Placeholder Tokenization & Decoding Functions

These dummy functions illustrate the expected behavior. In production, use a proper tokenizer and decoder.
*/

function tokenize(text) {
  // Production: Use a tokenizer library to convert text to token IDs.
  // This dummy implementation converts each character code mod 100.
  return text.split('').map(c => c.charCodeAt(0) % 100);
}

function decode(tokens) {
  // Production: Convert token IDs back into text using the correct vocabulary mapping.
  // This dummy implementation converts token numbers back to characters.
  return tokens.map(t => String.fromCharCode(t + 30)).join('');
}
