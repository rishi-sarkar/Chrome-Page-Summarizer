# Text Summarizer + Chatbot Chrome Extension

This repository contains a lightweight Chrome extension that performs two major tasks:
1. **Text Summarizer**: Scrapes and summarizes webpage content in real-time.
2. **Chatbot**: Simulates an AI “assistant” directly within the browser.

Both features are powered by a local NLP model in `.onnx` format, wrapped by a **Flask backend** to handle the summarization and chatbot logic. By running Flask locally, the extension ensures all processing is done on your own machine, preserving privacy.

---

## Features

1. **Local Inference Using Flask + ONNX**  
   - A lightweight Flask server hosts an ONNX-based NLP model.  
   - **All inference happens locally**—no external API calls required, keeping data private.

2. **One-Click Summaries**  
   - Summarize large chunks of text from any webpage with a single click.  
   - Summaries are concise and easy to read, enabling faster content digestion.

3. **Built-In Chatbot**  
   - Interact with a friendly AI “assistant” directly in the browser.  
   - Ask follow-up questions, re-summarize content in different ways, or chat freely—all handled by the local Flask server.

4. **Lightweight and User-Friendly**  
   - Installs like any standard Chrome extension—no bulky dependencies in the browser itself.  
   - Minimal overhead thanks to an efficient local model and Flask backend.

---

## Installation

1. **Clone or Download this Repository**  
   - Clone with Git:
     ```bash
     git clone https://github.com/YourUsername/Text-Summarizer-Chatbot-Extension.git
     ```
   - Or download and extract the ZIP from GitHub.

2. **Set up and Run the Flask Backend**  
   - Navigate to the repository’s root folder (where the Flask app is located).
   - (Optional) Create and activate a Python virtual environment:
     ```bash
     python -m venv venv
     source venv/bin/activate  # On Windows: venv\Scripts\activate
     ```
   - Install dependencies:
     ```bash
     pip install -r requirements.txt
     ```
   - Run the Flask server:
     ```bash
     python app.py
     ```
   - By default, Flask should run on `http://127.0.0.1:5000/`.  
   - **Keep the Flask server running** to handle summaries and chatbot queries locally.

3. **Load Extension in Chrome**  
   1. Open Chrome and go to `chrome://extensions/`.
   2. Enable “Developer Mode” in the top-right corner.
   3. Click “Load unpacked” and select the folder containing this project’s **Chrome extension files** (e.g., the `src/` and `manifest.json`).

4. **Verification**  
   - After loading, you should see a new extension icon in your browser toolbar.  
   - Clicking the icon should open the interface for both summarization and chatbot interactions.  
   - Make sure your Flask server is still running; otherwise, the extension will not be able to retrieve summaries or chat responses.

---

## Usage

1. **Summarizing Web Content**  
   - Navigate to any webpage with an article or content you’d like to summarize.  
   - Click the extension icon. A small popup will appear with a “Summarize” button.  
   - The extension sends the page’s text to the **Flask backend**, which uses the ONNX model to generate a concise summary.  
   - Wait a few seconds for the summary to appear in the popup window.

2. **Chatbot Interaction**  
   - Open the extension popup by clicking the icon.  
   - Type your queries or instructions in the chatbot text box.  
   - The chatbot uses the **Flask server** to process your queries. You can re-summarize the same webpage, fetch specific sections of text on demand, or hold a casual conversation.  
   - All data processing remains on your computer—no calls to remote APIs.

3. **Settings**  
   - There may be configurable settings (e.g., summary length, language preferences) in the extension’s “Options” page.  
   - Access these by right-clicking on the extension icon and selecting “Options.”

---

## How It Works

1. **Scraping & Processing**

   - A content script scrapes the webpage text. This text is then cleaned and sent to the extension’s local inference engine.

2. **Inference with ONNX**

   - The extension loads the ONNX model (stored in `model/onnx_flant5_small`) within the browser.
   - All inference is performed locally, ensuring user data never leaves the browser.

3. **Summary & Chat**
   - The summary or chatbot response is displayed in the extension’s popup.
   - The user can continue to refine summaries or interact with the chatbot for additional context.

---

## Development

1. **Prerequisites**

   - Node.js (for building, linting, or dependency management if you expand the project).
   - Basic knowledge of browser extension APIs (Manifest v2 or v3).

2. **Setup & Build**

   - Install dependencies (if any):
     ```bash
     npm install
     ```
   - Build/prepare the project:
     ```bash
     npm run build
     ```
   - Load the “dist” or build folder into Chrome as an “unpacked” extension.

3. **Testing**
   - Use Chrome’s extension tools or a testing framework like [Jest](https://jestjs.io/) (for pure JS) or [web extension test runner](https://github.com/GoogleChromeLabs/chrome-extension-cli) to validate functionality.
   - Regularly test content scraping, summary generation, and chatbot responses to ensure stability.

---

## Contributing

1. **Fork and Clone**
   - Fork the repository on GitHub, then clone your fork locally.
2. **Create a Branch**
   - Make a new branch for your feature or bug fix:
     ```bash
     git checkout -b feature/summarization-updates
     ```
3. **Commit and Push**
   - Commit your changes with clear messages:
     ```bash
     git commit -m "Add advanced summarization parameters"
     git push origin feature/summarization-updates
     ```
4. **Pull Request**
   - Open a PR against the main repository. Provide a clear description of your changes and testing steps.

---

## Roadmap

- **Advanced Language Support**: Add more languages beyond English for both summarization and chatbot interactions.
- **Custom Models**: Allow users to import or select different ONNX models.
- **Customization**: Fine-tune summary length, style, or level of detail through UI settings.
- **Optimization**: Explore WebAssembly or GPU-accelerated model inference for faster performance.

---

## License

This project is licensed under the [MIT License](LICENSE). Feel free to use, modify, and distribute this extension for both personal and commercial purposes.

---

## Disclaimer

- While all inference is handled locally, users should be aware of potential browser limitations and memory constraints when handling very large texts.
- Use this extension at your own risk. The maintainers do not guarantee perfect accuracy of the summarized content or chat responses.

---

## Acknowledgments

- **ONNX Runtime**: For making cross-platform, local inference simpler.
- **Chrome Extension Docs**: For providing detailed APIs and guides.
- The open-source community for providing tools, examples, and continuous inspiration.

**Enjoy summarizing and chatting within your browser—privacy-first and hassle-free!**
