document.getElementById('summarize').addEventListener('click', async () => {
  let [tab] = await chrome.tabs.query({ active: true, currentWindow: true });

  chrome.scripting.executeScript({
    target: { tabId: tab.id },
    func: getPageText
  }, async (results) => {
    if (results && results[0]) {
      const pageText = results[0].result;
      console.log(pageText);
      // Display the raw page text (optional)
      document.getElementById('summary').textContent = pageText;

      // Send pageText to your summarization API
      try {
        const response = await fetch('http://localhost:2850/summarize', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ text: pageText })
        });

        if (!response.ok) {
          console.error('Network response was not ok:', response.statusText);
          return;
        }

        let data;
        try {
          data = await response.json();
        } catch (err) {
          const text = await response.text();
          console.error('Failed to parse JSON. Raw response:', text);
          return;
        }
        if (data.summary) {
          // Display the summarized text
          document.getElementById('summary').textContent = data.summary;
        } else {
          console.error("Summarization error:", data);
        }
      } catch (err) {
        console.error("Failed to fetch summary:", err);
      }

      // Adjust the popup size if necessary
      adjustPopupSize(document.getElementById('summary'));
    }
  });
});

function getPageText() {
  return document.body.innerText;
}

function adjustPopupSize(container) {
  const contentHeight = container.scrollHeight + 100;
  const maxHeight = 600;
  const newHeight = Math.min(contentHeight, maxHeight);
  document.body.style.height = newHeight + 'px';
}
