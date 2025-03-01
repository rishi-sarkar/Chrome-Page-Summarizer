document.getElementById('summarize').addEventListener('click', async () => {
    let [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
  
    chrome.scripting.executeScript({
      target: { tabId: tab.id },
      func: getPageText
    }, (results) => {
      if (results && results[0]) {
        const pageText = results[0].result;
        // Display the page text in the popup
        const displayDiv = document.getElementById('displayData');
        displayDiv.textContent = pageText;
  
        // Optionally, adjust the popup size if necessary
        adjustPopupSize(displayDiv);
      }
    });
  });
  
  // Function that runs in the context of the active tab to get page text
  function getPageText() {
    return document.body.innerText;
  }
  
  // Optional: dynamically adjust the popup size based on content height
  function adjustPopupSize(container) {
    // Calculate the desired height based on content (with some max limit)
    const contentHeight = container.scrollHeight + 100; // extra space for padding etc.
    const maxHeight = 600; // maximum popup height
    const newHeight = Math.min(contentHeight, maxHeight);
    
    // Update the popup's body style; note that Chrome popups may have limits on auto-resizing
    document.body.style.height = newHeight + 'px';
  }
  