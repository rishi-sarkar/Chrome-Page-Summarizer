// This script runs automatically on all pages based on the manifest configuration.
(function() {
    const pageText = document.body.innerText;
    
    // Listen for messages from the popup to provide the page text.
    chrome.runtime.onMessage.addListener((msg, sender, sendResponse) => {
      if (msg.action === 'getText') {
        sendResponse({ text: pageText });
      }
    });
  })();
  