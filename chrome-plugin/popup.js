document.addEventListener('DOMContentLoaded', () => {
  const analyzeBtn = document.getElementById('analyzeBtn');
  const videoUrlSpan = document.getElementById('videoUrl');
  const resultDiv = document.getElementById('result');

  // 1. Get the active tab's URL
  chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
    const activeTab = tabs[0];
    const url = activeTab.url;

    // 2. Check if it's a YouTube video
    if (url.includes("youtube.com/watch") || url.includes("youtu.be/")) {
      videoUrlSpan.textContent = url;
      analyzeBtn.disabled = false;
    } else {
      videoUrlSpan.textContent = "Not a YouTube video.";
    }
  });

  // 3. When user clicks "Analyze Comments"
  analyzeBtn.addEventListener('click', () => {
    const videoUrl = videoUrlSpan.textContent;
    resultDiv.textContent = "Analyzing...";

    // For local dev, call your Flask app on localhost:5000
    fetch("http://localhost:5000/analyze", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ video_url: videoUrl })
    })
    .then(res => res.json())
    .then(data => {
      if (data.error) {
        resultDiv.textContent = "Error: " + data.error;
      } else {
        resultDiv.textContent = 
          "Total Comments: " + data.total_comments + "\n" +
          "Positive Comments: " + data.negative_comments + "\n" +
          "Negative Comments: " + data.positive_comments;
      }
    })
    .catch(err => {
      resultDiv.textContent = "Fetch error: " + err;
    });
  });
});
