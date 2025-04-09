document.addEventListener('DOMContentLoaded', () => {
  const analyzeBtn = document.getElementById('analyzeBtn');
  const videoUrlSpan = document.getElementById('videoUrl');
  const resultDiv = document.getElementById('result');
  let sentimentChart, forecastChart;

  // Get the active tab's URL
  chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
    const activeTab = tabs[0];
    const url = activeTab.url;
    if (url.includes("youtube.com/watch") || url.includes("youtu.be/")) {
      videoUrlSpan.textContent = url;
      analyzeBtn.disabled = false;
    } else {
      videoUrlSpan.textContent = "Not a YouTube video.";
    }
  });

  analyzeBtn.addEventListener('click', () => {
    const videoUrl = videoUrlSpan.textContent;
    resultDiv.textContent = "Analyzing comments...";

    fetch("http://localhost:5000/analyze", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ video_url: videoUrl })
    })
    .then(res => res.json())
    .then(data => {
      if (data.error) {
        resultDiv.textContent = "Error: " + data.error;
        return;
      }
      
      // Prepare results text
      const resultText = `
Total Comments: ${data.total_comments}
Unique Comments: ${data.unique_comments}
Positive Comments: ${data.positive_comments}
Negative Comments: ${data.negative_comments}
Average Rating (out of 5): ${data.average_rating.toFixed(2)}
      `;
      resultDiv.textContent = resultText;
      
      // Render sentiment pie chart
      const ctx = document.getElementById('sentimentChart').getContext('2d');
      if (sentimentChart) sentimentChart.destroy();
      sentimentChart = new Chart(ctx, {
        type: 'pie',
        data: {
          labels: ['Positive', 'Negative'],
          datasets: [{
            data: [data.positive_comments, data.negative_comments],
            backgroundColor: ['#28a745', '#dc3545']
          }]
        },
        options: {
          responsive: true,
          plugins: {
            legend: {
              position: 'bottom'
            }
          }
        }
      });
      
      // Render forecast line chart
      const fctx = document.getElementById('forecastChart').getContext('2d');
      if (forecastChart) forecastChart.destroy();
      forecastChart = new Chart(fctx, {
        type: 'line',
        data: {
          labels: data.time_series_forecast.dates,
          datasets: [{
            label: 'Forecast Comment Count',
            data: data.time_series_forecast.values,
            backgroundColor: 'rgba(54, 162, 235, 0.2)',
            borderColor: 'rgba(54, 162, 235, 1)',
            fill: true,
            tension: 0.4
          }]
        },
        options: {
          responsive: true,
          scales: {
            x: {
              display: true,
              title: {
                display: true,
                text: 'Date'
              }
            },
            y: {
              display: true,
              title: {
                display: true,
                text: 'Comments'
              },
              beginAtZero: true
            }
          }
        }
      });
    })
    .catch(err => {
      resultDiv.textContent = "Fetch error: " + err;
    });
  });
});
