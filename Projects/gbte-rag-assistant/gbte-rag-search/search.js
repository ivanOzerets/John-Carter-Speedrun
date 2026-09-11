function extractTimestamp(url) {
    const match = url.match(/[?&]t=(\d+)s/);

    return match ? parseInt(match[1], 10) : 0;
}

function formatTimestamp(seconds) {
    const hrs = Math.floor(seconds / 3600);
    const mins = Math.floor((seconds % 3600) / 60);
    const secs = seconds % 60;

    const paddedMins = String(mins).padStart(2, '0');
    const paddedSecs = String(secs).padStart(2, '0');

    if (hrs > 0) {
        return hrs + ':' + paddedMins + ':' + paddedSecs;
    }

    return mins + ':' + paddedSecs;
}

function formatCitations(citations) {
    let html = '<ul class="gbte-citations">';

    citations.forEach(function(citation) {
        const seconds = extractTimestamp(citation.url);
        const timeLabel = formatTimestamp(seconds);
        html += '<li><a href="' + citation.url + '" target="_blank">' + citation.title + ' (' + timeLabel + ')</a></li>';
    });

    html += "</ul>";

    return html;
}

function formatAnswer(text) {
    text = text.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
    text = text.replace(/\n/g, '<br>');

    return text;
}

let searchInProgress = false;

function runSearch() {
    if (searchInProgress) {
        return;
    }

    searchInProgress = true;

    const question = document.getElementById("gbte-question").value;
    const resultsDiv = document.getElementById("gbte-results");

    resultsDiv.innerHTML = "Searching...";

    fetch('https://pkzvniqch1.execute-api.us-east-2.amazonaws.com/search', {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: question })
    })
    .then(response => {
        if (!response.ok) {
            throw new Error("Request failed");
        }
        return response.json();
    })
    .then(data => {
        resultsDiv.innerHTML = "<h4>Answer</h4><p class=\"gbte-answer-text\">" + formatAnswer(data.answer) + "</p>" +
        "<p class=\"gbte-disclaimer\"><em>Answers are generated from GBTE's video library and may not always be fully accurate.</em></p>" +
        "<h4 class=\"gbte-sources-title\">Sources</h4>" + formatCitations(data.citations);
    })
    .catch(error => {
        resultsDiv.innerHTML = "Something went wrong. Please try again.";
    })
    .finally(() => {
        searchInProgress = false;
    });
}

document.getElementById("gbte-question").addEventListener("input", function(event) {
    const input = event.target;
    if (input.value.length > 0) {
        input.value = input.value.charAt(0).toUpperCase() + input.value.slice(1);
    }
});

document.getElementById("gbte-search-btn").addEventListener("click", runSearch);

document.getElementById("gbte-question").addEventListener("keydown", function(event) {
    if (event.key === "Enter") {
        runSearch();
    }
});
