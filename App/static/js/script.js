const uploadForm = document.getElementById('uploadForm');
const videoSection = document.getElementById('videoSection');
const videoPlayer = document.getElementById('videoPlayer');
const commentBox = document.getElementById('commentBox');
const loading = document.getElementById('loading');

const uploadButton = document.getElementById('uploadButton');
const analyzeButton = document.getElementById('analyzeButton');

let selectedFile = null;
let comments = [];

document.getElementById('videoInput').addEventListener('change', (e) => {
    selectedFile = e.target.files[0];
    if (selectedFile) {
        analyzeButton.disabled = false;
    }
});

uploadForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    // Upload button behavior - we separate now
});

analyzeButton.addEventListener('click', async () => {
    if (!selectedFile) return;

    const formData = new FormData();
    formData.append('video', selectedFile);

    // Show loading
    loading.style.display = 'block';
    videoSection.style.display = 'none';
    commentBox.style.opacity = 0;

    const response = await fetch('/upload', {
        method: 'POST',
        body: formData
    });

    const data = await response.json();
    loading.style.display = 'none';

    if (data.success) {
        videoPlayer.src = `/static/processed/${data.processed_video}`;
        comments = data.comments;
        videoSection.style.display = 'block';
    } else {
        alert('Error processing video.');
    }
});

let lastShownTime = null;
let fadeTimeout = null;

videoPlayer.addEventListener('timeupdate', () => {
    const currentTime = Math.floor(videoPlayer.currentTime);

    const currentComment = comments.find(comment => comment.timecode === currentTime);
    if (currentComment && lastShownTime !== currentTime) {
        commentBox.innerText = currentComment.text;
        commentBox.style.opacity = 1;

        lastShownTime = currentTime;

        if (fadeTimeout) {
            clearTimeout(fadeTimeout);
        }

        fadeTimeout = setTimeout(() => {
            if (!videoPlayer.paused) {
                commentBox.style.opacity = 0;
            }
        }, 2000);
    }
});

videoPlayer.addEventListener('pause', () => {
    // Keep comment visible if paused
    if (lastShownTime !== null) {
        commentBox.style.opacity = 1;
    }
});

videoPlayer.addEventListener('play', () => {
    // Restart fade after play resumes
    if (lastShownTime !== null) {
        if (fadeTimeout) {
            clearTimeout(fadeTimeout);
        }

        fadeTimeout = setTimeout(() => {
            if (!videoPlayer.paused) {
                commentBox.style.opacity = 0;
            }
        }, 2000);
    }
});


