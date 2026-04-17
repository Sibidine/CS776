let taskId = null;
let selectedFile = null;

const uploadArea = document.getElementById("uploadArea");
const fileInput = document.getElementById("fileInput");
const uploadText = document.getElementById("uploadText");
const progressBar = document.getElementById("progress");

// Click to upload
uploadArea.addEventListener("click", () => fileInput.click());

// File select
fileInput.addEventListener("change", () => {
    selectedFile = fileInput.files[0];
    uploadText.innerText = selectedFile.name;
});

// Drag & drop
uploadArea.addEventListener("dragover", (e) => {
    e.preventDefault();
    uploadArea.style.borderColor = "#4a90e2";
});

uploadArea.addEventListener("dragleave", () => {
    uploadArea.style.borderColor = "#bbb";
});

uploadArea.addEventListener("drop", (e) => {
    e.preventDefault();
    selectedFile = e.dataTransfer.files[0];
    uploadText.innerText = selectedFile.name;
});

// Upload
async function uploadVideo() {
    const status = document.getElementById("status");

    if (!selectedFile) {
        alert("Select a video first");
        return;
    }

    const formData = new FormData();
    formData.append("file", selectedFile);

    status.innerText = "Uploading...";
    progressBar.style.width = "20%";

    const res = await fetch("http://localhost:8000/upload", {
        method: "POST",
        body: formData
    });

    const data = await res.json();
    taskId = data.task_id;

    status.innerText = "Processing...";
    progressBar.style.width = "50%";

    pollStatus();
}

// Polling
async function pollStatus() {
    const status = document.getElementById("status");
    const video = document.getElementById("outputVideo");

    const interval = setInterval(async () => {
        const res = await fetch(`http://localhost:8000/result/${taskId}`);
        const data = await res.json();

        if (data.status === "done") {
            if (data.status === "done") {
                clearInterval(interval);
            
                progressBar.style.width = "100%";
            
                if (data.error) {
                    status.innerText = data.error;   // ❗ show message
                    status.style.color = "#d9534f";  // red
                } else {
                    status.innerText = "Processing complete!";
                    status.style.color = "green";
                }
            
                video.src = `http://localhost:8000${data.video_url}`;
                video.classList.add("visible");
            }
        } else {
            status.innerText = "Processing...";
        }
    }, 2000);
}