const carousel = document.querySelector(".carousel");
const carouselItems = carousel ? carousel.querySelectorAll(".carousel-item") : [];
const submissionResult = document.getElementById("submissionResult");
const linkInput = document.getElementById("link-input");
const filePicker = document.getElementById("fileInput");
const videoPreview = document.getElementById("video-preview");
const iFramePreview = document.getElementById("iframe-preview");
// Background video removed; keep null to avoid errors
const bgVideo = null;

var selectedFile = null;
var selectedLink = "";
var linkInputVisible = false;
let currentItem = 0;

// 1. Background image carousel
function changeImage() {
  if (!carouselItems || carouselItems.length === 0) return;
  carouselItems.forEach((item, index) => {
    if (index === currentItem) {
      item.classList.add("active");
    } else {
      item.classList.remove("active");
    }
  });
  currentItem = (currentItem + 1) % carouselItems.length;
}
setInterval(changeImage, 3000);

// 2. Box with title, description and (file uploader/link & submit button)
const box = document.querySelector(".box");
setTimeout(() => {
  box.classList.add("visible");
}, 2000);

// 3. File uploader
function openFilePicker() {
  hideLinkInput();
  filePicker.click();
}

filePicker.addEventListener("change", function () {
  selectedFile = this.files[0];
  document.getElementById("fileName").textContent =
    "Selected File: " + selectedFile.name;
  hideLinkInput();
  showVideoPreview(URL.createObjectURL(selectedFile));
});

// 4. Link
function toggleLinkInput() {
  hideVideoPreview(); 
  var linkInputContainer = document.getElementById("linkInputContainer");
  linkInputVisible = !linkInputVisible;
  if (linkInputVisible) {
    linkInputContainer.style.display = "block";
  } else {
    hideLinkInput();
  }
}

function hideLinkInput() {
  document.getElementById("linkInputContainer").style.display = "none";
  linkInput.value = "";
  selectedLink = "";
  hideIFramePreview();
}


function convertToEmbed(url) {
  // Regular expression to capture the video ID from the YouTube URL
  const videoIdPattern = /(?:v=|\/)([0-9A-Za-z_-]{11}).*/;
  const match = url.match(videoIdPattern);
  
  if (!match) {
      return null;  // Return null if no video ID is found
  }
  
  const videoId = match[1];
  const embedUrl = `https://www.youtube.com/embed/${videoId}`;
  return embedUrl;
}

linkInput.addEventListener("input", function () {
  selectedLink = this.value;
  selectedFile = null;
  document.getElementById("fileName").textContent = "";
  showIFramePreview(convertToEmbed(selectedLink));
});

// 5. Submit button
// Add smart comic options to the UI
function addSmartComicOptions() {
    const box = document.getElementById('box');
    const submitButton = box.querySelector('.submit-button');
    
    // Create options container
    const optionsDiv = document.createElement('div');
    optionsDiv.id = 'smart-options';
    optionsDiv.style.cssText = 'margin: 20px 0; padding: 15px; background: rgba(255,255,255,0.1); border-radius: 10px;';
    optionsDiv.innerHTML = `
        <h3 style="margin: 0 0 10px 0; font-size: 18px;">✨ Enhanced Comic Options</h3>
        <label style="display: block; margin: 5px 0; text-align: left; cursor: pointer;">
            <input type="checkbox" id="smart-mode" checked style="margin-right: 8px;">
            <span style="font-weight: bold;">Smart Frame Selection</span>
            <br>
            <span style="font-size: 12px; color: #aaa; margin-left: 25px;">
                Automatically selects the most engaging frames with perfect expressions
            </span>
        </label>
    `;
    
    // Insert before submit button
    box.insertBefore(optionsDiv, submitButton);
}

// Call on page load
document.addEventListener('DOMContentLoaded', addSmartComicOptions);

function submitForm() {
  // Get smart comic options
  const smartMode = document.getElementById('smart-mode').checked;
  
  // If file is selected
  if (selectedFile !== null && selectedLink === "") {
    submissionResult.textContent = "Your comic is being created";
    var formdata = new FormData();
    formdata.append("file", selectedFile);
    formdata.append("smart_mode", smartMode);

    var requestOptions = {
      method: "POST",
      body: formdata,
      redirect: "follow",
    };

    fetch("/uploader", requestOptions)
      .then((response) => response.text())
      .then((result) => {
        console.log(result);
        submissionResult.textContent = result;
      })
      .catch((error) => {
        console.log("error", error);
        alert(error);
      });
  }

  // If link is entered
  else if (selectedLink !== "" && selectedFile === null) {
    submissionResult.textContent = "Your comic is being created";

    var formdata = new FormData();
    formdata.append("link", linkInput.value);
    formdata.append("smart_mode", smartMode);

    var requestOptions = {
      method: "POST",
      body: formdata,
      redirect: "follow",
    };

    fetch("/handle_link", requestOptions)
      .then((response) => response.text())
      .then((result) => {
        console.log(result);
        submissionResult.textContent = result;
      })
      .catch((error) => {
        console.log("error", error);
        submissionResult.textContent = "An error has occurred";
      });
  } else {
    document.getElementById("submissionResult").textContent =
      "Please select either a file or enter a link.";
  }
}

// 6. Video preview
function showVideoPreview(url) {
  hideIFramePreview();
  videoPreview.src = url;
  videoPreview.style.display = "block";
  videoPreview.play();
  // no-op: background video removed
}

function hideVideoPreview() {
  videoPreview.src = "";
  videoPreview.style.display = "none";
  // no-op: background video removed
}

function showIFramePreview(url) {
  hideVideoPreview();
  iFramePreview.src = url;
  iFramePreview.style.display = "block";
  // no-op: background video removed
}

function hideIFramePreview() {
  iFramePreview.src = "";
  iFramePreview.style.display = "none";
  // no-op: background video removed
}
