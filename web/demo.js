import { OpenMakeup } from 'https://cdn.jsdelivr.net/npm/open-makeup-sdk/+esm';

/* =========================================
   CONFIGURATION
========================================= */

const API_URL = window.location.origin;

const $ = (selector) => document.querySelector(selector);

const video = $('#cam');
const canvas = $('#out');

const fileInput = $('#fileInput');
const scanBtn = $('#scanBtn');

const loading = $('#loading');
const error = $('#error');

const results = $('#results');
const scoresContainer = $('#scores');
const overlaysContainer = $('#overlays');

const report = $('#report');
const tryonContainer = $('#tryon-buttons');

const stage = $('.stage');
const cameraScanBtn = $('#btn-scan');
const cameraStartBtn = $('#btn-camera');

let selectedFile = null;
let originalImage = null;
let makeup = null;
let cameraStream = null;

let cameraReady = false;
let cameraInitializing = false;
let scanning = false;

/* =========================================
   UI HELPERS
========================================= */

function showError(message) {
  if (!error) return;

  error.textContent = message;
  error.style.display = 'block';
}

function hideError() {
  if (!error) return;

  error.textContent = '';
  error.style.display = 'none';
}

function setLoading(active, message = 'Analyzing image...') {
  if (!loading) return;

  loading.textContent = message;
  loading.style.display = active ? 'block' : 'none';
}

function showResults() {
  if (!results) return;

  // Override the HTML inline display:none.
  results.style.display = 'block';
  results.classList.add('active');
}

function hideResults() {
  if (!results) return;

  results.classList.remove('active');
  results.style.display = 'none';
}

function setButtonLoading(button, active, text) {
  if (!button) return;

  button.disabled = active;
  button.textContent = active ? 'Please wait...' : text;
}

async function parseResponse(response) {
  let data;

  try {
    data = await response.json();
  } catch {
    throw new Error('The server returned an invalid JSON response.');
  }

  if (!response.ok) {
    throw new Error(
      data.detail || data.error || `Request failed (${response.status})`
    );
  }

  return data;
}

/* =========================================
   RESPONSE NORMALIZATION
========================================= */

function getAnalysisData(data) {
  // Supports several common FastAPI response structures.
  return (
    data.analysis || data.scan || data.results || data.skin_analysis || data
  );
}

function getScores(data) {
  const analysis = getAnalysisData(data);

  return (
    analysis.scores || analysis.metrics || data.scores || data.metrics || {}
  );
}

function getOverlays(data) {
  const analysis = getAnalysisData(data);

  return (
    analysis.overlays ||
    analysis.heatmaps ||
    data.overlays ||
    data.heatmaps ||
    {}
  );
}

function getImageSource(value) {
  if (typeof value === 'string') {
    return value.startsWith('data:') ? value : `data:image/png;base64,${value}`;
  }

  if (value && typeof value === 'object') {
    return (
      value.data_url ||
      value.image_url ||
      value.url ||
      value.base64 ||
      value.data ||
      null
    );
  }

  return null;
}

/* =========================================
   IMAGE HELPERS
========================================= */

function loadImage(src) {
  return new Promise((resolve, reject) => {
    const img = new Image();

    img.onload = () => resolve(img);

    img.onerror = () => {
      reject(new Error('Could not load image.'));
    };

    img.src = src;
  });
}

function fileToDataURL(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();

    reader.onload = () => resolve(reader.result);

    reader.onerror = () => {
      reject(new Error('Could not read selected image.'));
    };

    reader.readAsDataURL(file);
  });
}

async function captureFrameBlob() {
  if (
    !video ||
    !video.videoWidth ||
    !video.videoHeight ||
    video.readyState < HTMLMediaElement.HAVE_CURRENT_DATA
  ) {
    throw new Error('Camera is not ready.');
  }

  const snapshot = document.createElement('canvas');

  snapshot.width = video.videoWidth;
  snapshot.height = video.videoHeight;

  const context = snapshot.getContext('2d');

  if (!context) {
    throw new Error('Could not create image snapshot.');
  }

  context.drawImage(video, 0, 0);

  return new Promise((resolve, reject) => {
    snapshot.toBlob(
      (blob) => {
        if (!blob) {
          reject(new Error('Failed to capture camera image.'));
          return;
        }

        resolve(blob);
      },
      'image/jpeg',
      0.92
    );
  });
}

/* =========================================
   API REQUESTS
========================================= */

async function scanUploadedImage(file) {
  const formData = new FormData();

  formData.append('image', file, file.name || 'upload.jpg');

  const response = await fetch(`${API_URL}/scan`, {
    method: 'POST',
    body: formData
  });

  return parseResponse(response);
}

async function scanAndRecommend(blob, questionnaire) {
  const formData = new FormData();

  formData.append('image', blob, 'camera-capture.jpg');

  formData.append('questionnaire_json', JSON.stringify(questionnaire));

  const response = await fetch(`${API_URL}/scan-and-recommend`, {
    method: 'POST',
    body: formData
  });

  return parseResponse(response);
}

/* =========================================
   DISPLAY SCORES
========================================= */

function displayScores(scores) {
  if (!scoresContainer) return;

  scoresContainer.replaceChildren();

  if (!scores || typeof scores !== 'object') {
    scoresContainer.textContent = 'No skin metrics available.';
    return;
  }

  const entries = Object.entries(scores).filter(([, value]) =>
    Number.isFinite(Number(value))
  );

  if (!entries.length) {
    scoresContainer.textContent = 'No skin metrics returned by the API.';
    return;
  }

  for (const [category, rawScore] of entries) {
    const score = Number(rawScore);

    // Keep normalized values within 0..1.
    const normalized = Math.max(0, Math.min(1, score));
    const percentage = normalized * 100;

    const card = document.createElement('article');
    card.className = 'score-card';

    const title = document.createElement('h3');
    title.textContent = category
      .replaceAll('_', ' ')
      .replace(/\b\w/g, (c) => c.toUpperCase());

    const value = document.createElement('div');
    value.className = 'score-value';
    value.textContent = `${percentage.toFixed(0)}%`;

    const progress = document.createElement('div');
    progress.className = 'score-progress';

    const track = document.createElement('div');
    track.className = 'score-progress-track';

    const bar = document.createElement('div');
    bar.className = 'score-progress-bar';

    bar.style.width = `${percentage}%`;

    track.appendChild(bar);
    progress.appendChild(track);

    card.append(title, value, progress);
    scoresContainer.appendChild(card);
  }
}

/* =========================================
   DISPLAY HEATMAPS
========================================= */

async function displayOverlays(overlays, baseImage = null) {
  if (!overlaysContainer) return;

  overlaysContainer.replaceChildren();

  if (!overlays || typeof overlays !== 'object') {
    overlaysContainer.textContent = 'No heatmaps available.';
    return;
  }

  const entries = Array.isArray(overlays)
    ? overlays.map((item, index) => [
        item.name || item.category || `Heatmap ${index + 1}`,
        item
      ])
    : Object.entries(overlays);

  if (!entries.length) {
    overlaysContainer.textContent = 'No heatmaps available.';
    return;
  }

  for (const [category, overlayData] of entries) {
    const card = document.createElement('article');
    card.className = 'overlay-card';

    const title = document.createElement('h4');
    title.textContent = String(category)
      .replaceAll('_', ' ')
      .replace(/\b\w/g, (c) => c.toUpperCase());

    const wrapper = document.createElement('div');
    wrapper.className = 'canvas-container';

    const output = document.createElement('canvas');
    output.className = 'heatmap-canvas';

    wrapper.appendChild(output);
    card.append(title, wrapper);

    overlaysContainer.appendChild(card);

    try {
      const src = getImageSource(overlayData);

      if (!src) {
        throw new Error('Invalid heatmap image data.');
      }

      const overlay = await loadImage(src);

      const width = baseImage?.naturalWidth || overlay.naturalWidth;

      const height = baseImage?.naturalHeight || overlay.naturalHeight;

      output.width = width;
      output.height = height;

      const context = output.getContext('2d');

      if (!context) {
        throw new Error('Canvas context unavailable.');
      }

      // Draw original image first.
      if (baseImage) {
        context.drawImage(baseImage, 0, 0, width, height);
      }

      // Draw heatmap on top.
      context.drawImage(overlay, 0, 0, width, height);
    } catch (err) {
      console.error(`Heatmap ${category}:`, err);

      const message = document.createElement('p');
      message.className = 'overlay-error';
      message.textContent = `Could not load ${category} heatmap.`;

      card.appendChild(message);
    }
  }
}

/* =========================================
   DISPLAY REPORT
========================================= */

function displayReport(data) {
  if (!report) return;

  const reportText =
    data.report_markdown ||
    data.report ||
    data.recommendation_report ||
    data.recommendations?.report ||
    '';

  report.textContent =
    typeof reportText === 'string'
      ? reportText
      : JSON.stringify(reportText, null, 2);

  report.style.display = reportText ? 'block' : 'none';
}

/* =========================================
   DISPLAY SCAN RESULTS
========================================= */

async function displayResults(data, baseImage = null) {
  console.log('API scan response:', data);

  if (!data || typeof data !== 'object') {
    throw new Error('Invalid scan response.');
  }

  const scores = getScores(data);
  const overlays = getOverlays(data);

  displayScores(scores);

  await displayOverlays(overlays, baseImage);

  // IMPORTANT: inline display:none must be overridden.
  showResults();

  results.scrollIntoView({
    behavior: 'smooth',
    block: 'start'
  });
}

/* =========================================
   FILE UPLOAD WORKFLOW
========================================= */

if (fileInput && scanBtn) {
  scanBtn.disabled = true;

  fileInput.addEventListener('change', async (event) => {
    const file = event.target.files?.[0];

    selectedFile = null;
    originalImage = null;

    scanBtn.disabled = true;

    hideError();

    if (!file) return;

    if (!['image/jpeg', 'image/png', 'image/webp'].includes(file.type)) {
      showError('Please select a JPEG, PNG or WebP image.');
      fileInput.value = '';
      return;
    }

    try {
      const dataURL = await fileToDataURL(file);
      originalImage = await loadImage(dataURL);

      selectedFile = file;
      scanBtn.disabled = false;
    } catch (err) {
      showError(err.message);
      fileInput.value = '';
    }
  });

  scanBtn.addEventListener('click', async () => {
    if (!selectedFile || scanning) return;

    scanning = true;

    hideError();
    hideResults();

    if (report) report.style.display = 'none';

    setLoading(true, 'Analyzing uploaded image...');
    setButtonLoading(scanBtn, true, 'Scan Image');

    try {
      const data = await scanUploadedImage(selectedFile);

      await displayResults(data, originalImage);

      // If the upload endpoint also returns recommendations,
      // display them.
      displayReport(data);
      displayTryOnButtons(data.tryon_presets || data.try_on_presets);
    } catch (err) {
      console.error('Upload scan error:', err);
      showError(err.message);
    } finally {
      scanning = false;

      setLoading(false);
      scanBtn.disabled = !selectedFile;
      scanBtn.textContent = 'Scan Image';
    }
  });
}

/* =========================================
   CAMERA INITIALIZATION
========================================= */

async function initializeCamera() {
  if (cameraReady) return;

  if (cameraInitializing) {
    throw new Error('Camera is already initializing.');
  }

  if (!navigator.mediaDevices?.getUserMedia) {
    throw new Error('Camera requires HTTPS or localhost.');
  }

  cameraInitializing = true;

  try {
    cameraStream = await navigator.mediaDevices.getUserMedia({
      audio: false,
      video: {
        facingMode: 'user',
        width: { ideal: 1280 },
        height: { ideal: 720 }
      }
    });

    video.srcObject = cameraStream;

    await video.play();

    if (stage) {
      stage.style.display = 'block';
    }

    if (!makeup) {
      makeup = new OpenMakeup({
        video,
        renderCanvas: canvas,

        assetsBaseUrl: 'https://cdn.jsdelivr.net/npm/open-makeup-sdk/assets',

        mediapipeBaseUrl:
          'https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh@0.4.1633559619'
      });

      await makeup.init();
    }

    cameraReady = true;

    console.log('Camera initialized.');
  } catch (err) {
    stopCamera();

    throw new Error(`Camera initialization failed: ${err.message}`);
  } finally {
    cameraInitializing = false;
  }
}

/* =========================================
   CAMERA CLEANUP
========================================= */

function stopCamera() {
  if (cameraStream) {
    cameraStream.getTracks().forEach((track) => track.stop());
    cameraStream = null;
  }

  if (video) {
    video.srcObject = null;
  }

  cameraReady = false;
}

/* =========================================
   CAMERA START BUTTON
========================================= */

cameraStartBtn?.addEventListener('click', async () => {
  hideError();

  cameraStartBtn.disabled = true;

  try {
    setLoading(true, 'Starting camera...');

    await initializeCamera();

    cameraStartBtn.textContent = 'Camera Ready';
  } catch (err) {
    showError(err.message);

    cameraStartBtn.disabled = false;
    cameraStartBtn.textContent = 'Start Camera';
  } finally {
    setLoading(false);
  }
});

/* =========================================
   QUESTIONNAIRE
========================================= */

function getQuestionnaire() {
  const consent = $('#consentGiven');

  if (!consent?.checked) {
    throw new Error('Please provide explicit consent before facial analysis.');
  }

  return {
    consent_given: true,
    skin_type: 'combination',
    known_allergens: ['fragrance', 'parfum', 'linalool'],
    pregnant_or_nursing: false,
    target_category: 'face-creams'
  };
}

/* =========================================
   VIRTUAL TRY-ON
========================================= */

function displayTryOnButtons(presets) {
  if (!tryonContainer) return;

  tryonContainer.replaceChildren();

  if (!Array.isArray(presets)) return;

  for (const preset of presets) {
    if (
      !preset ||
      typeof preset.label !== 'string' ||
      typeof preset.category !== 'string' ||
      !preset.opts ||
      typeof preset.opts !== 'object'
    ) {
      continue;
    }

    const button = document.createElement('button');

    button.type = 'button';
    button.textContent = `Try On ${preset.label}`;

    button.addEventListener('click', async () => {
      if (!makeup || !cameraReady) {
        showError('Start the camera before using virtual try-on.');
        return;
      }

      try {
        await makeup.apply(preset.category, preset.opts);
      } catch (err) {
        showError(`Virtual try-on failed: ${err.message}`);
      }
    });

    tryonContainer.appendChild(button);
  }
}

/* =========================================
   CAMERA SCAN WORKFLOW
========================================= */

cameraScanBtn?.addEventListener('click', async () => {
  if (scanning) return;

  scanning = true;

  hideError();
  hideResults();

  if (report) report.style.display = 'none';

  setButtonLoading(cameraScanBtn, true, 'Analyze Skin & Recommend');

  try {
    if (!cameraReady) {
      await initializeCamera();
    }

    const questionnaire = getQuestionnaire();

    setLoading(true, 'Capturing image...');

    if (makeup) {
      makeup.clearAll();
    }

    const blob = await captureFrameBlob();

    // Retain original camera capture for heatmap rendering.
    const captureURL = URL.createObjectURL(blob);

    let capturedImage;

    try {
      capturedImage = await loadImage(captureURL);
    } finally {
      // Keep the object URL alive until image is loaded.
      URL.revokeObjectURL(captureURL);
    }

    setLoading(true, 'Analyzing skin and generating recommendations...');

    const data = await scanAndRecommend(blob, questionnaire);

    await displayResults(data, capturedImage);

    displayReport(data);

    displayTryOnButtons(
      data.tryon_presets || data.try_on_presets || data.presets
    );

    console.log('Skin analysis complete:', data);
  } catch (err) {
    console.error('Camera scan error:', err);
    showError(err.message);
  } finally {
    scanning = false;

    setButtonLoading(cameraScanBtn, false, 'Analyze Skin & Recommend');

    setLoading(false);
  }
});

/* =========================================
   CLEANUP
========================================= */

window.addEventListener('pagehide', stopCamera);

console.log('Skin Scan frontend initialized.');
