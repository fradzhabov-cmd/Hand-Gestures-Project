import {
  analyzeHandGesture,
  Gesture,
  GESTURE_FILTERS,
  GESTURE_LABELS
} from "./gestureClassifier.js";

const MEDIAPIPE_TASKS_URL =
  "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/+esm";
const HAND_MODEL_URL =
  "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task";
const WASM_URL = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm";
const MAX_DEVICE_PIXEL_RATIO = 1.5;
const TRACKER_LOAD_TIMEOUT_MS = 20000;
const CLEAR_GESTURE_COOLDOWN_MS = 1200;
const BAYER_MATRIX = [
  [0, 8, 2, 10],
  [12, 4, 14, 6],
  [3, 11, 1, 9],
  [15, 7, 13, 5]
];
const HAND_CONNECTIONS = [
  [0, 1],
  [1, 2],
  [2, 3],
  [3, 4],
  [0, 5],
  [5, 6],
  [6, 7],
  [7, 8],
  [5, 9],
  [9, 10],
  [10, 11],
  [11, 12],
  [9, 13],
  [13, 14],
  [14, 15],
  [15, 16],
  [13, 17],
  [17, 18],
  [18, 19],
  [19, 20],
  [0, 17]
];

const video = document.querySelector("#camera");
const canvas = document.querySelector("#feed");
const drawingCanvas = document.querySelector("#drawing-layer");
const ctx = canvas.getContext("2d", { alpha: false, willReadFrequently: true });
const drawingCtx = drawingCanvas.getContext("2d");
const startButton = document.querySelector("#start-camera");
const permissionPanel = document.querySelector("#permission-panel");
const statusText = document.querySelector("#status");
const gestureName = document.querySelector("#gesture-name");
const filterName = document.querySelector("#filter-name");
const trackingStatus = document.querySelector("#tracking-status");
const modeTabs = document.querySelectorAll(".mode-tab");
const drawControls = document.querySelector("#draw-controls");
const colorChips = document.querySelectorAll(".color-chip");
const clearDrawingButton = document.querySelector("#clear-drawing");
const scratchCanvas = document.createElement("canvas");
const scratchCtx = scratchCanvas.getContext("2d", { alpha: false });

let handLandmarker;
let trackerState = "idle";
let running = false;
let latestLandmarks = null;
let latestAnalysis = null;
let activeGesture = Gesture.UNKNOWN;
let candidateGesture = Gesture.UNKNOWN;
let candidateFrames = 0;
let lastVideoTime = -1;
let activeMode = "filters";
let selectedColor = "#ffffff";
let lastDrawPoint = null;
let lastClearGestureAt = 0;
let animationStarted = false;

startButton.addEventListener("click", startExperience);
window.addEventListener("resize", resizeCanvases);
window.addEventListener("orientationchange", resizeCanvases);
clearDrawingButton.addEventListener("click", clearDrawing);

for (const tab of modeTabs) {
  tab.addEventListener("click", () => setMode(tab.dataset.mode));
}

for (const chip of colorChips) {
  chip.addEventListener("click", () => setDrawingColor(chip.dataset.color));
}

resizeCanvases();
updateHud(Gesture.UNKNOWN);
updateTrackingStatus("App v3 ready. Tap Start selfie camera.");

async function startExperience() {
  startButton.disabled = true;
  setStatus("Opening selfie camera...");
  updateTrackingStatus("Opening selfie camera...");

  try {
    await enterFullscreen();
    await startCamera();

    running = true;
    permissionPanel.classList.add("is-hidden");
    updateTrackingStatus("Camera is on. Loading MediaPipe hand tracker...");
    setStatus("Camera started.");
    startAnimationLoop();
    loadHandTracker();
  } catch (error) {
    console.error(error);
    startButton.disabled = false;
    const message = error.message || "Could not start the camera.";
    setStatus(message);
    updateTrackingStatus(message);
  }
}

async function enterFullscreen() {
  if (!document.fullscreenEnabled || document.fullscreenElement) {
    return;
  }

  try {
    await document.documentElement.requestFullscreen();
  } catch {
    // CSS still keeps the app viewport-sized where the Fullscreen API is absent.
  }
}

async function startCamera() {
  if (!navigator.mediaDevices?.getUserMedia) {
    throw new Error("This browser does not support camera capture.");
  }

  const stream = await navigator.mediaDevices.getUserMedia({
    audio: false,
    video: {
      facingMode: { ideal: "user" },
      width: { ideal: 640 },
      height: { ideal: 480 }
    }
  });

  video.srcObject = stream;
  await waitForMetadata(video);
  await video.play();
}

function startAnimationLoop() {
  if (animationStarted) {
    return;
  }

  animationStarted = true;
  requestAnimationFrame(tick);
}

async function loadHandTracker() {
  if (trackerState === "loading" || trackerState === "ready") {
    return;
  }

  trackerState = "loading";
  updateTrackingStatus("Loading MediaPipe hand tracker...");

  try {
    handLandmarker = await withTimeout(
      createHandLandmarker(),
      TRACKER_LOAD_TIMEOUT_MS,
      "MediaPipe hand tracker took too long to load. Check your network/CDN access."
    );
    trackerState = "ready";
    updateTrackingStatus("Tracker ready. Show one hand in the frame.");
  } catch (error) {
    console.error(error);
    trackerState = "error";
    updateTrackingStatus(error.message || "MediaPipe hand tracker failed to load.");
  }
}

async function createHandLandmarker() {
  const { FilesetResolver, HandLandmarker } = await import(MEDIAPIPE_TASKS_URL);
  const vision = await FilesetResolver.forVisionTasks(WASM_URL);

  try {
    return await createHandLandmarkerWithDelegate(HandLandmarker, vision, "GPU");
  } catch (error) {
    console.warn("GPU hand tracking failed; retrying with CPU delegate.", error);
    return createHandLandmarkerWithDelegate(HandLandmarker, vision, "CPU");
  }
}

function createHandLandmarkerWithDelegate(HandLandmarker, vision, delegate) {
  return HandLandmarker.createFromOptions(vision, {
    baseOptions: {
      modelAssetPath: HAND_MODEL_URL,
      delegate
    },
    numHands: 1,
    runningMode: "VIDEO",
    minHandDetectionConfidence: 0.2,
    minHandPresenceConfidence: 0.2,
    minTrackingConfidence: 0.2
  });
}

function waitForMetadata(mediaElement) {
  if (mediaElement.readyState >= 1) {
    return Promise.resolve();
  }

  return new Promise((resolve) => {
    mediaElement.addEventListener("loadedmetadata", resolve, { once: true });
  });
}

function tick(now) {
  if (!running) {
    return;
  }

  resizeCanvases();
  updateLandmarks(now);
  drawScene(now);
  requestAnimationFrame(tick);
}

function updateLandmarks(now) {
  if (trackerState === "loading") {
    return;
  }

  if (trackerState === "error") {
    latestLandmarks = null;
    latestAnalysis = null;
    activeGesture = Gesture.UNKNOWN;
    updateHud(activeGesture);
    lastDrawPoint = null;
    return;
  }

  if (!handLandmarker || !video.videoWidth || video.currentTime === lastVideoTime) {
    return;
  }

  lastVideoTime = video.currentTime;

  let results;
  try {
    results = handLandmarker.detectForVideo(video, now);
  } catch (error) {
    console.error("Hand tracking failed.", error);
    latestLandmarks = null;
    latestAnalysis = null;
    activeGesture = Gesture.UNKNOWN;
    updateHud(activeGesture);
    updateTrackingStatus("Hand tracking error. Refresh and try again.");
    lastDrawPoint = null;
    return;
  }

  latestLandmarks = results.landmarks?.[0] ?? null;

  if (!latestLandmarks) {
    latestAnalysis = null;
    activeGesture = stabilizeGesture(Gesture.UNKNOWN);
    updateHud(activeGesture);
    updateTrackingStatus("No hand detected. Hold one hand clearly in frame.");
    lastDrawPoint = null;
    return;
  }

  latestAnalysis = analyzeHandGesture(latestLandmarks);
  activeGesture = stabilizeGesture(latestAnalysis.gesture);
  updateHud(activeGesture);
  updateTrackingStatus(getTrackingMessage(latestAnalysis));
}

function stabilizeGesture(nextGesture) {
  if (nextGesture === Gesture.UNKNOWN) {
    candidateGesture = Gesture.UNKNOWN;
    candidateFrames = 0;
    return Gesture.UNKNOWN;
  }

  if (nextGesture === candidateGesture) {
    candidateFrames += 1;
  } else {
    candidateGesture = nextGesture;
    candidateFrames = 1;
  }

  if (candidateFrames >= 2 || activeGesture === Gesture.UNKNOWN) {
    return nextGesture;
  }

  return activeGesture;
}

function drawScene(now) {
  if (!video.videoWidth || !canvas.width || !canvas.height) {
    return;
  }

  const coverRect = getCoverRect(
    video.videoWidth,
    video.videoHeight,
    canvas.width,
    canvas.height
  );

  drawMirroredVideo(coverRect);

  const indexTip = latestLandmarks
    ? landmarkToCanvasPoint(latestLandmarks[8], coverRect)
    : null;
  const palmCenter = latestLandmarks
    ? landmarkToCanvasPoint(latestLandmarks[9], coverRect)
    : null;

  if (activeMode === "filters") {
    applyGestureFilter(activeGesture, indexTip, palmCenter, now);
  }

  if (latestLandmarks) {
    drawHandSkeleton(latestLandmarks, coverRect);
  }

  if (activeMode === "drawing" && latestAnalysis && latestLandmarks && indexTip) {
    updateDrawing(indexTip, latestAnalysis, latestLandmarks);
  }

  if (indexTip) {
    drawIndexDot(indexTip);
  }
}

function drawMirroredVideo(rect) {
  ctx.save();
  ctx.setTransform(-1, 0, 0, 1, canvas.width, 0);
  ctx.drawImage(
    video,
    canvas.width - rect.x - rect.width,
    rect.y,
    rect.width,
    rect.height
  );
  ctx.restore();
}

function applyGestureFilter(gesture, indexTip, palmCenter, now) {
  if (gesture === Gesture.FIST) {
    applyDither();
    return;
  }

  if (gesture === Gesture.PEACH_SIGN) {
    applyVhs(now);
    return;
  }

  if (gesture === Gesture.POINTING_FINGER && indexTip) {
    applySpotlight(indexTip);
    return;
  }

  if (gesture === Gesture.OPEN_HAND && palmCenter) {
    applyWaterRipple(palmCenter, now);
  }
}

function applyDither() {
  const image = ctx.getImageData(0, 0, canvas.width, canvas.height);
  const { data, width, height } = image;

  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      const index = (y * width + x) * 4;
      const luminance =
        data[index] * 0.299 + data[index + 1] * 0.587 + data[index + 2] * 0.114;
      const threshold = (BAYER_MATRIX[y % 4][x % 4] + 0.5) * 16;
      const value = luminance > threshold ? 255 : 12;

      data[index] = value;
      data[index + 1] = value;
      data[index + 2] = value;
    }
  }

  ctx.putImageData(image, 0, 0);
}

function applyVhs(now) {
  syncScratchCanvas();
  scratchCtx.drawImage(canvas, 0, 0);

  const jitter = Math.sin(now / 90) * 4;

  ctx.save();
  ctx.globalCompositeOperation = "screen";
  ctx.globalAlpha = 0.24;
  ctx.filter = "sepia(1) saturate(8) hue-rotate(-45deg)";
  ctx.drawImage(scratchCanvas, -5 + jitter, 0);
  ctx.filter = "sepia(1) saturate(8) hue-rotate(170deg)";
  ctx.drawImage(scratchCanvas, 5 + jitter, 0);
  ctx.restore();

  ctx.save();
  ctx.globalAlpha = 0.14;
  ctx.fillStyle = "#000";
  const lineHeight = Math.max(3, Math.round(canvas.height / 260));
  for (let y = 0; y < canvas.height; y += lineHeight * 2) {
    ctx.fillRect(0, y, canvas.width, lineHeight);
  }

  ctx.globalAlpha = 0.18;
  ctx.fillStyle = "#fff";
  const tearY = (Math.sin(now / 420) * 0.5 + 0.5) * canvas.height;
  ctx.fillRect(0, tearY, canvas.width, Math.max(1, canvas.height * 0.003));
  ctx.restore();

  drawRecBadge();
}

function applySpotlight(point) {
  const radius = Math.min(canvas.width, canvas.height) * 0.28;

  ctx.save();
  ctx.fillStyle = "rgba(0, 0, 0, 0.72)";
  ctx.fillRect(0, 0, canvas.width, canvas.height);

  ctx.globalCompositeOperation = "destination-out";
  const gradient = ctx.createRadialGradient(
    point.x,
    point.y,
    radius * 0.12,
    point.x,
    point.y,
    radius
  );
  gradient.addColorStop(0, "rgba(0, 0, 0, 1)");
  gradient.addColorStop(0.62, "rgba(0, 0, 0, 0.72)");
  gradient.addColorStop(1, "rgba(0, 0, 0, 0)");
  ctx.fillStyle = gradient;
  ctx.beginPath();
  ctx.arc(point.x, point.y, radius, 0, Math.PI * 2);
  ctx.fill();
  ctx.restore();
}

function applyWaterRipple(center, now) {
  const radius = Math.min(canvas.width, canvas.height) * 0.24;
  const x0 = Math.max(0, Math.floor(center.x - radius));
  const y0 = Math.max(0, Math.floor(center.y - radius));
  const x1 = Math.min(canvas.width, Math.ceil(center.x + radius));
  const y1 = Math.min(canvas.height, Math.ceil(center.y + radius));
  const width = x1 - x0;
  const height = y1 - y0;

  if (width <= 1 || height <= 1) {
    return;
  }

  const source = ctx.getImageData(x0, y0, width, height);
  const output = ctx.createImageData(width, height);
  const localCenterX = center.x - x0;
  const localCenterY = center.y - y0;
  const wavePhase = now / 95;

  output.data.set(source.data);

  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      const dx = x - localCenterX;
      const dy = y - localCenterY;
      const distanceFromCenter = Math.hypot(dx, dy);

      if (distanceFromCenter > radius || distanceFromCenter === 0) {
        continue;
      }

      const falloff = 1 - distanceFromCenter / radius;
      const offset =
        Math.sin(distanceFromCenter / 9 - wavePhase) * 10 * falloff;
      const sampleX = clamp(
        Math.round(x + (dx / distanceFromCenter) * offset),
        0,
        width - 1
      );
      const sampleY = clamp(
        Math.round(y + (dy / distanceFromCenter) * offset),
        0,
        height - 1
      );
      const targetIndex = (y * width + x) * 4;
      const sampleIndex = (sampleY * width + sampleX) * 4;

      output.data[targetIndex] = source.data[sampleIndex];
      output.data[targetIndex + 1] = source.data[sampleIndex + 1];
      output.data[targetIndex + 2] = source.data[sampleIndex + 2];
      output.data[targetIndex + 3] = source.data[sampleIndex + 3];
    }
  }

  ctx.putImageData(output, x0, y0);
  ctx.save();
  ctx.strokeStyle = "rgba(190, 235, 255, 0.5)";
  ctx.lineWidth = Math.max(2, canvas.width * 0.003);
  for (let i = 0; i < 3; i += 1) {
    const ringRadius =
      ((now / 12 + i * radius * 0.28) % radius) * 0.85 + radius * 0.08;
    ctx.globalAlpha = 1 - ringRadius / radius;
    ctx.beginPath();
    ctx.arc(center.x, center.y, ringRadius, 0, Math.PI * 2);
    ctx.stroke();
  }
  ctx.restore();
}

function updateDrawing(indexTip, analysis, landmarks) {
  if (isClearGesture(analysis, landmarks)) {
    const now = Date.now();
    if (now - lastClearGestureAt > CLEAR_GESTURE_COOLDOWN_MS) {
      clearDrawing();
      lastClearGestureAt = now;
      updateTrackingStatus("Two-finger clear gesture detected. Drawing cleared.");
    }
    lastDrawPoint = null;
    return;
  }

  const canDraw =
    analysis.fingerStates.index &&
    !analysis.fingerStates.middle &&
    !analysis.fingerStates.ring &&
    !analysis.fingerStates.pinky;

  if (!canDraw) {
    lastDrawPoint = null;
    return;
  }

  drawingCtx.save();
  drawingCtx.strokeStyle = selectedColor;
  drawingCtx.lineWidth = Math.max(7, Math.min(drawingCanvas.width, drawingCanvas.height) * 0.012);
  drawingCtx.lineCap = "round";
  drawingCtx.lineJoin = "round";
  drawingCtx.shadowColor = selectedColor;
  drawingCtx.shadowBlur = drawingCtx.lineWidth * 0.45;
  drawingCtx.beginPath();

  if (lastDrawPoint && distanceBetweenPoints(lastDrawPoint, indexTip) < drawingCanvas.width * 0.18) {
    drawingCtx.moveTo(lastDrawPoint.x, lastDrawPoint.y);
  } else {
    drawingCtx.moveTo(indexTip.x, indexTip.y);
  }

  drawingCtx.lineTo(indexTip.x, indexTip.y);
  drawingCtx.stroke();
  drawingCtx.restore();

  lastDrawPoint = indexTip;
}

function isClearGesture(analysis, landmarks) {
  if (
    !analysis.fingerStates.index ||
    !analysis.fingerStates.middle ||
    analysis.fingerStates.ring ||
    analysis.fingerStates.pinky
  ) {
    return false;
  }

  const palmSize = Math.max(distanceBetweenLandmarks(landmarks[0], landmarks[9]), 0.0001);
  const horizontalSpread = Math.abs(landmarks[8].x - landmarks[12].x);
  const verticalDifference = Math.abs(landmarks[8].y - landmarks[12].y);

  return horizontalSpread > palmSize * 0.24 && verticalDifference < palmSize * 0.55;
}

function clearDrawing() {
  drawingCtx.clearRect(0, 0, drawingCanvas.width, drawingCanvas.height);
  lastDrawPoint = null;
}

function drawHandSkeleton(landmarks, rect) {
  const points = landmarks.map((landmark) => landmarkToCanvasPoint(landmark, rect));
  const lineWidth = Math.max(2, Math.min(canvas.width, canvas.height) * 0.004);
  const pointRadius = Math.max(2.5, Math.min(canvas.width, canvas.height) * 0.0045);

  ctx.save();
  ctx.lineCap = "round";
  ctx.lineJoin = "round";
  ctx.lineWidth = lineWidth;
  ctx.strokeStyle = "rgba(65, 245, 255, 0.8)";
  ctx.shadowColor = "rgba(0, 0, 0, 0.5)";
  ctx.shadowBlur = lineWidth * 1.5;

  for (const [startIndex, endIndex] of HAND_CONNECTIONS) {
    const start = points[startIndex];
    const end = points[endIndex];
    ctx.beginPath();
    ctx.moveTo(start.x, start.y);
    ctx.lineTo(end.x, end.y);
    ctx.stroke();
  }

  ctx.shadowBlur = 0;
  ctx.fillStyle = "rgba(255, 255, 255, 0.92)";
  for (const point of points) {
    ctx.beginPath();
    ctx.arc(point.x, point.y, pointRadius, 0, Math.PI * 2);
    ctx.fill();
  }

  ctx.restore();
}

function drawIndexDot(point) {
  const dotRadius = Math.max(7, Math.min(canvas.width, canvas.height) * 0.012);

  ctx.save();
  ctx.shadowColor = "rgba(255, 255, 255, 0.95)";
  ctx.shadowBlur = dotRadius * 2.5;
  ctx.fillStyle = "#fff";
  ctx.beginPath();
  ctx.arc(point.x, point.y, dotRadius, 0, Math.PI * 2);
  ctx.fill();
  ctx.strokeStyle = "rgba(0, 0, 0, 0.45)";
  ctx.lineWidth = Math.max(2, dotRadius * 0.22);
  ctx.stroke();
  ctx.restore();
}

function drawRecBadge() {
  const padding = Math.max(14, canvas.width * 0.018);
  const radius = Math.max(5, canvas.width * 0.006);

  ctx.save();
  ctx.font = `${Math.max(16, canvas.width * 0.022)}px monospace`;
  ctx.fillStyle = "rgba(10, 10, 10, 0.54)";
  ctx.fillRect(padding, padding, padding * 5.1, padding * 2.1);
  ctx.fillStyle = "#ff2b2b";
  ctx.beginPath();
  ctx.arc(padding * 1.7, padding * 2.05, radius, 0, Math.PI * 2);
  ctx.fill();
  ctx.fillStyle = "#fff";
  ctx.fillText("VHS", padding * 2.35, padding * 2.25);
  ctx.restore();
}

function landmarkToCanvasPoint(landmark, rect) {
  return {
    x: rect.x + (1 - landmark.x) * rect.width,
    y: rect.y + landmark.y * rect.height
  };
}

function getCoverRect(sourceWidth, sourceHeight, targetWidth, targetHeight) {
  const scale = Math.max(targetWidth / sourceWidth, targetHeight / sourceHeight);
  const width = sourceWidth * scale;
  const height = sourceHeight * scale;

  return {
    x: (targetWidth - width) / 2,
    y: (targetHeight - height) / 2,
    width,
    height
  };
}

function resizeCanvases() {
  const bounds = canvas.getBoundingClientRect();
  const ratio = Math.min(window.devicePixelRatio || 1, MAX_DEVICE_PIXEL_RATIO);
  const width = Math.max(1, Math.round(bounds.width * ratio));
  const height = Math.max(1, Math.round(bounds.height * ratio));

  if (canvas.width !== width || canvas.height !== height) {
    canvas.width = width;
    canvas.height = height;
  }

  resizeDrawingCanvas(width, height);
}

function resizeDrawingCanvas(width, height) {
  if (drawingCanvas.width === width && drawingCanvas.height === height) {
    return;
  }

  const previous = document.createElement("canvas");
  previous.width = drawingCanvas.width;
  previous.height = drawingCanvas.height;

  if (previous.width && previous.height) {
    previous.getContext("2d").drawImage(drawingCanvas, 0, 0);
  }

  drawingCanvas.width = width;
  drawingCanvas.height = height;

  if (previous.width && previous.height) {
    drawingCtx.drawImage(previous, 0, 0, width, height);
  }
}

function syncScratchCanvas() {
  if (scratchCanvas.width !== canvas.width || scratchCanvas.height !== canvas.height) {
    scratchCanvas.width = canvas.width;
    scratchCanvas.height = canvas.height;
  }
}

function setMode(mode) {
  activeMode = mode === "drawing" ? "drawing" : "filters";
  lastDrawPoint = null;

  for (const tab of modeTabs) {
    tab.classList.toggle("is-active", tab.dataset.mode === activeMode);
  }

  drawControls.hidden = activeMode !== "drawing";
  updateHud(activeGesture);
  updateTrackingStatus(
    activeMode === "drawing"
      ? "Drawing mode. Point one index finger to draw; show two spread fingers to clear."
      : "Filter mode. Use fist, peace, pointing, or open hand."
  );
}

function setDrawingColor(color) {
  selectedColor = color || "#ffffff";

  for (const chip of colorChips) {
    chip.classList.toggle("is-active", chip.dataset.color === selectedColor);
  }

  updateHud(activeGesture);
}

function updateHud(gesture) {
  gestureName.textContent = GESTURE_LABELS[gesture] ?? GESTURE_LABELS.unknown;

  if (activeMode === "drawing") {
    filterName.textContent = `Drawing ${selectedColor}`;
    return;
  }

  filterName.textContent = GESTURE_FILTERS[gesture] ?? GESTURE_FILTERS.unknown;
}

function getTrackingMessage(analysis) {
  const fingers = Object.entries(analysis.fingerStates)
    .filter(([, extended]) => extended)
    .map(([finger]) => finger)
    .join(", ");
  const fingerSummary = fingers || "no extended fingers";

  if (activeMode === "drawing") {
    if (isClearGesture(analysis, latestLandmarks)) {
      return "Two spread fingers detected. Drawing will clear.";
    }

    if (analysis.fingerStates.index && !analysis.fingerStates.middle) {
      return `Drawing with ${selectedColor}. Extended: ${fingerSummary}.`;
    }
  }

  if (analysis.gesture === Gesture.UNKNOWN) {
    return `Hand detected. Extended: ${fingerSummary}. Adjust your gesture.`;
  }

  return `${GESTURE_LABELS[analysis.gesture]} detected. Extended: ${fingerSummary}.`;
}

function updateTrackingStatus(message) {
  trackingStatus.textContent = message;
}

function setStatus(message) {
  statusText.textContent = message;
}

function withTimeout(promise, timeoutMs, message) {
  let timeoutId;
  const timeout = new Promise((_, reject) => {
    timeoutId = window.setTimeout(() => reject(new Error(message)), timeoutMs);
  });

  return Promise.race([promise, timeout]).finally(() => window.clearTimeout(timeoutId));
}

function clamp(value, min, max) {
  return Math.min(Math.max(value, min), max);
}

function distanceBetweenPoints(a, b) {
  return Math.hypot(a.x - b.x, a.y - b.y);
}

function distanceBetweenLandmarks(a, b) {
  const dx = (a.x ?? 0) - (b.x ?? 0);
  const dy = (a.y ?? 0) - (b.y ?? 0);
  const dz = (a.z ?? 0) - (b.z ?? 0);
  return Math.hypot(dx, dy, dz);
}
