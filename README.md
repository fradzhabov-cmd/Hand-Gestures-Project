# MediaPipe Hand Gesture Camera Filters

Mobile-first browser app that opens the front-facing selfie camera full-screen,
tracks a white dot on the index fingertip with MediaPipe Hands, and maps hand
gestures to live visual filters on the camera feed.

## Features

- Full-screen smartphone camera view using `getUserMedia` with `facingMode: user`
- MediaPipe Hand Landmarker detection from the 21 hand landmarks
- White fingertip dot tracking landmark 8, the index finger tip
- Gesture recognition for:
  - Fist -> dither filter
  - Peach sign -> VHS filter with chromatic aberration
  - Pointing index finger -> spotlight filter
  - Open hand -> water ripple filter
- Mobile safe-area aware controls and heads-up gesture/filter display
- Drawing tab with persistent fingertip strokes, green default ink, color choices, manual clear, local two-finger erasing, and whiteboard mode
- Runtime tracker diagnostics that show camera/model loading, no-hand state, detected fingers, or model errors
- High-sensitivity tracking with an enhanced offscreen camera frame and brief landmark lock to reduce first-try flicker

## Run locally

This app is static HTML/CSS/JavaScript. Serve the repository from a local web
server:

```bash
npm start
```

Open the printed local URL on a phone or browser. Camera APIs require HTTPS in
production; `localhost` is allowed for local development.

## Drawing mode

Open the **Drawing** tab after starting the camera. Point with one index
finger to draw persistent strokes over the live selfie feed. Green is the
default ink color, and you can pick another color from the palette.

Use two spread fingers as a local eraser: move the two fingertips over the
area you want to remove. Tap **Clear** to wipe all drawing. Toggle
**Whiteboard** to draw on a blank whiteboard while the camera continues to
track your finger in the background.

## Tests

```bash
npm test
```

The tests cover the pure 21-landmark gesture classifier used by the camera app.

## Files

- `index.html` - app entry point
- `src/app.js` - camera, MediaPipe, canvas, and visual filter pipeline
- `src/gestureClassifier.js` - reusable landmark gesture classifier
- `src/styles.css` - full-screen mobile UI
- `tests/gestureClassifier.test.js` - classifier tests

## License

MIT License.
Hand Gestures Recognition 🤖✋

AI-powered real-time hand gesture recognition system using Python, OpenCV, and MediaPipe. The project detects and tracks hand movements through a webcam for touchless control, automation, gaming, accessibility tools, and interactive computer vision applications.


---

🚀 Features

Real-time hand tracking

Gesture recognition using AI & Computer Vision

Webcam input support

Lightweight and fast performance

Easy to customize and extend

Beginner-friendly project structure



---

🛠️ Technologies Used

Python

OpenCV

MediaPipe

NumPy



---

📦 Installation

Clone the repository:

git clone https://github.com/yourusername/hand-gestures.git
cd hand-gestures

Install dependencies:

pip install -r requirements.txt


---

▶️ Run the Project

python main.py


---

📌 Use Cases

Touchless PC control

AI interaction systems

Gesture-based gaming

Accessibility tools

Smart automation



---

🔮 Future Improvements

Custom gesture training

Deep learning integration

Multi-hand support

Virtual mouse & keyboard controls



---

🤝 Contributing

Contributions and ideas are welcome.


---

📄 License

MIT License.