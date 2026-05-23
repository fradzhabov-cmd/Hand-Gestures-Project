import assert from "node:assert/strict";
import test from "node:test";

import {
  analyzeHandGesture,
  classifyHandGesture,
  Gesture,
  getFingerStates
} from "../src/gestureClassifier.js";

test("classifies a fist from folded fingers", () => {
  const landmarks = makeHand({ gesture: Gesture.FIST });

  assert.equal(classifyHandGesture(landmarks), Gesture.FIST);
  assert.deepEqual(getFingerStates(landmarks), {
    thumb: false,
    index: false,
    middle: false,
    ring: false,
    pinky: false
  });
});

test("classifies a peach sign from separated index and middle fingers", () => {
  const landmarks = makeHand({ gesture: Gesture.PEACH_SIGN });
  const result = analyzeHandGesture(landmarks);

  assert.equal(result.gesture, Gesture.PEACH_SIGN);
  assert.equal(result.fingerStates.index, true);
  assert.equal(result.fingerStates.middle, true);
  assert.equal(result.fingerStates.ring, false);
  assert.equal(result.fingerStates.pinky, false);
});

test("classifies a pointing index finger", () => {
  const landmarks = makeHand({ gesture: Gesture.POINTING_FINGER });

  assert.equal(classifyHandGesture(landmarks), Gesture.POINTING_FINGER);
});

test("classifies an open hand when all fingers are extended", () => {
  const landmarks = makeHand({ gesture: Gesture.OPEN_HAND });

  assert.equal(classifyHandGesture(landmarks), Gesture.OPEN_HAND);
});

test("returns unknown for invalid landmark input", () => {
  const result = analyzeHandGesture([]);

  assert.equal(result.gesture, Gesture.UNKNOWN);
  assert.equal(result.confidence, 0);
});

function makeHand({ gesture }) {
  const landmarks = Array.from({ length: 21 }, () => point(0, 0));
  landmarks[0] = point(0.5, 0.9);

  const open = gesture === Gesture.OPEN_HAND;
  const peach = gesture === Gesture.PEACH_SIGN;
  const pointing = gesture === Gesture.POINTING_FINGER;

  setThumb(landmarks, open);
  setFinger(landmarks, [5, 6, 7, 8], 0.42, open || peach || pointing);
  setFinger(landmarks, [9, 10, 11, 12], 0.5, open || peach);
  setFinger(landmarks, [13, 14, 15, 16], 0.58, open);
  setFinger(landmarks, [17, 18, 19, 20], 0.66, open);

  if (peach) {
    landmarks[8] = point(0.36, 0.2);
    landmarks[12] = point(0.64, 0.2);
  }

  return landmarks;
}

function setThumb(landmarks, extended) {
  if (extended) {
    landmarks[1] = point(0.44, 0.8);
    landmarks[2] = point(0.35, 0.7);
    landmarks[3] = point(0.26, 0.62);
    landmarks[4] = point(0.17, 0.55);
    return;
  }

  landmarks[1] = point(0.44, 0.8);
  landmarks[2] = point(0.47, 0.73);
  landmarks[3] = point(0.5, 0.7);
  landmarks[4] = point(0.47, 0.68);
}

function setFinger(landmarks, indices, x, extended) {
  const [mcp, pip, dip, tip] = indices;
  landmarks[mcp] = point(x, 0.65);

  if (extended) {
    landmarks[pip] = point(x, 0.5);
    landmarks[dip] = point(x, 0.35);
    landmarks[tip] = point(x, 0.2);
    return;
  }

  landmarks[pip] = point(x, 0.72);
  landmarks[dip] = point(x + 0.03, 0.78);
  landmarks[tip] = point(x + 0.05, 0.73);
}

function point(x, y, z = 0) {
  return { x, y, z };
}
