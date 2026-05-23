export const Gesture = Object.freeze({
  FIST: "fist",
  PEACH_SIGN: "peach-sign",
  POINTING_FINGER: "pointing-finger",
  OPEN_HAND: "open-hand",
  UNKNOWN: "unknown"
});

const FINGERS = Object.freeze({
  thumb: [1, 2, 3, 4],
  index: [5, 6, 7, 8],
  middle: [9, 10, 11, 12],
  ring: [13, 14, 15, 16],
  pinky: [17, 18, 19, 20]
});

const DEFAULT_STATES = Object.freeze({
  thumb: false,
  index: false,
  middle: false,
  ring: false,
  pinky: false
});

export const GESTURE_LABELS = Object.freeze({
  [Gesture.FIST]: "Fist",
  [Gesture.PEACH_SIGN]: "Peach sign",
  [Gesture.POINTING_FINGER]: "Pointing finger",
  [Gesture.OPEN_HAND]: "Open hand",
  [Gesture.UNKNOWN]: "Unknown"
});

export const GESTURE_FILTERS = Object.freeze({
  [Gesture.FIST]: "Dither",
  [Gesture.PEACH_SIGN]: "VHS chromatic aberration",
  [Gesture.POINTING_FINGER]: "Spotlight",
  [Gesture.OPEN_HAND]: "Water ripple",
  [Gesture.UNKNOWN]: "None"
});

export function classifyHandGesture(landmarks) {
  return analyzeHandGesture(landmarks).gesture;
}

export function analyzeHandGesture(landmarks) {
  if (!Array.isArray(landmarks) || landmarks.length < 21) {
    return {
      gesture: Gesture.UNKNOWN,
      fingerStates: { ...DEFAULT_STATES },
      confidence: 0
    };
  }

  const fingerStates = getFingerStates(landmarks);

  if (
    fingerStates.index &&
    fingerStates.middle &&
    fingerStates.ring &&
    fingerStates.pinky
  ) {
    return {
      gesture: Gesture.OPEN_HAND,
      fingerStates,
      confidence: 0.94
    };
  }

  if (
    fingerStates.index &&
    fingerStates.middle &&
    !fingerStates.ring &&
    !fingerStates.pinky &&
    hasVSignSeparation(landmarks)
  ) {
    return {
      gesture: Gesture.PEACH_SIGN,
      fingerStates,
      confidence: 0.88
    };
  }

  if (
    fingerStates.index &&
    !fingerStates.middle &&
    !fingerStates.ring &&
    !fingerStates.pinky
  ) {
    return {
      gesture: Gesture.POINTING_FINGER,
      fingerStates,
      confidence: 0.9
    };
  }

  if (
    !fingerStates.index &&
    !fingerStates.middle &&
    !fingerStates.ring &&
    !fingerStates.pinky
  ) {
    return {
      gesture: Gesture.FIST,
      fingerStates,
      confidence: 0.86
    };
  }

  return {
    gesture: Gesture.UNKNOWN,
    fingerStates,
    confidence: 0.35
  };
}

export function getFingerStates(landmarks) {
  if (!Array.isArray(landmarks) || landmarks.length < 21) {
    return { ...DEFAULT_STATES };
  }

  const palmSize = getPalmSize(landmarks);

  return {
    thumb: isThumbExtended(landmarks, palmSize),
    index: isFingerExtended(landmarks, FINGERS.index, palmSize),
    middle: isFingerExtended(landmarks, FINGERS.middle, palmSize),
    ring: isFingerExtended(landmarks, FINGERS.ring, palmSize),
    pinky: isFingerExtended(landmarks, FINGERS.pinky, palmSize)
  };
}

function isFingerExtended(landmarks, indices, palmSize) {
  const [mcpIndex, pipIndex, dipIndex, tipIndex] = indices;
  const wrist = landmarks[0];
  const mcp = landmarks[mcpIndex];
  const pip = landmarks[pipIndex];
  const dip = landmarks[dipIndex];
  const tip = landmarks[tipIndex];
  const fingerLength =
    distance(mcp, pip) + distance(pip, dip) + distance(dip, tip);
  const straightness = distance(mcp, tip) / Math.max(fingerLength, 0.0001);
  const extendsBeyondKnuckle =
    distance(wrist, tip) > distance(wrist, mcp) + palmSize * 0.16;
  const extendsBeyondMiddleJoint =
    distance(wrist, tip) > distance(wrist, pip) + palmSize * 0.02;
  const reachesAwayFromPalm = extendsBeyondKnuckle && extendsBeyondMiddleJoint;

  return straightness > 0.62 && reachesAwayFromPalm;
}

function isThumbExtended(landmarks, palmSize) {
  const wrist = landmarks[0];
  const cmc = landmarks[1];
  const mcp = landmarks[2];
  const ip = landmarks[3];
  const tip = landmarks[4];
  const fingerLength =
    distance(cmc, mcp) + distance(mcp, ip) + distance(ip, tip);
  const straightness = distance(cmc, tip) / Math.max(fingerLength, 0.0001);
  const awayFromPalm =
    distance(tip, landmarks[5]) > palmSize * 0.45 &&
    distance(wrist, tip) > distance(wrist, mcp) + palmSize * 0.08;

  return straightness > 0.58 && awayFromPalm;
}

function hasVSignSeparation(landmarks) {
  const palmSize = getPalmSize(landmarks);
  return distance(landmarks[8], landmarks[12]) > palmSize * 0.55;
}

function getPalmSize(landmarks) {
  return Math.max(distance(landmarks[0], landmarks[9]), 0.0001);
}

function distance(a, b) {
  const dx = (a.x ?? 0) - (b.x ?? 0);
  const dy = (a.y ?? 0) - (b.y ?? 0);
  const dz = (a.z ?? 0) - (b.z ?? 0);
  return Math.hypot(dx, dy, dz);
}
