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
  const extendedNonThumb = [
    fingerStates.index,
    fingerStates.middle,
    fingerStates.ring,
    fingerStates.pinky
  ].filter(Boolean).length;

  if (extendedNonThumb >= 4) {
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
    !fingerStates.ring &&
    !fingerStates.pinky &&
    (!fingerStates.middle || isIndexDominantPoint(landmarks))
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
  const palmCenter = getPalmCenter(landmarks);
  const fingerLength =
    distance(mcp, pip) + distance(pip, dip) + distance(dip, tip);
  const straightness = distance(mcp, tip) / Math.max(fingerLength, 0.0001);
  const pipAngle = angle(mcp, pip, tip);
  const tipDistance = distance(palmCenter, tip);
  const pipDistance = distance(palmCenter, pip);
  const mcpDistance = distance(palmCenter, mcp);
  const tipAbovePip = pip.y - tip.y > palmSize * 0.02;
  const tipAboveDip = dip.y - tip.y > palmSize * 0.01;
  const reachesPastPip = tipDistance > pipDistance + palmSize * 0.015;
  const reachesPastMcp = tipDistance > mcpDistance + palmSize * 0.12;
  const pointsAwayFromWrist =
    distance(wrist, tip) > distance(wrist, pip) - palmSize * 0.04;

  return (
    (pipAngle > 142 && reachesPastPip) ||
    (straightness > 0.62 && reachesPastMcp && pointsAwayFromWrist) ||
    (tipAbovePip && tipAboveDip && pointsAwayFromWrist)
  );
}

function isThumbExtended(landmarks, palmSize) {
  const wrist = landmarks[0];
  const cmc = landmarks[1];
  const mcp = landmarks[2];
  const ip = landmarks[3];
  const tip = landmarks[4];
  const palmCenter = getPalmCenter(landmarks);
  const fingerLength =
    distance(cmc, mcp) + distance(mcp, ip) + distance(ip, tip);
  const straightness = distance(cmc, tip) / Math.max(fingerLength, 0.0001);
  const thumbAngle = angle(cmc, mcp, tip);
  const awayFromPalm =
    distance(tip, landmarks[5]) > palmSize * 0.38 &&
    distance(palmCenter, tip) > distance(palmCenter, mcp) + palmSize * 0.08 &&
    distance(wrist, tip) > distance(wrist, mcp) - palmSize * 0.03;

  return (straightness > 0.52 || thumbAngle > 132) && awayFromPalm;
}

function isIndexDominantPoint(landmarks) {
  const palmCenter = getPalmCenter(landmarks);
  const palmSize = getPalmSize(landmarks);
  const indexReach = distance(palmCenter, landmarks[8]);
  const middleReach = distance(palmCenter, landmarks[12]);
  const ringReach = distance(palmCenter, landmarks[16]);
  const pinkyReach = distance(palmCenter, landmarks[20]);

  return (
    indexReach > middleReach + palmSize * 0.08 &&
    indexReach > ringReach + palmSize * 0.14 &&
    indexReach > pinkyReach + palmSize * 0.14
  );
}

function hasVSignSeparation(landmarks) {
  const palmSize = getPalmSize(landmarks);
  return distance(landmarks[8], landmarks[12]) > palmSize * 0.24;
}

function getPalmCenter(landmarks) {
  const palmIndices = [0, 5, 9, 13, 17];
  const sum = palmIndices.reduce(
    (total, index) => ({
      x: total.x + (landmarks[index].x ?? 0),
      y: total.y + (landmarks[index].y ?? 0),
      z: total.z + (landmarks[index].z ?? 0)
    }),
    { x: 0, y: 0, z: 0 }
  );

  return {
    x: sum.x / palmIndices.length,
    y: sum.y / palmIndices.length,
    z: sum.z / palmIndices.length
  };
}

function getPalmSize(landmarks) {
  return Math.max(distance(landmarks[0], landmarks[9]), 0.0001);
}

function angle(a, b, c) {
  const ab = {
    x: (a.x ?? 0) - (b.x ?? 0),
    y: (a.y ?? 0) - (b.y ?? 0),
    z: (a.z ?? 0) - (b.z ?? 0)
  };
  const cb = {
    x: (c.x ?? 0) - (b.x ?? 0),
    y: (c.y ?? 0) - (b.y ?? 0),
    z: (c.z ?? 0) - (b.z ?? 0)
  };
  const denominator = Math.max(
    Math.hypot(ab.x, ab.y, ab.z) * Math.hypot(cb.x, cb.y, cb.z),
    0.0001
  );
  const cosine = clamp((ab.x * cb.x + ab.y * cb.y + ab.z * cb.z) / denominator, -1, 1);

  return (Math.acos(cosine) * 180) / Math.PI;
}

function distance(a, b) {
  const dx = (a.x ?? 0) - (b.x ?? 0);
  const dy = (a.y ?? 0) - (b.y ?? 0);
  const dz = (a.z ?? 0) - (b.z ?? 0);
  return Math.hypot(dx, dy, dz);
}

function clamp(value, min, max) {
  return Math.min(Math.max(value, min), max);
}
