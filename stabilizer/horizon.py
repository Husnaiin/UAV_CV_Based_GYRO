"""Horizon detection and roll estimation.

Design notes:

* Line fitting is total least squares (principal axis) rather than a slope-
  intercept fit. y = mx + b cannot represent a vertical line, and the horizon IS
  vertical at 90 deg of bank -- the slope goes to infinity exactly when the
  estimate matters most. A direction-vector fit is well conditioned at every
  bank angle.
* Angles are circular with period 180 deg. A line has no direction, so -89 deg
  and +89 deg are two degrees apart, not 178. Every angle operation goes through
  the helpers below; plain subtraction and arithmetic means are wrong here and
  fail hardest near inverted flight.
* Contour point extraction is vectorised -- a per-point Python loop over the
  contour dominates frame time otherwise.
* Estimates carry a confidence (RANSAC inlier fraction) and are rejected on
  implausible frame-to-frame jumps.
"""

from dataclasses import dataclass

import cv2
import numpy as np

from . import config


# ------------------------------------------------------------------ angle math
# Horizon angles are circular modulo 180 deg. Every angle operation goes through
# these helpers; none of them may be replaced with plain arithmetic.

def wrap180(a):
    """Wrap degrees into [-90, 90).

    A line has no direction, so +90 and -90 describe the same (vertical)
    orientation; this canonicalises that pair to -90. Callers must never compare
    two angles with `==` or `-`, only via angle_diff().
    """
    return (a + 90.0) % 180.0 - 90.0


def angle_diff(a, b):
    """Smallest signed difference a - b, in [-90, 90)."""
    return wrap180(a - b)


def circular_mean(angles):
    """Mean of angles with period 180 deg."""
    a = np.radians(np.asarray(angles, dtype=float) * 2.0)
    return float(np.degrees(np.arctan2(np.sin(a).mean(), np.cos(a).mean())) / 2.0)


def circular_ema(prev, new, alpha):
    """Exponential moving average that respects the 180 deg wrap."""
    if prev is None:
        return wrap180(new)
    return wrap180(prev + alpha * angle_diff(new, prev))


@dataclass
class HorizonEstimate:
    roll_deg: float | None = None
    confidence: float = 0.0
    # Line in detection-frame coords: point p0 and unit direction d.
    p0: tuple | None = None
    direction: tuple | None = None
    scale: float = 1.0     # detection frame -> full frame


class HorizonDetector:
    """Stateful horizon detector. One instance per camera."""

    def __init__(self):
        self.last_roll = None
        self.good_streak = 0
        self._sky_lo = np.array(config.HORIZON_SKY_LOWER, dtype=np.uint8)
        self._sky_hi = np.array(config.HORIZON_SKY_UPPER, dtype=np.uint8)

    def reset(self):
        self.last_roll = None
        self.good_streak = 0

    def detect(self, frame):
        """Estimate roll from a BGR frame. Returns a HorizonEstimate."""
        # picamera2's "RGB888" format hands back BGR byte order in the numpy
        # array, so the BGR conversions below are correct for it. If you switch
        # to a real RGB source, change these two constants.
        h, w = frame.shape[:2]
        scale = config.DETECT_WIDTH / float(w)
        if scale < 1.0:
            small = cv2.resize(frame, (config.DETECT_WIDTH, int(round(h * scale))),
                               interpolation=cv2.INTER_AREA)
        else:
            small = frame
            scale = 1.0

        pts = self._edge_points(small)
        if pts is None or len(pts) < config.HORIZON_MIN_POINTS:
            self._miss()
            return HorizonEstimate(scale=scale)

        # Predictive gating: once locked on, ignore points far from where the
        # horizon was last frame. Keeps cloud edges and ground clutter out.
        if (self.last_roll is not None
                and self.good_streak >= config.HORIZON_GATE_ENABLE_AFTER):
            pts = self._gate(pts, small.shape, scale)
            if len(pts) < config.HORIZON_MIN_POINTS:
                self._miss()
                return HorizonEstimate(scale=scale)

        fit = self._ransac(pts)
        if fit is None:
            self._miss()
            return HorizonEstimate(scale=scale)

        p0, direction, inlier_frac = fit
        if inlier_frac < config.HORIZON_MIN_INLIER_FRAC:
            self._miss()
            return HorizonEstimate(scale=scale)

        roll = wrap180(np.degrees(np.arctan2(direction[1], direction[0])))

        # Reject physically impossible jumps between consecutive frames.
        if self.last_roll is not None:
            if abs(angle_diff(roll, self.last_roll)) > config.HORIZON_MAX_JUMP_DEG:
                self._miss()
                return HorizonEstimate(scale=scale)

        self.last_roll = roll
        self.good_streak += 1
        return HorizonEstimate(roll_deg=roll, confidence=float(inlier_frac),
                               p0=tuple(p0), direction=tuple(direction), scale=scale)

    # -------------------------------------------------------------- internals
    def _miss(self):
        self.good_streak = 0

    def _edge_points(self, img):
        """Return an (N,2) float array of candidate horizon points, x/y order."""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        sky = cv2.inRange(hsv, self._sky_lo, self._sky_hi)
        combined = cv2.add(gray, sky)
        blur = cv2.bilateralFilter(combined, 5, 50, 50)
        _, mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        edges = cv2.Canny(mask, 50, 150)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not contours:
            return None

        largest = max(contours, key=cv2.contourArea)
        pts = largest[:, 0, :]                       # (N,2) as x,y
        if len(pts) < config.HORIZON_MIN_POINTS:
            return None

        # Keep only contour points that coincide with a Canny edge. Vectorised:
        # looping over contour points in Python dominates frame time.
        ys = np.clip(pts[:, 1], 0, edges.shape[0] - 1)
        xs = np.clip(pts[:, 0], 0, edges.shape[1] - 1)
        keep = edges[ys, xs] != 0
        pts = pts[keep]
        return pts.astype(np.float64) if len(pts) else None

    def _gate(self, pts, shape, scale):
        """Drop points far from the predicted horizon line."""
        h, w = shape[:2]
        theta = np.radians(self.last_roll)
        d = np.array([np.cos(theta), np.sin(theta)])
        p0 = np.array([w * 0.5, h * 0.5])
        rel = pts - p0
        perp = np.abs(rel[:, 0] * d[1] - rel[:, 1] * d[0])
        return pts[perp < config.HORIZON_GATE_PX * scale]

    def _ransac(self, pts):
        """Direction-vector RANSAC. Returns (p0, unit_direction, inlier_frac)."""
        n = len(pts)
        rng = np.random.default_rng()
        iters = config.HORIZON_RANSAC_ITERS
        thresh = config.HORIZON_INLIER_PX

        # Sample all candidate pairs at once.
        i = rng.integers(0, n, size=iters)
        j = rng.integers(0, n, size=iters)
        valid = i != j
        i, j = i[valid], j[valid]
        if len(i) == 0:
            return None

        a = pts[i]
        b = pts[j]
        d = b - a
        norm = np.linalg.norm(d, axis=1)
        ok = norm > 1e-6
        if not np.any(ok):
            return None
        a, d, norm = a[ok], d[ok], norm[ok]
        d = d / norm[:, None]

        # Perpendicular distance of every point to every candidate line.
        rel = pts[None, :, :] - a[:, None, :]
        perp = np.abs(rel[:, :, 0] * d[:, None, 1] - rel[:, :, 1] * d[:, None, 0])
        counts = (perp < thresh).sum(axis=1)

        best = int(np.argmax(counts))
        if counts[best] < config.HORIZON_MIN_POINTS:
            return None

        # Refit on the winning inlier set by total least squares, which is far
        # more accurate than the two-point sample that selected it.
        inliers = pts[perp[best] < thresh]
        p0 = inliers.mean(axis=0)
        centred = inliers - p0
        # Principal axis via SVD; handles vertical lines that polyfit cannot.
        _, _, vh = np.linalg.svd(centred, full_matrices=False)
        direction = vh[0]
        direction = direction / np.linalg.norm(direction)

        return p0, direction, counts[best] / float(n)


def draw_overlay(frame, est, roll_deg, mode, fps, extra=None):
    """Draw the detected line plus an artificial-horizon reference."""
    h, w = frame.shape[:2]
    out = frame

    if est is not None and est.direction is not None:
        p0 = np.array(est.p0) / est.scale
        d = np.array(est.direction)
        L = float(w)
        pt1 = tuple(np.round(p0 - d * L).astype(int))
        pt2 = tuple(np.round(p0 + d * L).astype(int))
        cv2.line(out, pt1, pt2, (0, 255, 0), 2)

    if roll_deg is not None:
        centre = (w // 2, h // 2)
        theta = np.radians(roll_deg)
        dx = int((w * 0.4) * np.cos(theta))
        dy = int((w * 0.4) * np.sin(theta))
        cv2.line(out, (centre[0] - dx, centre[1] - dy),
                 (centre[0] + dx, centre[1] + dy), (0, 0, 255), 1)
        cv2.circle(out, centre, 4, (0, 0, 255), -1)

    lines = [
        f"Mode: {mode}",
        f"Roll: {roll_deg:+.1f} deg" if roll_deg is not None else "Roll: --",
        f"FPS:  {fps:.1f}",
    ]
    if extra:
        lines.extend(extra)
    for k, text in enumerate(lines):
        cv2.putText(out, text, (10, 26 + 24 * k), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 255, 255), 2)
    return out
