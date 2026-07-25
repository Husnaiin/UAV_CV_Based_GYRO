#!/usr/bin/env python3
"""Horizon detector preview and tuning tool. No servo output, no pigpio.

Runs exactly the same HorizonDetector the flight program uses, so what you tune
here is what flies. Use it to set the sky HSV gate for your conditions and to
confirm the detector holds lock as you tilt the camera.

    python horizon_detector.py                  # live camera
    python horizon_detector.py --video clip.mp4 # replay a recording
    python horizon_detector.py --tune           # HSV trackbars

Keys:  q quit    p pause    s save frame
"""

import argparse
import time
from collections import deque

import cv2
import numpy as np

from stabilizer import config
from stabilizer.horizon import HorizonDetector, circular_ema, draw_overlay


def open_source(args):
    if args.video:
        cap = cv2.VideoCapture(args.video)
        if not cap.isOpened():
            raise SystemExit(f"Cannot open {args.video}")
        return lambda: cap.read()[1], cap.release

    from picamera2 import Picamera2
    picam2 = Picamera2(config.CAMERA_INDEX)
    picam2.configure(picam2.create_preview_configuration(
        main={"format": "RGB888", "size": config.CAMERA_SIZE}))
    picam2.start()
    time.sleep(0.5)
    return picam2.capture_array, picam2.stop


def make_tuner(detector):
    win = "sky gate"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    lo, hi = config.HORIZON_SKY_LOWER, config.HORIZON_SKY_UPPER
    for name, val, top in (("H lo", lo[0], 179), ("S lo", lo[1], 255), ("V lo", lo[2], 255),
                           ("H hi", hi[0], 179), ("S hi", hi[1], 255), ("V hi", hi[2], 255)):
        cv2.createTrackbar(name, win, val, top, lambda _v: None)

    def apply():
        g = lambda n: cv2.getTrackbarPos(n, win)
        detector._sky_lo = np.array([g("H lo"), g("S lo"), g("V lo")], dtype=np.uint8)
        detector._sky_hi = np.array([g("H hi"), g("S hi"), g("V hi")], dtype=np.uint8)

    return apply


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", help="replay a video file instead of the camera")
    ap.add_argument("--tune", action="store_true", help="show HSV trackbars")
    args = ap.parse_args()

    grab, close = open_source(args)
    detector = HorizonDetector()
    apply_tuning = make_tuner(detector) if args.tune else None

    roll = None
    fps_hist = deque(maxlen=30)
    paused = False
    saved = 0
    frame = None

    try:
        while True:
            if not paused:
                t0 = time.monotonic()
                new = grab()
                if new is None:
                    print("End of stream.")
                    break
                frame = new

                if apply_tuning is not None:
                    apply_tuning()

                est = detector.detect(frame)
                if est.roll_deg is not None:
                    dt = time.monotonic() - t0
                    alpha = min(dt / (config.ROLL_FILTER_TAU_S + dt), 1.0)
                    roll = circular_ema(roll, est.roll_deg, alpha)

                fps_hist.append(1.0 / max(time.monotonic() - t0, 1e-6))
                status = "LOCK" if est.roll_deg is not None else "LOST"
                disp = draw_overlay(
                    frame.copy(), est, roll, status, float(np.mean(fps_hist)),
                    extra=[f"conf {est.confidence:.2f}  streak {detector.good_streak}"])
                cv2.imshow("horizon", disp)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("p"):
                paused = not paused
            if key == ord("s") and frame is not None:
                name = f"capture_{saved:03d}.png"
                cv2.imwrite(name, frame)
                print(f"saved {name}")
                saved += 1
    finally:
        close()
        cv2.destroyAllWindows()
        if args.tune:
            print(f"\nFinal gate -- copy into stabilizer/config.py:\n"
                  f"HORIZON_SKY_LOWER = {tuple(int(v) for v in detector._sky_lo)}\n"
                  f"HORIZON_SKY_UPPER = {tuple(int(v) for v in detector._sky_hi)}")


if __name__ == "__main__":
    main()
