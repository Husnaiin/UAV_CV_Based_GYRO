#!/usr/bin/env python3
"""Hardware-free verification of the control and vision maths.

Runs on any machine with numpy + opencv -- no pigpio, no picamera2, no camera.
Everything here is a property that has to hold before the code goes near an
airframe. Run it after any change to stabilizer/.

    python selftest.py
"""

import sys
import types

import numpy as np

# --- stub pigpio so stabilizer.rc imports on a desktop --------------------
if "pigpio" not in sys.modules:
    fake = types.ModuleType("pigpio")
    fake.INPUT = 0
    fake.OUTPUT = 1
    fake.EITHER_EDGE = 2
    fake.PUD_DOWN = 3
    fake.tickDiff = lambda a, b: (b - a) & 0xFFFFFFFF
    sys.modules["pigpio"] = fake

import cv2  # noqa: E402

from stabilizer import config  # noqa: E402
from stabilizer.horizon import (  # noqa: E402
    HorizonDetector, angle_diff, circular_ema, circular_mean, wrap180)
from stabilizer.pid import PID  # noqa: E402
from stabilizer.rc import ServoOutput  # noqa: E402


FAILURES = []


def check(name, cond, detail=""):
    if cond:
        print(f"  PASS  {name}")
    else:
        print(f"  FAIL  {name}  {detail}")
        FAILURES.append(name)


def section(title):
    print(f"\n{title}\n" + "-" * len(title))


# ============================================================ angle maths
def test_angles():
    section("Circular angle maths (period 180 deg)")

    check("wrap180 folds +100 to -80", abs(wrap180(100.0) - (-80.0)) < 1e-9)
    # +90 and -90 are the same vertical line; wrap180 canonicalises to -90, and
    # angle_diff must therefore call them identical.
    check("wrap180 canonicalises +90 to -90", abs(wrap180(90.0) - (-90.0)) < 1e-9)
    check("+90 and -90 are the same orientation", abs(angle_diff(90.0, -90.0)) < 1e-9)

    # The property that makes a plain rolling mean unsafe on these angles.
    d = angle_diff(89.0, -89.0)
    check("89 and -89 are 2 deg apart, not 178", abs(abs(d) - 2.0) < 1e-9,
          f"got {d}")

    m = circular_mean([89.0, -89.0])
    check("circular_mean([89,-89]) is near +/-90, not 0", abs(abs(m) - 90.0) < 1e-6,
          f"got {m}")
    # Why circular_mean exists: an arithmetic mean of the same pair reports
    # ~0 deg -- "wings level" -- while the aircraft is very nearly inverted.
    check("arithmetic mean is unusable for these angles",
          abs(np.mean([89.0, -89.0])) < 1e-9)

    # EMA must cross the wrap without a 178 deg excursion.
    r = 88.0
    for _ in range(12):
        r = circular_ema(r, -88.0, 0.35)
    check("circular_ema crosses the +/-90 wrap smoothly", abs(angle_diff(r, -88.0)) < 6.0,
          f"got {r}")


# ============================================================ PID
def test_pid():
    section("PID controller")

    # Time-correctness. Integrating a constant error of 1.0 for one simulated
    # second must give the same result regardless of the step size used.
    outs = []
    for dt, n in ((0.01, 100), (0.001, 1000), (0.05, 20)):
        p = PID(kp=0.0, ki=1.0, kd=0.0, integral_limit=1e6, output_limit=1e6)
        for _ in range(n):
            out = p.update(measurement=-1.0, dt=dt, setpoint=0.0)
        outs.append(out)
    check("I term is rate-independent", max(outs) - min(outs) < 1e-6,
          f"outputs {outs}")

    # Why dt is explicit: accumulating per iteration instead makes the same
    # one-second integral vary by orders of magnitude with loop rate alone.
    per_iter = []
    for _, n in ((0.01, 100), (0.001, 1000), (0.05, 20)):
        integral = 0.0
        for _ in range(n):
            integral += 1.0
        per_iter.append(integral)
    check("per-iteration accumulation would be rate-dependent",
          max(per_iter) - min(per_iter) > 100.0, f"outputs {per_iter}")

    # Derivative on a known ramp. measurement ramps at +10 units/s, so
    # d(error)/dt = -10, and the D term should approach kd * -10.
    p = PID(kp=0.0, ki=0.0, kd=1.0, integral_limit=1e6, output_limit=1e6,
            derivative_tau=0.02)
    dt = 0.005
    meas = 0.0
    for _ in range(400):
        meas += 10.0 * dt
        out = p.update(meas, dt)
    check("D term tracks a ramp to kd*-rate", abs(out - (-10.0)) < 0.5, f"got {out}")

    # Derivative must not amplify as dt shrinks.
    ds = []
    for dt in (0.02, 0.005, 0.001):
        p = PID(kp=0.0, ki=0.0, kd=1.0, integral_limit=1e6, output_limit=1e6,
                derivative_tau=0.02)
        meas = 0.0
        for _ in range(int(2.0 / dt)):
            meas += 10.0 * dt
            out = p.update(meas, dt)
        ds.append(out)
    check("D term is rate-independent", max(ds) - min(ds) < 0.5, f"outputs {ds}")

    # Anti-windup: sustained saturating error must not run the integral away.
    p = PID(kp=1.0, ki=5.0, kd=0.0, integral_limit=50.0, output_limit=10.0)
    for _ in range(2000):
        p.update(measurement=-100.0, dt=0.01)
    check("integral is bounded under saturation", abs(p.integral) <= 50.0 + 1e-9,
          f"integral {p.integral}")
    # And recovery is prompt once the error reverses.
    steps = 0
    while steps < 500:
        out = p.update(measurement=0.0, dt=0.01)
        steps += 1
        if out <= 0.0:
            break
    check("recovers from saturation within 5 s", steps < 500, f"took {steps} steps")

    # reset() must clear state, so a mode change cannot jolt the surface.
    p.reset()
    check("reset clears the integral", p.integral == 0.0)
    check("first update after reset has no D spike",
          abs(p.update(measurement=50.0, dt=0.02)) < 1e6)

    # Output limit is honoured.
    p = PID(kp=1000.0, ki=0.0, kd=0.0, integral_limit=10.0, output_limit=250.0)
    out = p.update(measurement=-100.0, dt=0.02)
    check("output respects the authority limit", abs(out) <= 250.0 + 1e-9, f"got {out}")


# ============================================================ servo output
class FakePi:
    def __init__(self):
        self.sent = []

    def set_mode(self, *a):
        pass

    def set_servo_pulsewidth(self, gpio, us):
        self.sent.append(us)


def test_servo():
    section("Servo output")

    pi = FakePi()
    s = ServoOutput(pi, 18)

    # Hard clamp.
    s.write_immediate(5000)
    check("clamps above SERVO_MAX_US", s.current_us == config.SERVO_MAX_US)
    s.write_immediate(0)
    check("clamps below SERVO_MIN_US", s.current_us == config.SERVO_MIN_US)

    # Slew limiting: a full-scale step in one 50 Hz tick must be rate limited.
    s.write_immediate(config.SERVO_NEUTRAL_US)
    dt = 1.0 / config.CONTROL_HZ
    before = s.current_us
    s.write(config.SERVO_MAX_US, dt)
    step = s.current_us - before
    expected = config.SERVO_SLEW_US_PER_S * dt
    check("slew limiter caps a step command", step <= expected + 1e-6,
          f"step {step} > {expected}")

    # It should still get there over time.
    for _ in range(200):
        s.write(config.SERVO_MAX_US, dt)
    check("slew limiter still reaches the target", s.current_us == config.SERVO_MAX_US)

    # Never emits an out-of-range pulse.
    bad = [v for v in pi.sent if v != 0 and not (config.SERVO_MIN_US <= v <= config.SERVO_MAX_US)]
    check("no out-of-range pulse ever emitted", not bad, f"{bad[:5]}")


# ============================================================ mode selector
def test_modes():
    section("Mode selection")

    from flight_stabilizer import MANUAL, STABILIZE, TRAINER, ModeSelector

    sel = ModeSelector()
    check("defaults to MANUAL", sel.mode == MANUAL)
    check("lost mode channel forces MANUAL",
          sel.update(1800.0, live=False) == MANUAL)

    sel = ModeSelector()
    check("low pulse selects MANUAL", sel.update(1100.0, True) == MANUAL)
    check("mid pulse selects STABILIZE", sel.update(1500.0, True) == STABILIZE)
    check("high pulse selects TRAINER", sel.update(1900.0, True) == TRAINER)

    # Hysteresis: dithering on a threshold must not chatter.
    sel = ModeSelector()
    sel.update(1500.0, True)
    edge = config.MODE_MANUAL_BELOW_US
    modes = {sel.update(edge + off, True) for off in (-5, 5, -5, 5, -5, 5)}
    check("hysteresis prevents chatter at a threshold", len(modes) == 1,
          f"saw {modes}")


# ============================================================ horizon
def synth_frame(roll_deg, size=(640, 480), noise=6.0):
    """Sky above / ground below a line through centre at `roll_deg`."""
    w, h = size
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    th = np.radians(roll_deg)
    # Signed perpendicular distance from the line through the image centre.
    side = -(xx - w / 2) * np.sin(th) + (yy - h / 2) * np.cos(th)

    img = np.empty((h, w, 3), dtype=np.uint8)
    sky = np.array([170, 120, 70], dtype=np.float64)     # BGR, inside the HSV gate
    ground = np.array([40, 60, 50], dtype=np.float64)    # BGR, outside it
    img[:] = np.where(side[..., None] < 0, sky, ground).astype(np.uint8)

    if noise > 0:
        rng = np.random.default_rng(0)
        img = np.clip(img.astype(np.float64) + rng.normal(0, noise, img.shape),
                      0, 255).astype(np.uint8)
    return img


def test_horizon():
    section("Horizon detection")

    truths = [0, 5, -5, 15, -20, 35, -45, 60, -70, 80, -85, 88]
    errors = []
    misses = []
    for t in truths:
        det = HorizonDetector()          # fresh: no gating, no jump rejection
        est = det.detect(synth_frame(t))
        if est.roll_deg is None:
            misses.append(t)
            continue
        errors.append(abs(angle_diff(est.roll_deg, t)))

    check("detects the horizon at every test angle", not misses,
          f"missed {misses}")
    if errors:
        worst = max(errors)
        check("roll error within 2.5 deg at all angles", worst < 2.5,
              f"worst {worst:.2f} deg")
        print(f"        mean error {np.mean(errors):.3f} deg, worst {worst:.3f} deg")

    # The case np.polyfit could not represent at all: a vertical horizon.
    det = HorizonDetector()
    est = det.detect(synth_frame(90.0))
    ok = est.roll_deg is not None and abs(angle_diff(est.roll_deg, 90.0)) < 2.5
    check("handles a vertical horizon (90 deg bank)", ok,
          f"got {est.roll_deg}")

    # Confidence should be meaningful on a clean synthetic scene.
    det = HorizonDetector()
    est = det.detect(synth_frame(12.0))
    check("reports high confidence on a clean frame", est.confidence > 0.5,
          f"conf {est.confidence:.2f}")

    # A featureless frame must yield no estimate rather than a bogus one.
    det = HorizonDetector()
    blank = np.full((480, 640, 3), 128, dtype=np.uint8)
    check("returns None on a featureless frame", det.detect(blank).roll_deg is None)

    # Jump rejection: a locked detector must refuse a physically impossible step.
    det = HorizonDetector()
    for _ in range(4):
        det.detect(synth_frame(0.0))
    est = det.detect(synth_frame(70.0))
    check("rejects an impossible frame-to-frame jump", est.roll_deg is None,
          f"accepted {est.roll_deg}")

    # Sign convention: rotating the scene must move roll the same way.
    det_a, det_b = HorizonDetector(), HorizonDetector()
    a = det_a.detect(synth_frame(10.0)).roll_deg
    b = det_b.detect(synth_frame(20.0)).roll_deg
    check("roll increases with scene rotation (sign is consistent)",
          a is not None and b is not None and angle_diff(b, a) > 5.0,
          f"a={a} b={b}")


def test_timing():
    section("Detection cost")

    det = HorizonDetector()
    frame = synth_frame(12.0)
    det.detect(frame)                      # warm up
    import time
    t0 = time.perf_counter()
    n = 30
    for _ in range(n):
        det.detect(frame)
    ms = (time.perf_counter() - t0) / n * 1000.0
    print(f"        {ms:.1f} ms/frame on this machine "
          f"(detection width {config.DETECT_WIDTH}px)")
    check("detection cost is sane on this machine", ms < 200.0, f"{ms:.1f} ms")


def main():
    print(f"OpenCV {cv2.__version__}, NumPy {np.__version__}")
    test_angles()
    test_pid()
    test_servo()
    test_modes()
    test_horizon()
    test_timing()

    print()
    if FAILURES:
        print(f"FAILED: {len(FAILURES)} check(s): {', '.join(FAILURES)}")
        return 1
    print("All checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
