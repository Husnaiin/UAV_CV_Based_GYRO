#!/usr/bin/env python3
"""Vision-based roll stabilizer for a fixed-wing RC aircraft.

Reads the pilot's aileron channel and a 3-position mode switch from the receiver,
estimates roll from the horizon in the camera image, and drives the aileron servo.

    MANUAL     pilot input passed straight through, stabilizer bypassed
    STABILIZE  PID correction added on top of pilot input
    TRAINER    PID drives to wings-level from neutral, pilot roll ignored

Run with --bench first. See README.md; do not fly this without completing the
pre-flight checklist there, and do not fly it without a hardware failsafe.
"""

import argparse
import csv
import signal
import sys
import time

import cv2
import numpy as np

from stabilizer import config
from stabilizer.camera import CameraThread
from stabilizer.horizon import HorizonDetector, circular_ema, draw_overlay
from stabilizer.pid import PID
from stabilizer.rc import PWMReader, ServoOutput


MANUAL, STABILIZE, TRAINER, FAILSAFE = "MANUAL", "STABILIZE", "TRAINER", "FAILSAFE"


class ModeSelector:
    """Maps the mode-switch pulse width to a mode, with hysteresis."""

    def __init__(self):
        self.mode = MANUAL

    def update(self, pulse_us, live):
        # Lost the mode channel: fall back to passthrough. If the pilot's stick
        # channel is alive we must not take authority we cannot verify.
        if not live:
            self.mode = MANUAL
            return self.mode

        h = config.MODE_HYSTERESIS_US
        lo, hi = config.MODE_MANUAL_BELOW_US, config.MODE_STABILIZE_BELOW_US

        if self.mode == MANUAL:
            if pulse_us > lo + h:
                self.mode = STABILIZE if pulse_us <= hi + h else TRAINER
        elif self.mode == STABILIZE:
            if pulse_us < lo - h:
                self.mode = MANUAL
            elif pulse_us > hi + h:
                self.mode = TRAINER
        else:  # TRAINER
            if pulse_us < lo - h:
                self.mode = MANUAL
            elif pulse_us < hi - h:
                self.mode = STABILIZE
        return self.mode


class Stabilizer:
    def __init__(self, args):
        self.args = args
        self.detector = HorizonDetector()
        self.selector = ModeSelector()

        limit = config.MAX_CORRECTION_US
        self.pids = {
            STABILIZE: PID(**config.PID_STABILIZE,
                           integral_limit=config.PID_INTEGRAL_LIMIT_US,
                           output_limit=limit,
                           derivative_tau=config.PID_DERIVATIVE_TAU_S),
            TRAINER: PID(**config.PID_TRAINER,
                         integral_limit=config.PID_INTEGRAL_LIMIT_US,
                         output_limit=limit,
                         derivative_tau=config.PID_DERIVATIVE_TAU_S),
        }

        self.roll = None                 # filtered roll estimate, degrees
        self.last_good_roll_t = 0.0
        self.last_conf = 0.0
        self.prev_mode = None
        self.running = True

        self.pi = None
        self.aileron_in = None
        self.mode_in = None
        self.servo = None
        self.cam = None
        self.picam2 = None
        self.writer = None
        self.csv_file = None
        self.csv = None

    # ------------------------------------------------------------- lifecycle
    def setup_hardware(self):
        import pigpio
        self.pi = pigpio.pi()
        if not self.pi.connected:
            raise SystemExit("Cannot reach pigpiod. Start it with: sudo pigpiod")

        self.aileron_in = PWMReader(self.pi, config.GPIO_AILERON_IN)
        self.mode_in = PWMReader(self.pi, config.GPIO_MODE_IN)
        if not self.args.no_servo:
            self.servo = ServoOutput(self.pi, config.GPIO_AILERON_OUT)

    def setup_camera(self):
        from picamera2 import Picamera2
        self.picam2 = Picamera2(config.CAMERA_INDEX)
        cfg = self.picam2.create_preview_configuration(
            main={"format": "RGB888", "size": config.CAMERA_SIZE})
        self.picam2.configure(cfg)
        self.picam2.start()
        self.cam = CameraThread(self.picam2).start()

        # Wait for the first frame so we do not enter the loop blind.
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            frame, _ = self.cam.latest()
            if frame is not None:
                return
            time.sleep(0.05)
        raise SystemExit("Camera produced no frames within 5s.")

    def setup_logging(self):
        if config.VIDEO_LOG_ENABLED or self.args.preview:
            # mp4v is present in every stock OpenCV build. Codecs such as X264
            # are frequently absent and fail silently, producing a 0-byte file.
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self.writer = cv2.VideoWriter(config.VIDEO_LOG_PATH, fourcc,
                                          config.VIDEO_LOG_FPS, config.CAMERA_SIZE)
            if not self.writer.isOpened():
                print("WARNING: video writer failed to open, video logging disabled")
                self.writer = None

        if config.CSV_LOG_ENABLED:
            self.csv_file = open(config.CSV_LOG_PATH, "w", newline="")
            self.csv = csv.writer(self.csv_file)
            self.csv.writerow(["t", "mode", "roll_deg", "conf", "aileron_us",
                               "switch_us", "rc_live", "horizon_ok",
                               "correction_us", "output_us", "dt"])

    def shutdown(self):
        print("\nShutting down...")
        if self.cam is not None:
            self.cam.stop()
        if self.picam2 is not None:
            try:
                self.picam2.stop()
            except Exception:
                pass
        if self.servo is not None:
            # Centre before cutting pulses, so a powered servo on the bench does
            # not snap to a stop as the signal disappears.
            self.servo.write_immediate(config.SERVO_NEUTRAL_US)
            time.sleep(0.1)
            self.servo.stop()
        for reader in (self.aileron_in, self.mode_in):
            if reader is not None:
                reader.cancel()
        if self.pi is not None:
            self.pi.stop()
        if self.writer is not None:
            self.writer.release()
        if self.csv_file is not None:
            self.csv_file.close()

    # ------------------------------------------------------------- vision
    def update_vision(self, frame, now):
        """Run detection and fold the result into the filtered roll estimate."""
        est = self.detector.detect(frame)
        if est.roll_deg is None:
            return est

        # Circular EMA -- deliberately light. Heavy smoothing here is the classic
        # way to make a vision stabilizer oscillate, because the phase lag lands
        # directly in front of the derivative term.
        dt_v = max(now - self.last_good_roll_t, 1e-3) if self.last_good_roll_t else None
        if dt_v is None:
            alpha = 1.0
        else:
            alpha = min(dt_v / (config.ROLL_FILTER_TAU_S + dt_v), 1.0)

        self.roll = circular_ema(self.roll, est.roll_deg, alpha)
        self.last_good_roll_t = now
        self.last_conf = est.confidence
        return est

    def horizon_fresh(self, now):
        return (self.last_good_roll_t > 0.0
                and (now - self.last_good_roll_t) < config.HORIZON_LOST_TIMEOUT_S
                and self.roll is not None)

    # ------------------------------------------------------------- control
    def compute_output(self, mode, aileron_us, rc_live, horizon_ok, dt):
        """Return (output_us, correction_us, effective_mode)."""
        # --- Failsafe: pilot channel gone. We cannot pass anything through.
        if not rc_live:
            if horizon_ok:
                # Hold wings level and let the airframe fly a stable glide.
                pid = self.pids[TRAINER]
                corr = config.CORRECTION_SIGN * pid.update(self.roll, dt, 0.0)
                return config.SERVO_NEUTRAL_US + corr, corr, FAILSAFE
            return config.SERVO_NEUTRAL_US, 0.0, FAILSAFE

        if mode == MANUAL:
            return aileron_us, 0.0, MANUAL

        # --- Horizon lost. Hand control back to the pilot rather than acting on
        # a stale or absent estimate. In TRAINER this is essential: that mode
        # ignores the stick, so continuing blind would leave the pilot with no
        # way to recover.
        if not horizon_ok:
            return aileron_us, 0.0, MANUAL

        pid = self.pids[mode]
        corr = config.CORRECTION_SIGN * pid.update(self.roll, dt, 0.0)
        base = aileron_us if mode == STABILIZE else config.SERVO_NEUTRAL_US
        return base + corr, corr, mode

    def on_mode_change(self, mode):
        """Reset integrators so a stale integral cannot jolt the surface."""
        for pid in self.pids.values():
            pid.reset()
        print(f"mode -> {mode}")

    # ------------------------------------------------------------- main loop
    def run(self):
        period = 1.0 / config.CONTROL_HZ
        next_tick = time.monotonic()
        last_t = time.monotonic()
        last_seq = -1
        frames = 0
        fps = 0.0
        fps_t = time.monotonic()
        stats_t = time.monotonic()
        vision_errors = 0

        while self.running:
            now = time.monotonic()
            dt = now - last_t
            last_t = now

            # --- inputs
            aileron_us, ail_live = self.aileron_in.read()
            switch_us, sw_live = self.mode_in.read()

            # --- vision, only when a new frame has actually arrived
            est = None
            frame, seq = self.cam.latest()
            if frame is not None and seq != last_seq:
                last_seq = seq
                try:
                    est = self.update_vision(frame, now)
                except Exception as exc:
                    # A detector fault must never stop the control loop. It
                    # degrades to "horizon lost", which hands the pilot control.
                    vision_errors += 1
                    if vision_errors <= 5:
                        print(f"vision error: {exc!r}")
                frames += 1
                if now - fps_t >= 1.0:
                    fps = frames / (now - fps_t)
                    frames, fps_t = 0, now

            horizon_ok = self.horizon_fresh(now)

            # --- mode
            mode = self.selector.update(switch_us, sw_live)
            if mode != self.prev_mode:
                self.on_mode_change(mode)
                self.prev_mode = mode

            # --- control
            out_us, corr, eff_mode = self.compute_output(
                mode, aileron_us, ail_live, horizon_ok, dt)

            if self.servo is not None:
                sent = self.servo.write(out_us, dt)
            else:
                sent = out_us

            # --- logging
            if self.csv is not None:
                self.csv.writerow([
                    f"{now:.4f}", eff_mode,
                    f"{self.roll:.2f}" if self.roll is not None else "",
                    f"{self.last_conf:.2f}", f"{aileron_us:.0f}", f"{switch_us:.0f}",
                    int(ail_live), int(horizon_ok), f"{corr:.1f}", f"{sent:.0f}",
                    f"{dt:.4f}"])

            if self.writer is not None and frame is not None and est is not None:
                overlay = draw_overlay(frame.copy(), est, self.roll, eff_mode, fps,
                                       extra=[f"out {sent:.0f}us  corr {corr:+.0f}"])
                self.writer.write(overlay)
                if self.args.preview:
                    cv2.imshow("stabilizer", overlay)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        self.running = False

            if self.args.bench and now - stats_t >= 1.0:
                stats_t = now
                print(f"[{eff_mode:9s}] roll={_fmt(self.roll)} conf={self.last_conf:.2f} "
                      f"ail={aileron_us:4.0f} sw={switch_us:4.0f} "
                      f"live={int(ail_live)}{int(sw_live)} hz_ok={int(horizon_ok)} "
                      f"corr={corr:+6.1f} out={sent:4.0f} vfps={fps:4.1f}")

            # --- fixed-rate pacing
            next_tick += period
            sleep = next_tick - time.monotonic()
            if sleep > 0:
                time.sleep(sleep)
            else:
                # Overran the budget; resync rather than spiral into catch-up.
                next_tick = time.monotonic()


def _fmt(v):
    return f"{v:+6.1f}" if v is not None else "   --"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bench", action="store_true",
                    help="print a status line every second (ground testing)")
    ap.add_argument("--no-servo", action="store_true",
                    help="run the full loop but never drive the servo output")
    ap.add_argument("--preview", action="store_true",
                    help="show an OpenCV preview window (ground testing only)")
    args = ap.parse_args()

    app = Stabilizer(args)

    def handle_signal(signum, _frame):
        app.running = False

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    try:
        app.setup_hardware()
        app.setup_camera()
        app.setup_logging()
        if args.no_servo:
            print("--no-servo: servo output is DISABLED")
        print(f"Running at {config.CONTROL_HZ:.0f} Hz. Ctrl-C to stop.")
        app.run()
    finally:
        app.shutdown()
        if args.preview:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
