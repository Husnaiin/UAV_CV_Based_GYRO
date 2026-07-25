"""RC PWM input and servo output via pigpio.

Reading a receiver channel means timing the pulse yourself: subscribe to both
edges and difference the tick counters, which is what PWMReader does. pigpio
ticks are microseconds and wrap at 2^32 (~72 minutes), so tickDiff() must be used
rather than plain subtraction.

A common trap worth stating explicitly: pigpio's get_servo_pulsewidth() is not a
way to read an input. It reports the pulsewidth pigpio is itself transmitting on
a pin, and returns 0 on a pin it has never driven.
"""

import threading
import time

import pigpio

from . import config


class PWMReader:
    """Measures incoming RC pulse width on a GPIO, with signal-loss detection."""

    def __init__(self, pi, gpio):
        self.pi = pi
        self.gpio = gpio
        self._lock = threading.Lock()
        self._high_tick = None
        self._width_us = 0.0
        self._last_valid = 0.0
        self._glitches = 0

        pi.set_mode(gpio, pigpio.INPUT)
        # Light pull-down so a disconnected signal wire reads as a dead channel
        # rather than floating and producing plausible-looking noise.
        pi.set_pull_up_down(gpio, pigpio.PUD_DOWN)
        self._cb = pi.callback(gpio, pigpio.EITHER_EDGE, self._on_edge)

    def _on_edge(self, gpio, level, tick):
        # Runs on pigpio's callback thread.
        if level == 1:
            self._high_tick = tick
        elif level == 0 and self._high_tick is not None:
            width = pigpio.tickDiff(self._high_tick, tick)
            self._high_tick = None
            if config.PWM_VALID_MIN_US <= width <= config.PWM_VALID_MAX_US:
                with self._lock:
                    self._width_us = float(width)
                    self._last_valid = time.monotonic()
            else:
                self._glitches += 1

    def read(self):
        """Return (width_us, is_live). width_us is the last valid pulse seen."""
        with self._lock:
            width = self._width_us
            last = self._last_valid
        live = (last > 0.0) and (time.monotonic() - last) < config.RC_TIMEOUT_S
        return width, live

    @property
    def glitch_count(self):
        return self._glitches

    def cancel(self):
        if self._cb is not None:
            self._cb.cancel()
            self._cb = None


class ServoOutput:
    """Slew-rate-limited, hard-clamped servo output."""

    def __init__(self, pi, gpio):
        self.pi = pi
        self.gpio = gpio
        self._current = float(config.SERVO_NEUTRAL_US)
        pi.set_mode(gpio, pigpio.OUTPUT)

    def write(self, target_us, dt):
        """Command `target_us`, clamped and slew-limited. Returns what was sent."""
        target = _clamp(float(target_us), config.SERVO_MIN_US, config.SERVO_MAX_US)

        max_step = config.SERVO_SLEW_US_PER_S * max(dt, 0.0)
        delta = target - self._current
        if delta > max_step:
            target = self._current + max_step
        elif delta < -max_step:
            target = self._current - max_step

        self._current = target
        self.pi.set_servo_pulsewidth(self.gpio, int(round(target)))
        return target

    def write_immediate(self, target_us):
        """Bypass the slew limiter. Used for failsafe and shutdown only."""
        target = _clamp(float(target_us), config.SERVO_MIN_US, config.SERVO_MAX_US)
        self._current = target
        self.pi.set_servo_pulsewidth(self.gpio, int(round(target)))
        return target

    def stop(self):
        """Stop sending pulses. The servo goes limp -- ground use only."""
        self.pi.set_servo_pulsewidth(self.gpio, 0)

    @property
    def current_us(self):
        return self._current


def _clamp(v, lo, hi):
    return lo if v < lo else (hi if v > hi else v)
