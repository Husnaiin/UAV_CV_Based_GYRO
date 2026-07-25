"""Time-correct PID controller.

Design notes, all of which matter in flight:

* dt is explicit, so the I and D gains keep their meaning when the loop rate
  varies. Integrating per iteration instead would make both gains a function of
  however fast the loop happened to be running.
* Derivative is taken on the measurement, not the error, and low-pass filtered.
  Differentiating a noisy vision-derived angle raw puts that noise straight onto
  the servo.
* Conditional anti-windup: integration is suspended while the output is
  saturated and the error would push it further into the stop.
* reset() clears all state. Callers use it on every mode transition so a stale
  integral cannot produce a jolt when the stabilizer is re-engaged.
"""


class PID:
    def __init__(self, kp, ki, kd, *, integral_limit, output_limit,
                 derivative_tau=0.0):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.integral_limit = abs(integral_limit)
        self.output_limit = abs(output_limit)
        self.derivative_tau = derivative_tau
        self.reset()

    def reset(self):
        self._integral = 0.0
        self._prev_measurement = None
        self._deriv = 0.0

    def update(self, measurement, dt, setpoint=0.0):
        """Return the control output for `measurement`, given elapsed `dt` seconds."""
        if dt <= 0.0:
            return self._last_output_guess()

        error = setpoint - measurement

        # --- proportional
        p_term = self.kp * error

        # --- derivative, on measurement, sign-corrected, then low-passed.
        # d(error)/dt == -d(measurement)/dt for a constant setpoint, and using the
        # measurement avoids a derivative spike when the setpoint steps.
        if self._prev_measurement is None:
            raw_deriv = 0.0
        else:
            raw_deriv = -(measurement - self._prev_measurement) / dt
        self._prev_measurement = measurement

        if self.derivative_tau > 0.0:
            alpha = dt / (self.derivative_tau + dt)
            self._deriv += alpha * (raw_deriv - self._deriv)
        else:
            self._deriv = raw_deriv
        d_term = self.kd * self._deriv

        # --- integral with conditional anti-windup.
        # Provisionally integrate, then undo it if that only deepens saturation.
        prev_integral = self._integral
        self._integral += error * dt
        self._integral = _clamp(self._integral, -self.integral_limit, self.integral_limit)
        i_term = self.ki * self._integral

        output = p_term + i_term + d_term
        if abs(output) > self.output_limit and (output > 0) == (error > 0):
            self._integral = prev_integral
            i_term = self.ki * self._integral
            output = p_term + i_term + d_term

        self._last = _clamp(output, -self.output_limit, self.output_limit)
        return self._last

    def _last_output_guess(self):
        return getattr(self, "_last", 0.0)

    @property
    def integral(self):
        return self._integral


def _clamp(v, lo, hi):
    return lo if v < lo else (hi if v > hi else v)
