"""Central configuration.

Every tunable lives here; nothing else in the package hard-codes a constant.

Gain units
----------
The controller is time-correct: it integrates and differentiates against a
measured dt rather than per iteration. Consequently `ki` is in output-units per
degree-second and `kd` is in output-units per degree-per-second. Gains do not
need adjusting when the loop rate changes, but they are not interchangeable with
gains taken from a controller that omits dt.
"""

# ---------------------------------------------------------------- GPIO (BCM)
GPIO_AILERON_IN = 4      # PWM from RX aileron channel
GPIO_MODE_IN = 17        # PWM from RX 3-position mode switch
GPIO_AILERON_OUT = 18    # PWM to aileron servo (hardware PWM capable pin)

# ------------------------------------------------------------- Servo limits
SERVO_MIN_US = 1000       # absolute hard limit, never exceeded
SERVO_MAX_US = 2000
SERVO_NEUTRAL_US = 1500

# Maximum microseconds the stabilizer may add on top of pilot input. This caps
# control authority so a bad horizon estimate cannot slam the surface to a stop.
# 250us on a 1000-2000 range is +/-50% of half-travel. Start smaller, work up.
MAX_CORRECTION_US = 250.0

# Sign of the correction. If the bench test in the README shows the aileron
# driving the wrong way (bank the airframe right and the surface commands
# further right), flip this to -1. An inverted sign makes the loop diverge, so
# verify it on the bench every time the camera is remounted.
CORRECTION_SIGN = +1

# Slew rate limit on the output, us per second. 3000 us/s = full 1000->2000
# sweep in 1/3 s. Prevents step commands from shock-loading the linkage.
SERVO_SLEW_US_PER_S = 3000.0

# ------------------------------------------------------------- RC input
# Pulses outside this window are treated as glitches and discarded.
PWM_VALID_MIN_US = 700
PWM_VALID_MAX_US = 2300

# If a channel produces no valid pulse for this long, it is declared lost.
# Standard RC frames are 20ms, so 150ms = 7 missed frames.
RC_TIMEOUT_S = 0.15

# Mode switch thresholds (us) with hysteresis to stop chatter on the boundary.
MODE_MANUAL_BELOW_US = 1300
MODE_STABILIZE_BELOW_US = 1700
MODE_HYSTERESIS_US = 40

# ------------------------------------------------------------- Control loop
CONTROL_HZ = 50.0            # fixed-rate control loop
CAMERA_SIZE = (640, 480)     # capture resolution
CAMERA_INDEX = 1             # Picamera2 camera index
DETECT_WIDTH = 320           # detection runs downscaled; roll is scale-invariant

# ------------------------------------------------------------- PID gains
# STABILIZE: correction is added on top of pilot input (pilot keeps authority).
PID_STABILIZE = dict(kp=2.0, ki=0.4, kd=0.06)

# TRAINER: controller drives to wings-level from neutral, pilot roll ignored.
# Needs more authority than STABILIZE because it is doing all the flying.
PID_TRAINER = dict(kp=4.0, ki=0.8, kd=0.12)

# Integral clamp, in output microseconds. Also bounded by MAX_CORRECTION_US.
PID_INTEGRAL_LIMIT_US = 120.0

# Derivative low-pass time constant (s). The roll estimate is noisy; differentiating
# it raw injects that noise straight into the servo. ~50ms rolls off above ~3Hz.
PID_DERIVATIVE_TAU_S = 0.05

# ------------------------------------------------------------- Horizon detect
HORIZON_MIN_POINTS = 25          # fewer inlier points than this = no estimate
HORIZON_RANSAC_ITERS = 60
HORIZON_INLIER_PX = 3.0
HORIZON_MIN_INLIER_FRAC = 0.45   # inlier fraction below this = low confidence, rejected

# Predictive gating: points further than this from the predicted horizon line are
# ignored. Keeps ground clutter and cloud edges out of the fit once locked on.
HORIZON_GATE_PX = 40.0
HORIZON_GATE_ENABLE_AFTER = 3    # consecutive good frames before gating engages

# Reject an estimate that jumps more than this from the previous one. A fixed-wing
# cannot roll 60 deg between two 20ms frames; such a jump is a detection error.
HORIZON_MAX_JUMP_DEG = 25.0

# Sky colour gate (HSV). Widen for hazy/overcast conditions; see the tuning tool.
HORIZON_SKY_LOWER = (100, 20, 40)
HORIZON_SKY_UPPER = (140, 255, 180)

# Roll smoothing time constant (s). Kept SHORT on purpose: heavy smoothing adds
# phase lag directly ahead of the D-term and will make the loop oscillate.
ROLL_FILTER_TAU_S = 0.04

# How long the horizon may be missing before the stabilizer stands down.
HORIZON_LOST_TIMEOUT_S = 0.5

# ------------------------------------------------------------- Logging
VIDEO_LOG_ENABLED = False        # costs CPU; off by default for flight
VIDEO_LOG_PATH = "flight_log.mp4"
VIDEO_LOG_FPS = 20
CSV_LOG_ENABLED = True
CSV_LOG_PATH = "flight_log.csv"
