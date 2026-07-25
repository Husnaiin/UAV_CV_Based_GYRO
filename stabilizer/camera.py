"""Threaded frame capture.

Control rate must be steady and independent of vision rate. Capturing inline
with the control loop would tie servo updates to whatever the camera happened to
deliver, and any slow frame would stall the loop. Instead the PID runs at a fixed
CONTROL_HZ and uses the most recent roll estimate this thread has produced, so a
dropped or slow frame degrades estimate freshness rather than control.
"""

import threading
import time


class CameraThread:
    """Continuously captures frames, keeping only the newest."""

    def __init__(self, picam2):
        self._picam2 = picam2
        self._frame = None
        self._seq = 0
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="capture", daemon=True)
        self.error = None

    def start(self):
        self._thread.start()
        return self

    def _run(self):
        while not self._stop.is_set():
            try:
                frame = self._picam2.capture_array()
            except Exception as exc:          # camera failure must not kill control
                self.error = exc
                time.sleep(0.05)
                continue
            with self._lock:
                self._frame = frame
                self._seq += 1

    def latest(self):
        """Return (frame, sequence). frame is None until the first capture."""
        with self._lock:
            return self._frame, self._seq

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=2.0)
