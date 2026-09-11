import time
import math
import numpy as np
from typing import Dict, Tuple, List, Optional
from collections import deque


class SwimmerStateTracker:
    """Tracks temporal kinematics and distress accumulation for an individual swimmer ID."""

    def __init__(self, track_id: int, window_sec: float = 3.5, fps: float = 30.0):
        self.track_id = track_id
        self.window_sec = window_sec
        self.fps = fps
        self.max_history = int(window_sec * fps)

        # Centroid history for velocity calculation [(timestamp, x, y)]
        self.centroid_history = deque(maxlen=self.max_history)
        # Threat score history
        self.threat_history = deque(maxlen=self.max_history)
        
        # State tracking
        self.first_distress_time: Optional[float] = None
        self.current_state: str = "SAFE"  # SAFE, WATCH, CRITICAL
        self.threat_score: float = 0.0
        self.distress_duration: float = 0.0
        self.last_torso_angle: float = 0.0
        self.last_velocity: float = 0.0
        self.last_splash_energy: float = 0.0
        self.reason: str = "Normal Swimming"

    def update(
        self,
        centroid: Tuple[float, float],
        torso_angle_deg: float,
        splash_energy: float,
        current_time: float
    ) -> Dict:
        """
        Updates kinematics, computes 4-quadrant truth table, and evaluates temporal accumulator.
        """
        self.centroid_history.append((current_time, centroid[0], centroid[1]))
        self.last_torso_angle = torso_angle_deg
        self.last_splash_energy = splash_energy

        # Calculate lateral velocity (pixels per second) over a rolling 1-second window
        velocity_px_sec = self._compute_velocity(current_time, window_duration=1.0)
        self.last_velocity = velocity_px_sec

        # 4-Quadrant Truth Table Evaluation
        # High Angle = Vertical (>65°), Low Velocity = Stuck in place (<20 px/s)
        is_vertical = torso_angle_deg >= 65.0
        is_stationary = velocity_px_sec < 25.0
        is_splashing = splash_energy >= 0.35

        # Determine instantaneous frame threat (0.0 to 1.0)
        if is_vertical and is_stationary:
            if is_splashing:
                # Quadrant 3: Active Drowning (Panicked struggle / clawing water)
                frame_threat = 0.95
                instant_reason = "Active Drowning (Vertical + Panic Splash)"
            else:
                # Quadrant 4: Passive Drowning (Unconscious / Silent sinking)
                frame_threat = 0.90
                instant_reason = "Passive Drowning (Vertical Submersion)"
        elif is_vertical and not is_stationary:
            # Treading / moving while vertical (e.g. water polo, stepping)
            frame_threat = 0.40
            instant_reason = "Vertical Motion (Treading)"
        elif not is_vertical and is_stationary:
            # Floating on back / resting horizontally
            frame_threat = 0.15
            instant_reason = "Horizontal Float (Relaxing)"
        else:
            # Quadrant 1 & 2: Normal Swimming (Horizontal + Moving)
            frame_threat = 0.05
            instant_reason = "Normal Swimming"

        self.threat_history.append(frame_threat)
        # Smoothed rolling threat score
        self.threat_score = float(np.mean(self.threat_history))

        # Temporal Accumulator: Must sustain threat > 0.65 for >= window_sec
        if self.threat_score >= 0.65:
            if self.first_distress_time is None:
                self.first_distress_time = current_time
            self.distress_duration = round(current_time - self.first_distress_time, 1)

            if self.distress_duration >= self.window_sec:
                self.current_state = "CRITICAL"
                self.reason = f"🚨 {instant_reason} ({self.distress_duration}s)"
            else:
                self.current_state = "WATCH"
                self.reason = f"⚠️ Potential Distress ({self.distress_duration}s / {self.window_sec}s)"
        else:
            # Cooldown / reset distress timer if normal movement resumes
            self.first_distress_time = None
            self.distress_duration = 0.0
            self.current_state = "SAFE"
            self.reason = instant_reason

        return {
            "track_id": self.track_id,
            "state": self.current_state,
            "threat_score": round(self.threat_score, 2),
            "torso_angle": round(self.last_torso_angle, 1),
            "velocity": round(self.last_velocity, 1),
            "splash_energy": round(self.last_splash_energy, 2),
            "distress_duration": self.distress_duration,
            "reason": self.reason
        }

    def _compute_velocity(self, current_time: float, window_duration: float = 1.0) -> float:
        if len(self.centroid_history) < 2:
            return 0.0
        
        # Look back up to window_duration seconds
        t_target = current_time - window_duration
        idx = 0
        for i, (t, _, _) in enumerate(self.centroid_history):
            if t >= t_target:
                idx = i
                break

        t_start, x_start, y_start = self.centroid_history[idx]
        t_end, x_end, y_end = self.centroid_history[-1]

        dt = t_end - t_start
        if dt <= 0.05:
            return 0.0

        dist = math.hypot(x_end - x_start, y_end - y_start)
        return dist / dt


class DistressEngine:
    """Manages multi-swimmer behavioral states and pool-wide threat triage."""

    def __init__(self, alert_window_sec: float = 3.5, target_fps: float = 30.0):
        self.alert_window_sec = alert_window_sec
        self.target_fps = target_fps
        self.swimmers: Dict[int, SwimmerStateTracker] = {}

    def process_swimmer(
        self,
        track_id: int,
        centroid: Tuple[float, float],
        torso_angle_deg: float,
        splash_energy: float,
        current_time: Optional[float] = None
    ) -> Dict:
        if current_time is None:
            current_time = time.time()

        if track_id not in self.swimmers:
            self.swimmers[track_id] = SwimmerStateTracker(
                track_id=track_id,
                window_sec=self.alert_window_sec,
                fps=self.target_fps
            )

        return self.swimmers[track_id].update(
            centroid=centroid,
            torso_angle_deg=torso_angle_deg,
            splash_energy=splash_energy,
            current_time=current_time
        )

    def cleanup(self, active_track_ids: List[int]):
        active_set = set(active_track_ids)
        stale = [tid for tid in self.swimmers if tid not in active_set]
        for tid in stale:
            del self.swimmers[tid]
