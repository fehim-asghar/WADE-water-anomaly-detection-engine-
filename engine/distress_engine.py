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
        self.threat_history = deque(maxlen=self.max_history)
        self.aspect_history = deque(maxlen=self.max_history)
        
        # State tracking
        self.last_seen_time = 0.0
        self.fall_impact_detected = False
        self.fall_impact_time = 0.0
        self.first_distress_time: Optional[float] = None
        self.current_state: str = "SAFE"
        self.threat_score: float = 0.0
        self.distress_duration: float = 0.0
        self.last_torso_angle: float = 0.0
        self.last_velocity: float = 0.0
        self.last_splash_energy: float = 0.0
        self.reason: str = "Normal"

    def update(
        self,
        centroid: Tuple[float, float],
        torso_angle_deg: float,
        splash_energy: float,
        current_time: float,
        bbox: Optional[List[float]] = None
    ) -> Dict:
        """
        Updates kinematics with Dynamic Fall Detection (Descent Velocity Spike + Post-Impact Ground Stillness).
        """
        self.last_seen_time = current_time
        self.centroid_history.append((current_time, centroid[0], centroid[1]))
        self.last_torso_angle = torso_angle_deg
        self.last_splash_energy = splash_energy

        # Calculate horizontal and vertical velocities
        vx, vy, v_total = self._compute_split_velocities(current_time, window_duration=0.5)
        self.last_velocity = v_total

        aspect = 1.0
        if bbox:
            w_box = max(1.0, bbox[2] - bbox[0])
            h_box = max(1.0, bbox[3] - bbox[1])
            aspect = h_box / w_box
            self.aspect_history.append((current_time, aspect))

        is_flat_on_floor = (torso_angle_deg <= 35.0 or aspect < 0.75)
        is_stationary = v_total < 25.0

        # Detect Sudden Dynamic Fall Impact: rapid downward plunge (vy > 50 px/s)
        # or rapid aspect ratio collapse within 0.8s
        is_sudden_plunge = vy > 50.0
        is_aspect_collapsed = False
        if len(self.aspect_history) > 3:
            past_aspect = self.aspect_history[0][1]
            if past_aspect > 1.1 and aspect < 0.8:
                is_aspect_collapsed = True

        if (is_sudden_plunge or is_aspect_collapsed) and (current_time - self.fall_impact_time > 3.0):
            self.fall_impact_detected = True
            self.fall_impact_time = current_time

        # High Threat only if subject experienced a fall event OR is flat on floor in a corridor/room motionless
        # Seated diners have aspect > 1.0 (vertical torso), while collapsed subjects have aspect < 0.8
        if self.fall_impact_detected or (is_flat_on_floor and aspect < 0.8):
            if is_stationary:
                frame_threat = 0.95
                instant_reason = "🚨 UNRESPONSIVE COLLAPSE ON FLOOR"
            else:
                frame_threat = 0.70
                instant_reason = "⚠️ FALL IMPACT DETECTED (Recovering)"
        elif is_stationary:
            # Sitting calmly at a table or standing in place
            frame_threat = 0.05
            instant_reason = "Seated / Resting"
        else:
            frame_threat = 0.05
            instant_reason = "Normal Movement"

        self.threat_history.append(frame_threat)
        self.threat_score = float(np.mean(self.threat_history))

        # Temporal Accumulator: Must sustain threat > 0.65 for >= window_sec
        if self.threat_score >= 0.65:
            if self.first_distress_time is None:
                self.first_distress_time = current_time
            self.distress_duration = round(current_time - self.first_distress_time, 1)

            if self.distress_duration >= self.window_sec:
                self.current_state = "CRITICAL"
                self.reason = instant_reason
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

    def _compute_split_velocities(self, current_time: float, window_duration: float = 0.5) -> Tuple[float, float, float]:
        if len(self.centroid_history) < 2:
            return 0.0, 0.0, 0.0

        t_target = current_time - window_duration
        idx = 0
        for i, (t, _, _) in enumerate(self.centroid_history):
            if t >= t_target:
                idx = i
                break

        t_start, x_start, y_start = self.centroid_history[idx]
        t_end, x_end, y_end = self.centroid_history[-1]

        dt = t_end - t_start
        if dt <= 0.04:
            return 0.0, 0.0, 0.0

        vx = abs(x_end - x_start) / dt
        vy = (y_end - y_start) / dt  # Positive means downward plunge
        v_total = math.hypot(x_end - x_start, y_end - y_start) / dt
        return vx, vy, v_total

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
        current_time: Optional[float] = None,
        bbox: Optional[List[float]] = None
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
            current_time=current_time,
            bbox=bbox
        )

    def cleanup(self, active_track_ids: List[int], current_time: Optional[float] = None, timeout_sec: float = 2.5):
        if current_time is None:
            current_time = time.time()
        active_set = set(active_track_ids)
        stale = [
            tid for tid, tracker in self.swimmers.items()
            if tid not in active_set and (current_time - tracker.last_seen_time > timeout_sec)
        ]
        for tid in stale:
            del self.swimmers[tid]
