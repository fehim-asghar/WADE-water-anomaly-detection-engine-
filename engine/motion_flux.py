import cv2
import numpy as np
from typing import Dict, Tuple, Optional


class MotionFluxAnalyzer:
    """
    Stream 2: Bounding-Box Optical Flux Analyzer
    Measures hydrodynamic churn and splash turbulence using temporal frame differencing.
    Near-zero computational cost (<0.2ms) with zero additional neural network models.
    """

    def __init__(self, energy_scale: float = 1.0):
        # Maps person track_id -> previous grayscale bounding-box ROI
        self.prev_rois: Dict[int, np.ndarray] = {}
        self.energy_scale = energy_scale

    def calculate_splash_energy(
        self,
        frame_gray: np.ndarray,
        track_id: int,
        bbox: Tuple[int, int, int, int]
    ) -> float:
        """
        Calculates local motion flux energy inside the subject's bounding box.
        
        Args:
            frame_gray: Grayscale full frame.
            track_id: Persistent tracking ID assigned by ByteTrack.
            bbox: (x1, y1, x2, y2) coordinates.
            
        Returns:
            Normalized splash energy float (0.0 to 1.0).
        """
        h, w = frame_gray.shape[:2]
        x1, y1, x2, y2 = [int(v) for v in bbox]
        
        # Clamp coordinates within frame boundaries
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        if (x2 - x1) <= 5 or (y2 - y1) <= 5:
            return 0.0

        curr_roi = frame_gray[y1:y2, x1:x2]

        if track_id not in self.prev_rois:
            self.prev_rois[track_id] = curr_roi.copy()
            return 0.0

        prev_roi = self.prev_rois[track_id]

        # Resize previous ROI if bbox dimensions changed slightly due to motion
        if prev_roi.shape != curr_roi.shape:
            prev_roi = cv2.resize(prev_roi, (curr_roi.shape[1], curr_roi.shape[0]))

        # Compute instantaneous absolute pixel flux
        diff = cv2.absdiff(prev_roi, curr_roi)
        mean_diff = float(np.mean(diff))

        # Save current ROI for next frame
        self.prev_rois[track_id] = curr_roi.copy()

        # Normalize: mean difference typically ranges 0 to 45 in water turbulence
        # Values > 25 represent intense white-water splashing / thrashing
        normalized_energy = min(1.0, (mean_diff / 30.0) * self.energy_scale)
        return round(normalized_energy, 3)

    def cleanup_missing_tracks(self, active_track_ids: list):
        """Removes memory for swimmers who left the pool or were lost."""
        active_set = set(active_track_ids)
        stale_ids = [tid for tid in self.prev_rois if tid not in active_set]
        for tid in stale_ids:
            del self.prev_rois[tid]
