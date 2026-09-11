"""
Automated Behavioral Verification Test for WADE
Validates zero false alarms on normal swimming and verified alarm trigger on drowning.
"""
import sys
import os
import time
import cv2

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from engine.pose_detector import PoseDetector
from engine.motion_flux import MotionFluxAnalyzer
from engine.distress_engine import DistressEngine


def test_video(video_path, expected_alarm=False):
    print(f"\n[*] Testing video: {video_path}")
    cap = cv2.VideoCapture(video_path)
    detector = PoseDetector()
    flux = MotionFluxAnalyzer()
    engine = DistressEngine(alert_window_sec=3.5, target_fps=30.0)

    critical_triggered = False
    critical_timestamp = None
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        t_now = frame_idx / 30.0  # Normalized video time
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        detections, _ = detector.detect(frame)
        active_ids = []

        for det in detections:
            tid = det["track_id"]
            active_ids.append(tid)
            splash = flux.calculate_splash_energy(frame_gray, tid, det["bbox"])
            state = engine.process_swimmer(tid, det["centroid"], det["torso_angle"], splash, t_now)

            if state["state"] == "CRITICAL" and not critical_triggered:
                critical_triggered = True
                critical_timestamp = t_now
                print(f"  🚨 [EVENT TRIGGERED] Track #{tid} marked CRITICAL at {t_now:.1f}s | Reason: {state['reason']}")

        flux.cleanup_missing_tracks(active_ids)
        engine.cleanup(active_ids)

    cap.release()

    if expected_alarm:
        assert critical_triggered, f"FAILED: Expected alarm on {video_path}, but none fired!"
        print(f"  ✅ PASS: Drowning successfully caught at {critical_timestamp:.1f}s")
    else:
        assert not critical_triggered, f"FAILED: False alarm fired on {video_path} at {critical_timestamp:.1f}s!"
        print(f"  ✅ PASS: Zero false alarms detected on normal swimming")


if __name__ == "__main__":
    print("=" * 65)
    print("🧪 WADE AUTOMATED BEHAVIORAL ACCURACY VERIFICATION")
    print("=" * 65)
    test_video("assets/demo_normal_swimming.mp4", expected_alarm=False)
    test_video("assets/demo_drowning_incident.mp4", expected_alarm=True)
    print("\n" + "=" * 65)
    print("🏆 ALL BEHAVIORAL TESTS PASSED WITH 100% ACCURACY!")
    print("=" * 65)
