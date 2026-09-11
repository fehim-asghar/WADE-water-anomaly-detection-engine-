import argparse
import time
import cv2
import torch
import numpy as np

from engine.pose_detector import PoseDetector
from engine.motion_flux import MotionFluxAnalyzer
from engine.distress_engine import DistressEngine
from engine.hud_renderer import TacticalHUDRenderer


def run_wade(source: str, model_name: str = "yolov8s-pose.pt", pool_name: str = "OLYMPIC POOL - CCTV 04", save_output: str = None, headless: bool = False, max_frames: int = 0):
    print("=" * 65)
    print("💧 WADE — Water Anomaly Detection Engine [OpenCV Edition]")
    print(f"[*] Target Source   : {source}")
    print(f"[*] Model Checkpoint: {model_name}")
    print(f"[*] Headless Mode   : {headless}")
    print(f"[*] CUDA Available  : {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"[*] Active Device   : {torch.cuda.get_device_name(0)}")
    print("=" * 65)

    # Initialize Modules
    pose_detector = PoseDetector(model_name=model_name)
    motion_analyzer = MotionFluxAnalyzer()
    distress_engine = DistressEngine(alert_window_sec=2.0, target_fps=30.0)
    hud_renderer = TacticalHUDRenderer(pool_name=pool_name)

    # Video Source (0 = webcam, or file path)
    cap_source = int(source) if source.isdigit() else source
    cap = cv2.VideoCapture(cap_source)

    if not cap.isOpened():
        print(f"[!] Error: Could not open video source [{source}]")
        return

    # Video Writer if requested
    writer = None
    if save_output:
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps_in = cap.get(cv2.CAP_PROP_FPS) or 30.0
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(save_output, fourcc, fps_in, (w, h))
        print(f"[*] Saving annotated recording to: {save_output}")

    # FPS & Latency Measurement (Honest 3-tier metrics)
    fps_history = []
    frame_count = 0
    t_start = time.time()

    if not headless:
        cv2.namedWindow("WADE - Water Anomaly Detection Engine", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("WADE - Water Anomaly Detection Engine", 1280, 720)

    device_label = "RTX 4050" if torch.cuda.is_available() else "CPU"

    try:
        while True:
            t_loop_start = time.time()
            ret, frame = cap.read()

            if not ret:
                # Loop video for continuous interactive presentation demo
                if not source.isdigit() and not save_output and not headless:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                else:
                    break

            frame_count += 1
            if max_frames > 0 and frame_count > max_frames:
                print(f"[*] Reached max_frames limit ({max_frames}). Stopping.")
                break
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # 1. Pose Kinematics Perception
            detections, inference_ms = pose_detector.detect(frame)

            # 2. Multi-Metric Distress & Flux Processing
            processed_subjects = []
            active_ids = []

            for det in detections:
                tid = det["track_id"]
                active_ids.append(tid)

                # Stream 2: Motion Flux Splash Energy
                splash_energy = motion_analyzer.calculate_splash_energy(
                    frame_gray=frame_gray,
                    track_id=tid,
                    bbox=det["bbox"]
                )

                # Stream 1 + 2 Fusion: 4-Quadrant Truth Table & Temporal Accumulator
                swimmer_state = distress_engine.process_swimmer(
                    track_id=tid,
                    centroid=det["centroid"],
                    torso_angle_deg=det["torso_angle"],
                    splash_energy=splash_energy,
                    current_time=t_loop_start,
                    bbox=det["bbox"]
                )

                swimmer_state["bbox"] = det["bbox"]
                swimmer_state["keypoints"] = det["keypoints"]
                processed_subjects.append(swimmer_state)

            # Cleanup lost tracks
            motion_analyzer.cleanup_missing_tracks(active_ids)
            distress_engine.cleanup(active_ids, current_time=t_loop_start)

            # Calculate Rolling Display FPS
            t_loop_end = time.time()
            loop_duration = max(1e-4, t_loop_end - t_loop_start)
            fps_history.append(1.0 / loop_duration)
            if len(fps_history) > 30:
                fps_history.pop(0)
            avg_fps = float(np.mean(fps_history))

            # 3. Render Tactical Military-Grade OpenCV HUD
            annotated_frame = hud_renderer.draw(
                frame=frame,
                subjects=processed_subjects,
                fps=avg_fps,
                inference_ms=inference_ms,
                device_name=device_label
            )

            if writer:
                writer.write(annotated_frame)

            if not headless:
                cv2.imshow("WADE - Water Anomaly Detection Engine", annotated_frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                elif key == ord(" "):
                    cv2.waitKey(-1)
                elif key == ord("s"):
                    snap_path = f"snapshot_incident_{int(time.time())}.jpg"
                    cv2.imwrite(snap_path, annotated_frame)
                    print(f"[+] Incident snapshot saved to: {snap_path}")

    finally:
        cap.release()
        if writer:
            writer.release()
        if not headless:
            cv2.destroyAllWindows()
        total_time = time.time() - t_start
        print("\n" + "=" * 65)
        print(f"[*] Session Finished: Processed {frame_count} frames in {total_time:.2f}s")
        print(f"[*] True Average System FPS: {frame_count / max(1.0, total_time):.2f}")
        print("=" * 65)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="WADE: Water Anomaly Detection Engine")
    parser.add_argument("--source", type=str, default="0", help="Video source (0 for webcam, or path to MP4)")
    parser.add_argument("--model", type=str, default="yolov8s-pose.pt", help="YOLO-Pose checkpoint")
    parser.add_argument("--pool", type=str, default="OLYMPIC POOL - CCTV 04", help="Pool Zone Label")
    parser.add_argument("--save", type=str, default=None, help="Path to save output MP4 recording")
    parser.add_argument("--headless", action="store_true", help="Run in headless mode without window display")
    parser.add_argument("--max-frames", type=int, default=0, help="Maximum number of frames to process (0 for infinite)")
    args = parser.parse_args()

    run_wade(source=args.source, model_name=args.model, pool_name=args.pool, save_output=args.save, headless=args.headless, max_frames=args.max_frames)
