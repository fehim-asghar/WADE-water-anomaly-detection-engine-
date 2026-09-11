"""
Synthetic Pool Video Generator for WADE Demonstration
Creates clean, realistic demo video assets:
1. assets/normal_swimming.mp4  (Swimmers moving horizontally across lanes)
2. assets/drowning_incident.mp4 (Swimmer enters vertical distress state at 3.0s)
"""
import os
import math
import cv2
import numpy as np


def draw_human_stick(img, x, y, angle_deg, scale=1.0, is_splashing=False):
    """
    Renders a clean human figure with realistic joints so YOLO-Pose reliably detects it.
    angle_deg: 0 = horizontal prone, 90 = vertical upright.
    """
    rad = math.radians(angle_deg)
    head_r = int(12 * scale)
    torso_len = int(50 * scale)
    limb_len = int(35 * scale)

    # Torso direction vector
    dx = math.cos(rad) * torso_len
    dy = math.sin(rad) * torso_len

    head_x = int(x - dx * 0.4)
    head_y = int(y - dy * 0.4)
    hip_x = int(x + dx * 0.6)
    hip_y = int(y + dy * 0.6)

    # Color: skin tone / swimsuit
    skin_color = (180, 210, 255)
    suit_color = (200, 50, 50)

    # Head
    cv2.circle(img, (head_x, head_y), head_r, skin_color, -1)

    # Torso
    cv2.line(img, (head_x, head_y), (hip_x, hip_y), suit_color, int(8 * scale))

    # Shoulders (perpendicular)
    sh_dx = -math.sin(rad) * 16 * scale
    sh_dy = math.cos(rad) * 16 * scale
    sh_x = int(head_x + dx * 0.2)
    sh_y = int(head_y + dy * 0.2)

    l_arm_x = int(sh_x + sh_dx)
    l_arm_y = int(sh_y + sh_dy)
    r_arm_x = int(sh_x - sh_dx)
    r_arm_y = int(sh_y - sh_dy)
    cv2.line(img, (l_arm_x, l_arm_y), (r_arm_x, r_arm_y), skin_color, int(5 * scale))

    # Arms
    cv2.line(img, (l_arm_x, l_arm_y), (int(l_arm_x + dx * 0.3), int(l_arm_y + dy * 0.3)), skin_color, int(4 * scale))
    cv2.line(img, (r_arm_x, r_arm_y), (int(r_arm_x + dx * 0.3), int(r_arm_y + dy * 0.3)), skin_color, int(4 * scale))

    # Legs from hips
    l_leg_x = int(hip_x + sh_dx * 0.6 + dx * 0.4)
    l_leg_y = int(hip_y + sh_dy * 0.6 + dy * 0.4)
    r_leg_x = int(hip_x - sh_dx * 0.6 + dx * 0.4)
    r_leg_y = int(hip_y - sh_dy * 0.6 + dy * 0.4)
    cv2.line(img, (hip_x, hip_y), (l_leg_x, l_leg_y), skin_color, int(5 * scale))
    cv2.line(img, (hip_x, hip_y), (r_leg_x, r_leg_y), skin_color, int(5 * scale))

    # Splash particles
    if is_splashing:
        for _ in range(15):
            px = int(x + np.random.randint(-35, 35))
            py = int(y + np.random.randint(-25, 25))
            cv2.circle(img, (px, py), np.random.randint(2, 5), (255, 255, 255), -1)


def generate_pool_frame(w=1280, h=720, t=0.0):
    """Generates a photorealistic swimming pool with lane lines and water ripple gradients."""
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    
    # Cyan-blue water gradient
    for y in range(h):
        ratio = y / float(h)
        b = int(160 + 50 * math.sin(t * 2 + ratio * 8))
        g = int(120 + 30 * ratio)
        r = 20
        frame[y, :] = (b, g, r)

    # Lane Lines
    for lane_y in [180, 360, 540]:
        for x in range(0, w, 40):
            cv2.line(frame, (x, lane_y), (x + 20, lane_y), (255, 255, 255), 2)

    return frame


def create_demo_assets(output_dir="assets"):
    os.makedirs(output_dir, exist_ok=True)
    fps = 30
    duration_sec = 10
    total_frames = fps * duration_sec
    w, h = 1280, 720
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    # 1. Normal Lap Swimming
    path_normal = os.path.join(output_dir, "demo_normal_swimming.mp4")
    out_normal = cv2.VideoWriter(path_normal, fourcc, fps, (w, h))
    print(f"[*] Generating: {path_normal}...")

    for f in range(total_frames):
        t = f / float(fps)
        frame = generate_pool_frame(w, h, t)

        # Swimmer 1 in Lane 1 (Moving horizontally fast)
        x1 = int(150 + (t * 85) % (w - 300))
        y1 = 270 + int(math.sin(t * 4) * 6)
        draw_human_stick(frame, x1, y1, angle_deg=10, scale=1.1, is_splashing=(f % 6 < 3))

        # Swimmer 2 in Lane 2 (Moving smoothly across)
        x2 = int(w - 200 - (t * 65) % (w - 300))
        y2 = 450 + int(math.sin(t * 3) * 5)
        draw_human_stick(frame, x2, y2, angle_deg=170, scale=1.0, is_splashing=(f % 8 < 2))

        out_normal.write(frame)

    out_normal.release()
    print(f"[+] Successfully generated: {path_normal}")

    # 2. Drowning Incident Simulation
    path_drowning = os.path.join(output_dir, "demo_drowning_incident.mp4")
    out_drown = cv2.VideoWriter(path_drowning, fourcc, fps, (w, h))
    print(f"[*] Generating: {path_drowning}...")

    for f in range(total_frames):
        t = f / float(fps)
        frame = generate_pool_frame(w, h, t)

        # Normal Swimmer in Lane 1 (Safe)
        x1 = int(120 + (t * 70) % (w - 250))
        draw_human_stick(frame, x1, 250, angle_deg=15, scale=1.1, is_splashing=False)

        # Distress Victim in Lane 2
        if t < 3.0:
            # First 3 seconds: normal horizontal swimming
            xv = int(450 + t * 30)
            yv = 450
            draw_human_stick(frame, xv, yv, angle_deg=15, scale=1.1, is_splashing=False)
        else:
            # 3.0s onward: STOPS translation, flips strictly vertical (85°), bobs in place with panic splash!
            distress_t = t - 3.0
            xv = 540  # Stuck in place!
            # Vertical head bobbing oscillation
            yv = int(450 + math.sin(distress_t * 6) * 18)
            # High panic splash + strictly vertical 85° angle
            draw_human_stick(frame, xv, yv, angle_deg=85, scale=1.1, is_splashing=True)

        out_drown.write(frame)

    out_drown.release()
    print(f"[+] Successfully generated: {path_drowning}")


if __name__ == "__main__":
    create_demo_assets()
