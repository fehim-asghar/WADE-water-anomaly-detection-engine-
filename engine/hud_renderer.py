import cv2
import numpy as np
import time
from typing import List, Dict, Tuple


class TacticalHUDRenderer:
    """
    Direct OpenCV Tactical HUD Renderer.
    Burns military-grade security surveillance telemetry directly onto the video frame.
    Zero browser dependencies, zero network latency.
    """

    COLOR_SAFE = (0, 230, 115)       # Neon Emerald Green (BGR)
    COLOR_WATCH = (0, 191, 255)      # Electric Amber / Orange-Yellow
    COLOR_CRITICAL = (30, 30, 255)   # Intense Crimson Red
    COLOR_CYAN = (240, 200, 0)       # Cyber Cyan
    COLOR_BG_DARK = (15, 15, 18)     # Deep Obsidian Black

    def __init__(self, pool_name: str = "OLYMPIC POOL - CCTV 04"):
        self.pool_name = pool_name
        self.strobe_counter = 0

    def draw(
        self,
        frame: np.ndarray,
        subjects: List[Dict],
        fps: float,
        inference_ms: float,
        device_name: str = "RTX 4050"
    ) -> np.ndarray:
        """
        Renders complete tactical overlay onto frame.
        """
        out = frame.copy()
        h, w = out.shape[:2]
        scale = max(0.45, min(1.0, w / 1280.0))
        self.strobe_counter += 1

        has_critical = any(s["state"] == "CRITICAL" for s in subjects)
        has_watch = any(s["state"] == "WATCH" for s in subjects)

        # 1. Full-Screen Flashing Red Border on Critical Alert
        if has_critical and (self.strobe_counter % 8 < 5):
            border_thick = 12
            cv2.rectangle(out, (0, 0), (w, h), self.COLOR_CRITICAL, border_thick)

        # 2. Render Swimmer Bounding Boxes & Diagnostics
        for s in subjects:
            bbox = s["bbox"]
            state = s["state"]
            track_id = s["track_id"]
            angle = s["torso_angle"]
            vel = s["velocity"]
            splash = s["splash_energy"]
            threat = s["threat_score"]
            reason = s["reason"]

            x1, y1, x2, y2 = [int(v) for v in bbox]

            # Choose color
            if state == "CRITICAL":
                color = self.COLOR_CRITICAL
                thickness = 3
            elif state == "WATCH":
                color = self.COLOR_WATCH
                thickness = 2
            else:
                color = self.COLOR_SAFE
                thickness = 2

            # Draw Tactical Bounding Box with Corner Accents
            cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness)
            self._draw_corner_brackets(out, x1, y1, x2, y2, color, length=14, thickness=3)

            # Swimmer HUD Tag Header
            tag_scale = max(0.38, 0.50 * scale)
            tag_text = f"ID #{track_id} | {state}"
            if state != "SAFE":
                tag_text += f" ({s['distress_duration']}s)"

            (tw, th), _ = cv2.getTextSize(tag_text, cv2.FONT_HERSHEY_SIMPLEX, tag_scale, 1)
            cv2.rectangle(out, (x1, y1 - th - 6), (x1 + tw + 8, y1), color, -1)
            cv2.putText(out, tag_text, (x1 + 4, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, tag_scale, (0, 0, 0), 1, cv2.LINE_AA)

            # Telemetry Metrics below bounding box
            diag_scale = max(0.32, 0.42 * scale)
            diag_text = f"ANG: {angle:.0f}* | VEL: {vel:.0f}px/s | FLUX: {splash:.2f}"
            (dw, dh), _ = cv2.getTextSize(diag_text, cv2.FONT_HERSHEY_SIMPLEX, diag_scale, 1)
            cv2.rectangle(out, (x1, y2), (x1 + dw + 6, y2 + dh + 6), (20, 20, 20), -1)
            cv2.putText(out, diag_text, (x1 + 3, y2 + dh + 1), cv2.FONT_HERSHEY_SIMPLEX, diag_scale, (220, 220, 220), 1, cv2.LINE_AA)

            # If Critical, Draw Emergency Pulsing Beacon Box Above Head
            if state == "CRITICAL":
                crit_scale = max(0.48, 0.65 * scale)
                alert_banner = "🚨 DROWNING ANOMALY DETECTED"
                (aw, ah), _ = cv2.getTextSize(alert_banner, cv2.FONT_HERSHEY_SIMPLEX, crit_scale, 2)
                bx = max(10, x1 - 20)
                by = max(ah + 10, y1 - 20)
                cv2.rectangle(out, (bx - 4, by - ah - 4), (bx + aw + 4, by + 4), self.COLOR_CRITICAL, -1)
                cv2.putText(out, alert_banner, (bx, by - 2), cv2.FONT_HERSHEY_SIMPLEX, crit_scale, (255, 255, 255), 2, cv2.LINE_AA)

        # 3. Top Tactical Surveillance Banner (Dark semi-transparent header)
        self._render_top_banner(out, w, fps, inference_ms, device_name, len(subjects), has_critical, has_watch)

        return out

    def _draw_corner_brackets(self, img, x1, y1, x2, y2, color, length=12, thickness=2):
        # Top-Left
        cv2.line(img, (x1, y1), (x1 + length, y1), color, thickness)
        cv2.line(img, (x1, y1), (x1, y1 + length), color, thickness)
        # Top-Right
        cv2.line(img, (x2, y1), (x2 - length, y1), color, thickness)
        cv2.line(img, (x2, y1), (x2, y1 + length), color, thickness)
        # Bottom-Left
        cv2.line(img, (x1, y2), (x1 + length, y2), color, thickness)
        cv2.line(img, (x1, y2), (x1, y2 - length), color, thickness)
        # Bottom-Right
        cv2.line(img, (x2, y2), (x2 - length, y2), color, thickness)
        cv2.line(img, (x2, y2), (x2, y2 - length), color, thickness)

    def _render_top_banner(self, img, w, fps, inf_ms, device, active_count, is_crit, is_watch):
        scale = max(0.45, min(1.0, w / 1280.0))
        header_h = int(46 * scale)
        if header_h < 28:
            header_h = 28

        # Draw translucent top bar
        overlay = img.copy()
        cv2.rectangle(overlay, (0, 0), (w, header_h), self.COLOR_BG_DARK, -1)
        cv2.addWeighted(overlay, 0.85, img, 0.15, 0, img)
        cv2.line(img, (0, header_h), (w, header_h), (60, 60, 60), 1)

        font_scale = 0.50 * scale
        thickness = 1 if scale < 0.7 else 2

        # Left Branding
        brand = f"WADE v1.0 // {self.pool_name}"
        cv2.putText(img, brand, (int(10 * scale), int(header_h * 0.68)), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

        # Right Telemetry HUD (Live FPS, Latency, Target GPU)
        telemetry = f"FPS: {fps:.1f} | {inf_ms:.1f}ms | {device} | TRACKS: {active_count}"
        (tw, _), _ = cv2.getTextSize(telemetry, cv2.FONT_HERSHEY_SIMPLEX, font_scale * 0.9, 1)
        cv2.putText(img, telemetry, (w - tw - int(10 * scale), int(header_h * 0.68)), cv2.FONT_HERSHEY_SIMPLEX, font_scale * 0.9, (0, 220, 255), 1, cv2.LINE_AA)

        # Center Threat Status (Only if width is wide enough to avoid overlap)
        brand_w, _ = cv2.getTextSize(brand, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
        space_avail = (w - tw) - brand_w

        if is_crit:
            status_text = "🚨 ALARM: CRITICAL DISTRESS"
            status_color = self.COLOR_CRITICAL
        elif is_watch:
            status_text = "⚠️ WATCH: ELEVATED DISTRESS"
            status_color = self.COLOR_WATCH
        else:
            status_text = "● ALL ZONES CLEAR"
            status_color = self.COLOR_SAFE

        (sw, _), _ = cv2.getTextSize(status_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        if space_avail > sw + 20:
            cv2.putText(img, status_text, ((w - sw) // 2, int(header_h * 0.68)), cv2.FONT_HERSHEY_SIMPLEX, font_scale, status_color, thickness, cv2.LINE_AA)

