import sys
import os
import cv2
import yt_dlp

# Add project root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from main import run_wade


def get_youtube_stream_url(youtube_url: str) -> str:
    """Extracts direct mp4 stream URL from YouTube without saving any file to disk."""
    print(f"[*] Resolving direct video stream for: {youtube_url}...")
    ydl_opts = {
        "format": "best[ext=mp4]/best",
        "quiet": True,
        "no_warnings": True,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(youtube_url, download=False)
        # Return direct playable stream URL
        return info["url"]


def browse_file() -> str:
    """Opens native Windows File Picker dialog so user can select any video file."""
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        file_path = filedialog.askopenfilename(
            title="Select Pool Video for WADE",
            filetypes=[("Video Files", "*.mp4 *.avi *.mov *.mkv *.webm"), ("All Files", "*.*")]
        )
        root.destroy()
        return file_path
    except Exception as e:
        print("[!] File picker error:", e)
        return ""


def main():
    os.system("cls" if os.name == "nt" else "clear")
    print("=" * 65)
    print("🛡️  CARE / WADE — Autonomous CCTV Fall & Anomaly Sentinel")
    print("    [Team larperchud // Real-Time RTX 4050 Acceleration]")
    print("=" * 65)
    print("  [1] 🏪 Store Entrance Slip & Fall          (Camera 01 — Real CCTV)")
    print("  [2] 🏥 Hospital Corridor Patient Collapse   (Camera 08 — 1080p HD)")
    print("  [3] 🍽️ Restaurant Hallway Sudden Collapse  (Camera 03 — Real CCTV)")
    print("  [4] 🏢 Office Reception Door Slip & Fall    (CCTV 02)")
    print("  [5] ⚠️ Kitchen Wet-Floor Severe Slip        (Camera 01)")
    print("  [6] ☕ Cafeteria Breakroom Slip & Fall      (Camera 04)")
    print("  [7] 💊 Pharmacy Dispensary Worker Collapse  (Camera 13)")
    print("  -------------------------------------------------------------")
    print("  [8] 📹 Live Interactive Webcam Mode         (Test Yourself Live!)")
    print("  [9] 📂 Browse Any Video File from Computer  (Windows File Picker)")
    print(" [10] 🌐 Play directly from YouTube Link      (Direct Stream, No DL)")
    print("=" * 65)

    choice = input("\n👉 Choose scenario (1-10) [Default 1]: ").strip() or "1"

    source = ""
    pool_label = "SURVEILLANCE SENTINEL"

    if choice == "1":
        source = "assets/clean_store_fall.mp4"
        pool_label = "STORE ENTRANCE - CAM 01"
    elif choice == "2":
        source = "assets/cctv_hospital_fall.mp4"
        pool_label = "HOSPITAL WARD - CAM 08"
    elif choice == "3":
        source = "assets/clean_cctv_collapse.mp4"
        pool_label = "RESTAURANT CORRIDOR - CAM 03"
    elif choice == "4":
        source = "assets/cctv_reception_fall.mp4"
        pool_label = "RECEPTION ENTRANCE - CCTV 02"
    elif choice == "5":
        source = "assets/cctv_wetfloor_slip.mp4"
        pool_label = "COMMERCIAL KITCHEN - CAM 01"
    elif choice == "6":
        source = "assets/cctv_staged_fall.mp4"
        pool_label = "CAFETERIA BREAKROOM - CAM 04"
    elif choice == "7":
        source = "assets/security_fall_2.mp4"
        pool_label = "PHARMACY DISPENSARY - CAM 13"
    elif choice == "8":
        source = "0"
        pool_label = "STAGE WEBCAM - LIVE TEST"
    elif choice == "9":
        source = browse_file()
        if not source:
            print("[!] No file selected. Exiting.")
            return
        pool_label = os.path.basename(source)
    elif choice == "10":
        yt_url = input("\n🔗 Paste YouTube URL: ").strip()
        if not yt_url:
            print("[!] Empty URL.")
            return
        try:
            source = get_youtube_stream_url(yt_url)
            pool_label = "LIVE YOUTUBE STREAM"
        except Exception as e:
            print(f"[!] Could not stream from YouTube: {e}")
            return
    else:
        source = "assets/clean_store_fall.mp4"
        pool_label = "STORE ENTRANCE - CAM 01"

    print(f"\n🚀 Launching CARE Sentinel on [{pool_label}]...")
    run_wade(source=source, pool_name=pool_label)


if __name__ == "__main__":
    main()
