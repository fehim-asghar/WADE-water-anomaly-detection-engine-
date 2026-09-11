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
    print("=" * 60)
    print("💧 WADE — Interactive Demo Launcher (Zero Friction)")
    print("=" * 60)
    print("  [1] 🚨 Real Drowning Incident Drill (Water Rescue)")
    print("  [2] 🏊 Real Overhead Community Pool (CCTV Angle)")
    print("  [3] 📹 Live Interactive Webcam Mode (Test Yourself on Stage!)")
    print("  [4] 📂 Browse Any Video File from Computer (File Picker)")
    print("  [5] 🌐 Play directly from YouTube Link (NO Downloading!)")
    print("=" * 60)

    choice = input("\n👉 Choose option (1-5) [Default 1]: ").strip() or "1"

    source = ""
    pool_label = "OLYMPIC POOL - CCTV 04"

    if choice == "1":
        source = "assets/drowning_incident.mp4"
        pool_label = "RESCUE ZONE - CCTV 02"
    elif choice == "2":
        source = "assets/public_pool_aerial.mp4"
        pool_label = "COMMUNITY POOL - CCTV 01"
    elif choice == "3":
        source = "0"
        pool_label = "STAGE WEBCAM - LIVE TEST"
    elif choice == "4":
        source = browse_file()
        if not source:
            print("[!] No file selected. Exiting.")
            return
        pool_label = os.path.basename(source)
    elif choice == "5":
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
        source = "assets/drowning_incident.mp4"

    print(f"\n🚀 Launching WADE on [{pool_label}]...")
    run_wade(source=source, pool_name=pool_label)


if __name__ == "__main__":
    main()
