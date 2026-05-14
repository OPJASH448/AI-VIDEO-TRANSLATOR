from pathlib import Path
import shutil
import subprocess

from fastapi import HTTPException


def _get_ffmpeg_path() -> str:
    ffmpeg_path = shutil.which("ffmpeg")
    if not ffmpeg_path:
        raise HTTPException(status_code=500, detail="ffmpeg not found in PATH")
    return ffmpeg_path


def extract_audio(video_path: Path, outputs_dir: Path) -> Path:
    audio_name = f"{video_path.stem}.wav"
    audio_path = outputs_dir / audio_name

    ffmpeg_path = _get_ffmpeg_path()

    try:
        subprocess.run(
            [ffmpeg_path, "-y", "-i", str(video_path), str(audio_path)],
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as exc:
        raise HTTPException(
            status_code=500,
            detail=f"ffmpeg failed: {exc.stderr.strip()}",
        )

    return audio_path


def merge_audio(video_path: Path, audio_path: Path, outputs_dir: Path) -> Path:
    output_name = f"{video_path.stem}_translated.mp4"
    output_path = outputs_dir / output_name

    ffmpeg_path = _get_ffmpeg_path()

    try:
        subprocess.run(
            [
                ffmpeg_path,
                "-y",
                "-i",
                str(video_path),
                "-i",
                str(audio_path),
                "-map",
                "0:v",
                "-map",
                "1:a",
                "-c:v",
                "copy",
                "-shortest",
                str(output_path),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as exc:
        raise HTTPException(
            status_code=500,
            detail=f"ffmpeg merge failed: {exc.stderr.strip()}",
        )

    return output_path
