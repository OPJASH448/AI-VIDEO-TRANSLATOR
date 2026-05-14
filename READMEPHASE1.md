# Phase 1 - Project Contents

This document summarizes the files and roles in the current Phase 1 build.

## Backend
- `backend/main.py`: FastAPI app and main pipeline endpoints (`/upload-video`, `/process-video`).
- `backend/services/whisper_service.py`: Whisper model load and transcription logic.
- `backend/services/translation_service.py`: Translation model load and text translation logic.
- `backend/services/tts_service.py`: gTTS text-to-speech generation for translated audio.
- `backend/services/ffmpeg_service.py`: FFmpeg audio extraction and audio-video merge.

## Project Files
- `.gitignore`: ignores venv, outputs, uploads, temp, and OS artifacts.
- `PHASE1_README.md`: phase overview and requirements.

## Outputs and Runtime Folders
- `uploads/`: incoming uploaded videos.
- `outputs/`: extracted audio, translated audio, and dubbed videos.
- `temp/`: scratch directory for processing.
