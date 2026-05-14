# AI Video Translator - Phase 1

End-to-end pipeline for video translation:
- Video upload
- Audio extraction (FFmpeg)
- Transcription (Whisper)
- Translation (NLLB)
- Text-to-speech (gTTS)
- Audio merged back into video

## Backend API
- `POST /upload-video`: upload an MP4 and extract audio
- `POST /process-video`: full pipeline, returns original text, translated text, translated audio, and translated video

## Requirements
- Python 3.9+
- FFmpeg available on PATH
- Python packages: `fastapi`, `uvicorn`, `whisper`, `transformers`, `gtts`

## Notes
- Outputs are written to the `outputs/` folder.
- Uploads are written to the `uploads/` folder.
