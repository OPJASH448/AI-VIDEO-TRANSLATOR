from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, File, HTTPException, UploadFile

from services.ffmpeg_service import extract_audio, merge_audio
from services.translation_service import translate_text
from services.tts_service import synthesize_speech
from services.whisper_service import transcribe_audio

app = FastAPI()

PROJECT_ROOT = Path(__file__).resolve().parents[1]
UPLOADS_DIR = PROJECT_ROOT / "uploads"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
CHUNK_SIZE = 1024 * 1024


async def _save_upload(file: UploadFile, suffix: str, target_dir: Path) -> Path:
    safe_name = Path(file.filename).name
    target_name = f"{Path(safe_name).stem}_{uuid4().hex}{suffix}"
    target_path = target_dir / target_name

    with target_path.open("wb") as target_file:
        while True:
            chunk = await file.read(CHUNK_SIZE)
            if not chunk:
                break
            target_file.write(chunk)

    return target_path




@app.get("/")
def home():
    return {"message": "AI Video Translator Running"}


@app.post("/upload-video")
async def upload_video(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="Missing filename")

    ext = Path(file.filename).suffix.lower()
    if ext != ".mp4":
        raise HTTPException(status_code=400, detail="Only .mp4 files are supported")

    target_path = await _save_upload(file, ext, UPLOADS_DIR)
    audio_path = extract_audio(target_path, OUTPUTS_DIR)

    return {
        "filename": target_path.name,
        "path": str(target_path),
        "audio": str(audio_path),
    }


@app.post("/process-video")
async def process_video(
    file: UploadFile = File(...),
    translation_model: str = "facebook/nllb-200-distilled-600M",
    src_lang: str = "eng_Latn",
    tgt_lang: str = "tel_Telu",
    tts_lang: str = "te",
):
    if not file.filename:
        raise HTTPException(status_code=400, detail="Missing filename")

    ext = Path(file.filename).suffix.lower()
    if ext != ".mp4":
        raise HTTPException(status_code=400, detail="Only .mp4 files are supported")

    video_path = await _save_upload(file, ext, UPLOADS_DIR)
    audio_path = extract_audio(video_path, OUTPUTS_DIR)

    original_text = transcribe_audio(str(audio_path))
    if not original_text:
        raise HTTPException(status_code=422, detail="No speech detected in audio")

    translated_text = translate_text(
        original_text,
        translation_model,
        src_lang,
        tgt_lang,
    )

    translated_audio_path = synthesize_speech(
        translated_text,
        tts_lang,
        OUTPUTS_DIR,
        video_path.stem,
    )

    translated_video_path = merge_audio(
        video_path,
        translated_audio_path,
        OUTPUTS_DIR,
    )

    return {
        "original_text": original_text,
        "translated_text": translated_text,
        "translated_audio": str(translated_audio_path),
        "translated_video": str(translated_video_path),
    }
