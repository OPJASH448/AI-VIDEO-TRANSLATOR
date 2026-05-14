from pathlib import Path

from gtts import gTTS


def synthesize_speech(text: str, lang: str, outputs_dir: Path, base_name: str) -> Path:
    audio_name = f"{base_name}_translated.mp3"
    audio_path = outputs_dir / audio_name

    tts = gTTS(text=text, lang=lang)
    tts.save(str(audio_path))

    return audio_path
