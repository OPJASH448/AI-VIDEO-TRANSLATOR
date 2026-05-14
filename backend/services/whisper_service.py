import whisper


model = whisper.load_model("small")


def transcribe_audio(path: str) -> str:
    result = model.transcribe(
        path,
        language="te",
        fp16=False,
    )

    return result.get("text", "").strip()
