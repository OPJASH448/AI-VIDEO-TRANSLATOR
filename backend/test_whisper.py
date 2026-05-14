import whisper


def main():
    model = whisper.load_model("base")
    result = model.transcribe("outputs/audio1.wav")
    print(result["text"])


if __name__ == "__main__":
    main()
