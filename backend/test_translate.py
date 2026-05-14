import argparse
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline


def translate_with_model(text: str, model_name: str, src_lang: str, tgt_lang: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name, src_lang=src_lang)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    inputs = tokenizer(text, return_tensors="pt")
    if hasattr(tokenizer, "lang_code_to_id"):
        forced_bos_token_id = tokenizer.lang_code_to_id[tgt_lang]
    else:
        forced_bos_token_id = tokenizer.convert_tokens_to_ids(tgt_lang)
    generated = model.generate(
        **inputs,
        forced_bos_token_id=forced_bos_token_id,
        max_length=256,
    )
    return tokenizer.batch_decode(generated, skip_special_tokens=True)[0]


def translate_with_pipeline(text: str, model_name: str, task: str):
    translator = pipeline(task, model=model_name)
    result = translator(text)
    return result[0]["translation_text"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", default="Hello everyone welcome back to my channel")
    parser.add_argument(
        "--input-file",
        default="outputs/transcript_2.txt",
        help="If set, load input text from this file.",
    )
    parser.add_argument("--model", default="facebook/nllb-200-distilled-600M")
    parser.add_argument("--src-lang", default="eng_Latn")
    parser.add_argument("--tgt-lang", default="tel_Telu")
    parser.add_argument(
        "--task",
        default="translation",
        help="Use 'translation' for NLLB or 'translation_en_to_te' for opus-mt.",
    )
    parser.add_argument(
        "--output-file",
        default="outputs/translation_1.txt",
        help="Write the translated text to this file in UTF-8.",
    )
    args = parser.parse_args()

    text = args.text
    if args.input_file:
        encodings = ["utf-8", "utf-16", "cp1252"]
        for encoding in encodings:
            try:
                with open(args.input_file, "r", encoding=encoding) as handle:
                    text = handle.read().strip() or args.text
                break
            except UnicodeDecodeError:
                continue

    if args.task == "translation":
        translated = translate_with_model(text, args.model, args.src_lang, args.tgt_lang)
    else:
        translated = translate_with_pipeline(text, args.model, args.task)

    with open(args.output_file, "w", encoding="utf-8") as handle:
        handle.write(translated)

    print(translated)


if __name__ == "__main__":
    main()
