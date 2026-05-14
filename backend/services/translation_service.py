from functools import lru_cache

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


@lru_cache(maxsize=4)
def get_translation_assets(model_name: str, src_lang: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name, src_lang=src_lang)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return tokenizer, model


def translate_text(text: str, model_name: str, src_lang: str, tgt_lang: str) -> str:
    tokenizer, model = get_translation_assets(model_name, src_lang)
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
