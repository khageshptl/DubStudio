from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tok = AutoTokenizer.from_pretrained('facebook/nllb-200-distilled-600M')
mod = AutoModelForSeq2SeqLM.from_pretrained('facebook/nllb-200-distilled-600M')
tok.src_lang = 'eng_Latn'
inp = tok('hello this is a test', return_tensors='pt')
out = mod.generate(**inp, forced_bos_token_id=tok.convert_tokens_to_ids('hin_Deva'))
print(f'OUTPUT: {tok.batch_decode(out, skip_special_tokens=True)[0]}')
