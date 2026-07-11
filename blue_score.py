import sacrebleu
from jiwer import wer, cer

import re
import unicodedata


def remove_diacritics(text: str) -> str:
    # Normalize to NFD (decompose characters into base + diacritic)
    nfkd_form = unicodedata.normalize('NFD', text)
    # Keep only base characters (category != 'Mn' means not a diacritic)
    return ''.join([c for c in nfkd_form if unicodedata.category(c) != 'Mn'])


def normalize(text):
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)  # remove punctuation
    return text

ref_norm = "Ṣugbọn, niwọn bi metamorphosis ti awọn ẹya ti a ṣeto si n tẹsiwaju diẹ sii laiyara, aipe awọn nkan yẹn yoo waye."
hyp_norm = "ilé ìṣèwọ̀pọ̀ ni wọ́n bí lè ta náà pàṣísì gbogbo ìlà jẹ́ àṣẹ̀pẹ̀sì wà tí gẹ̀ láyìn lára àbìí ẹ̀rẹ́kọ̀ọ́ mi lọ́la jẹ̩́́"


ref_norm = normalize(ref_norm)
hyp_norm = normalize(hyp_norm)

# Example reference and candidate
ref_norm = [remove_diacritics(ref_norm)]
hyp_norm = [remove_diacritics(hyp_norm)]


print(ref_norm); print(hyp_norm)

# Compute BLEU score
bleu = sacrebleu.corpus_bleu(hyp_norm, [ref_norm])
print("BLEU score:", bleu.score)



print("WER:", wer(ref_norm, hyp_norm))
print("CER:", cer(ref_norm, hyp_norm))

