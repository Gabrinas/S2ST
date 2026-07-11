import sacrebleu


ref = ["Pupọ kere si erogba ati hydrogen ni fọọmu ti a ṣe deede fun ilana atẹgun ju ni ibamu si atẹgun ti o gba ninu ẹdọforo."]
hyp = ["àwọn òkèké rẹ̀ sí ìrùbá àti agbóṣèmi ni fọ́ọ̀mù tí a ṣe díndín fi máa tẹ̀gbọ́n gbọ́gbé ìgbámù sí àtẹ̀wò tí ìrùbá ìgbẹ́ ẹgbẹ́."]



# Compute BLEU score
bleu = sacrebleu.corpus_bleu(hyp, [ref])
print("BLEU score:", bleu.score)

