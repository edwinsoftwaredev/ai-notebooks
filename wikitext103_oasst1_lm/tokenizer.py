import sentencepiece as spm
from itertools import chain

from wikitext103_oasst1_lm.datasets import load_tokenizer_datasets

USER_TOKEN = "<user>"
ASSISTANT_TOKEN = "<assistant>"


def train_tokenizer():
    def wikitext_gen_fn(sets):
        for s in sets:
            s = s.compute()
            for row in s.itertuples():
                yield row.text

    def oasst1_gen_fn(sets):
        for s in sets:
            s = s.compute()
            for row in s.itertuples():
                if row.role == "prompter":
                    yield f"{USER_TOKEN} {row.text}"
                else:
                    yield f"{ASSISTANT_TOKEN} {row.text}"

    wikitext_parts, oasst1_parts = load_tokenizer_datasets()

    spm.SentencePieceTrainer.Train(
        sentence_iterator=chain(
            wikitext_gen_fn(wikitext_parts), oasst1_gen_fn(oasst1_parts)
        ),
        model_prefix="m",
        vocab_size=32000,
        character_coverage=1.0,
        model_type="unigram",
        max_sentence_length=10000,
        user_defined_symbols=[USER_TOKEN, ASSISTANT_TOKEN],
        pad_id=3,
    )

    return spm
