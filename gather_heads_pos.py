#%%
"""
Takes a giant corpus of POS-tagged sentences, runs them through the model,
and saves the attention patterns for each sentence along with the POS tags.
"""
import nltk
from transformer_lens import HookedTransformer, ActivationCache
import torch as t
from typing import Tuple

MAIN = __name__ == '__main__'
# %%
nltk.download('brown')

categories = ['adventure', 'hobbies', 'lore', 'mystery', 'romance', 'science_fiction']
childrens_tagged_sents = nltk.corpus.brown.tagged_sents(categories=categories)

childrens_tagged_sents
#%%
# Function to reconstruct sentence from tokens
def reconstruct_sentence(tokens) -> str:
    sentence = ' ' # start with a space for better tokenization
    for i, (token, _) in enumerate(tokens):
        if token in ['.', '!', '?', ':', ';', ',', ')', ']', '}']:
            sentence += token
        elif token in ['(', '[', '{', '``', "''"]:
            sentence += ' ' + token
        # Handling the case where the previous token is a quotation mark.
        elif i > 0 and tokens[i-1][0] in ['``', "''"]:
            sentence += token
        else:
            # Regular word or word with a leading punctuation.
            sentence += ' ' + token
    return sentence.strip()

# %%
model = HookedTransformer.from_pretrained('gpt2-small')
#%%
def get_pos_dict(sentence: str, tagged_sentence: list) -> dict:
    # Build up dictionary of sequence position to POS tag
    str_tokens = model.to_str_tokens(sentence)
    pos_dict = {}
    last_pos = 0
    for word, pos in tagged_sentence:
        # Find the token that matches the word (in the substring that hasn't been matched yet)
        for i, token in enumerate(str_tokens[last_pos:]):
            if token.strip() == word.strip():
                pos_dict[i+last_pos] = pos
                last_pos = i+last_pos
                break
    return pos_dict
# %%
# Process each sentence, tokenize, pass through model, associate with POS
if MAIN:
    results = []
    for sent_idx, tagged_sentence in enumerate(childrens_tagged_sents):
        print(sent_idx)
        # Ignore short sentences
        if len(tagged_sentence) < 15:
            continue

        # Reconstruct the sentence from tokens
        original_sentence = reconstruct_sentence(tagged_sentence)

        # Build up dictionary of sequence position to POS tag
        pos_dict = get_pos_dict(original_sentence, tagged_sentence)

        # Pass the sentence through the model
        outs, cache = model.run_with_cache(original_sentence)
        pattern_cache = [cache['pattern', i] for i in range(model.cfg.n_layers)]
        
        # Save the results
        results.append({
            'original_sentence': original_sentence,
            'pos_dict': pos_dict,
            'pattern_cache': pattern_cache,
        })

    t.save(results, 'attn_pattern_results.pt')

# %%
