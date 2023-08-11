#%%
"""
Mostly messing around with embeddings spaces, trying to do vector arithmetic
to see if there's a "noun direction"

Conclusion: probably not, GPT embeddings seem less semantically meaningful than Word2Vec
"""
from transformer_lens import HookedTransformer
import torch as t
# import spacy
import re
import nltk
from nltk.corpus import wordnet, words
import plotly.express as px
import pandas as pd
from typing import Tuple, Dict, Union, Optional
from enum import Enum
# import enchant


#%%
model = HookedTransformer.from_pretrained('gpt2-small')
model.eval()
# spacy_model = spacy.load("en_core_web_sm")
nltk.download('punkt')
nltk.download('words')
# dictionary = enchant.Dict("en_US")

# %%
# def pos_tokens() -> Tuple[Dict[int, str], Dict[int, str]]:
#     noun_tokens = {}
#     verb_tokens = {}
#     for token, id in model.tokenizer.get_vocab().items():
#         if token.startswith('Ġ'):
#             token = token[1:]
#         else:
#             continue

#         synset = wordnet.synsets(token)
#         synset_filtered = [s for s in synset if s.name().split('.')[0] == token]
#         posset = set([s.pos() for s in synset_filtered])
#         if len(posset) != 1:
#             continue
#         if 'n' in posset:
#             pos_nltk = 'n'
#         elif 'v' in posset:
#             pos_nltk = 'v'
#         else:
#             continue

#         spacy_tokens = spacy_model(token)
#         if len(spacy_tokens) != 1:
#             continue
#         if spacy_tokens[0].pos_ == 'NOUN':
#             # noun_tokens[id] = token
#             pos_spacy = 'n'
#         elif spacy_tokens[0].pos_ == 'VERB':
#             # verb_tokens[id] = token
#             pos_spacy = 'v'
#         else:
#             continue
        
#         if pos_spacy != pos_nltk:
#             print(token, pos_spacy, pos_nltk)
#             continue
#         if pos_spacy == 'n':
#             noun_tokens[id] = token
#         elif pos_spacy == 'v':
#             verb_tokens[id] = token
#     return noun_tokens, verb_tokens
def pos_tokens() -> Tuple[Dict[int, str], Dict[int, str]]:
    noun_tokens = {}
    verb_tokens = {}
    for token, id in model.tokenizer.get_vocab().items():
        if token.startswith('Ġ'):
            token = token[1:]
        else:
            continue

        # if token.lower() not in words.words():
        if re.search(r'[^a-zA-Z]', token):
            # print('Not a word', token)
            continue

        is_noun = wordnet.synsets(token, pos=wordnet.NOUN)
        is_verb = wordnet.synsets(token, pos=wordnet.VERB)
        if is_noun and is_verb:
            # print('Confusion', token)
            continue
        elif is_noun:
            noun_tokens[id] = token
        elif is_verb:
            verb_tokens[id] = token

    return noun_tokens, verb_tokens

noun_tokens, verb_tokens = pos_tokens()

#%%
# find average noun direction in unembedding space
noun_vecs = []
for token_id in noun_tokens.keys():
    noun_vecs.append(model.W_U[:, token_id].detach())
mean_noun_vec = t.stack(noun_vecs, dim=1).mean(dim=1)
# find average verb direction in unembedding space
verb_vecs = []
for token_id in verb_tokens.keys():
    verb_vecs.append(model.W_U[:, token_id].detach())
mean_verb_vec = t.stack(verb_vecs, dim=1).mean(dim=1)

noun_to_verb = mean_verb_vec - mean_noun_vec
# %%
def visualize_next_token_preds(prompt: str):
    outs = model(prompt)
    next_token_logits = outs[0, -1, :]
    next_token_probs = next_token_logits.softmax(dim=-1)
    next_token_probs = next_token_probs.detach().numpy()
    # plot the top 10 with plotly
    top10 = pd.DataFrame({
        'token': [model.tokenizer.decode([i]) for i in next_token_probs.argsort()[-10:][::-1]],
        'prob': next_token_probs[next_token_probs.argsort()[-10:][::-1]]
    })
    fig = px.bar(top10, x='token', y='prob')
    fig.update_layout(title_text=prompt+'...')
    fig.show()

visualize_next_token_preds('I find it interesting that the most common')

# %%
def visualize_embeddings(vector: t.Tensor, top_n: int = 10,
                         use_unembed: bool = False,
                         title: Optional[str] = None):
    # find the top n closest tokens
    if use_unembed:
        logits = model.unembed(vector[None, None, :]).squeeze().detach()
    else:
        logits = (model.W_E @ vector).detach()
    top_n_indices = logits.argsort(descending=True)[:top_n]
    top_n = pd.DataFrame({
        'token': [model.tokenizer.decode([i]) for i in top_n_indices],
        'prob': logits[top_n_indices]
    })
    fig = px.bar(top_n, x='token', y='prob')
    if title is not None:
        fig.update_layout(title_text=title)
    fig.show()

#%%
def apply_vec_to_token(token: str, vec: t.Tensor, alpha: float = 1.0):
    token_id = model.to_tokens(token)[0, 1]
    token_vec = model.W_U[:, token_id]
    shifted_vec = token_vec + alpha * (token_vec.norm() / vec.norm()) * vec

    visualize_embeddings(shifted_vec, use_unembed=True,
                         title=f'"{token}" shifted by a vector ({alpha=})')

apply_vec_to_token(' insight', noun_to_verb)
# %%
class Op(Enum):
    PLUS = '+'
    MINUS = '-'

    def __str__(self):
        return self.value

def embedding_arithmetic(*args: Union[str, Op], use_unembed: bool = False,
                         visualize: bool = True) -> t.Tensor:
    tokens = args[::2]
    ops = args[1::2]

    token_ids = [model.to_tokens(token)[0, 1] for token in tokens]
    token_vecs = [model.W_U[:, token_id] if use_unembed else model.W_E[token_id]
                  for token_id in token_ids]
    # detach all vecs
    token_vecs = [vec.detach() for vec in token_vecs]

    result_vec = token_vecs[0].clone()
    for i in range(1, len(token_vecs)):
        if ops[i-1] == Op.PLUS:
            result_vec += token_vecs[i]
        elif ops[i-1] == Op.MINUS:
            result_vec -= token_vecs[i]
        else:
            raise ValueError('Invalid operation')

    if visualize:
        title = ' '.join([f'"{a}"' if type(a) is str else str(a) for a in args])
        if use_unembed:
            title += ' (in W_U)'
        else:
            title += ' (in W_E)'
        visualize_embeddings(result_vec, use_unembed=use_unembed, title=title)

    return result_vec
    
embedding_arithmetic(' integrate', Op.PLUS, ' difference', Op.MINUS, ' differentiate');
# %%
def visualize_adjacent_tokens(token: str, use_unembed: bool = False):
    token_id = model.to_tokens(token)[0, 1]
    token_vec = model.W_U[:, token_id] if use_unembed else model.W_E[token_id]
    visualize_embeddings(token_vec, use_unembed=use_unembed,
                         title=f'"{token}" (in W_U)' if use_unembed
                                else f'"{token}" (in W_E)')

visualize_adjacent_tokens(' integrate', use_unembed=True)

# %%
