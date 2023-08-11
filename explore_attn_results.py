#%%
"""
Take the results from gather_heads_pos.py and explore them.

Goal: Are there heads that pay a lot more attention to certain POS tags?
"""
import torch as t
import plotly.express as px
import circuitsvis as cv
from transformer_lens import HookedTransformer

#%%
data = t.load('attn_pattern_results.pt')
len(data), data[0].keys(), len(data[0]['pattern_cache']), data[0]['pattern_cache'][0].shape
#%%
model = HookedTransformer.from_pretrained('gpt2-small')
# %%
# nouns: NN*, verbs: VB*, adjectives: JJ*
# for each head, find the average attention to each of the POS tags
noun_attns = {f'{l}.{h}': [] for l in range(12) for h in range(12)}
verb_attns = {f'{l}.{h}': [] for l in range(12) for h in range(12)}
adj_attns = {f'{l}.{h}': [] for l in range(12) for h in range(12)}
for sent in data:
    # find the positions of the parts of speech we're analyzing right now
    noun_positions = []
    verb_positions = []
    adj_positions = []
    for i, pos in sent['pos_dict'].items():
        if pos.startswith('NN'):
            noun_positions.append(i)
        elif pos.startswith('VB'):
            verb_positions.append(i)
        elif pos.startswith('JJ'):
            adj_positions.append(i)

    # for each head, find the average attention to each of the POS tags
    for layer_idx, layer in enumerate(sent['pattern_cache']):
        for head_idx, head_pattern in enumerate(layer[0]):
            # average attention paid to each POS tag
            head_name = f'{layer_idx}.{head_idx}'
            if len(noun_positions) > 0:
                noun_attns[head_name].append(head_pattern[:, noun_positions].mean())
            if len(verb_positions) > 0:
                verb_attns[head_name].append(head_pattern[:, verb_positions].mean())
            if len(adj_positions) > 0:
                adj_attns[head_name].append(head_pattern[:, adj_positions].mean())

noun_attns = t.tensor([sum(v)/len(v) for v in noun_attns.values()]).reshape(12, 12)
verb_attns = t.tensor([sum(v)/len(v) for v in verb_attns.values()]).reshape(12, 12)
adj_attns = t.tensor([sum(v)/len(v) for v in adj_attns.values()]).reshape(12, 12)

noun_attns, verb_attns, adj_attns
# %%
px.imshow(noun_attns, title='Average attention to nouns')
# %%
px.imshow(verb_attns, title='Average attention to verbs')
# %%
px.imshow(adj_attns, title='Average attention to adjectives')
# %%
# print heads above a certain threshold
[
    ('noun', t.nonzero(noun_attns > 0.043)),
    ('verb', t.nonzero(verb_attns > 0.0465)),
    ('adj', t.nonzero(adj_attns > 0.042)),
]
# %%
# what about the ones with the biggest differences?
for name, diff, thresh in [('noun-verb', noun_attns - verb_attns, 0.0238),
                           ('noun-adj', noun_attns - adj_attns, 0.0129),
                           ('verb-adj', verb_attns - adj_attns, 0.021)]:
    heads = t.nonzero(diff.abs() > thresh)
    print(name)
    for head in heads:
        print(f'L{head[0].item()}H{head[1].item()}: {diff[head[0], head[1]]:.3f}')
    print()
# %%
# let's examine head 2.8
def show_sentence_attn(sent_idx, layer_idx, head_idx):
    sent = data[sent_idx]
    text = sent['original_sentence']
    print(text)
    head_pattern = sent['pattern_cache'][layer_idx][0, head_idx]
    return cv.attention.attention_pattern(
        tokens=model.to_str_tokens(text),
        attention=head_pattern,
    )

show_sentence_attn(2963, 2, 8)
# %%
