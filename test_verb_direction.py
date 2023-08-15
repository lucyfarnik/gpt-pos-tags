#%%
import torch as t
from sklearn.metrics import f1_score
import numpy as np
from typing import Optional, Tuple, Callable
from gather_heads_pos import model, childrens_tagged_sents

MAIN = __name__ == '__main__'
# %%
verb_direction = t.load('verb_direction.pt')

# %%
def is_verb_classifier1(word: str) -> Optional[bool]:
    # prepend space, remove any trailing spaces
    if word[0] != ' ':
        word = ' ' + word
    word = word.rstrip()

    # tokenize
    token = model.to_tokens(word)[0, 1:]
    # # TODO add handling of multi-token words
    if token.size(0) != 1:
        return None
    token = token.item()
    
    # embed the token (the pos embedding acts as a bias)
    word_vec = model.W_E[token] + model.W_pos[1]

    # token = model.to_tokens(word)
    # word_vec = model.embed(token) + model.pos_embed(token)
    # print(model.pos_embed(token))

    # return the dot product with the verb direction, if it's positive then it's a verb
    return t.dot(word_vec, verb_direction).item() > 0

[is_verb_classifier1(x) for x in ['think', 'different', 'bro']]
# %%
def test_classifier(classifier: Callable[[str], Optional[bool]] = is_verb_classifier1):
    true_positive = 0
    false_positive = 0
    true_negative = 0
    false_negative = 0
    for sent in childrens_tagged_sents:
        for word, tag in sent:
            y = tag[:2] == 'VB'
            y_hat = classifier(word)
            if y_hat is None:
                continue

            if y and y_hat:
                true_positive += 1
            elif y and not y_hat:
                false_negative += 1
            elif not y and y_hat:
                false_positive += 1
            elif not y and not y_hat:
                true_negative += 1
            else:
                raise ValueError
    print(f'{true_positive=:,} {false_positive=:,} {true_negative=:,} {false_negative=:,}')
    print(f'Accuracy: {(true_positive + true_negative) / (true_positive + true_negative + false_positive + false_negative):.1%}')
    print(f'Precision: {true_positive / (true_positive + false_positive):.1%}')
    print(f'Recall: {true_positive / (true_positive + false_negative):.1%}')
    print(f'F1: {2 * true_positive / (2 * true_positive + false_positive + false_negative):.1%}')

test_classifier(is_verb_classifier1)
# %%
# finding the optimal threshold
def get_dot_and_cos_sim(word: str) -> Tuple[Optional[float], Optional[float]]:
    # prepend space, remove any trailing spaces
    if word[0] != ' ':
        word = ' ' + word
    word = word.rstrip()

    # tokenize
    token = model.to_tokens(word)[0, 1:]
    # # TODO add handling of multi-token words
    if token.size(0) != 1:
        return None, None
    token = token.item()
    
    # embed the token (the pos embedding acts as a bias)
    word_vec = model.W_E[token] + model.W_pos[1]

    # return the dot product with the verb direction
    return t.dot(word_vec, verb_direction).item(), t.cosine_similarity(word_vec, verb_direction, dim=0).item()

get_dot_and_cos_sim('think')
# %%
verb_dot_cos = []
for sent in childrens_tagged_sents:
    for word, tag in sent:
        is_verb = tag[:2] == 'VB'
        dot, cos = get_dot_and_cos_sim(word)
        if dot is None:
            continue
        verb_dot_cos.append((is_verb, dot, cos))
len(verb_dot_cos)
# %%
# for both dot and cosine similarity, find the threshold that maximizes the f1 score
def get_f1_score(threshold: float, use_cos: bool = False) -> float:
    y_true = [x[0] for x in verb_dot_cos]
    y_pred = [x[2 if use_cos else 1] > threshold for x in verb_dot_cos]
    return f1_score(y_true, y_pred)

def explore_thresh_range(min: float, max: float, num: int, use_cos: bool = False):
    thresholds = np.linspace(min, max, num)
    f1_scores = [get_f1_score(threshold, use_cos) for threshold in thresholds]
    best_threshold = thresholds[np.argmax(f1_scores)]
    best_f1_score = np.max(f1_scores)
    print(f'Best threshold: {best_threshold:.3f} (F1 score: {best_f1_score:.1%}, {use_cos=})')

explore_thresh_range(-0.5, 0.5, 10)
# %%
explore_thresh_range(-5, 5, 10)
# %%
explore_thresh_range(3.5, 4.5, 10)
# %%
explore_thresh_range(2.5, 3.5, 10)
# %%
explore_thresh_range(2.9, 3.2, 10)
# %%
explore_thresh_range(3, 3.3, 20)
# %%
explore_thresh_range(3.1, 3.25, 100)
# Best threshold: 3.214 (F1 score: 49.9%, use_cos=False)
# %%
explore_thresh_range(-0.5, 0.5, 10, True)
# %%
explore_thresh_range(0, 0.2, 10, True)
# %%
explore_thresh_range(0.02, 0.08, 100, True)
# Best threshold: 0.059 (F1 score: 50.5%, use_cos=True)
# %%
def is_verb_classifier2(word: str) -> Optional[bool]:
    # prepend space, remove any trailing spaces
    if word[0] != ' ':
        word = ' ' + word
    word = word.rstrip()

    # tokenize
    token = model.to_tokens(word)[0, 1:]
    # # TODO add handling of multi-token words
    if token.size(0) != 1:
        return None
    token = token.item()
    
    # embed the token (the pos embedding acts as a bias)
    word_vec = model.W_E[token] + model.W_pos[1]

    # return the dot product with the verb direction, if it's positive then it's a verb
    return t.cosine_similarity(word_vec, verb_direction, dim=0).item() > 0.059

[is_verb_classifier2(x) for x in ['think', 'different', 'bro']]
# %%
test_classifier(is_verb_classifier2)
# %%
