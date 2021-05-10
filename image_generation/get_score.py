import numpy as np
import math
import os
import re

re_digits = re.compile(r'(\d+)')

def emb_numbers(s):
    pieces=re_digits.split(s)
    pieces[1::2]=map(int,pieces[1::2])
    return pieces

def sort_strings_with_emb_numbers(alist):
    aux = [(emb_numbers(s),s) for s in alist]
    aux.sort()
    return [s for __,s in aux]

def sort_strings_with_emb_numbers2(alist):
    return sorted(alist, key=emb_numbers)


score_list = []
file_dir = input('input the file directory: ')
for i in sort_strings_with_emb_numbers(os.listdir(file_dir)):
    print(i)
    if i.endswith('.npz') and i != 'noise_norm_list.npz':
        scores = np.load(os.path.join(file_dir, i))
        score_list.append([[np.mean(scores['fid']), np.std(scores['fid'])],
                           [np.mean(scores['inception']), np.std(scores['inception'])],
                           [np.mean(scores['mmd2']), np.std(scores['mmd2'])]])
score_list = np.asarray(score_list)
print(score_list[:, 0, 0])
print(score_list[:, 1, 0])
print(score_list[:, 2, 0])

print(np.min(score_list[:, 0, 0]), score_list[np.argmin(score_list[:, 0, 0]), 0, 1],
      np.max(score_list[:, 1, 0]), score_list[np.argmin(score_list[:, 1, 0]), 1, 1],
      np.min(score_list[:, 2, 0]), score_list[np.argmin(score_list[:, 2, 0]), 2, 1])