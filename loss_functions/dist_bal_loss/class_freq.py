import mmcv
import numpy as np
import pandas as pd
from collections import Counter
from itertools import chain


def save_class_freq(args, dest_file='class_freq.pkl'):
    if args.dataset == 'noisy_actions':
        src_file = 'Data/noisyActionsMultiTrainTestList/trainlist100k.txt'
    else:
        print('Add more arguments for other datasets')
        assert 2 == 1
    freq_info = dict()
    data = pd.read_csv(src_file, sep=' ', names=['vid', 'labels'])
    data['labels'] = data['labels'].apply(lambda x: x.split('|'))
    data_len = len(data['labels'])
    all_labels = list(chain(*data['labels'].tolist()))
    all_labels = list(map(int, all_labels))
    class_freq = dict(Counter(all_labels))
    class_freq = dict(sorted(class_freq.items()))
    #print(class_freq)
    freq_info['class_freq'] = np.array(list(class_freq.values()))
    freq_info['neg_class_freq'] = data_len - freq_info['class_freq']
    # print(freq_info)
    mmcv.dump(freq_info, dest_file)
    print("Class information saved!")