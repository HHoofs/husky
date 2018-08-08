import csv
import os


def read_csv_encoding(file='data/train/train_ship_segmentations.csv', length=None):
    img_encoding = dict()
    _file = file
    with open(_file, 'r') as _csv:
        _csv_lines = csv.reader(_csv)
        next(_csv_lines)
        for i, _csv_line in enumerate(_csv_lines):
            sample = os.path.splitext(_csv_line[0])[0]
            if not img_encoding.get(sample):
                img_encoding[sample] = {'encoding': [], 'class': 0, 'count': 0}
            img_encoding[sample]['encoding'].append(_csv_line[1])
            if len(_csv_line[1]) > 0:
                img_encoding[sample]['class'] = 1
                img_encoding[sample]['count'] += 1
            if i == length:
                break
    return img_encoding


def ids_for_each_count(img_encoding):
    ids_per_count = {}
    ids_for_empty = {}
    for sample, features in img_encoding.items():
        count = features['count']
        if count == 0:
            if not ids_for_empty.get(count):
                ids_for_empty[count] = []
            ids_for_empty[count].append(sample)
        else:
            if not ids_per_count.get(count):
                ids_per_count[count] = []
            ids_per_count[count].append(sample)

    return ids_for_empty, ids_per_count
