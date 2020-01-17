import argparse

ap = argparse.ArgumentParser()
ap.add_argument('--src_data', default='data/coco.data')
ap.add_argument('--dst_data', default='data/coco-person-bag.data')
args = ap.parse_args()

classes = [l.strip() for l in open(args.src_data)]
classes = filter(lambda x:'names=' in x, classes)
classes = list(classes)[0].replace('names=', '')
classes = [l.strip() for l in open(classes)]

## CHOOSE BAG CLASS NAMES ##
sub_classes = ['person', 'backpack', 'handbag', 'suitcase']
sub_indices = [str(classes.index(s)) for s in sub_classes]

cfg = {}
with open(args.src_data) as fr:
    for l in fr:
        k,v = map(str.strip, l.split('='))
        cfg[k] = v

import os.path as osp
outputs = osp.splitext(args.dst_data)[0]

import os

from glob import glob
from tqdm import tqdm
from shutil import copy


def maybe_create_folder(folder_name):
    if osp.exists(folder_name)==False:
        os.mkdir(folder_name)


maybe_create_folder(outputs)
maybe_create_folder(osp.join(outputs, 'labels'))
maybe_create_folder(osp.join(outputs, 'images'))

for split, saveAs in zip(map(osp.expanduser, [cfg['valid'], cfg['train']]),
                         ['valid.txt', 'train.txt']):
    useful = []
    for l in tqdm(open(split).readlines(), desc=osp.basename(split)):
        image_path = l.strip()
        label_path = image_path.replace('images', 'labels').replace('.jpg', '.txt')

        image = osp.basename(image_path)
        label = osp.splitext(image)[0]+'.txt'

        if osp.exists(label_path)==False: continue
        p = [] # getting lines containing elements of the subclass
        for line in open(label_path):
            i = line.split()
            if i[0] in sub_indices:
                i[0] = str(sub_indices.index(i[0]))
                p.append(' '.join(i) + os.linesep)
        if len(p)==0: continue
        image = osp.join(outputs, 'images', image)
        useful.append(image)
        copy(image_path, image)
        with open(osp.join(outputs, 'labels', label), 'w') as fw:
            for line in p:
                fw.writelines(line)
    cfg[saveAs.split('.')[0]] = label_list_path = osp.join(outputs, saveAs)
    with open(label_list_path, 'w') as fw:
        for u in useful:
            print(u.strip(), file=fw)

cfg['names'] = args.dst_data.replace('.data', '.names')
with open(cfg['names'], 'w') as f:
    for s in sub_classes:
        print(s, file=f)
cfg['classes'] = len(sub_classes)
cfg['eval'] = 'single'
with open(args.dst_data, 'w') as fw:
    for k,v in cfg.items():
        print(f'{k}={v}', file=fw)
