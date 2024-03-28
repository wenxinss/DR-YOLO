import glob
import os
import csv
from pathlib import Path
from sys import prefix
import numpy as np
import pickle

keys = ['img_name', 'person', 'car', 'bus', 'bicycle', 'motorbike']
img_formats = ['bmp', 'jpg', 'jpeg', 'png', 'tif', 'tiff', 'dng', 'webp', 'mpo']
lines = []
img_name = []
cls = []
def img2label_paths(img_paths):
    # Define label paths as a function of image paths
    sa, sb = os.sep + 'images' + os.sep, os.sep + 'labels' + os.sep  # /images/, /labels/ substrings
    return ['txt'.join(x.replace(sa, sb, 1).rsplit(x.split('.')[-1], 1)) for x in img_paths]

def LoadImagesAndLabels(path):
    f = []
    for p in path if isinstance(path, list) else [path]:
        p = Path(p)  # os-agnostic
        if p.is_dir():  # dir
            f += glob.glob(str(p / '**' / '*.*'), recursive=True)

        elif p.is_file():  # file
            with open(p, 'r') as t:
                t = t.read().strip().splitlines()

                for x in t:
                    f += [x]

        else:
            raise Exception(f'{prefix}{p} does not exist')
    img_files = sorted([x.replace('/', os.sep) for x in f if x.split('.')[-1].lower() in img_formats])
    label_files = img2label_paths(img_files)
    return img_files, label_files
def statistics(img_files,label_files,flag, classes, save_path):
    with open(save_path, 'a', newline='', encoding='utf-8') as f:
        for i in range(len(label_files)):
            with open(label_files[i]) as f1:
                writer = csv.DictWriter(f, fieldnames=keys)
                if flag == 0:
                    writer.writeheader()
                    flag += 1
                dic = {'img_name': img_files[i], 'person': 0, 'car': 0, 'bus': 0, 'bicycle': 0, 'motorbike': 0}
                print(img_files[i])
                while True:
                    line = f1.readline()
                    if not line:
                        break
                    if classes[int(line[0])] in dic.keys():
                        dic[classes[int(line[0])]] = 1
                    # lines.append((line))
                writer.writerow(dic)

def write_csv():
    with open('train_val1.csv', 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
def make_adj_file(train_csv, classes, pick_save_path):
    tmp = np.loadtxt(train_csv, dtype=np.str, delimiter=',')
    times = tmp[1:, 1:]


    adj_matrix = np.zeros(shape=(len(classes), len(classes)))
    nums_matrix = np.zeros(shape=(len(classes)))

    for index in range(len(times)):
        data = times[index]
        for i in range(len(classes)):
            if int(data[i]) >= 1:
                nums_matrix[i] += 1
                for j in range(len(classes)):
                    if j != i:
                        if int(data[j]) >= 1:
                            adj_matrix[i][j] += 1

    adj = {'adj': adj_matrix,
           'nums': nums_matrix}
    print(adj)
    pickle.dump(adj, open(pick_save_path, 'wb'), pickle.HIGHEST_PROTOCOL)
if __name__ == '__main__':
    path = 'datasets/mydata/train.txt'  # the train dataset path
    classes = ['person', 'car', 'bus', 'bicycle', 'motorbike']  # class
    save_path = ""  # the save path of statistics
    pick_save_path = ""
    img_files, label_files = LoadImagesAndLabels(path)
    statistics(img_files=img_files, label_files=label_files, classes=classes, flag=0, save_path=save_path)
    make_adj_file(save_path, classes, pick_save_path)

