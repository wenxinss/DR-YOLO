import torch
# 如果没有torchtext包 需要使用命令 pip install torchtext  安装torchtext包
import torchtext.vocab as vocab
import numpy as np
import pickle
import scipy as scipy
import sklearn as sklearn

# 计算余弦相似度
def Cos(x, y):
    cos = torch.matmul(x, y.view((-1,))) / (
            (torch.sum(x * x) + 1e-9).sqrt() * torch.sum(y * y).sqrt())
    return cos


if __name__ == '__main__':
    classes = ['person', 'car', 'bus', 'bicycle',  'motorbike']
    total = np.array([])
    glove = vocab.GloVe(name="42B", dim=300)

    for cls in classes:
        a = glove.vectors[glove.stoi[cls]]
        total = np.append(total, a.numpy())
    total = total.reshape(len(classes), -1)
    # 保存对应类别的word embedding
    pickle.dump(total, open('/home8T/swx/yolov7/datasets/VOC_Dark/wordEmbedding.pkl', 'wb'), pickle.HIGHEST_PROTOCOL)
