import mxnet as mx
from mxnet import gluon, nd, init, npx
from mxnet.gluon import nn, Block, HybridBlock
from operator import itemgetter
import numpy as np


def auc(pred, label):
    label = label.asnumpy()
    pred = pred.asnumpy()
    tmp = []
    for i in range(pred.shape[0]):
        tmp.append((label[i], pred[i]))
    tmp = sorted(tmp, key=itemgetter(1), reverse=True)
    label_sum = label.sum()
    if label_sum == 0 or label_sum == label.size:
        raise Exception("AUC with one class is undefined")
    label_one_num = np.count_nonzero(label)
    label_zero_num = len(label) - label_one_num
    total_area = label_zero_num * label_one_num
    height = 0
    width = 0
    area = 0
    for a, _ in tmp:
        if a == 1.0:
            height += 1.0
        else:
            width += 1.0
            area += height
    return area / total_area


class DeepFM(nn.Block):
    def __init__(self, small_field_num, large_field_num, feature_num,
                 embedding_dim, mlp_dims):
        super(DeepFM, self).__init__()
        self.field_num = small_field_num + large_field_num
        self.small_field_num = small_field_num
        self.large_field_num = large_field_num
        self.embedding_dim = embedding_dim
        self.embedding = nn.Embedding(feature_num,
                                      embedding_dim,
                                      sparse_grad=False)
        self.fc = nn.Embedding(feature_num, 1, sparse_grad=False)
        self.linear_layer = nn.Dense(1, use_bias=True)
        input_dim = self.embed_output_dim = self.field_num * embedding_dim
        self.mlp = nn.Sequential()
        for dim in mlp_dims:
            self.mlp.add(nn.Dense(dim, 'relu', True, in_units=input_dim))
            self.mlp.add(nn.Dropout(rate=0.1))
            input_dim = dim
        self.mlp.add(nn.Dense(in_units=input_dim, units=1))

    def forward(self, small_embed_ids, large_embed_ids, embedding_input):
        ids = nd.concat(small_embed_ids, large_embed_ids, dim=1)
        embed_x = self.embedding(small_embed_ids)
        embedding_input = nd.reshape(embedding_input,
                                     shape=(-1, self.large_field_num,
                                            self.embedding_dim))
        embed_x = nd.concat(embed_x, embedding_input, dim=1)
        square_of_sum = nd.sum(embed_x, axis=1) * nd.sum(embed_x, axis=1)
        sum_of_square = nd.sum(embed_x * embed_x, axis=1)
        inputs = nd.reshape(embed_x, (-1, self.embed_output_dim))
        x = self.linear_layer(
            self.fc(ids).sum(1)) + 0.5 * (square_of_sum - sum_of_square).sum(
                1, keepdims=True) + self.mlp(inputs)
        # x = nd.concat(self.mlp(inputs), x, dim=1)
        # x = mx.nd.sigmoid(x)
        return x
