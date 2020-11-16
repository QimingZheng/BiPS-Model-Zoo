import argparse
from rec_model import *
import numpy as np
import mxnet as mx
from recsys.mxnet.ops import MXRecsys
import recsys
import sys
import time
import logging
import os

logging.basicConfig(level=logging.INFO)


parser = argparse.ArgumentParser(
    description="Train Recommendation Models with RecSys",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--num-epoch',
                    type=int,
                    default=1,
                    help='number of epochs to train')
parser.add_argument('--batch-size',
                    type=int,
                    default=2048,
                    help='number of examples per batch')
parser.add_argument(
    '--log-interval',
    type=int,
    default=1,
    help='number of batches to wait before logging training status')
parser.add_argument('--cluster-config',
                    type=str,
                    help='cluster configuration file')
parser.add_argument('--large-field',
                    type=int,
                    default=1,
                    help='number of large embedding field')
parser.add_argument('--small-field',
                    type=int,
                    default=38,
                    help='number of small embedding field')
parser.add_argument('--large-feature',
                    type=int,
                    default=2000000,
                    help='number of large embedding feature')
parser.add_argument('--small-feature',
                    type=int,
                    default=1000000,
                    help='number of small embedding feature')
parser.add_argument('--model',
                    type=str,
                    default="deepfm",
                    help='Model name [widedeep, deepfm]')

parser.add_argument('--train-data',
                    type=str,
                    help='Path to the training data')
parser.add_argument('--train-label',
                    type=str,
                    help='Path to the training label')
parser.add_argument('--eval-data',
                    type=str,
                    help='Path to the evaluation data')
parser.add_argument('--eval-label',
                    type=str,
                    help='Path to the evaluation label')
parser.add_argument('--checkpoint-path',
                    type=str,
                    help='checkpoint folder path')


def get_model(model_name, small_field_num, large_field_num, small_feature_num,
              dev):
    if model_name == "deepfm":
        return DeepFM(small_field_num, large_field_num, small_feature_num, 80,
                      [512, 256, 128])
    if model_name == "widedeep":
        return WideDeep(small_field_num, large_field_num, small_feature_num, 80,
                      [512, 256, 128])
    else:
        raise NotImplementedError


def run_server(bips):
    bips.run_server()
    return


def run_worker(bips, args):
    dev = bips.get_dev_id()
    small_feature_num = args.small_feature
    large_feature_num = args.large_feature
    large_field_num = args.large_field
    small_field_num = args.small_field
    epoch = args.num_epoch
    batch_size = args.batch_size
    log_interval = args.log_interval
    dim = 80  # each feature dim = 80
    embedding_shape = [large_feature_num, dim]
    bips.init_meta(0, embedding_shape)
    bips.init_embedding(0, embedding_shape)

    if args.model == "widedeep":
        small_dim = 1
        small_embedding_shape = [large_feature_num, small_dim]
        bips.init_meta(1, small_embedding_shape)
        bips.init_embedding(1, small_embedding_shape)

    # Read Dataset
    CRITEO_FIELD_NUM = large_field_num + small_field_num
    """
    train_data = mx.io.CSVIter(data_csv=args.train_data + "." + str(dev), data_shape=(CRITEO_FIELD_NUM,),
                               label_csv=args.train_label + "." + str(dev), label_shape=(1,),
                               batch_size=batch_size, round_batch=False,
                               prefetching_buffer=4)
    """
    train_data = mx.io.CSVIter(data_csv=args.train_data, data_shape=(CRITEO_FIELD_NUM,),
                               label_csv=args.train_label, label_shape=(1,),
                               batch_size=batch_size, round_batch=False,
                               prefetching_buffer=4)
    eval_data = mx.io.CSVIter(data_csv=args.eval_data, data_shape=(CRITEO_FIELD_NUM,),
                              label_csv=args.eval_label, label_shape=(1,),
                              batch_size=batch_size, round_batch=False,
                              prefetching_buffer=4)
    # Build Model
    model = get_model(args.model, small_field_num, large_field_num,
                      small_feature_num, dev)
    ctx = [mx.gpu(dev)]
    # model.initialize(mx.init.Uniform(), ctx=ctx)
    # model.initialize(mx.init.Constant(0.0001), ctx=ctx)
    model.initialize(mx.init.Xavier(), ctx=ctx)
    # model.initialize(mx.init.Normal(sigma=0.05), ctx=ctx)
    model.hybridize()

    loss = gluon.loss.SigmoidBinaryCrossEntropyLoss()
    # loss = gluon.loss.SoftmaxCrossEntropyLoss()
    trainer = mx.gluon.Trainer(
        model.collect_params(),
        optimizer='adam',
        optimizer_params={
            'learning_rate': 0.0001,
        })

    embedding_weight = mx.nd.zeros((batch_size, large_field_num * dim),
                                   ctx=mx.gpu(dev))
    if args.model == "widedeep":
        small_embedding_weight = mx.nd.zeros((batch_size, large_field_num * small_dim),
                                       ctx=mx.gpu(dev))
    embedding_weight.attach_grad()
    start = time.time()
    train_data_iter = iter(train_data)
    eval_data_iter = iter(eval_data)
    for epoch_num in range(epoch):
        batch_num = 0
        for batch in train_data_iter:
            all_embedding_ids = mx.nd.split(batch.data[0],
                                            axis=1,
                                            num_outputs=CRITEO_FIELD_NUM)
            if small_field_num > 1:
                small_embedding_id = mx.nd.Concat(
                    *(all_embedding_ids[:small_field_num]), dim=1).astype(
                        np.int64)
            else:
                small_embedding_id = all_embedding_ids[0].astype(np.int64)
            if large_field_num > 1:
                embedding_id = mx.nd.Concat(
                    *(all_embedding_ids[small_field_num:]), dim=1).astype(
                        np.int64)
            else:
                embedding_id = all_embedding_ids[-1].astype(np.int64)
            label = batch.label[0]
            label = label.copyto(mx.gpu(dev))
            small_embedding_id = small_embedding_id.copyto(mx.gpu(dev))
            embedding_id = embedding_id.copyto(mx.gpu(dev))
            mx.nd.waitall()

            bips.pull_embedding_weights(0, embedding_weight, embedding_id)
            if args.model == "widedeep":
                bips.pull_embedding_weights(1, small_embedding_weight, embedding_id)

            # do some graph computation
            with mx.autograd.record():
                if args.model == "widedeep":
                    output = model(small_embedding_id, embedding_weight, small_embedding_weight)
                else:
                    output = model(small_embedding_id, embedding_id,
                                   embedding_weight)
                _loss = loss(output, label)

            _loss.backward()
            grads = [
                i.grad(mx.gpu(dev)) for i in model.collect_params().values()
            ]
            embedding_grad = embedding_weight.grad

            for grad in grads:
                bips.all_reduce(grad)

            trainer.update(batch_size, ignore_stale_grad=True)

            bips.push_embedding_grads(0, embedding_grad, embedding_id)
            if args.model == "widedeep":
                bips.push_embedding_grads(1, small_embedding_weight, embedding_id)

            if batch_num % log_interval == 0 and batch_num > 0:
                elapsed = time.time() - start
                print(
                    "Epoch [%d] Batch [%d], Loss = %f, Auc = %f, Throughput = %f k samples/sec"
                    % (epoch_num, batch_num, np.mean(
                        _loss.asnumpy()), auc(mx.nd.sigmoid(output), label),
                        batch_size * log_interval * 1.0 / (elapsed * 1000.0)))
                start = time.time()

            batch_num += 1

        train_data_iter.reset()

        if args.checkpoint_path != None:
            bips.save_weight(0, large_feature_num, dim)
            bips.load_weight(0, large_feature_num, dim)
            if args.model == "widedeep":
                    bips.save_weight(1, large_feature_num, small_dim)
                    bips.load_weight(1, large_feature_num, small_dim)

        eval_loss = 0.0
        eval_auc = 0.0
        eval_batch_num = 0

        for batch in eval_data_iter:
            all_embedding_ids = mx.nd.split(batch.data[0],
                                            axis=1,
                                            num_outputs=CRITEO_FIELD_NUM)
            if small_field_num > 1:
                small_embedding_id = mx.nd.Concat(
                    *(all_embedding_ids[:small_field_num]), dim=1).astype(
                        np.int64)
            else:
                small_embedding_id = all_embedding_ids[0].astype(np.int64)
            if large_field_num > 1:
                embedding_id = mx.nd.Concat(
                    *(all_embedding_ids[small_field_num:]), dim=1).astype(
                        np.int64)
            else:
                embedding_id = all_embedding_ids[-1].astype(np.int64)
            label = batch.label[0]
            label = label.copyto(mx.gpu(dev))
            small_embedding_id = small_embedding_id.copyto(mx.gpu(dev))
            embedding_id = embedding_id.copyto(mx.gpu(dev))
            mx.nd.waitall()

            bips.pull_embedding_weights(0, embedding_weight, embedding_id)
            if args.model == "widedeep":
                bips.pull_embedding_weights(1, small_embedding_weight, embedding_id)

            # do some graph computation
            with mx.autograd.record():
                if args.model == "widedeep":
                    output = model(small_embedding_id, embedding_weight, small_embedding_weight)
                else:
                    output = model(small_embedding_id, embedding_id,
                                   embedding_weight)
                _loss = loss(output, label)

            mx.nd.waitall()

            eval_loss += np.mean(_loss.asnumpy())
            eval_auc += auc(mx.nd.sigmoid(output), label)
            eval_batch_num += 1.0
        eval_data_iter.reset()
        print("Epoch [%d], Eval-Loss = %f, Eval-Auc = %f"
              % (epoch_num, eval_loss/eval_batch_num, eval_auc/eval_batch_num))
    bips.shuts_down()
    return


def main(args):
    bips = MXRecsys(args.cluster_config)
    bips.build_replica()  # must build replica first
    if bips.is_worker():
        if args.checkpoint_path != None:
            bips.set_checkpoint_path(0, args.checkpoint_path + ".0")
            if args.model == "widedeep":
                bips.set_checkpoint_path(1, args.checkpoint_path + ".1")
        run_worker(bips, args)
    else:
        run_server(bips)
    return


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
