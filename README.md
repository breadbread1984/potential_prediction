# Introduction

this project is to predict xc potential from electron density.

# Usage

## prepare dataset

### Tensorflow

```shell
python3 create_dataset.py --input_dir <path/to/raw/dataset> --output_dir <path/to/directory/for/tfrecords> --eval_dists dist1,dist2,...
```

|param | description |
|------|---------|
|input_dir| path to directory containing npy files |
|output_dir| path to directory to hold generated tfrecord files |
|eval_dists| a list of distances to use as evaluation dataset | 

### Pytorch

nothing is needed for pytorch.

## training

### Tensorflow

train with Tensorflow keras

```shell
python3 train_keras.py --dataset <path/to/directory/for/tfrecord> [--ckpt <path/to/checkpoint>] [--batch_size <batch size>] [--lr <learning rate>] [--dist]
```

train with Tensorflow eager

```shell
python3 train_eager.py --dataset <path/to/directory/for/tfrecord> [--ckpt <path/to/checkpoint>] [--batch_size <batch size>] [--lr <learning rate>]
```

|param | description |
|------|-------------|
|dataset| path to directory containing tfrecord files|
|ckpt| path to directory to hold checkpoints, default path is ./ckpt |
|batch_size| batch size|
|lr | the initial learning rate|
|dist| whether to use distribute training. if set data parallelism is employed among visible GPUs |

### Pytorch

train with Pytorch

```shell
python3 train_torch.py --dataset <path/to/directory/for/npy> [--ckpt <path/to/checkpoint>] [--batch_size <batch size>] [--lr <learning rate>] [--workers <number of workers>] [--device (cpu|cuda)]
```

|param | description |
|------|-------------|
|dataset| path to directory containing npy files|
|ckpt| path to directory to hold checkpoints, default path is ./ckpt |
|batch_size| batch size|
|lr | the initial learning rate|
|workers| number of workers for data loader |
|device | cpu or cuda|

