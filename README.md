# Introduction

this project is to predict xc potential from electron density.

# Usage

## prepare dataset

```shell
python3 create_dataset.py --input_dir <path/to/raw/dataset> --output_dir <path/to/directory/for/tfrecords> --eval_dists dist1,dist2,...
```

## training

```shell
python3 train.py --dataset <path/to/directory/for/tfrecord>
```
