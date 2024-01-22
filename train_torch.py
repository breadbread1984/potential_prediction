#!/usr/bin/python3

from absl import flags, app
from create_dataset_torch import RhoDataset

FLAGS = flags.FLAGS

def add_options():
  flags.DEFINE_string('dataset', default = None, help = 'path to directory containing train and test set')
  flags.DEFINE_string('ckpt', default = 'ckpt', help = 'path to directory for checkpoints')
  flags.DEFINE_integer('channels', default = 768, help = 'output channel')
  flags.DEFINE_integer('groups', default = 1, help = 'group number for conv')
  flags.DEFINE_integer('batch_size', default = 128, help = 'batch size')
  flags.DEFINE_integer('save_freq', default = 1000, help = 'checkpoint save frequency')
  flags.DEFINE_integer('epochs', default = 600, help = 'epochs to train')
  flags.DEFINE_float('lr', default = 0.01, help = 'learning rate')
  flags.DEFINE_integer('decay_steps', default = 200000, help = 'decay steps')
  flags.DEFINE_list('eval_dists', default = ['1.7',], help = 'bond distances which are used as evaluation dataset')

def main(unused_argv):
  eval_dists = [int(float(d) * 1000) for d in FLAGS.eval_dists]
  trainset = RhoDataset(FLAGS.dataset, eval_dists, True)
  evalset = RhoDataset(FLAGS.dataset, eval_dists, False)
  print('trainset size: %d, evalset size: %d' % (len(trainset), len(evalset)))
  train_dataloader = DataLoader(trainset, batch_size = FLAGS.batch_size, shuffle = True)
  eval_dataloader = DataLoader(evalset, batch_size = FLAGS.batch_size, shuffle = True)
  for sample, label in train_dataloader:
    print(sample.shape, label.shape)
    break

if __name__ == "__main__":
  add_options()
  app.run(main)

