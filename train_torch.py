#!/usr/bin/python3

from absl import flags, app
from os import mkdir
from os.path import exists, join
from torch import device, save, no_grad
from torch.nn import L1Loss
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchmetrics
from create_dataset_torch import RhoDataset
from models_torch import PredictorSmall

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
  flags.DEFINE_integer('workers', default = 4, help = 'number of workers')
  flags.DEFINE_enum('device', default = 'cuda', enum_values = ['cpu', 'cuda'], help = 'device')

def main(unused_argv):
  eval_dists = [int(float(d) * 1000) for d in FLAGS.eval_dists]
  trainset = RhoDataset(join(FLAGS.dataset, 'train'))
  evalset = RhoDataset(join(FLAGS.dataset, 'val'))
  print('trainset size: %d, evalset size: %d' % (len(trainset), len(evalset)))
  train_dataloader = DataLoader(trainset, batch_size = FLAGS.batch_size, shuffle = True, num_workers = FLAGS.batch_size)
  eval_dataloader = DataLoader(evalset, batch_size = FLAGS.batch_size, shuffle = True, num_workers = FLAGS.batch_size)
  model = PredictorSmall(in_channel = 4, out_channel = FLAGS.channels, groups = FLAGS.groups)
  model.to(device(FLAGS.device))
  mae = L1Loss()
  optimizer = Adam(model.parameters(), lr = FLAGS.lr)
  scheduler = CosineAnnealingWarmRestarts(optimizer, T_0 = 5)
  tb_writer = SummaryWriter(log_dir = join(FLAGS.ckpt, 'summaries'))
  global_steps = 0
  if not exists(FLAGS.ckpt): mkdir(FLAGS.ckpt)
  if exists(join(FLAGS.ckpt, 'model.pth')): model = load(join(FLAGS.ckpt, 'model.pth'))
  for epoch in range(FLAGS.epochs):
    model.train()
    for x, y in train_dataloader:
      rho, potential = x.to(device(FLAGS.device)), y.to(device(FLAGS.device))
      preds = model(rho)
      loss = mae(potential, preds)
      loss.backward()
      optimizer.step()
      global_steps += 1
      if global_steps % 100 == 0:
        print('#%d steps #%d epoch: loss %f' % (global_steps, epoch, loss))
        tb_writer.add_scalar('loss', loss, global_steps)
      if global_steps % FLAGS.save_freq == 0:
        save(model, join(FLAGS.ckpt, 'model.pth'))
    scheduler.step()
    with no_grad():
      model.eval()
      for x, y in eval_dataloader:
        rho, potential = x.to(device(FLAGS.device)), y.to(device(FLAGS.device))
        preds = model(rho)
        print('evaluate: loss %f' % torchmetrics.functional.mean_absolute_error(preds, potential))

if __name__ == "__main__":
  add_options()
  app.run(main)

