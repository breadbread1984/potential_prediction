#!/usr/bin/python3

from absl import flags, app
from os import mkdir
from os.path import exists, join
from torch import device, save, load, no_grad, any, isnan, autograd
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
  flags.DEFINE_integer('batch_size', default = 200, help = 'batch size')
  flags.DEFINE_integer('save_freq', default = 1000, help = 'checkpoint save frequency')
  flags.DEFINE_integer('epochs', default = 600, help = 'epochs to train')
  flags.DEFINE_float('lr', default = 1e-4, help = 'learning rate')
  flags.DEFINE_integer('decay_steps', default = 200000, help = 'decay steps')
  flags.DEFINE_list('eval_dists', default = ['1.7',], help = 'bond distances which are used as evaluation dataset')
  flags.DEFINE_integer('workers', default = 4, help = 'number of workers')
  flags.DEFINE_enum('device', default = 'cuda', enum_values = ['cpu', 'cuda'], help = 'device')

def main(unused_argv):
  autograd.set_detect_anomaly(True)
  eval_dists = [int(float(d) * 1000) for d in FLAGS.eval_dists]
  trainset = RhoDataset(join(FLAGS.dataset, 'train'))
  evalset = RhoDataset(join(FLAGS.dataset, 'val'))
  print('trainset size: %d, evalset size: %d' % (len(trainset), len(evalset)))
  train_dataloader = DataLoader(trainset, batch_size = FLAGS.batch_size, shuffle = True, num_workers = FLAGS.batch_size)
  eval_dataloader = DataLoader(evalset, batch_size = FLAGS.batch_size, shuffle = True, num_workers = FLAGS.batch_size)
  model = PredictorSmall()
  model.to(device(FLAGS.device))
  mae = L1Loss()
  optimizer = Adam(model.parameters(), lr = FLAGS.lr)
  scheduler = CosineAnnealingWarmRestarts(optimizer, T_0 = 5, T_mult = 2)
  tb_writer = SummaryWriter(log_dir = join(FLAGS.ckpt, 'summaries'))
  start_epoch = 0
  if not exists(FLAGS.ckpt): mkdir(FLAGS.ckpt)
  if exists(join(FLAGS.ckpt, 'model.pth')):
    ckpt = load(join(FLAGS.ckpt, 'model.pth'))
    model.load_state_dict(ckpt['state_dict'])
    optimizer.load_state_dict(ckpt['optimizer'])
    scheduler = ckpt['scheduler']
    start_epoch = ckpt['epoch']
  for epoch in range(start_epoch, FLAGS.epochs - start_epoch):
    model.train()
    for step, (x, y) in enumerate(train_dataloader):
      optimizer.zero_grad()
      rho, potential = x.to(device(FLAGS.device)), y.to(device(FLAGS.device))
      preds = model(rho)
      if any(isnan(preds)):
        print('there is nan in prediction results!')
        continue
      loss = mae(potential, preds)
      if any(isnan(loss)):
        print('there is nan in loss!')
        continue
      loss.backward()
      optimizer.step()
      global_steps = epoch * len(train_dataloader) + step
      if global_steps % 100 == 0:
        print('Step #%d Epoch #%d: loss %f, lr %f' % (global_steps, epoch, loss, scheduler.get_last_lr()[0]))
        tb_writer.add_scalar('loss', loss, global_steps)
      if global_steps % FLAGS.save_freq == 0:
        ckpt = {'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler}
        save(ckpt, join(FLAGS.ckpt, 'model.pth'))
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

