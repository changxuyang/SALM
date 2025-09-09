import torch
from torch import nn
import argparse
from torch.utils.data import DataLoader, Dataset
from models.lstm_unet import UNet_ConvLSTM
import pandas as pd
from tqdm import tqdm
import h5py

class H5Dataset(Dataset):
    def __init__(self, h5_path, gt_type='velocity', transform=None):
        """
        h5_path: path to HDF5 file
        gt_type: which output to use, options: 'structure', 'velocity', 'direction'
        transform: optional data augmentation
        """
        self.h5_path = h5_path
        self.gt_type = gt_type
        self.transform = transform
        with h5py.File(self.h5_path, 'r') as f:
            self.num_samples = f['/input'].shape[0]

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        with h5py.File(self.h5_path, 'r') as f:
            input_data = f['/input'][idx]
            gt_data = f[f'/{self.gt_type}'][idx]
        # convert to torch tensor and normalize to [0,1]
        input_tensor = torch.from_numpy(input_data).float() / 255.0
        gt_tensor = torch.from_numpy(gt_data).float() / 255.0

        if self.transform:
            input_tensor = self.transform(input_tensor)
            gt_tensor = self.transform(gt_tensor)

        return input_tensor, gt_tensor, idx


parser = argparse.ArgumentParser()
parser.add_argument('--h5_file', default='./SALM1_Dataset.H5', help='Path to HDF5 training data file')
parser.add_argument('--checkpoint_dir', default='./Checkpoint/', help='Checkpoint directory')
parser.add_argument('--stat_dir', type=str, default='./statistics/', help='Statistics directory')
parser.add_argument('--im_size', type=int, default=256, help='High-resolution image size')
parser.add_argument('--batch_size', type=int, default=2, help='Batch size')
parser.add_argument('--start_epoch', type=int, default=0, help='Start epoch')
parser.add_argument('--epochs_til_ckpt', type=int, default=10, help='Save checkpoint every N epochs')
parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
parser.add_argument('--suffix', type=str, default='20250909', help='Experiment name suffix')
parser.add_argument('--device', type=str, default='cuda:0', help='Device to use')
parser.add_argument('--gt_type', type=str, default='velocity',
                    help='Ground truth type: structure, velocity')
opt = parser.parse_args()

device = torch.device(opt.device)

# initialize model
model = UNet_ConvLSTM(n_channels=1, n_classes=1, use_LSTM=True, parallel_encoder=False, lstm_layers=1)
model = model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.85)
MSEloss = nn.MSELoss().to(device)
loss_data = pd.DataFrame(columns=['Epoch', 'Total Loss', 'MSE Loss', 'LR'])

# dataset
train_dataset = H5Dataset(opt.h5_file, gt_type=opt.gt_type)
train_data_loader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True)

# training loop
for epoch in range(301):
    epoch_true = epoch + opt.start_epoch
    model.train()

    running_loss = 0.0
    MSE_loss = 0.0
    total_loss = 0.0

    for input_data, truth, filename in tqdm(train_data_loader, desc=f'Epoch {epoch_true + 1}/301', unit='batch',
                                            ncols=100):
        learning_rate = optimizer.param_groups[0]['lr']
        input_data, truth = input_data.to(device), truth.to(device)

        # upsample input
        B, F, C, H, W = input_data.shape
        input_data = input_data.view(B * F, C, H, W)
        input_data = torch.nn.functional.interpolate(input_data, size=opt.im_size, mode='bilinear', align_corners=False)
        input_data = input_data.view(B, F, C, opt.im_size, opt.im_size)

        optimizer.zero_grad()
        output = model(input_data)[0]

        loss = MSEloss(output, truth.unsqueeze(1))
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        MSE_loss += loss.item()
        total_loss += loss.item() * B

    scheduler.step()
    avg_total_loss = running_loss / len(train_data_loader)
    avg_mse_loss = MSE_loss / len(train_data_loader)

    print(f'Epoch {epoch_true + 1}, LR: {learning_rate}, Loss: {avg_total_loss}, MSELoss: {avg_mse_loss}')

    new_row = pd.DataFrame({'Epoch': [epoch_true + 1],
                            'LR': [learning_rate],
                            'Total Loss': [avg_total_loss],
                            'MSE Loss': [avg_mse_loss]})
    loss_data = pd.concat([loss_data, new_row], ignore_index=True)
    loss_data.to_excel(opt.stat_dir + '/training_loss-' + opt.suffix + '.xlsx', index=False)

    # save model
    if epoch_true % opt.epochs_til_ckpt == 0:
        torch.save(model.state_dict(), opt.checkpoint_dir + f'epoch_{epoch_true}_{opt.suffix}.pkl')
