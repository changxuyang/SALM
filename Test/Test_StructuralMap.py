import torch
from torch.utils.data import DataLoader, Dataset
import h5py
import numpy as np
from PIL import Image
import os
import torch.nn.functional as F
from tqdm import tqdm
from models.lstm_unet import UNet_ConvLSTM

# ===== Paths and parameters =====
TestMode = 'SALM3'
assert TestMode in {'SALM1', 'SALM2', 'SALM3'}
CHECKPOINT = f"Checkpoint/{TestMode}_Checkpoint_StructuralMap.pkl"
INPUT_H5   = f"TestData/{TestMode}_TestData.H5"
OUTPUT_PNG = f"{TestMode}_Results/StructuralMap.png"

DEVICE = "cuda:0"   # or "cpu"
BATCH_SIZE = 1
WINDOW_SIZE = 5
IM_SIZEX = 544
IM_SIZEY = 752
# ============================================


class ExpDataset_H5_Test(Dataset):
    def __init__(self, h5_filepath, window_size=5):
        """
        h5_filepath: path to H5 file (dataset key assumed to be '/Input')
        window_size: number of consecutive frames per sample
        """
        self.h5_file = h5py.File(h5_filepath, 'r')
        self.data = self.h5_file['/Input']   # assumed shape: (num_frames, H, W)
        self.window_size = window_size
        self.num_frames = self.data.shape[0]
        self.num_samples = self.num_frames // self.window_size

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        start = idx * self.window_size
        end = start + self.window_size
        sample = self.data[start:end, :, :].astype('float32') / 255.0
        return torch.from_numpy(sample).unsqueeze(1)  # (T, 1, H, W)

    def __del__(self):
        self.h5_file.close()


def save_png_from_tensor(tensor_2d, out_path):
    """Normalize a 2D tensor to 0–255 and save as PNG."""
    diff = (tensor_2d.max() - tensor_2d.min()).item()
    if diff > 0:
        tensor_2d = (tensor_2d - tensor_2d.min()) / diff * 255.0
    img_uint8 = tensor_2d.clamp(0, 255).cpu().numpy().astype(np.uint8)
    os.makedirs(os.path.dirname(out_path), exist_ok=True) if os.path.dirname(out_path) else None
    Image.fromarray(img_uint8).save(out_path)


def main():
    device = torch.device(DEVICE)

    # dataset and loader
    dataset = ExpDataset_H5_Test(INPUT_H5, window_size=WINDOW_SIZE)
    data_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    # load model and checkpoint
    model = UNet_ConvLSTM(n_channels=1, n_classes=1, use_LSTM=True,
                          parallel_encoder=False, lstm_layers=1).to(device)
    model.load_state_dict(torch.load(CHECKPOINT, map_location=device))
    model.eval()

    # accumulate over all samples
    group_sum = torch.zeros((IM_SIZEX, IM_SIZEY), device=device)
    group_cnt = 0

    with torch.no_grad():
        for sample in tqdm(data_loader, desc="Inference", unit="batch"):
            sample = sample.to(device)  # (B, T, 1, H, W)
            B, T, C, H, W = sample.shape
            sample_flat = sample.view(B * T, C, H, W)
            sample_up = F.interpolate(sample_flat, size=(IM_SIZEX, IM_SIZEY),
                                      mode='bilinear', align_corners=False)
            sample_up = sample_up.view(B, T, C, IM_SIZEX, IM_SIZEY)

            # forward pass
            out = model(sample_up)[0].squeeze()
            group_sum += out
            group_cnt += 1

    if group_cnt > 0:
        group_avg = group_sum / float(group_cnt)
        save_png_from_tensor(group_avg, OUTPUT_PNG)
        print(f"Saved PNG: {OUTPUT_PNG}")
    else:
        print("[Warning] No samples found; nothing saved.")


if __name__ == '__main__':
    main()
