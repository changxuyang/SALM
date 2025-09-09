import os
import h5py
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import scipy.io as sio
from models.lstm_unet_Direction import UNet_ConvLSTM

# ===== Paths and parameters =====
TestMode = 'SALM3'
assert TestMode in {'SALM1', 'SALM2', 'SALM3'}
CHECKPOINT = f"Checkpoint/{TestMode}_Checkpoint_DirectionMap.pkl"
INPUT_H5   = f"TestData/{TestMode}_TestData.H5"
OUTPUT_DIR = f"{TestMode}_Results"

OUTPUT_MAT = os.path.join(OUTPUT_DIR, "DirectionMap.mat")
OUTPUT_PSEUDOCOLOR_PNG = os.path.join(OUTPUT_DIR, "DirectionMap_color.png")
MASK_PATH  = f"TestData/Mask.mat"   # root folder; variable name 'bw'

DEVICE = "cuda:0"   # or "cpu"
BATCH_SIZE = 1
WINDOW_SIZE = 5
IM_SIZEX = 544
IM_SIZEY = 752
EPS = 0.0          # if >0, treat |label|<=EPS as zero (ignored)
# =================================


class ExpDataset_H5_Test(Dataset):
    def __init__(self, h5_filepath, window_size=5):
        self.h5_file = h5py.File(h5_filepath, 'r')
        self.data = self.h5_file['/Input']   # shape: (num_frames, H, W)
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
        try:
            self.h5_file.close()
        except Exception:
            pass


def save_mat_int32(arr2d: torch.Tensor, path: str, var_name: str = "DirectionMap"):
    os.makedirs(os.path.dirname(path), exist_ok=True) if os.path.dirname(path) else None
    sio.savemat(path, {var_name: arr2d.cpu().numpy().astype(np.int32)})


def save_pseudocolor_png(labels_0_8: torch.Tensor, path: str):
    """Save a 0–8 label map as a palette PNG with legend attached."""
    import numpy as np
    from PIL import Image, ImageDraw

    # ---- Tensor -> numpy -> PIL.Image ----
    labels_np = labels_0_8.clamp(0, 8).to(torch.uint8).cpu().numpy()
    label_img = Image.fromarray(labels_np, mode="P")

    palette = [
        0, 0, 0,          # 0
        31, 119, 180,     # 1
        77, 187, 214,     # 2
        44, 160, 44,      # 3
        106, 204, 106,    # 4
        255, 127, 14,     # 5
        255, 180, 85,     # 6
        214, 39, 40,      # 7
        240, 128, 128,    # 8
    ] + [0, 0, 0] * (256 - 9)
    label_img.putpalette(palette)

    # ---- legend ----
    swatch_size = 40
    legend = Image.new("RGB", (swatch_size * 9, swatch_size), (255, 255, 255))
    draw = ImageDraw.Draw(legend)
    colors = [(palette[i*3], palette[i*3+1], palette[i*3+2]) for i in range(9)]
    for i, c in enumerate(colors):
        draw.rectangle([i*swatch_size, 0, (i+1)*swatch_size, swatch_size], fill=c)
        draw.text((i*swatch_size+12, 10), str(i), fill=(255, 255, 255))

    # ---- combine result + legend ----
    w = max(label_img.width, legend.width)
    combined = Image.new("RGB", (w, label_img.height + legend.height), (255, 255, 255))
    combined.paste(label_img.convert("RGB"), (0, 0))
    combined.paste(legend, (0, label_img.height))

    combined.save(path)
    print(f"Saved pseudocolor PNG with legend: {path}")


def main():
    # device guard
    device = torch.device(DEVICE if (DEVICE.startswith("cuda") and torch.cuda.is_available()) else "cpu")

    # dataset & loader
    dataset = ExpDataset_H5_Test(INPUT_H5, window_size=WINDOW_SIZE)
    data_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    # model
    model = UNet_ConvLSTM(n_channels=1, use_LSTM=True, parallel_encoder=False, lstm_layers=1)
    state = torch.load(CHECKPOINT, map_location="cpu")
    model.load_state_dict(state)
    model = model.to(device).eval()

    # prepare accumulation tensors
    group_sum   = torch.zeros((IM_SIZEX, IM_SIZEY), device=device, dtype=torch.float32)
    group_count = torch.zeros((IM_SIZEX, IM_SIZEY), device=device, dtype=torch.float32)

    # load and prep mask 'bw'
    if os.path.isfile(MASK_PATH):
        mat = sio.loadmat(MASK_PATH)
        if "bw" not in mat:
            raise KeyError("mask.mat does not contain variable 'bw'.")
        bw = torch.from_numpy(mat["bw"]).to(device)
        if bw.dim() > 2:
            bw = bw.squeeze()
        bw = (bw > 0).to(torch.float32)
        if bw.shape != (IM_SIZEX, IM_SIZEY):
            bw = F.interpolate(bw.unsqueeze(0).unsqueeze(0),
                               size=(IM_SIZEX, IM_SIZEY),
                               mode="bilinear", align_corners=False
                               ).squeeze(0).squeeze(0)
            bw = (bw > 0.5).to(torch.float32)
    else:
        # if no mask provided, use all-ones (no masking)
        bw = torch.ones((IM_SIZEX, IM_SIZEY), device=device, dtype=torch.float32)

    with torch.no_grad():
        for sample in tqdm(data_loader, desc="Inference", unit="batch"):
            sample = sample.to(device)  # (B, T, 1, H, W)
            B, T, C, H, W = sample.shape

            # resize to target shape
            sample_flat = sample.view(B * T, C, H, W)
            sample_up = F.interpolate(sample_flat, size=(IM_SIZEX, IM_SIZEY),
                                      mode='bilinear', align_corners=False).contiguous()
            sample_up = sample_up.view(B, T, C, IM_SIZEX, IM_SIZEY)

            # forward
            out = model(sample_up)[0]
            predicted = torch.argmax(out, dim=1).to(torch.float32).squeeze()  # (H, W), int [0..num_classes-1]

            # only accumulate non-zero (or >EPS) and finite
            if EPS > 0:
                mask = (predicted.abs() > EPS) & torch.isfinite(predicted)
            else:
                mask = (predicted != 0) & torch.isfinite(predicted)

            # per-pixel accumulate
            group_sum[mask]   += predicted[mask]
            group_count[mask] += 1.0

    if group_count.sum().item() == 0:
        print("[Warning] No valid (non-zero & finite) pixels found; nothing saved.")
        return

    # per-pixel average over counted samples
    group_avg = torch.zeros_like(group_sum)
    valid = group_count > 0
    group_avg[valid] = group_sum[valid] / group_count[valid]

    group_avg = group_avg.round().clamp(0, 8).to(torch.int64)
    group_avg = (group_avg.to(torch.float32) * bw).to(torch.int64)

    # save MAT (int32)
    save_mat_int32(group_avg, OUTPUT_MAT, var_name="DirectionMap")
    print(f"Saved MAT: {OUTPUT_MAT}")

    # save pseudocolor PNG
    save_pseudocolor_png(group_avg, OUTPUT_PSEUDOCOLOR_PNG)
    print(f"Saved pseudocolor PNG: {OUTPUT_PSEUDOCOLOR_PNG}")


if __name__ == '__main__':
    main()
