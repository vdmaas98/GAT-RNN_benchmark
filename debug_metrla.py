import os
import shutil
import numpy as np
import torch
import kagglehub

from utils import load_metr_la
from models import PlainLSTM


def main():
    torch.manual_seed(0)
    np.random.seed(0)

    repo_root = os.path.abspath(os.path.dirname(__file__))
    data_dir = os.path.join(repo_root, "data", "METR-LA")
    os.makedirs(data_dir, exist_ok=True)

    path = kagglehub.dataset_download("annnnguyen/metr-la-dataset")

    shutil.copy(os.path.join(path, "METR-LA.h5"),
                os.path.join(data_dir, "metr-la.h5"))
    shutil.copy(os.path.join(path, "adj_METR-LA.pkl"),
                os.path.join(data_dir, "adj_mx.pkl"))

    train_loader, _, _, edge_index, edge_attr, scaler = load_metr_la(
        data_dir=data_dir,
        seq_len=12,
        pred_len=12,
        batch_size=2,
    )

    sample_x, sample_y = next(iter(train_loader))

    print("sample_x shape:", sample_x.shape)
    print("sample_y shape:", sample_y.shape)

    x0 = sample_x[0]
    y0 = sample_y[0]

    print("\nFirst sample, first 3 timesteps, first 3 nodes, all features (normalized):")
    print(x0[:3, :3, :])

    print("\nFirst sample, first 3 prediction steps, first 3 nodes, target (normalized speed channel):")
    print(y0[:3, :3])

    node_feat_dim = sample_x.shape[-1]
    model = PlainLSTM(
        node_feat_dim=node_feat_dim,
        hidden_dim=32,
        output_dim=12,
        num_layers=1,
        dropout=0.0,
    )

    with torch.no_grad():
        y_pred_norm = model(sample_x)

    print("\nPredictions (normalized space), first sample, first 3 nodes, first 5 steps:")
    print(y_pred_norm[0, :3, :5])

    y_true_flat = sample_y.reshape(-1)
    y_pred_flat = y_pred_norm.transpose(1, 2).reshape(-1)

    y_true_rescaled = scaler.inverse_transform(y_true_flat)
    y_pred_rescaled = scaler.inverse_transform(y_pred_flat)

    print("\nRescaled ground truth speeds (first 20):")
    print(y_true_rescaled[:20])

    print("\nRescaled predicted speeds (first 20):")
    print(y_pred_rescaled[:20])

    print("\nRescaled stats:")
    print("  y_true mean/std:", float(y_true_rescaled.mean()), float(y_true_rescaled.std()))
    print("  y_pred mean/std:", float(y_pred_rescaled.mean()), float(y_pred_rescaled.std()))


if __name__ == "__main__":
    main()
