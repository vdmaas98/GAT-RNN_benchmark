# GAT-RNN Benchmark on METR-LA

Benchmark implementation of **GAT-LSTM** and **GAT-GRU** models for spatio-temporal traffic forecasting on the METR-LA dataset.

## Overview

This repository provides a standardized benchmark for evaluating Graph Attention Network (GAT) combined with recurrent architectures (LSTM/GRU) on traffic speed forecasting. The models leverage graph structure to capture spatial dependencies and recurrent units to model temporal dynamics.

### Models

- **GAT-LSTM**: Combines GATv2Conv layers with LSTM gates for spatio-temporal modeling
- **GAT-GRU**: Combines GATv2Conv layers with GRU gates for spatio-temporal modeling

Both models use:
- Multi-head attention (default: 2 heads)
- Layer normalization
- Dropout regularization
- Configurable hidden dimensions and prediction horizons

## Dataset: METR-LA

**METR-LA** is a widely-used traffic forecasting benchmark containing:
- **207 sensors** on highways in Los Angeles County
- **4 months** of data (March 2012 - June 2012)
- **5-minute intervals** (34,272 timesteps)
- **Traffic speed** measurements in miles per hour

The dataset is split into:
- Train: 70% (first ~24,000 timesteps)
- Validation: 10% (~3,400 timesteps)
- Test: 20% (last ~6,800 timesteps)

### Download Dataset

1. Download METR-LA from [Google Drive](https://drive.google.com/drive/folders/10FOTa6HXPqX8Pf5WRoRwcFnW9BrNZEIX) or the original [DCRNN repository](https://github.com/liyaguang/DCRNN)

2. Place the following files in `./data/METR-LA/`:
   ```
   data/METR-LA/
   ├── metr-la.h5           # Traffic speed data
   ├── adj_mx.pkl           # Adjacency matrix
   └── distances_la_2012.csv # Distance matrix (optional)
   ```

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/GAT-RNN_benchmark.git
cd GAT-RNN_benchmark

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install PyTorch Geometric dependencies (if needed)
pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.2.2+cu121.html
```

**Note**: Adjust the PyTorch Geometric wheel URL based on your CUDA version. See [PyG installation guide](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html).

## Usage

### Training

Train GAT-LSTM on METR-LA:

```bash
python train.py --model gatlstm --data_dir ./data/METR-LA --gpu 0
```

Train GAT-GRU:

```bash
python train.py --model gatgru --data_dir ./data/METR-LA --gpu 0
```

### Key Arguments

```
--model          Model architecture: 'gatlstm' or 'gatgru'
--data_dir       Path to METR-LA data directory
--seq_len        Input sequence length (default: 12 = 1 hour)
--pred_len       Prediction horizon (default: 12 = 1 hour)
--hidden_dim     Hidden dimension (default: 64)
--heads          Number of attention heads (default: 2)
--dropout        Dropout rate (default: 0.3)
--batch_size     Batch size (default: 64)
--epochs         Number of epochs (default: 100)
--lr             Learning rate (default: 0.001)
--gpu            GPU device ID (-1 for CPU)
```

### Example with Custom Settings

```bash
python train.py \
    --model gatlstm \
    --data_dir ./data/METR-LA \
    --hidden_dim 128 \
    --heads 4 \
    --batch_size 32 \
    --epochs 150 \
    --lr 0.0005 \
    --gpu 0
```

## Evaluation Metrics

The benchmark reports three standard metrics:

- **MAE** (Mean Absolute Error): Average absolute difference in speed (mph)
- **RMSE** (Root Mean Squared Error): Square root of average squared error
- **MAPE** (Mean Absolute Percentage Error): Average percentage error

Metrics are reported for:
- **Overall**: Across all 12 prediction steps
- **Horizon 3** (15 minutes ahead)
- **Horizon 6** (30 minutes ahead)
- **Horizon 12** (60 minutes ahead)

## Expected Results

Baseline results on METR-LA (12-step ahead prediction):

| Model     | MAE   | RMSE  | MAPE  |
|-----------|-------|-------|-------|
| GAT-LSTM  | TBD   | TBD   | TBD   |
| GAT-GRU   | TBD   | TBD   | TBD   |

**Reference baselines** (from literature):
- DCRNN: MAE 2.77, RMSE 5.38, MAPE 7.3%
- Graph WaveNet: MAE 2.69, RMSE 5.15, MAPE 6.9%
- STGCN: MAE 2.88, RMSE 5.74, MAPE 7.6%

## Model Architecture

### Input Format

```python
x_seq: [batch_size, seq_len, num_nodes, node_feat_dim]
edge_index: [2, num_edges]  # Static graph structure
edge_attr: [num_edges, edge_feat_dim]
```

### Output Format

```python
output: [batch_size, num_nodes, pred_len]
```

### Forward Pass

1. For each timestep in the input sequence:
   - Apply GAT layers to aggregate spatial information
   - Update recurrent hidden states (LSTM or GRU)
2. Project final hidden state to prediction horizon

## Project Structure

```
GAT-RNN_benchmark/
├── models/
│   ├── __init__.py
│   ├── gat_lstm.py       # GAT-LSTM implementation
│   └── gat_gru.py        # GAT-GRU implementation
├── utils/
│   ├── __init__.py
│   ├── data_loader.py    # METR-LA dataset loader
│   └── metrics.py        # Evaluation metrics
├── data/
│   └── METR-LA/          # Place dataset here
├── checkpoints/          # Saved models (created during training)
├── train.py              # Training script
├── requirements.txt
└── README.md
```

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{gat-rnn-benchmark,
  author = {Your Name},
  title = {GAT-RNN Benchmark for Spatio-Temporal Forecasting},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/yourusername/GAT-RNN_benchmark}
}
```

### Dataset Citation

```bibtex
@inproceedings{li2018dcrnn,
  title={Diffusion Convolutional Recurrent Neural Network: Data-Driven Traffic Forecasting},
  author={Li, Yaguang and Yu, Rose and Shahabi, Cyrus and Liu, Yan},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2018}
}
```

## License

MIT License

## Acknowledgments

- METR-LA dataset from [DCRNN](https://github.com/liyaguang/DCRNN)
- PyTorch Geometric for graph neural network layers
- Evaluation metrics follow standard traffic forecasting conventions

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.

## Contact

For questions or issues, please open a GitHub issue or contact [your-email@example.com].
