### Running the Project for Part 1 of Technical Overview

**1. Configure settings** (optional):

Edit `config.yaml` to adjust hyperparameters:

```yaml
device: "cuda"          # Change to "cpu" if no GPU
dataset:
  batch_size: 32        # Reduce to 16 if running out of memory
  num_workers: 0        # Keep 0 on Windows
training:
  epochs: 10
  lr: 1e-4
```

**2. Train the model**:

```bash
python train.py
```

This will:
- Download SST-2 dataset automatically (first run only)
- Train for up to 10 epochs with early stopping
- Save checkpoints to `checkpoints/`
- Generate attention visualizations in `attention_plots/`
- Log metrics to TensorBoard in `runs/`

**3. Monitor training with TensorBoard**:

```bash
python -m tensorboard.main --logdir=runs
```

Then open http://localhost:6006 in your browser.

**4. View attention visualizations**:

After training, check the `attention_plots/` directory for heatmaps showing which tokens the model attends to.