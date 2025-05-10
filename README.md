# Interpreting Motion Planning Neural Networks

## Setup
### Downloading Pretrained Models
1. Go to https://github.com/leggedrobotics/viplanner and download their model checkpoint and config (near the bottom of their README)
2. create a directory named `models` and copy both of those files as `model.pt` and `model.yaml`

### Local Setup
Create a local conda environment with
```
conda env create -f env.yml
conda activate viplanner
```

Reconstruct the point cloud by running:
```
python viplanner/viplanner/depth_reconstruct.py
```

Then construct the cost map with
```
python viplanner/viplanner/cost_builder.py
```

## Reproducing Experiments

### Statistical Analysis
```
python statistical_analysis.py 
```