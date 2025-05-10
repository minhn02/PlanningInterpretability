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

Install viplanner as a local directory with
```
cd viplanner
pip install -e .
```

Reconstruct the point cloud by running:
```
python viplanner/viplanner/depth_reconstruct.py
```
NOTE: this seems to run out of memory on Macs, it is also avaiable to download here:
https://drive.google.com/file/d/1RnQFUz468x2i9e_V1y9PV5hJz6TYA3f3/view?usp=sharing.

After downloading drop this in the `carla` folder

Then construct the cost map with
```
python viplanner/viplanner/cost_builder.py
```

NOTE: this seems to run out of memory on Macs, it is also avaiable to download here:
https://drive.google.com/file/d/1jWdIjuDi4zO6NCIXZ8qrX82zpqZfn3Ld/view?usp=sharing

Also drop this in the `carla` folder after uncompressing.


## Reproducing Experiments

### Statistical Analysis
```
python statistical_analysis.py 
```