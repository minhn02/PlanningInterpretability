# Interpreting Motion Planning Neural Networks

## Setup
### Downloading Pretrained Models
1. Go to https://drive.google.com/file/d/1PY7XBkyIGESjdh1cMSiJgwwaIT0WaxIc/view?usp=sharing to download the latest viplanner model checkpoint
2. copy this in the `models` folder as `model.pt`

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

At the end your `carla` folder should have: depth, maps, semantics, camera_extrinsic.txt, cloud.ply, and intrinsics.txt


## Reproducing Experiments

### Statistical Analysis + Linear Probing
```
python statistical_analysis.py 
```

This will display plots and dump `data` to be used in other experiments

### Guided Backprop
```
python guided_backprop.py
```

This will display and dump plots into `plots`.

## Neuron Patching
```
python neuron_ablations.py
```

These will dump plots into the `plots` folder.