# Environment Settings
```
python==3.10.13
networkx==3.2.1
numpy==1.26.0
scikit-learn=1.3.0
scipy==1.11.3
torch==1.13.1
torch_geometric==2.4.0
ogb==1.3.6
```


# Usage
In this code, we only provide GraphGPS+WaveGC on PS and Photo because of the limitation of size.

For GraphGPS+WaveGC on PS:

1. Go into the folder ./wave_gps_graph/
2. Run the following command line

```
CUDA_VISIBLE_DEVICES=0 python main.py --cfg configs/wlgt/pf.yaml --repeat 3  wandb.use False
```

For GraphGPS+WaveGC on Photo:

1. Go into the folder ./wave_gps_node/datasets/, unzip the Amazon.zip and then run generate_eig.py
2. Back to ./wave_gps_node/, and run the following command line

```
CUDA_VISIBLE_DEVICES=0 python main.py --cfg configs/wlgt/photo.yaml --repeat 10  wandb.use False
```

