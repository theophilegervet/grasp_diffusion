## Conda Environment Setup

```
conda create -n analogical_grasping python=3.10;
conda activate analogical_grasping;
conda install -c conda-forge scikit-sparse;
pip install -e .;

git clone git@github.com:TheCamusean/mesh_to_sdf.git;
cd mesh_to_sdf; pip install -e .; cd ..;
```

## Download Data and Trained Models

Refer to original Readme to download data and trained models. 

## Train

Train pointcloud conditioned model
```
python scripts/train/train_pointcloud_6d_grasp_diffusion.py
```

Train partial pointcloud conditioned model
```
python scripts/train/train_partial_pointcloud_6d_grasp_diffusion.py
```

If you are training on a machine with a display, you can add `--summary 1` to the commands above to log visualizations of the generated grasps and SDF during training.

To overfit a single object for debugging, you can add `--overfit_one_object 1`.
