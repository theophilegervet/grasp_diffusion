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
They should be in the same directory.