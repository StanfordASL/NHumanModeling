**UPDATE**: There is a newer and much faster version of this codebase implemented in PyTorch! Take a look **[here](https://github.com/StanfordASL/DynSTGModeling)**!

# N-Human Modeling

This repository contains the code for [Generative Modeling of Multimodal Multi-Human Behavior](https://arxiv.org/abs/1803.02015) by Boris Ivanovic, Edward Schmerling, Karen Leung, and Marco Pavone.

**Note**: We use [Git LFS](https://git-lfs.github.com) to version large files (such as model checkpoints and data). 

## Installation ##

First, we'll create a conda environment to hold the dependencies.
```
conda create --name modeling python=2.7 -y
source activate modeling
pip install -r requirements.txt
```

Then, since this project uses IPython notebooks, we'll install this conda environment as a kernel.
```
python -m ipykernel install --user --name modeling --display-name "Python 2.7 (NHumanModeling)"
```

Now, you can start a Jupyter session and view/run all the notebooks with
```
jupyter notebook
```

When you're done, don't forget to deactivate the conda environment with
```
source deactivate
```

## Datasets ##

The preprocessed datasets are available in this repository, under `data/` folders (e.g. `nba-dataset/data/`).

If you want the *original* traffic weaving or NBA datasets, I obtained them from here: [Traffic Weaving Dataset](https://github.com/StanfordASL/TrafficWeavingCVAE) and [NBA Dataset](https://github.com/linouk23/NBA-Player-Movements).
