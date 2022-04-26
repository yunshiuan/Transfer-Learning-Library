
# Setup 

## conda environment

- cpu (`transfer`)
  - `conda create -n transfer pytorch==1.7.1 torchvision==0.8.2`
- gpu (`transfer-gpu` with cuda 10.2)
  - `conda create -n transfer-gpu pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=10.2 -c pytorch`

`pip install -r requirements.txt`

`conda install pandas`
`conda install scikit-learn`

- optional:
`conda install gpustat`

## other

- to import from folder
  - at the terminal: `export PYTHONPATH=.`

- check if gpu is available
  - in python terminal: `torch.cuda.is_available()`

------------------

# Sandbox

- following:
  - `https://tl.thuml.ai/get_started/quickstart.html`

## TODO

- evaluate on both the source and the target domain. Currently, the model is only evaluated on the target domain. I should partition the source domain into train and test, and also partition the target domain into train and test. I would like to know if the DANN performs equally well on both the source test set and the target test set.
  - () run the model and evaluate on the val & test set of both the source and the target domains

- apply the DANN model to the stance detection dataset.


------------------

# How to create a custom dataset

- should create a child class of `ImageList`
  - for an example, see `tllib/vision/datasets/office31_v2.py`
- run `sandbox/dann.py` with `DATASET = Office31_v2`

------------------

# run with nohup

- at the repo dir:
  - `nohup python -u sandbox/dann_v2.py &`