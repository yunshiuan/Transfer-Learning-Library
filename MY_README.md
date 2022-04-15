
# Setup 

## conda environment

- cpu (`transfer`)
  - `conda create -n transfer pytorch==1.7.1 torchvision==0.8.2`
- gpu (`transfer-gpu` with cuda 10.2)
  - `conda create -n transfer-gpu pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=10.2 -c pytorch`

`pip install -r requirements.txt`

- optional:
`conda install gpustat`

## other

- to import from folder
  - at the terminal: `export PYTHONPATH=.`

- check if gpu is available
  - in python terminal: `torch.cuda.is_available()`
# Sandbox

- following:
  - `https://tl.thuml.ai/get_started/quickstart.html`