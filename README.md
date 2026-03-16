## This repository is the official implementation of the paper "[ROOT: Rethinking Offline Optimization as Distributional Translation via Probabilistic Bridge](https://arxiv.org/abs/2509.16300)" (NeurIPS 2025 Spotlight)

![Overview of ROOT](overview.png)


## Abstract
This paper studies the black-box optimization task which aims to find the maxima
of a black-box function using a static set of its observed input-output pairs. This is
often achieved via learning and optimizing a surrogate function with that offline
data. Alternatively, it can also be framed as an inverse modeling task that maps a
desired performance to potential input candidates that achieve it. Both approaches
are constrained by the limited amount of offline data. To mitigate this limitation,
we introduce a new perspective that casts offline optimization as a distributional
translation task. This is formulated as learning a probabilistic bridge transforming
an implicit distribution of low-value inputs (i.e., offline data) into another distribution of high-value inputs (i.e., solution candidates). Such probabilistic bridge can
be learned using low- and high-value inputs sampled from synthetic functions that
resemble the target function. These synthetic functions are constructed as the mean
posterior of multiple Gaussian processes fitted with different parameterizations on
the offline data, alleviating the data bottleneck. The proposed approach is evaluated on an extensive benchmark comprising most recent methods, demonstrating
significant improvement and establishing a new state-of-the-art performance.

## Requirements
To set up the environment, let's follow the below instruction:
```commandline
conda env create -f environment.yml
conda activate ROOT

# install pytorch
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch

# Mujoco Installation
wget https://www.roboti.us/download/mujoco200_linux.zip -O mujoco200_linux.zip
mkdir -p ~/.mujoco
unzip mujoco200_linux.zip -d ~/.mujoco
mv ~/.mujoco/mujoco200_linux ~/.mujoco/mujoco200
rm mujoco200_linux.zip

wget https://www.roboti.us/file/mjkey.txt -O ~/.mujoco/mjkey.txt

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/.mujoco/mujoco200/bin

# follow https://github.com/openai/mujoco-py/issues/627 to install mujoco-py without root privileges 
conda install -c conda-forge glew
conda install -c conda-forge mesalib
conda install -c menpo glfw3
export CPATH=$CONDA_PREFIX/include
python3 -m pip install patchelf
python3 -m pip install Cython==0.29.36 numpy==1.21.5 mujoco_py==2.0.2.3

# Read the comment https://github.com/openai/mujoco-py/issues/627#issuecomment-1094396677 to solve the IGL issue or -IOSMesa issue

# Design-Bench Installation
python3 -m pip install design-bench==2.0.12
python3 -m pip install robel==0.1.2 morphing_agents==1.5.1 transforms3d --no-dependencies
python3 -m pip install botorch==0.6.4 gpytorch==1.6.0
python3 -m pip install gym==0.12.5

# Download Design-Bench Offline Datasets: 
python3 -m pip install gdown
python3 -m pip uninstall charset-normalizer
python3 -m pip install charset-normalizer
# Dataset can be downloaded from https://huggingface.co/datasets/beckhamc/design_bench_data or our Google Drive as below:
gdown 'https://drive.google.com/uc?id=1n5R0p_7OAejDts6B_WH6qbBRfT8BEiiN'
unzip design_bench_data.zip
rm -rf design_bench_data.zip
mv -v design_bench_data $CONDA_PREFIX/lib/python3.9/site-packages
python3 -m pip install tensorflow==2.11.0
python3 -m pip install wandb
# python3 -m pip uninstall numpy
python3 -m pip install numpy==1.22.4
python3 -m pip install omegaconf
python3 -m pip install einops
export PYTHONPATH=$CONDA_PREFIX/lib/python3.9/site-packages/:$PYTHONPATH
```
## Experiments

To reproduce our results, run the following command: 
```
bash scripts/bash.sh
```

## Acknowledgement
Our code is implemented based on Brownian Bridge Diffusion Models 

[Brownian Bridge Diffusion Models](https://github.com/xuekt98/BBDM)  

## Citation
```
@misc{dao2025rootrethinkingofflineoptimization,
      title={ROOT: Rethinking Offline Optimization as Distributional Translation via Probabilistic Bridge}, 
      author={Manh Cuong Dao and The Hung Tran and Phi Le Nguyen and Thao Nguyen Truong and Trong Nghia Hoang},
      year={2025},
      eprint={2509.16300},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2509.16300}, 
}
```
