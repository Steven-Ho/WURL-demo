# Wasserstein Unsupervised Reinforcement Learning
This repo is an example code repo for implementing WURL, including RL backends, URL baselines, WURL core files and simple environments used in paper.

## Requirements
* PyTorch >= 1.7.0
* Gym 0.18.0
* MuJoCo 1.50
* mujoco-py 1.50.1.0
* mujoco-maze 0.2.0
* python 3.7.5

## Installation
### Install the Dependencies with ANACONDA
[Anaconda3](https://www.anaconda.com/) is recommanded for managing the experiment environment.
```bash
conda env create -f environment.yaml
conda activate wurl
```
### Install MuJoCo 150 

Follow the [instructions](https://github.com/openai/mujoco-py/tree/1.50.1.0) to install MuJoCo version 1.50

### Compile mujoco_py==1.50.1.0 from source code
```bash
# get into one tmp directory and clone the source code
cd <tmp_mujoco_py_path>
git clone -b 1.50.1.0 git@github.com:openai/mujoco-py.git
cd mujoco-py
pip install -e .
```

### Install mujoco-maze==0.2.0
```bash
pip install mujoco-maze==0.2.0
```

### Install Pytorch >= 1.7.0
Get the proper version of [PyTorch](https://pytorch.org/get-started/locally/)


## Experiments in MuJoCo Environments
You can reproduce the results in MuJoCo environments("HalfCheetah-v3", "AntCustom-v0", "Humanoid-v3") by the scripts `train.py` & `test.py`.

All the configs can be modified in 'mujoco_configs/train/' & 'mujoco_configs/test' and via experiment arguments(please refer to `config.py` for details).

### Train New Policies
We provide the script `train.py` including the algorithms "APWD", "DIAYN-I", "DADS-I".

**Experiment Arguments**
* `scenario`:   The environment to run the experiment, defaults to `"AntCustom-v0"`. (`["HalfCheetah-v3", "AntCustom-v0", "Humanoid-v3"]`).
* `exp_name`:   The algorithm to train the diverse policies, defaults to `"wurl"`. Please note that all the algorithm are the *N policies* version which means the `"diayn"` equals to "DIAYN-I" and `"dads"` equals to "DADS-I".(`["wurl", "diayn", "dads"]`).
* `cuda`:       Run on GPU or not, defaults to `True`.
* `num_modes`:  The number of the policies/skills to train, defaults to `10`.
* `run`:        The run id to name the directory of these learned policies, and you should change this id of the individual run(like `--run 1`, etc.), defaults to `10086` which means the debug id.

**Examples**
```bash
# The 1st run to train APWD in HalfCheetah-v3 with 10 modes.
python train.py --scenario HalfCheetah-v3 --run 1

# The 1st run to train DIAYN-I in Humanoid-v3 with 10 modes.
python train.py --scenario Humanoid-v3 --exp_name diayn --run 1

# The 1st run to train DADS-I in AntCustom-v0 with 10 modes.
python train.py --scenario AntCustom-v0 --exp_name dads --run 1
```

The training details(some configs, training seed & training time) will be saved into "results/`<scenario>`/`<exp_name>`/run`<run>`.csv"

### Test Learned Policies
We provide the script `test.py` to calculate the metrics in our paper including *Success Rate of the Discriminator* & the *Wasserstein Distance*.

**Test Arguments**
* `scenario`:   The environment to run the experiment, defaults to `"AntCustom-v0"`. (`["HalfCheetah-v3", "AntCustom-v0", "Humanoid-v3"]`).
* `cuda`:       Run on GPU or not, defaults to `True`.
* `num_modes`:  The number of the modes of the policies/skills, defaults to `10`.
* `backend`:    The backend of the learned policies, defaults to `apwd`. (`["apwd", "diayn", "dads"]`)
* `prefix`:     model loading path prefix, defaults to `"models/AntCustom-v0/wurl/apwd/run"`. Please note the learned models from "DIAYN-I" and "DADS-I" have different path prefixes "models/AntCustom-v0/diayn/run" & "models/AntCustom-v0/dads/run". And all the pretrained models are in the folder "pretrained/".
* `skill_run`:   The experiment id of the learned policies which is exactly the `run` argument you used for the `train.py`, defaults to `10086` which means the debug id. Please change it to the run id you used to train before.

**Examples**
```bash
# Test the 1st run learned by APWD in HalfCheetah-v3.
python test.py --scenario HalfCheetah-v3 --backend apwd --prefix "models/HalfCheetah-v3/wurl/apwd/run"  --skill_run 1

# Test the 1st run learned by DIAYN-I in Humanoid-v3.
python test.py --scenario Humanoid-v3 --backend diayn --prefix "models/Humanoid-v3/diayn/run" --skill_run 1

# Test the 1st run learned by DADS-I in AntCustom-v0.
python test.py --scenario AntCustom-v0 --backend dads --prefix "models/AntCustom-v0/dads/run" --skill_run 1
```
The test results('Disc' means *Success Rate of the Discriminator* & 'WD' means *Wasserstein Distance*) will also be saved in the same csv file.

### Hierarchical Reinforcement Learning
We provide the script `hrl.py` & `hrl-test.py` to run the HRL experiments.

**Experiment Arguments**
* `scenario`:       The environment, defaults to `"AntCustom-v0"`.
* `subscenario`:    The sub environment, defaults to `"AntCustom-v0"`.
* `cuda`:           Run on GPU or not, defaults to `True`.
* `backend`:        The backend of the learned policies, defaults to `apwd`. (`["apwd", "diayn", "dads"]`)
* `prefix`:         model loading path prefix, defaults to `"models/AntCustom-v0/wurl/apwd/run1/"`. Different from the argument `prefix` for `test.py` and here the skill_id should be included in the prefix.
* `skill_run`:      The experiment id of the learned policies which is exactly the `run` argument you used for the `train.py`, defaults to `10086` which means the debug id. Please change it to the run id you used to train before.
* `run`:            The run id of the HRL experiment, defaults to `10086` which means the debug id. Please note this argument is different from `skill_run`.

**Examples**
```bash
# 1st HRL experiment with the 1st run policies learned by APWD in AntCustom-v0.
python hrl.py --skill_run 1 --backend apwd --prefix "models/AntCustom-v0/wurl/apwd/run1/" --run 1
python hrl-test.py --skill_run 1 --backend apwd --prefix "models/AntCustom-v0/wurl/apwd/run1/" --run 1

# 2nd HRL experiment with the 1st run policies learned by APWD in AntCustom-v0.
python hrl.py --skill_run 1 --backend apwd --prefix "models/AntCustom-v0/wurl/apwd/run1/" --run 2
python hrl-test.py --skill_run 1 --backend apwd --prefix "models/AntCustom-v0/wurl/apwd/run1/" --run 2

# 1st HRL experiment with the 1st run policies learned by DIAYN-I in AntCustom-v0.
python hrl.py --skill_run 1 --backend diayn --prefix "models/AntCustom-v0/diayn/run1/" --run 1
python hrl-test.py --skill_run 1 --backend diayn --prefix "models/AntCustom-v0/diayn/run1/" --run 1

# 1st HRL experiment with the 1st run policies learned by DADS-I in AntCustom-v0.
python hrl.py --skill_run 1 --backend dads --prefix "models/AntCustom-v0/dads/run1/" --run 1
python hrl-test.py --skill_run 1 --backend dads --prefix "models/AntCustom-v0/dads/run1/" --run 1
```

The results will be saved into "results/hrl/AntCustom-v0/`<backend>`/skill`<skill_run>`-run`<run>`.csv"

# OLD Version
The code to run the experiment in mujoco-maze is in the backup folder. You must move the scripts to the root path of this project and then to run the code.
## Code brief
`wasserstein.py` - framework instantiation, train 2 policies at the same time, support all primal and all dual (test functions) methods.

`wasserstein-multi.py` - train N policies at the same time, support all primal methods (amortized or not).

`wasserstein-sequential.py` - train N policies in sequence, support all primal methods.

`diayn.py` - train baseline DIAYN in N skills.

`dads.py` - train baseline DADS in N skills.

`test-mujoco.py` - test and render each policy of mujoco tasks.

`test-render-freerun.py` - test and draw all trajectories in FreeRun.

`test-render-treemaze.py` - test and draw all trajectories in TreeMaze.

## Examples

`python diayn.py --num_episodes 5000 --num_modes 10 --scenario "FreeRun-v0"` train 10 skills at the same time in FreeRun with DIAYN.

`python wasserstein-multi.py --reward_scale 10 --sr_algo "apwd"` train with amortized PWD.

`python wasserstein-multi.py --reward_scale 1000 --sr_algo "pwd"` train with PWD.

`python wasserstein-sequential.py --reward_scale 10 --sr_algo "apwd"` train with amortized PWD one policy by one policy.

`python test.py --scenario "FreeRun-v0" --num_modes 10 --backend "diayn" --prefix "pretrained/FreeRun-v0/n=10/"` test mean discriminability and Wasserstein distance from pretrained models. 

`python test-mujoco.py --scenario "HalfCheetah" --num_modes 12 --backend "apwd" --prefix "pretrained/HalfCheetah-v3/run1/"` render all policies in specific MuJoCo tasks.

