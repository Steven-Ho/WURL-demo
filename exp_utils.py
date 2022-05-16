import os
import pandas as pd
import datetime
import random
import numpy as np
import torch
import json

COLUMNS = ["ID", "Disc", "WD", "Reward Scale", "Tau", "Start Step", "Num Episodes", "Train Seed", "Train Time", "Test Seed", "Test Time"]
SEED_MIN = 10000000
SEED_MAX = 99999999


def save_exp_config(args, exp_time, exp_name):
    dir_path = "results/{}/{}".format(args.scenario, exp_name)

    # read data from csv
    try:
        df = pd.read_csv(os.path.join(dir_path, "run{}.csv".format(args.run)), index_col=0)
    except:
        df = pd.DataFrame(columns=COLUMNS)

    # add new data
    df.loc[df.shape[0]] = [args.run, None, None, args.reward_scale, args.tau, args.start_steps, args.num_episodes, args.seed, exp_time, None, None]

    # save
    assert_path(dir_path)
    df.to_csv(os.path.join(dir_path, "run{}.csv".format(args.run)))


def save_exp_result(disc_score, wd_score, args, exp_time, exp_name, idx=None):
    dir_path = "results/{}/{}".format(args.scenario, exp_name)
    df = pd.read_csv(os.path.join(dir_path, "run{}.csv".format(args.skill_run)), index_col=0)

    assert idx is not None
    df.loc[idx, "Disc"] = disc_score
    df.loc[idx, "WD"] = wd_score
    df.loc[idx, "Test Seed"] = args.seed
    df.loc[idx, "Test Time"] = exp_time

    df.to_csv(os.path.join(dir_path, "run{}.csv".format(args.skill_run)))


def assert_path(dir_path):
    dirs = dir_path.split("/")
    for i in range(len(dirs)):
        cur_path = os.path.join(*dirs[:i+1])
        if not os.path.exists(cur_path):
            os.makedirs(cur_path)


def save_args(json_config, args, save_path, key_set_list=None):
    args_dict = vars(args)
    
    if key_set_list is None:
        key_set_list = ["common", "env", args_dict["algo"], args.exp_name]         # these set of keys will be saved
        
    exp_config = {}
    for key_set in key_set_list:
        exp_config[key_set] = {}
        for key in json_config[key_set].keys():
            if key in args_dict.keys():
                exp_config[key_set][key] = args_dict[key]
            else:
                exp_config[key_set][key] = json_config[key_set][key]

    with open(os.path.join(save_path, "{}.json".format(args_dict["scenario"])), "w") as f:
        json.dump(exp_config, f, indent=4, separators=(',', ': '))


def get_start_time():
    start_time = datetime.datetime.now().strftime("%Y.%m.%d - %H:%M:%S")
    time_seed = int(datetime.datetime.now().strftime("%M%S%f"))
    random.seed(time_seed)
    exp_seed = random.randint(SEED_MIN, SEED_MAX)
    
    return start_time, exp_seed


def setup_seed(seed: int):
    # disable hash randomization
    os.environ['PYTHONHASHSEED'] = str(seed)

    # set seed for Python random module
    # random.seed(seed)

    # set seed for Numpy module
    np.random.seed(seed)

    # set seed for cpu
    torch.manual_seed(seed)
    # set seed for current gpu
    torch.cuda.manual_seed(seed)
    # set seed for all gpus
    torch.cuda.manual_seed_all(seed)
    # disable the optimization for convolution in cudnn
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
