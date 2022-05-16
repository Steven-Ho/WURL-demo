#!/bin/bash
set -v
# -----------------------------------------------------------------
export CUDA_VISIBLE_DEVICES=0 &&            # set GPU_ID

# set RUNNING CONFIGS
SKILL_ID=0                                  # 0 for the pretrained model. Or u can change it to other skill_id
EXP_NAME=wurl                               # wurl, dads or diayn
BACKEND=apwd                                # apwd, dads or diayn
SCENARIO="AntCustom-v0"

MODEL_PREFIX="pretrained/"                   # "pretrained/" or "models/"
# MODEL_PREFIX="models/"
MODEL="wurl/apwd/"

# train skill
# python train.py --scenario $SCENARIO --exp_name $EXP_NAME --run $SKILL_ID &&

# hrl
for i in `seq 21 25`
do
    echo "--------------------------------------------------------------"
    echo "Its running run_id=="$i" of skill_id="$SKILL_ID" now!"
    python hrl.py --skill_run $SKILL_ID --backend $BACKEND --prefix $MODEL_PREFIX$SCENARIO"/"$MODEL"run"$SKILL_ID"/" --run $i &&
    python hrl-test.py --skill_run $SKILL_ID --backend $BACKEND --prefix $MODEL_PREFIX$SCENARIO"/"$MODEL"run"$SKILL_ID"/" --run $i
done
