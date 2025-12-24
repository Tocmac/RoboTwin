# 1. Install
```
cd policy/ACT

pip install pyquaternion pyyaml rospkg pexpect mujoco==2.3.7 dm_control==1.0.14 opencv-python matplotlib einops packaging h5py ipython

cd detr && pip install -e . && cd ..
```

# 2. Collect Data
```
bash collect_data.sh ${task_name} ${task_config} ${gpu_id}
# Clean Data Example: bash collect_data.sh beat_block_hammer demo_clean 0
# Radomized Data Example: bash collect_data.sh beat_block_hammer demo_randomized 0

bash collect_data.sh blocks_ranking_rgb aloha_clean_500 0
```


# 2. Prepare Training Data
This step performs data preprocessing, converting the original RoboTwin 2.0 data into the format required for ACT training. The expert_data_num parameter specifies the number of trajectory pairs to be used as training data.

```
bash process_data.sh ${task_name} ${task_config} ${expert_data_num}
# bash process_data.sh beat_block_hammer demo_clean 50

bash process_data.sh blocks_ranking_rgb piper_randomized_500 50
bash process_data.sh blocks_ranking_rgb demo_clean 50

bash process_data.sh blocks_ranking_rgb aloha_clean_500 300
```

# 3. Train Policy
This step launches the training process. By default, the model is trained for 6,000 steps.

```
bash train.sh ${task_name} ${task_config} ${expert_data_num} ${seed} ${gpu_id}
# bash train.sh beat_block_hammer demo_clean 50 0 0

bash train.sh blocks_ranking_rgb piper_clean_50 50 0 0

bash train.sh blocks_ranking_rgb demo_clean 50 0 0 30000

bash train.sh blocks_ranking_rgb aloha_clean_500 200 0 1 30000
```

# 4. Eval Policy
The task_config field refers to the evaluation environment configuration, while the ckpt_setting field refers to the training data configuration used during policy learning.

```
bash eval.sh ${task_name} ${task_config} ${ckpt_setting} ${expert_data_num} ${seed} ${gpu_id}

# bash eval.sh beat_block_hammer demo_clean demo_clean 50 0 0
# This command trains the policy using the `demo_clean` setting ($ckpt_setting)
# and evaluates it using the same `demo_clean` setting ($task_config).
#
# To evaluate a policy trained on the `demo_clean` setting and tested on the `demo_randomized` setting, run:
# bash eval.sh beat_block_hammer demo_randomized demo_clean 50 0 0

bash eval.sh blocks_ranking_rgb piper_clean_50 piper_clean_50 50 0 0

bash eval.sh blocks_ranking_rgb demo_clean demo_clean 50 0 0 30000

bash eval.sh blocks_ranking_rgb demo_randomized demo_clean 50 0 0

bash eval.sh blocks_ranking_rgb aloha_clean_500 aloha_clean_500 200 0 0 20000

bash eval.sh blocks_ranking_rgb aloha_clean_500 aloha_clean_500 200 0 0 30000
```

The evaluation results, including videos, will be saved in the eval_result directory under the project root.