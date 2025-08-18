set -x 
# export OPENBLAS_NUM_THREADS=4  # or a lower number like 2-8
# export OMP_NUM_THREADS=4
# Increase user process limits
ulimit -u unlimited  # or set to a high number like 65535
# export OPENBLAS_NUM_THREADS=64
# export GOTO_NUM_THREADS=64
# export OMP_NUM_THREADS=64
# For permanent change, add to /etc/security/limits.conf:
# * soft nproc 65535
export NUMEXPR_MAX_THREADS=128  # 设置为CPU核心数或更高
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
# * hard nproc 65535
# /root/autodl-tmp/qwen1.5-moe
ray job submit --address="http://127.0.0.1:8267" \
   --runtime-env-json='{"working_dir": "/root/MoE-Post-Tuning"}' \
   -- python3 -m openrlhf.cli.train_ppo_ray \
    --ref_num_nodes 1 \
    --ref_num_gpus_per_node 1 \
    --reward_num_nodes 0 \
    --reward_num_gpus_per_node 0 \
    --critic_num_nodes 1 \
    --critic_num_gpus_per_node 1 \
    --actor_num_nodes 1 \
    --actor_num_gpus_per_node 1 \
    --vllm_num_engines 1 \
    --vllm_tensor_parallel_size 1 \
    --pretrain /root/autodl-fs/models/LlamaMoEForCausalLM/Random-positive/llama3.2-1b-16Select4-gate_proj-HardBCE \
    --critic_pretrain /root/autodl-fs/llama3.2-1b \
    --remote_rm_url /root/MoE-Post-Tuning/openrlhf/trainer/ppo_utils/hard_reward_label.py \
    --eval_n_samples_per_prompt 1 \
    --ckpt_path /root/autodl-fs/save \
    --save_steps 10 \
    --micro_train_batch_size 1 \
    --train_batch_size 8 \
    --micro_rollout_batch_size 2 \
    --rollout_batch_size 64 \
    --max_epochs 1 \
    --num_episodes 10 \
    --label_key answer \
    --prompt_max_len 1024 \
    --generate_max_len 1024 \
    --zero_stage 2 \
    --bf16 \
    --eval_steps 1 \
    --actor_learning_rate 5e-7 \
    --critic_learning_rate 9e-6 \
    --init_kl_coef 0.01 \
    --prompt_data /root/MoE-Post-Tuning/data/train.json \
    --input_key input \
    --max_samples 10000000 \
    --packing_samples \
    --normalize_reward \
    --adam_offload \
    --vllm_sync_backend nccl \
    --gradient_checkpointing \
    --use_wandb b718e83ae2e0c39c7d64d7ffc96cae4c4a8253c6
    #     --apply_chat_template \
    # --flash_attn False\

      #   --eval_dataset /root/OpenRLHF/data/test.json \
