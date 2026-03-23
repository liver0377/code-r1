#!/bin/bash
# 单轮代码生成RL训练脚本
# 使用Qwen2.5-3B + GRPO + DeepSeek API

set -x

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

export CUDA_VISIBLE_DEVICES=0

export DEEPSEEK_API_KEY="${DEEPSEEK_API_KEY:-your_deepseek_api_key_here}"
export DEEPSEEK_BASE_URL="${DEEPSEEK_BASE_URL:-https://api.deepseek.com}"
export SANDBOX_URL="${SANDBOX_URL:-http://10.250.2.24:8090/run_code}"

TRAIN_DATA="$PROJECT_DIR/dataset/python-codes-5k.parquet"
VAL_DATA="$PROJECT_DIR/dataset/python-codes-5k.parquet"

MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2.5-3B-Instruct}"

NOW=$(date +%Y%m%d_%H%M%S)
export WANDB_DIR="$PROJECT_DIR/logs/code_single_turn_${NOW}"
export WANDB_PROJECT="code_r1_single_turn"
export WANDB_EXP="qwen2.5-3b-${NOW}"

mkdir -p "$WANDB_DIR"

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files="$TRAIN_DATA" \
    data.val_files="$VAL_DATA" \
    data.train_batch_size=64 \
    data.val_batch_size=32 \
    data.max_prompt_length=512 \
    data.max_response_length=1024 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.shuffle=True \
    actor_rollout_ref.model.path="$MODEL_PATH" \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.lora_rank=32 \
    actor_rollout_ref.model.lora_alpha=32 \
    actor_rollout_ref.model.target_modules=all-linear \
    actor_rollout_ref.actor.optim.lr=1e-5 \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.1 \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0.001 \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.actor.fsdp_config.fsdp_size=-1 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
    actor_rollout_ref.rollout.n=4 \
    actor_rollout_ref.rollout.max_model_len=2048 \
    actor_rollout_ref.rollout.max_num_batched_tokens=2048 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.load_format=safetensors \
    actor_rollout_ref.rollout.layered_summon=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.critic_warmup=0 \
    trainer.val_before_train=True \
    trainer.logger='[console, wandb]' \
    trainer.project_name="$WANDB_PROJECT" \
    trainer.experiment_name="$WANDB_EXP" \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    trainer.save_freq=500 \
    trainer.test_freq=100 \
    trainer.total_epochs=3 \
    trainer.default_local_dir="$PROJECT_DIR/checkpoints/code_single_turn" \
    custom_reward_function.path="$PROJECT_DIR/reward_functions/python_code_reward.py" \
    custom_reward_function.name=compute_score \
    reward_model.reward_manager=prime \
    $@ 2>&1 | tee "$WANDB_DIR/training.log"

echo "训练完成! 日志目录: $WANDB_DIR"
