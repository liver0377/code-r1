# Code-R1 开发规范文档

## 项目概述

本项目基于 verl 框架，实现了一个代码生成 Agent 的强化学习训练系统。通过 GRPO 算法训练 Qwen2.5-3B 模型，使其能够生成可执行的 Python 代码来解决用户问题。

---

## 文件修改清单

### 新增文件

| 文件路径 | 作用 |
|----------|------|
| `scripts/convert_data.py` | 数据转换脚本，将 JSONL 格式转换为 Parquet 格式 |
| `scripts/run_code_single_turn.sh` | 单轮训练启动脚本 |
| `scripts/run_code_multiturn.sh` | 多轮训练启动脚本 |
| `reward_functions/python_code_reward.py` | 自定义 Reward 函数，集成 DeepSeek API 和 Sandbox |
| `DEV_SPEC.md` | 本文档 |

### 修改文件

| 文件路径 | 修改内容 |
|----------|----------|
| `examples/sglang_multiturn/config/tool_config/sandbox_fusion_tool_config.yaml` | 更新 sandbox_fusion_url 为本地地址 `http://10.250.2.24:8090/run_code` |

---

## 项目架构

```
code-R1/
├── scripts/
│   ├── convert_data.py              # 数据转换
│   ├── run_code_single_turn.sh      # 单轮训练
│   └── run_code_multiturn.sh        # 多轮训练
├── reward_functions/
│   └── python_code_reward.py        # Reward计算
├── dataset/
│   ├── python-codes-25k.jsonl       # 原始数据
│   └── python-codes-5k.parquet      # 转换后数据
├── logs/                            # 训练日志
├── checkpoints/                     # 模型检查点
└── examples/sglang_multiturn/config/tool_config/
    └── sandbox_fusion_tool_config.yaml  # Sandbox配置
```

---

## 训练配置参数

### 核心参数

| 参数 | 单轮模式 | 多轮模式 | 说明 |
|------|----------|----------|------|
| `algorithm.adv_estimator` | grpo | grpo | GRPO算法 |
| `data.train_batch_size` | 64 | 64 | 训练批次大小 |
| `data.max_prompt_length` | 512 | 512 | 最大提示长度 |
| `data.max_response_length` | 1024 | 2048 | 最大响应长度 |
| `actor_rollout_ref.model.path` | Qwen/Qwen2.5-3B-Instruct | 同左 | 模型路径 |
| `actor_rollout_ref.model.lora_rank` | 32 | 32 | LoRA秩 |
| `actor_rollout_ref.rollout.name` | vllm | sglang | 推理引擎 |
| `actor_rollout_ref.rollout.n` | 4 | 4 | 每个prompt生成n个响应 |
| `actor_rollout_ref.rollout.multi_turn.enable` | False | True | 多轮模式 |
| `actor_rollout_ref.rollout.multi_turn.max_assistant_turns` | - | 3 | 最大轮次 |
| `trainer.n_gpus_per_node` | 1 | 1 | GPU数量 |
| `trainer.total_epochs` | 3 | 3 | 训练轮数 |

### 资源配置

| 资源 | 配置 |
|------|------|
| GPU | H100 80GB × 1 |
| 显存占用 | 单轮 ~35GB / 多轮 ~45GB |
| Sandbox服务 | http://10.250.2.24:8090/run_code |
| LLM API | DeepSeek API (10并发) |

---

## 训练流程

### Step 1: 环境准备

```bash
# 设置环境变量
export DEEPSEEK_API_KEY="your_deepseek_api_key"
export DEEPSEEK_BASE_URL="https://api.deepseek.com"
export SANDBOX_URL="http://10.250.2.24:8090/run_code"
export WANDB_API_KEY="your_wandb_api_key"  # 可选

# 确认Sandbox服务可用
curl -X POST http://10.250.2.24:8090/run_code \
  -H "Content-Type: application/json" \
  -d '{"code": "print(1+1)", "language": "python"}'
```

### Step 2: 数据转换

```bash
cd /path/to/code-R1
python scripts/convert_data.py \
  --input dataset/python-codes-25k.jsonl \
  --output dataset/python-codes-5k.parquet \
  --num_samples 5000 \
  --use_instruction_only
```

**输出数据格式：**
```python
{
  "prompt": "Help me set up my daily to-do list!",
  "data_source": "python_codes",
  "extra_info": {
    "user_message": "Help me set up my daily to-do list!",
    "ground_truth": "```python\n...\n```"
  }
}
```

### Step 3: 启动训练

**单轮模式：**
```bash
bash scripts/run_code_single_turn.sh
```

**多轮模式：**
```bash
bash scripts/run_code_multiturn.sh
```

### Step 4: 监控训练

- **WandB**: 查看实时训练曲线
- **日志文件**: `logs/code_single_turn_YYYYMMDD_HHMMSS/training.log`
- **检查点**: `checkpoints/code_single_turn/`

---

## Agentic RL 设计

### 整体架构

```
┌─────────────────────────────────────────────────────────────────┐
│                    H100 80GB (单卡训练)                          │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │                    verl 训练框架                          │    │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐      │    │
│  │  │ Actor Model │  │  Rollout    │  │ GRPO Update │      │    │
│  │  │ (3B + LoRA) │◄─┤ (vLLM/SGL)  │◄─┤ Algorithm   │      │    │
│  │  └─────────────┘  └─────────────┘  └─────────────┘      │    │
│  │         ▲                │                                 │    │
│  │         │                ▼                                 │    │
│  │  ┌─────────────────────────────────────────────────┐    │    │
│  │  │              Reward Manager (Prime)              │    │    │
│  │  │  ┌─────────────┐  ┌─────────────────────────┐   │    │    │
│  │  │  │   Sandbox   │  │  DeepSeek LLM Judge     │   │    │    │
│  │  │  │  代码执行    │  │  答案正确性判断          │   │    │    │
│  │  │  └─────────────┘  └─────────────────────────┘   │    │    │
│  │  └─────────────────────────────────────────────────┘    │    │
│  └─────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
```

### Agent 输出格式

模型被训练为输出特定格式的响应：

```
eless
[思考过程：分析问题，规划解决方案]
Done
<code>
[生成的Python代码]
</code>
<observation>
[代码执行结果]
</observation>
eless
[根据执行结果进行反思]
Done
<answer>
[最终答案]
</answer>
```

### Reward 计算逻辑

```python
def compute_score(data_source, solution_str, ground_truth, extra_info):
    # 1. 格式验证
    is_valid, _ = is_valid_sequence(solution_str)
    
    if is_valid:
        score = 0.5  # 格式正确基础分
        
        # 2. 代码执行
        stdout, stderr = code_result(solution_str)
        
        if 'error' in stderr.lower() or 'traceback' in stderr.lower():
            score -= 0.5  # 执行失败
        else:
            score += 0.5  # 执行成功
            
            # 3. LLM判断答案正确性
            user_message = extra_info['user_message']
            score += answer_reward(user_message, solution_str)
            # +1.0 (正确) 或 -1.0 (错误)
        
        return score
    else:
        # 格式不正确，返回部分格式分
        return format_score  # 0.0 ~ 0.26
```

**Score 范围：**
- 最高分：0.5 + 0.5 + 1.0 = **2.0** (格式正确 + 执行成功 + 答案正确)
- 最低分：**0.0** (格式完全错误)
- 格式正确但执行失败：**0.0**
- 格式正确 + 执行成功 + 答案错误：**1.0**

### 单轮 vs 多轮对比

| 特性 | 单轮模式 | 多轮模式 |
|------|----------|----------|
| 交互次数 | 1次 | 最多3次 |
| Tool调用 | 无 | Sandbox Tool |
| 输出复杂度 | 简单 | 可迭代修正 |
| 训练时间 | ~3小时 | ~5-6小时 |
| 适用场景 | 快速验证 | 生产级训练 |

### 多轮交互流程

```
用户问题
    │
    ▼
┌─────────────┐
│  Agent思考   │ ◄──┐
└─────────────┘    │
    │              │
    ▼              │
┌─────────────┐    │
│  生成代码    │    │
└─────────────┘    │
    │              │
    ▼              │
┌─────────────┐    │
│ Sandbox执行  │    │ 可迭代 (最多3轮)
└─────────────┘    │
    │              │
    ▼              │
┌─────────────┐    │
│ 观察结果     │ ───┘ (如需修正)
└─────────────┘
    │
    ▼
┌─────────────┐
│  输出答案    │
└─────────────┘
```

---

## 成本估算

### 训练成本 (5k数据, 3 epochs)

| 项目 | 单轮模式 | 多轮模式 |
|------|----------|----------|
| DeepSeek API | ~¥20-30 | ~¥30-50 |
| H100 80GB × 1 | ~¥75-100 (3h) | ~¥125-150 (5h) |
| **总计** | **~¥95-130** | **~¥155-200** |

### 时间成本

| 模式 | 预估时间 |
|------|----------|
| 单轮 | 2.5-3.5小时 |
| 多轮 | 4.5-7小时 |

---

## 常见问题

### Q1: Sandbox服务连接失败

```bash
# 检查服务是否可用
curl -X POST http://10.250.2.24:8090/run_code \
  -H "Content-Type: application/json" \
  -d '{"code": "print(1)", "language": "python"}'
```

### Q2: DeepSeek API 调用限流

- 默认配置为10并发
- 如遇限流，降低 `reward_model.num_workers`

### Q3: 显存不足

```bash
# 开启更多offload选项
actor_rollout_ref.actor.fsdp_config.param_offload=True
actor_rollout_ref.actor.fsdp_config.optimizer_offload=True
actor_rollout_ref.rollout.gpu_memory_utilization=0.3  # 降低
```

### Q4: 如何恢复训练

```bash
bash scripts/run_code_single_turn.sh \
  trainer.resume_mode=auto \
  trainer.resume_from_path=checkpoints/code_single_turn/last
```

---

## 后续优化方向

1. **Reward函数优化**
   - 添加代码质量评分（复杂度、可读性）
   - 引入测试用例覆盖度

2. **模型优化**
   - 尝试更大的基础模型（7B/14B）
   - 调整LoRA参数

3. **训练优化**
   - 增加训练数据量
   - 调整GRPO超参数

4. **部署优化**
   - 本地部署Reward模型替代API
   - 优化Sandbox执行效率
