# StepSearch 源码深度导读（面向简历与算法实习面试）

> 目标：你已经跑通训练，本教程帮助你“讲明白项目”、能独立复现同类系统，并应对面试追问。

---

## 1. 先给你一个“电梯介绍”（可直接写简历）

StepSearch 是一个“**带检索工具调用的多跳问答 RL 训练框架**”：
- 用 **PPO** 训练 LLM，不只看最终答案，还对每一步 `<search>... </search>` + `<information>... </information>` 给 token-level step reward。
- 训练时由模型自己决定何时搜索、搜什么词；外部检索服务返回文档后再继续推理，直到 `<answer>`。
- 奖励由三部分组成：**答案正确性**、**搜索关键词匹配度**、**逐步信息增益/冗余惩罚**。

一句话简历版：

> 基于 Ray+veRL 搭建多 GPU Step-wise PPO 训练管线，联动本地检索服务实现“思考-搜索-观察-回答”闭环，通过信息增益与冗余惩罚的 token 级奖励优化多跳检索推理能力。

---

## 2. 代码库分层架构（你要能画出来）

你可以把项目分为 5 层：

1. **数据层（离线预处理）**  
   把 MuSiQue 原始样本改造成 RL 训练格式（prompt + ground_truth + support_docs）。
2. **检索层（在线工具）**  
   FastAPI 检索服务 + FAISS/BM25，接受 query，返回 top-k 文档。
3. **交互生成层（Agent Loop）**  
   LLM 多轮生成：plan/search -> retrieval observation -> plan... -> answer。
4. **奖励层（Rule-based RM）**  
   根据生成轨迹计算 final reward + step rewards。
5. **训练调度层（PPO + Ray）**  
   actor/critic/ref 多 worker 协同 rollout、估计优势、反向更新。

---

## 3. 训练入口与关键配置：先理解 `train.sh`

`train.sh` 是完整实验配置样例：
- 指定 base model、数据 parquet、batch size、最大长度。
- 指定 rollout 后端（vLLM）、并行参数、学习率等。
- 打开 StepSearch 的关键开关：
  - `trainer.answer_check_method=step`
  - `trainer.redundancy_penalty=true`
  - `trainer.information_gain=true`
  - `trainer.search_steps_reward=true`
  - `trainer.search_key_reward=true`
- 检索接口来自 `retriever.url`。

你面试时可说：**这个项目不是单纯改 loss，而是通过配置驱动把 tool-use RL 实验串起来。**

---

## 4. 数据是怎么喂给 PPO 的：`scripts/data_process/musi_search.py`

这份脚本做了三件关键事：

1. **构造系统提示模板**（要求模型按 `<plan>/<search>/<observation>/<answer>` 格式输出）。
2. **包装 ground truth**：
   - `target`: 答案及别名
   - `search_keys`: 子问题关键词（训练时用于 query reward）
3. **包装 support_docs**：
   - 每个样本附带参考证据文档（用于信息增益 reward）

最终导出 parquet 给 PPO 训练。

你要理解：**StepSearch 的 step reward 成立依赖数据里有可对齐的 support_docs 与 search_keys。**

---

## 5. 检索服务细节：`search_r1/search/retrieval_server.py`

它提供可插拔检索：
- **DenseRetriever**：FAISS + encoder embedding（默认可用 e5）
- **BM25Retriever**：Pyserini Lucene

核心流程：
1. query 编码（对 e5/bge 有特定前缀模板）；
2. 索引搜索 top-k；
3. 返回带 title/text 的文档列表。

工程点：
- 支持 `faiss_gpu` 把索引搬到 GPU shard。
- 检索与训练解耦（HTTP 调用），方便后续替换 Google/Serper 等在线搜索。

---

## 6. 多轮“思考-搜索-观察”循环：`search_r1/llm_agent/generation.py`

这是项目最像 Agent 的地方。

`LLMGenerationManager` 负责：
- 维护 rolling context（历史 prompt/response/information）；
- 每轮让 actor 生成；
- 截断到 `</search>` 或 `</answer>`；
- 对 `<search>` 调外部检索并拼接 `<information>`；
- 更新 active mask，直到到达 `max_turns` 或样本结束。

面试高频问法：“为什么要自己写 loop 不直接一次性生成？”
你可答：
- 多跳检索本质是**交互式决策过程**；
- 每一轮 observation 会改变下一轮 query 分布；
- RL 才能优化“查什么、何时停”的策略。

---

## 7. 奖励函数是项目灵魂：`verl/trainer/main_ppo.py` + `verl/utils/reward_score/qa_step.py`

### 7.1 RewardManager（主入口）

`RewardManager.__call__` 对每条样本：
- 解码完整序列；
- 调 `qa_step.compute_score_f1_steps_plan_with_support_docs` 算分；
- 将最终分写到最后 token；
- 若有 step_scores，把每步得分写回对应 `</search>...</information>` 结束 token 位置。

这意味着：**PPO 学到的是 dense-ish token reward，而非只在最后一位 sparse reward。**

### 7.2 qa_step 打分结构

`compute_score_f1_steps_plan_with_support_docs` 的关键逻辑：
1. 抽取 `<answer>`，算和 `target` 的 F1（answer_correct）；
2. 检查 `<answer>` 是否在结尾（格式非法直接负分）；
3. 提取每步 `<information>`，与 `support_docs` 做相似度，得信息增益；
4. 对重复信息做 redundancy penalty；
5. 提取 `<search>` 序列，与 `search_keys` 算匹配分；
6. 按配置把上述项加权成 final score + step_scores。

可面试表达：
- 这是把“过程监督”显式写进 reward，降低只追最终答案造成的投机行为。

---

## 8. PPO 训练主干：`verl/trainer/ppo/ray_trainer.py`

你重点记这条链：
1. rollout 采样得到 response + (optional) info mask；
2. reward_fn 产出 `token_level_scores`；
3. `apply_kl_penalty` 加参考模型 KL 约束 -> `token_level_rewards`；
4. `compute_advantage`（GAE/GRPO）算优势；
5. actor/critic 按 mini-batch/micro-batch 更新；
6. 记录 answer_correct / search_key_score / step_scores 等指标。

项目定制点：
- 指标不仅有 reward/kl，还有 retrieval task 定制统计，便于判断是“答案提升”还是“搜索行为提升”。

---

## 9. 你可以“独立复刻”同类项目的最小蓝图

如果你要自己写一个 mini 版（面试很加分），建议 4 周路线：

1. **Week1：单轮 tool-use SFT**
   - 固定模板 `<search>/<information>/<answer>`，先训可执行格式。
2. **Week2：接检索服务**
   - 单独 FastAPI + FAISS，训练只做 inference loop。
3. **Week3：接 PPO**
   - 先只做 final answer reward。
4. **Week4：加 step reward**
   - 加信息增益+冗余惩罚，再做 ablation（去掉某项看曲线变化）。

---

## 10. 简历写法（可直接粘贴后改数字）

- 设计并实现基于 Ray + veRL 的多 GPU PPO 训练框架，完成 Qwen-3B 在多跳检索问答任务的策略优化。  
- 构建“计划-检索-观察-回答”交互式生成闭环，联动 FAISS 检索服务实现在线工具调用。  
- 实现 token 级 step reward（信息增益、冗余惩罚、搜索关键词匹配）并集成 KL 正则，提升检索轨迹质量与最终答案正确率。  
- 完成训练/验证全流程复现、指标分析与消融实验，沉淀可复用的 RLHF for Tool-Use 实验模板。

---

## 11. 面试官高频拷问与回答模板

1. **Q：为什么 final reward 不够？**  
   A：多跳检索是长链决策，只有终点奖励太稀疏，credit assignment 难；step reward 提供中间信号，能更快学会有效 query。

2. **Q：信息增益怎么定义？**  
   A：每一步将检索到的信息与标注 support docs 做匹配，计算相对上一轮的新覆盖增量；重复信息会被 penalty 抵消。

3. **Q：为什么要 KL 到参考模型？**  
   A：防止策略漂移导致语言退化或 reward hacking，维持可读性与稳定训练。

4. **Q：这个系统最容易崩在哪里？**  
   A：格式约束失效（标签不闭合）、检索噪声累积、长上下文截断导致 early evidence 丢失。

5. **Q：你做过哪些工程优化？**  
   A：多 GPU batch 对齐、rollout padding trim、检索服务解耦、定制化 metric logging。

---

## 12. 你接下来该怎么“深入源码”

建议按下面顺序读，每读完写 5 行总结：

1. `train.sh`（知道实验怎么配）
2. `scripts/data_process/musi_search.py`（知道数据和 reward 监督从哪里来）
3. `search_r1/llm_agent/generation.py`（知道 agent loop）
4. `verl/utils/reward_score/qa_step.py`（知道打分）
5. `verl/trainer/main_ppo.py`（知道 reward 如何挂到 trainer）
6. `verl/trainer/ppo/ray_trainer.py`（知道 PPO 主流程）
7. `search_r1/search/retrieval_server.py`（知道检索系统）

如果你愿意，我下一步可以继续给你一版：
- “**逐函数级源码走读清单**”（每个函数：输入/输出/张量形状/坑点/可问面试题）
- 并附上“**一页纸面试速记版**”。

---

## 13. 逐函数级源码走读清单（输入/输出/张量形状/坑点/面试题）

> 使用方式：按顺序过一遍，每个函数自己补一份“最小样例输入”。

### A. 训练入口与奖励挂载（`verl/trainer/main_ppo.py`）

#### 1) `_select_rm_score_fn(data_source)`
- **输入**：`data_source: str`
- **输出**：打分函数对象（当前固定返回 `qa_step.compute_score_f1_steps_plan_with_support_docs`）
- **形状**：无张量
- **坑点**：当前没有按数据源分发，后续扩展多数据集时容易漏改。
- **可问面试题**：如果接入数学题/代码题，你会如何做 reward router？

#### 2) `_extract_qwen_solution(sequences_str)`
- **输入**：`str`（完整 decode 文本）
- **输出**：`str`（assistant 之后的内容）
- **形状**：无张量
- **坑点**：强依赖 Qwen chat 模板分隔符 `<|im_start|>assistant`。
- **可问面试题**：不同模型模板不一致时怎么做通用解析？

#### 3) `RewardManager.__call__(data: DataProto)`
- **输入**：`DataProto`
  - `batch['prompts']`: `[B, P]`
  - `batch['responses']`: `[B, R]`
  - `batch['attention_mask']`: `[B, P+R]`
- **输出**：
  - `reward_tensor`: `[B, R]`（token 级 reward）
  - `answer_correct`: `List[float|int], len=B`
  - `search_key_score`: `List[float], len=B`
  - `step_scores`: `List[float], len=B`（每条样本的步进分汇总）
- **关键逻辑**：末 token 放 final score；step 分写到 step 对齐 token。
- **坑点**：
  - `valid_response_length-1` 写分前要保证响应非空；
  - 文本解析失败会导致 step reward 丢失；
  - tokenizer encode/decode 不可逆时 token 位置可能偏移。
- **可问面试题**：为什么把 final reward 放最后一个有效 token 而不是均匀分配？

#### 4) `RewardManager._compute_step_reward_tensor(...)`
- **输入**：
  - `reward_tensor`: `[R]`
  - `valid_response_ids`: `[Rv]`
  - `step_scores: List[float]`
- **输出**：更新后的 `reward_tensor: [R]`
- **形状细节**：通过正则定位 `</search>... </information>` 结束字符，再转 token 位置。
- **坑点**：
  - 正则和模板强绑定；
  - `search_end_positions[i]-1` 可能越界（极短响应场景）；
  - step 数量与匹配片段数量可能不一致。
- **可问面试题**：如果 step 奖励错位，会如何定位（文本级 vs token级）？

### B. PPO 主干（`verl/trainer/ppo/ray_trainer.py`）

#### 5) `apply_kl_penalty(data, kl_ctrl, kl_penalty='kl')`
- **输入**：
  - `old_log_probs`: `[B, R]`
  - `ref_log_prob`: `[B, R]`（可选）
  - `token_level_scores`: `[B, R]`
  - `attention_mask` 或 `info_mask`: `[B, P+R]`
- **输出**：
  - `data.batch['token_level_rewards']`: `[B, R]`
  - `metrics`: `{'critic/kl', 'critic/kl_coeff'}`
- **坑点**：
  - `info_mask` 分支会改变哪些 token 参与 KL；
  - ref 缺失时 beta=0，训练可能漂移更快。
- **可问面试题**：Adaptive KL 控制器如何影响探索-稳定性平衡？

#### 6) `compute_advantage(data, adv_estimator, gamma, lam, num_repeat)`
- **输入**：`token_level_rewards [B,R]`, `values [B,R]`, `response_mask [B,R]`
- **输出**：`advantages [B,R]`, `returns [B,R]`
- **坑点**：
  - GAE 与 GRPO 逻辑不同；
  - mask 错会把 padding 位置也纳入优势估计。
- **可问面试题**：为什么长链 tool-use 任务通常更依赖好的 advantage 估计？

#### 7) `_compute_response_info(batch)`
- **输入**：`responses [B,Rmax]`, `attention_mask [B,P+Rmax]`
- **输出**：`response_mask [B,Rmax]`, `prompt_length [B]`, `response_length [B]`
- **坑点**：变长序列统计错误会污染所有监控指标。
- **可问面试题**：你如何验证 response length 指标可信？

#### 8) `compute_data_metrics(batch, use_critic=True)`
- **输入**：scores/rewards/advantages/returns/values
- **输出**：多项标量指标（mean/max/min）
- **坑点**：`sequence_answer_correct` 等是 python list，不是 tensor；需注意跨卡聚合一致性。
- **可问面试题**：为什么要单独记录 search_key_score 与 step_scores？

### C. Agent 交互循环（`search_r1/llm_agent/generation.py`）

#### 9) `LLMGenerationManager._postprocess_responses(responses)`
- **输入**：`responses: Tensor [B, Rgen]`
- **输出**：
  - `responses_ids: Tensor [B, Rtrim]`
  - `responses_str: List[str]`
- **逻辑**：截到 `</search>` 或 `</answer>`，防止模型一次吐太多无效内容。
- **坑点**：截断策略若过激，会损伤完整推理链。
- **可问面试题**：为何在 rollout 阶段强约束格式比后处理更重要？

#### 10) `_process_next_obs(next_obs)`
- **输入**：`List[str]` 检索观察文本
- **输出**：`next_obs_ids: Tensor [B, O]`（`O<=max_obs_length`）
- **坑点**：obs 截断过短会丢关键证据；过长会挤占上下文 budget。
- **可问面试题**：max_obs_length 你会如何调参？

#### 11) `_update_rolling_state(rollings, cur_responses, next_obs_ids)`
- **输入**：
  - `rollings.input_ids [B, Lold]`
  - `cur_responses [B, Rcur]`
  - `next_obs_ids [B, O]`
- **输出**：新 `DataProto`，字段维度约 `[B, Lnew<=max_prompt_length]`
- **坑点**：位置编码与 attention mask 必须同步更新；否则模型行为异常但不一定报错。
- **可问面试题**：为什么要 cut 到 effective len 再裁到 max_prompt_length？

#### 12) `_info_masked_concatenate_with_padding(...)`
- **输入**：prompt/response/info 多段 tensor
- **输出**：
  - `padded_tensor`
  - `padded_tensor_with_info`（information 区域置 pad id 掩码）
- **形状**：均为 `[B, Lcat]`
- **坑点**：左/右 pad 语义混用会导致 token 对齐错误。
- **可问面试题**：为什么需要 `responses_with_info_mask`？

#### 13) `_generate_with_gpu_padding(active_batch)`
- **输入**：`active_batch.batch['input_ids'] [B, L]`
- **输出**：去掉补齐样本后的生成结果
- **逻辑**：若 `B % num_gpus != 0`，复制首样本补齐，生成后再裁掉。
- **坑点**：meta_info 也要同步裁剪；不然统计错位。
- **可问面试题**：这种 padding 策略的副作用是什么？

#### 14) `run_llm_loop(gen_batch, initial_input_ids)`
- **输入**：初始 batch
- **输出**：多轮后的 left/right side 轨迹、统计信息
- **关键状态**：`active_mask`, `turns_stats`, `valid_search_stats`
- **坑点**：
  - done 样本与 active 样本切换容易索引错位；
  - max_turns 到达时末轮行为要定义清楚（是否强制 answer）。
- **可问面试题**：如何证明你的多轮 loop 没有 label leakage？

### D. 奖励细则（`verl/utils/reward_score/qa_step.py`）

#### 15) `extract_solution(solution_str)`
- **输入**：模型输出字符串
- **输出**：最后一个 `<answer>...</answer>` 的内容或 `None`
- **坑点**：多 answer 标签时取最后一个，可能掩盖前面错误策略。
- **可问面试题**：为何选择“最后答案”而不是“第一个答案”？

#### 16) `query_f1_score(prediction, golden_answers)`
- **输入**：预测文本 + 参考答案列表
- **输出**：`[0,1]` 的 F1（四舍五入）
- **坑点**：normalize 会去标点和冠词，某些任务可能过宽松。
- **可问面试题**：EM 与 F1 在多别名场景的取舍？

#### 17) `step_information_gain(informations, golden_info)`
- **输入**：
  - `informations: List[List[str]]`（每步若干文段）
  - `golden_info: List[dict]`
- **输出**：
  - `information_gains: List[float]`（每一步）
  - `redundancy_penalty: List[float]`
- **坑点**：依赖 tf-idf cosine 子模块，短文本/噪声文段波动大。
- **可问面试题**：为什么不用 cross-encoder 做更强语义匹配？

#### 18) `step_search_keys_match(searches, keys)`
- **输入**：搜索词列表、关键词组列表
- **输出**：`[0,1]` 匹配得分
- **坑点**：best-match 分配不是匈牙利算法，可能重复匹配同组。
- **可问面试题**：你会如何改成全局最优匹配？

#### 19) `compute_score_f1_steps_plan_with_support_docs(...)`
- **输入**：config、solution_str、ground_truth、support_docs
- **输出**：dict，至少包含
  - `score`（最终奖励）
  - `answer_correct`
  - `step_scores`
  - `search_key_score`
- **坑点**：
  - 格式失败直接 `score=-1`，训练早期可能过多负样本；
  - 多 reward 项开关组合多，实验要做 ablation matrix。
- **可问面试题**：如何验证 step reward 真正提升了“搜索质量”而非只改变输出长度？

### E. 检索侧（`search_r1/search/retrieval_server.py`）

#### 20) `Encoder.encode(query_list, is_query=True)`
- **输入**：`List[str] | str`
- **输出**：`np.ndarray [B, D]` float32
- **坑点**：
  - e5/bge 需要正确 prompt 前缀；
  - GPU 显存与 batch size 强相关。
- **可问面试题**：为什么 dense retriever 要做向量归一化？

#### 21) `DenseRetriever._search(query, num, return_score)`
- **输入**：query 文本
- **输出**：文档列表（可含 score）
- **形状**：faiss 返回 `scores [1,k], idxs [1,k]`
- **坑点**：索引与语料版本不一致会“查得到但全是错文档”。
- **可问面试题**：如何做线上检索质量回归测试？

#### 22) `BM25Retriever._search(...)`
- **输入**：query
- **输出**：top-k 文档（可含 score）
- **坑点**：索引是否包含原文内容会决定读取路径（Lucene raw / corpus 文件）。
- **可问面试题**：BM25 与 dense 在多跳问答中各自优势？

---

## 14. 一页纸面试速记版（可打印）

### 14.1 30 秒项目概述
- StepSearch = Tool-Use RL for Multi-hop QA。  
- 用 PPO 学策略：何时搜索、搜什么、何时回答。  
- 奖励 = 最终答案 F1 + 搜索关键词匹配 + 每步信息增益 - 冗余惩罚。  
- 训练框架：Ray + veRL，多 worker 协同 actor/critic/ref + vLLM rollout。

### 14.2 关键技术亮点（背这 4 条）
1. **Step-wise token reward 对齐**：step 分写回对应 action token，缓解 sparse reward。  
2. **多轮 Agent Loop**：plan/search/information/observation 闭环，支持交互式决策。  
3. **KL 正则稳定 PPO**：约束策略漂移，减少 reward hacking。  
4. **检索训练解耦**：HTTP retriever 可替换 FAISS/BM25/在线搜索。

### 14.3 你做了什么（实习面试口吻）
- 我复现并理解了 StepSearch 训练链路，能定位从数据构造到 reward 回传的每个节点。  
- 我重点分析了 `run_llm_loop` 与 `RewardManager` 的 token 对齐机制，明确了 step reward 的落点。  
- 我能独立解释并改造奖励项（信息增益、冗余惩罚、query key 匹配）并设计 ablation。

### 14.4 面试常见追问（快答）
- **为何 step reward 有用？**
  - 给中间决策密集反馈，改善 credit assignment。
- **如何防止模型乱搜？**
  - search_key reward + 信息增益导向 + 冗余惩罚 + KL 稳定。
- **如何判断提升来自“会搜了”而非“更会编”？**
  - 看 search_key_score、step_scores、retrieval 命中率与答案 F1 的协同变化。
- **最大工程难点？**
  - 多轮变长序列的 mask 对齐和跨 GPU batch 对齐。

### 14.5 你可以主动反问面试官
1. 你们线上 tool-use 任务更偏检索增强还是执行器增强？
2. 奖励建模里更关注结果对齐还是过程对齐？
3. 训练系统目前瓶颈在 rollout、reward 还是并行策略？

### 14.6 最后 10 秒收尾
> 这个项目让我具备了从“可复现”到“可解释、可改造”的能力：我不仅能跑 PPO+检索训练，还能针对 reward 设计、轨迹质量和系统稳定性做有依据的优化。
