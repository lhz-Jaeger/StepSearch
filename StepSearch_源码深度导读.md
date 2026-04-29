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
