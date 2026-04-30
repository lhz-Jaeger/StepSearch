# StepSearch 重点源码精读指南

这份文档只从代码角度理解项目。上一份 `PROJECT_RESUME_INTERVIEW_GUIDE.md` 更像论文、简历和面试总览；这一份的目标是带你顺着源码走完整条链路：

```text
parquet 样本
  -> DataLoader / DataProto
  -> vLLM rollout 生成 <search>
  -> FastAPI 检索器返回 <information>
  -> 多轮拼接成 response trajectory
  -> rule-based reward 写入 token-level reward tensor
  -> KL / GAE / PPO loss
  -> actor / critic 更新
```

读完你应该能回答：

1. 一条训练样本从 parquet 进来后，在哪些字段里流动。
2. 模型生成 `<search>` 后，代码如何真的调用检索器。
3. `<information>` 为什么不会参与 actor policy loss。
4. step reward 和 final reward 分别写在 reward tensor 的哪里。
5. PPO 的 old log prob、ref log prob、advantage、loss mask 在哪里产生和使用。

## 0. 先看当前入口脚本

当前项目里有几个入口：

```text
launch_retrieval.sh          启动检索服务
train_4.sh                   4 GPU 常规训练脚本
train.sh                     当前是 resume 训练脚本，会先切分 parquet，再从 checkpoint 继续训
scripts/eval_checkpoint.sh   用 checkpoint 做 val_only 推理评估
scripts/eval.py              对 predictions JSON 算 EM/F1
```

注意：你当前打开的 `train.sh` 已经不是最原始的从 base model 开始训练脚本，而是 resume 版本。它会调用 `scripts/slice_parquet_after_steps.py` 跳过已经训练过的数据，然后使用：

```text
actor_rollout_ref.model.path=$ACTOR_CKPT
critic.model.path=$CRITIC_CKPT
```

如果你想读最标准的训练配置，建议先看 `train_4.sh`；如果你想理解断点续训，再看当前 `train.sh`。

## 1. 全局源码调用链

先把大图刻在脑子里，后面每个文件都能挂回这张图：

```text
train_4.sh / train.sh
  -> python3 -m verl.trainer.main_ppo
    -> main(config)
      -> main_task(config)
        -> 初始化 tokenizer
        -> 选择 ActorRolloutRefWorker / CriticWorker / RefPolicy
        -> 构造 RewardManager
        -> 构造 RayPPOTrainer
          -> _create_dataloader()
            -> RLHFDataset 读 parquet，tokenize prompt
          -> init_workers()
          -> fit()
            -> 从 train_dataloader 取 batch
            -> LLMGenerationManager.run_llm_loop()
              -> actor_rollout_wg.generate_sequences()
              -> 解析 <search> / <answer>
              -> batch_search() 请求检索服务
              -> 拼接 <information>
              -> 返回完整 prompts/responses/input_ids/info_mask
            -> compute_log_prob() 得到 old_log_probs
            -> ref_policy_wg.compute_ref_log_prob()
            -> critic_wg.compute_values()
            -> RewardManager(batch)
              -> qa_step.compute_score_f1_steps_plan_with_support_docs()
              -> 生成 token_level_scores
            -> apply_kl_penalty()
            -> compute_advantage()
            -> critic_wg.update_critic()
            -> _create_loss_mask()
            -> actor_rollout_wg.update_actor()
```

检索服务是另一条进程链：

```text
launch_retrieval.sh
  -> search_r1/search/retrieval_server.py
    -> 加载 FAISS index
    -> 加载 corpus jsonl
    -> 加载 E5 encoder
    -> FastAPI /retrieve
```

## 2. 先理解 DataProto

文件：`verl/protocol.py`

你不需要一开始读完整个 veRL，但必须先理解 `DataProto`。它是 veRL 在 trainer、worker、reward function 之间传 batch 的统一容器。

核心结构：

```python
DataProto(
    batch=TensorDict(...),          # tensor 字段
    non_tensor_batch={...},         # Python 对象 / numpy object 字段
    meta_info={...},                # 配置、采样参数、统计信息
)
```

在 `verl/protocol.py` 里重点看：

1. `DataProto` 定义：约第 164 行。
2. `__getitem__`：约第 189 行，取单条样本时会同时取 tensor 和 non-tensor。
3. `union_tensor_dict`：约第 66 行，把新 tensor 字段并入已有 batch。
4. `pad_dataproto_to_divisor` / `unpad_dataproto`：约第 40 行，多 GPU 推理时补齐 batch。

本项目中你最常见的字段如下。

### 2.1 Tensor 字段

```text
input_ids               prompt 或 prompt+response token
attention_mask          哪些 token 有效
position_ids            position id
prompts                 原始 prompt token
responses               actor 生成的 response trajectory
responses_with_info_mask 用于构造 info_mask 的辅助字段
info_mask               屏蔽 <information> 后的有效 token mask
old_log_probs           rollout 轨迹在旧策略下的 log prob
ref_log_prob            reference policy 的 log prob
values                  critic 估计的 value
token_level_scores      reward function 直接给的分数
token_level_rewards     token_level_scores 扣 KL 后的 reward
advantages              GAE/GRPO 估计出的 advantage
returns                 critic 训练目标
loss_mask               actor update 时真正使用的 mask
```

### 2.2 Non-tensor 字段

```text
data_source             数据集名，例如 musi
prompt                  原始 chat prompt
ability                 fact-reasoning
reward_model            ground_truth，包括 target/search_keys
extra_info              support_docs、split、index
index                   当前样本编号
uid                     PPO/GRPO 分组或样本标识
```

读源码时，看到 `batch.batch[...]` 就想成 tensor；看到 `batch.non_tensor_batch[...]` 就想成原始数据、答案、support docs 这些 Python 对象。

## 3. 数据处理：jsonl 如何变成训练样本

文件：`scripts/data_process/musi_search.py`

这是你理解 reward 字段来源的第一站。

### 3.1 `make_prefix`

位置：`scripts/data_process/musi_search.py:16`

它把原始 question 包装成 prompt，要求模型按固定格式输出：

```text
<plan>...</plan>
<search>...</search>
<information>...</information>
<observation>...</observation>
<answer>...</answer>
```

源码里有两个 `prefix = ...`，第二个会覆盖第一个，所以真实生效的是第 25 行开始的 plan/search/observation/answer 版本。

你要注意这一点：论文里常写 `<think>`，但当前训练数据 prompt 里要求的是 `<plan>` 和 `<observation>`。不过代码真正提取动作时只看 `<search>` 和 `<answer>`，所以 plan/observation 更多是格式约束和语言习惯。

### 3.2 `process_fn`

位置：`scripts/data_process/musi_search.py:67`

这段代码最重要：

```python
solution = {
    "target": [example[args.answer_name]] + example['answer_aliases'],
    "search_keys": example['sub_searchs'] if not args.test else []
}
```

含义：

1. `target` 用于最终答案 F1。
2. `search_keys` 用于 search query 的 F1 奖励。
3. 测试集不提供 `search_keys`，避免把过程监督泄漏给评估。

接着它构造：

```python
"extra_info": {
    "split": split,
    "index": idx,
    "support_docs": example['sub_support_docs'] if not args.test else []
}
```

含义：

1. `support_docs` 用于 step information gain。
2. 测试集里置空，所以验证时 step reward 可能没有训练集那么完整，主要看 final answer。

### 3.3 读完这段你要能回答

1. `ground_truth['target']` 从哪里来？
2. `ground_truth['search_keys']` 从哪里来？
3. `support_docs` 从哪里来？
4. 为什么 test 模式下 `search_keys` 和 `support_docs` 为空？

## 4. Dataset：parquet 如何变成 token

文件：`verl/utils/dataset/rl_dataset.py`

这里是 veRL 读训练数据的地方。

### 4.1 `collate_fn`

位置：`verl/utils/dataset/rl_dataset.py:26`

它做一件事：把 list of dict 分成 tensor 和 non-tensor 两类。

```text
torch.Tensor -> stack 成 batch tensor
其他对象 -> np.array(dtype=object)
```

所以 parquet 中的 `reward_model`、`extra_info` 这类字典不会被 token 化，它们会进入 `non_tensor_batch`。

### 4.2 `RLHFDataset.__getitem__`

位置：`verl/utils/dataset/rl_dataset.py:122`

它的流程：

1. `row_dict = self.dataframe.iloc[item].to_dict()` 取一行 parquet。
2. `chat = row_dict.pop(self.prompt_key)` 取出 prompt。
3. 如果 tokenizer 有 chat template，就 `apply_chat_template`。
4. 调 `tokenize_and_postprocess_data` 得到 `input_ids` 和 `attention_mask`。
5. 计算 `position_ids`。
6. 把 `input_ids/attention_mask/position_ids` 放回 `row_dict`。
7. 从 `extra_info.index` 写入 `row_dict["index"]`。

这一步结束后，dataloader 每条样本已经同时带有：

```text
tensor:
  input_ids, attention_mask, position_ids

non-tensor:
  data_source, ability, reward_model, extra_info, index ...
```

### 4.3 `RayPPOTrainer._create_dataloader`

文件：`verl/trainer/ppo/ray_trainer.py`

位置：约第 404 行。

它把 `data.train_files`、`data.val_files` 传给 `RLHFDataset`，再创建 PyTorch `DataLoader`：

```python
self.train_dataloader = DataLoader(
    dataset=self.train_dataset,
    batch_size=self.config.data.train_batch_size,
    shuffle=self.config.data.shuffle_train_dataloader,
    drop_last=True,
    collate_fn=collate_fn,
)
```

训练集 `drop_last=True`，所以如果你做断点续训切数据，最后不满一个 batch 的部分会被丢掉。你当前 `scripts/slice_parquet_after_steps.py` 打印的 `remaining_rows_dropped_by_drop_last` 就是在提醒这件事。

## 5. 检索服务：query 如何变成 documents

文件：`search_r1/search/retrieval_server.py`

启动入口：`launch_retrieval.sh`

```bash
python search_r1/search/retrieval_server.py \
  --index_path data/musi.index \
  --corpus_path data/musi.jsonl \
  --retriever_name e5 \
  --retriever_model intfloat/e5-base-v2 \
  --topk 3 \
  --faiss_gpu
```

### 5.1 `Encoder.encode`

位置：`search_r1/search/retrieval_server.py:77`

重点：

```python
if "e5" in self.model_name.lower():
    if is_query:
        query_list = [f"query: {query}" for query in query_list]
    else:
        query_list = [f"passage: {query}" for query in query_list]
```

E5 模型要求 query/passsage 使用不同前缀。检索时只会走 query 前缀；建索引时 passage 应该用 passage 前缀。

随后 tokenizer、模型前向、mean pooling、normalize，最后得到 numpy float32 embedding。

### 5.2 `DenseRetriever.__init__`

位置：`search_r1/search/retrieval_server.py:208`

它加载三件东西：

```text
FAISS index       self.index = faiss.read_index(...)
corpus jsonl      self.corpus = load_corpus(...)
encoder model     self.encoder = Encoder(...)
```

如果 `--faiss_gpu` 打开，会把 index 搬到多 GPU：

```python
co = faiss.GpuMultipleClonerOptions()
co.useFloat16 = True
co.shard = True
self.index = faiss.index_cpu_to_all_gpus(self.index, co=co)
```

### 5.3 `DenseRetriever._batch_search`

位置：`search_r1/search/retrieval_server.py:242`

这是实际检索逻辑：

```text
query_batch
  -> encoder.encode(query_batch)
  -> self.index.search(batch_emb, k=num)
  -> 得到 batch_idxs
  -> load_docs(self.corpus, flat_idxs)
  -> 按每个 query 的 topk chunk 回来
```

它还打印耗时：

```text
total / encode / faiss / load_docs
```

训练时如果你发现 rollout 特别慢，先看这些日志。

### 5.4 `/retrieve`

位置：`search_r1/search/retrieval_server.py:353`

请求格式：

```json
{
  "queries": ["where is KBQI located"],
  "topk": 3,
  "return_scores": true
}
```

返回格式：

```json
{
  "result": [
    [
      {"document": {...}, "score": 0.82},
      {"document": {...}, "score": 0.79}
    ]
  ]
}
```

这个格式会被 `generation.py` 的 `_passages2string` 消费。

## 6. 训练入口：main_ppo 如何启动整个系统

文件：`verl/trainer/main_ppo.py`

训练脚本最终都会执行：

```bash
python3 -m verl.trainer.main_ppo ...
```

### 6.1 `main`

位置：`verl/trainer/main_ppo.py:137`

Hydra 会读取 `verl/trainer/config/ppo_trainer.yaml`，再叠加 shell 脚本里的一堆 override。

`main` 里初始化 Ray：

```python
ray.init(runtime_env={
    'env_vars': {
        'VLLM_ATTENTION_BACKEND': 'XFORMERS',
        'TOKENIZERS_PARALLELISM': 'true',
        'NCCL_DEBUG': 'WARN'
    }
})
```

然后调用远程任务：

```python
ray.get(main_task.remote(config))
```

### 6.2 `main_task`

位置：`verl/trainer/main_ppo.py:155`

它做 5 件事：

1. 解析 config，打印完整配置。
2. 加载 actor 模型路径对应的 tokenizer。
3. 根据 `actor.strategy` 选择 FSDP 或 Megatron worker。
4. 构造 `RewardManager`。
5. 构造 `RayPPOTrainer` 并启动 `trainer.fit()`。

最关键的 worker 映射：

```python
role_worker_mapping = {
    Role.ActorRollout: ray.remote(ActorRolloutRefWorker),
    Role.Critic: ray.remote(CriticWorker),
    Role.RefPolicy: ray.remote(ActorRolloutRefWorker),
}
```

这里 ActorRolloutRefWorker 一人多职：actor 训练、rollout 生成、reference logprob 都由同一个 worker class 的不同角色承担。

### 6.3 `RewardManager`

位置：`verl/trainer/main_ppo.py:38`

这是 rule-based reward 和 PPO tensor 对接的桥。

它不直接判断 search quality，而是调用：

```python
qa_step.compute_score_f1_steps_plan_with_support_docs(...)
```

然后把返回的 Python 分数写进 `reward_tensor`。

## 7. Rollout 核心：LLMGenerationManager

文件：`search_r1/llm_agent/generation.py`

这是全项目最重要的源码。它实现的是“LLM + 搜索环境”的多轮交互。

### 7.1 `GenerationConfig`

位置：`search_r1/llm_agent/generation.py:14`

字段：

```text
max_turns              最多搜索交互轮数
max_start_length       初始 prompt 保留长度
max_prompt_length      滚动上下文最大长度
max_response_length    单次生成最大长度
max_obs_length         单次检索 observation 最大长度
num_gpus               多 GPU padding 用
search_url             检索服务 URL
topk                   每次检索 top-k
```

这些值来自 `ray_trainer.py` 里的 config。

### 7.2 `_postprocess_responses`

位置：`search_r1/llm_agent/generation.py:55`

模型一次生成可能很长，但环境只需要执行第一个动作。这里会把输出截断到第一个 `</search>` 或 `</answer>`：

```python
responses_str = [
    resp.split('</search>')[0] + '</search>'
    if '</search>' in resp
    else resp.split('</answer>')[0] + '</answer>'
    if '</answer>' in resp
    else resp
    for resp in responses_str
]
```

面试时可以说：这一步把自由生成变成工具调用环境中的“一步 action”。

### 7.3 `_info_masked_concatenate_with_padding`

位置：`search_r1/llm_agent/generation.py:121`

这是理解 mask 的关键函数。

它拼接：

```text
已有 responses
当前 actor 生成 response
检索器返回 info
```

同时维护一个 `responses_with_info_mask`。对于 actor 自己生成的 response，保留原 token；对于外部 `info`，填成 pad token：

```python
info_mask = torch.full(info.size(), pad_id, dtype=info.dtype, device=info.device)
tensors_with_mask.append(info_mask)
```

后面 `create_attention_mask(responses_with_info_mask)` 时：

```text
actor 生成 token -> 非 pad -> mask=1
外部 information -> pad -> mask=0
```

这就是 `<information>` 不参与 policy loss 的根。

### 7.4 `run_llm_loop`

位置：`search_r1/llm_agent/generation.py:221`

主循环结构：

```python
for step in range(self.config.max_turns):
    1. 只取 active 样本
    2. actor_rollout_wg.generate_sequences(...)
    3. 截断到 search/answer
    4. execute_predictions(...)
    5. 更新 active_mask
    6. 把 next_obs token 化
    7. 更新 rolling context
    8. 更新 original_right_side

最后再做一次 final rollout，不再真的搜索
```

几个变量很重要：

```text
active_mask           哪些样本还没 answer
turns_stats           每条样本用了几轮
valid_action_stats    合法 action 数
valid_search_stats    合法 search 数
rollings              下一轮要喂给模型的上下文
original_right_side   最终要返回给 PPO 的完整 response trajectory
```

### 7.5 `execute_predictions`

位置：`search_r1/llm_agent/generation.py:361`

它是真正的环境 step。

先解析 action：

```python
cur_actions, contents = self.postprocess_predictions(predictions)
```

如果有 search：

```python
search_queries = [content for action, content in zip(cur_actions, contents) if action == 'search']
search_results = self.batch_search(search_queries)
```

然后逐条处理：

```text
answer -> done=True，不追加 observation
search -> 追加 <information>...</information>，done=False
非法格式 -> 追加 warning，done=False
inactive -> 空 observation，done=True
```

这段要和 `_postprocess_responses` 一起读：前者截断生成，后者执行动作。

### 7.6 `postprocess_predictions`

位置：`search_r1/llm_agent/generation.py:424`

正则只识别两种标签：

```python
pattern = r'<(search|answer)>(.*?)</\1>'
```

这解释了为什么 prompt 可以要求 `<plan>` 和 `<observation>`，但环境只关心 `<search>` 和 `<answer>`。

### 7.7 `batch_search`

位置：`search_r1/llm_agent/generation.py:455`

它请求检索服务：

```python
payload = {
    "queries": queries,
    "topk": self.config.topk,
    "return_scores": True
}
response = requests.post(self.config.search_url, json=payload)
```

随后 `_passages2string` 把结构化 doc 变成模型可读文本：

```text
Doc 1<## Title: ... ##> ...
Doc 2<## Title: ... ##> ...
```

注意这个格式后面会被 reward 函数解析。

### 7.8 `_compose_final_output`

位置：`search_r1/llm_agent/generation.py:329`

最后返回给 PPO trainer 的 DataProto 包含：

```text
prompts
responses
input_ids = prompts + responses
attention_mask
info_mask
position_ids
```

`info_mask` 是后面 KL 和 actor loss mask 的基础。

## 8. Reward 源码：字符串如何变成分数

文件：`verl/utils/reward_score/qa_step.py`

这个文件处理的是“完整 response 字符串”。

### 8.1 基础标准化

位置：`verl/utils/reward_score/qa_step.py:10`

`normalize_answer` 会：

1. lowercase
2. 去标点
3. 去英文冠词
4. 合并空格

答案 F1 和搜索关键词 F1 都基于它。

### 8.2 `extract_solution`

位置：`verl/utils/reward_score/qa_step.py:26`

提取最后一个 `<answer>...</answer>`。

如果没有 answer，返回 `None`。后续 `query_f1_score(None, ...)` 会给 0。

### 8.3 `extract_content_from_information`

位置：`verl/utils/reward_score/qa_step.py:54`

它假设 information 里文档格式是：

```text
Doc 1<## Title: XXX ##> content
Doc 2<## Title: YYY ##> content
```

然后通过：

```python
information = information.split('##>')[1:]
information = [item.split('<##')[0].strip() for item in information]
```

抽出每篇文档正文。

这就是为什么 `generation.py` 的 `_passages2string` 格式不能随便改。

### 8.4 `query_f1_score`

位置：`verl/utils/reward_score/qa_step.py:69`

它计算预测文本和多个 gold answer/key 的最大 word-level F1。答案奖励和 search key reward 都用这个函数。

### 8.5 `step_information_gain`

位置：`verl/utils/reward_score/qa_step.py:106`

这是 StepSearch 最核心的 reward。

输入：

```text
informations: List[List[str]]
  每轮 search 返回的文档正文列表

golden_info: List[dict]
  每条 dict 里有 paragraph_text
```

内部状态：

```python
golden_infos = [normalize_answer(info['paragraph_text']) for info in golden_info]
previous_match_degree = [0.0 for _ in golden_infos]
```

对每一轮 search：

1. 初始化当前轮对每个 golden doc 的匹配度。
2. 遍历当前轮检索到的每个 doc。
3. 用 `subm_tfidf_cosine(input_str=info, concept_units=golden_infos)` 算它和所有 golden doc 的相似度。
4. 对每个 golden doc 保留当前轮最大相似度。
5. 只奖励比历史更高的部分：

```python
max(current_match_degree[i] - previous_match_degree[i], 0)
```

6. 更新历史最大匹配度。

直觉：

```text
第一轮找到了 gold doc A -> 有 gain
第二轮又找到 A -> 没有 gain
第二轮新找到了 gold doc B -> 有 gain
```

### 8.6 redundancy penalty

位置：`verl/utils/reward_score/qa_step.py:143`

这里不是语义相似重复，而是文档字符串完全重复：

```python
info_gotten = set()
for i, information in enumerate(informations):
    for info in information:
        if info in info_gotten:
            redundancy_penalty[i] += 1/len(information)
        else:
            info_gotten.add(info)
```

如果当前轮 top-k 文档中有重复历史文档，就按比例扣分。

### 8.7 `step_search_keys_match`

位置：`verl/utils/reward_score/qa_step.py:176`

输入：

```text
searches: 模型发出的 query 列表
keys: gold search key groups
```

逻辑：

1. 每个模型 query 和每组 gold keys 算 F1。
2. 对每组 gold keys，保留被任意 query 命中的最高分。
3. 对所有 key group 求平均。

这鼓励模型覆盖每个子问题需要的 search key。

### 8.8 `compute_score_f1_steps_plan_with_support_docs`

位置：`verl/utils/reward_score/qa_step.py:218`

总控函数：

1. 提取最终 answer。
2. 用 `query_f1_score` 算 answer_correct。
3. 用 `answer_last_check` 确保 answer 在最后；否则直接返回 `score=-1`。
4. 用正则找所有 `<search>...</search><information>...</information>` 对。
5. 提取每轮 information 中的文档正文。
6. 计算 information gain 和 redundancy penalty。
7. 按 config 开关合成 step scores：

```python
step_scores = [
    gain - penalty
    for gain, penalty in zip(information_gains, redundancy_penalty)
]
```

8. 提取所有 `<search>` query，计算 search_key_score。
9. final score：

```python
final_score = answer_f1 + search_key_score * 0.618
```

如果对应开关关闭，相关项会变成 0。

## 9. RewardManager：分数如何写进 reward tensor

文件：`verl/trainer/main_ppo.py`

### 9.1 `RewardManager.__call__`

位置：`verl/trainer/main_ppo.py:48`

输入是完整 `DataProto` batch。它逐条样本处理：

```python
prompt_ids = data_item.batch['prompts']
response_ids = data_item.batch['responses']
```

根据 `attention_mask` 找有效长度，然后：

```python
sequences = torch.cat((valid_prompt_ids, valid_response_ids))
sequences_str = self.tokenizer.decode(sequences)
```

如果是 Qwen chat template，会截掉 `<|im_start|>assistant` 前面的部分，只保留 assistant 输出。

接着调用：

```python
score = compute_score_fn(
    self.config,
    solution_str=sequences_str,
    ground_truth=ground_truth,
    support_docs=data_item.non_tensor_batch['extra_info']['support_docs'],
)
```

返回的 `score` 是 Python dict：

```text
score['score']             final score
score['answer_correct']    answer F1
score['step_scores']       每轮 search step 分数
score['search_key_score']  search key F1
```

### 9.2 step reward 位置

位置：`verl/trainer/main_ppo.py:119`

`_compute_step_reward_tensor` 会重新 decode response，并找到每个：

```text
<search>...</search><information>...</information>
```

中 `</search>` 的 token 位置：

```python
search_end_indices = [
    m.start() + m.group().find('</search>') + len('</search>')
    for m in re.finditer(...)
]
search_end_positions = [
    len(self.tokenizer.encode(response[:search_end_index]))
    for search_end_index in search_end_indices
]
```

然后把每个 `step_scores[i]` 写到：

```python
reward_tensor[search_end_positions[i]-1] = step_scores[i]
```

这意味着：step reward 大致落在 `</search>` 结束 token 附近，而不是 `<information>` 结束 token。

### 9.3 final reward 位置

位置：`verl/trainer/main_ppo.py:97`

最终奖励写在最后一个有效 response token：

```python
reward_tensor[i, valid_response_length - 1] = score['score']
```

所以一个 response 的 reward tensor 里通常只有少数非零点：

```text
第 1 次 search 结束处: information_gain - redundancy_penalty
第 2 次 search 结束处: information_gain - redundancy_penalty
...
最后一个 token: answer_f1 + 0.618 * search_key_score
```

### 9.4 一个代码级小例子

假设 response 是：

```text
<plan>...</plan>
<search>KBQI location</search>
<information>Doc 1<## Title: KBQI ##> KBQI is in Albuquerque...</information>
<observation>...</observation>
<search>Albuquerque county state</search>
<information>Doc 1<## Title: Albuquerque ##> ... Bernalillo County...</information>
<answer>Bernalillo County, New Mexico</answer>
```

reward 侧会：

1. 提取 answer，和 target 算 F1。
2. 提取两个 search query，和 search_keys 算 search_key_score。
3. 提取两段 information，和 support_docs 算两个 step_scores。
4. 把两个 step_scores 写到两个 `</search>` 位置。
5. 把 final_score 写到最后 `</answer>` 位置。

## 10. PPO 主循环：RayPPOTrainer.fit

文件：`verl/trainer/ppo/ray_trainer.py`

### 10.1 `fit` 总流程

位置：`verl/trainer/ppo/ray_trainer.py:742`

建议你把第 782 到 912 行反复读几遍，这是训练核心。

流程分解：

```text
1. 从 train_dataloader 取 batch_dict
2. DataProto.from_single_dict(batch_dict)
3. batch.repeat(n_agent)
4. pop input_ids/attention_mask/position_ids 给 rollout
5. 如果 do_search=true，调用 generation_manager.run_llm_loop
6. compute_log_prob 得到 old_log_probs
7. 把 generation output union 回 batch
8. balance batch
9. ref policy 计算 ref_log_prob
10. critic 计算 values
11. reward_fn 计算 token_level_scores
12. apply_kl_penalty 得到 token_level_rewards
13. compute_advantage 得到 advantages/returns
14. update_critic
15. create_loss_mask
16. update_actor
17. validate / save / log
```

### 10.2 生成轨迹

位置：`verl/trainer/ppo/ray_trainer.py:812`

```python
first_input_ids = gen_batch.batch['input_ids'][:, -gen_config.max_start_length:].clone().long()
final_gen_batch_output = generation_manager.run_llm_loop(
    gen_batch=gen_batch,
    initial_input_ids=first_input_ids,
)
```

`gen_batch` 最初只有 prompt 的 token。`final_gen_batch_output` 会多出：

```text
prompts
responses
input_ids
attention_mask
info_mask
position_ids
```

### 10.3 old log prob

位置：`verl/trainer/ppo/ray_trainer.py:826`

```python
output = self.actor_rollout_wg.compute_log_prob(final_gen_batch_output)
final_gen_batch_output = final_gen_batch_output.union(output)
```

这里算的是 rollout 时旧策略对这条 response trajectory 的 log probability。PPO 后面要比较新旧策略概率比。

### 10.4 reference log prob

位置：`verl/trainer/ppo/ray_trainer.py:854`

```python
ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
batch = batch.union(ref_log_prob)
```

这个用于 KL penalty，防止 RL 把模型拉太偏。

### 10.5 critic values

位置：`verl/trainer/ppo/ray_trainer.py:860`

```python
values = self.critic_wg.compute_values(batch)
batch = batch.union(values)
```

critic 给每个 response token 一个 value estimate，GAE 要用。

### 10.6 reward function

位置：`verl/trainer/ppo/ray_trainer.py:876`

```python
reward_tensor, answer_correct, search_key_score, step_scores = self.reward_fn(batch)
batch.batch['token_level_scores'] = reward_tensor
```

注意这里字段名叫 `token_level_scores`。还没扣 KL。

### 10.7 KL penalty

位置：`verl/trainer/ppo/ray_trainer.py:93`

函数 `apply_kl_penalty` 里：

```python
attention_mask = data.batch['info_mask'] if 'info_mask' in data.batch else data.batch['attention_mask']
response_mask = attention_mask[:, -response_length:]
kld = kl_penalty(old_log_probs, ref_log_prob) * response_mask
token_level_rewards = token_level_scores - beta * kld
```

这里也用了 `info_mask`，所以外部 information token 不参与 KL 统计。

### 10.8 advantage

位置：`verl/trainer/ppo/ray_trainer.py:125`

如果 `algorithm.adv_estimator=gae`：

```python
advantages, returns = core_algos.compute_gae_advantage_return(
    token_level_rewards=token_level_rewards,
    values=values,
    eos_mask=response_mask,
    gamma=gamma,
    lam=lam,
)
```

注意这里的 `response_mask` 使用的是普通 `attention_mask`，不是 `info_mask`。真正屏蔽 information 的 actor loss 在后面 `_create_loss_mask` 和 `dp_actor.py` 中发生。

### 10.9 loss mask

位置：`verl/trainer/ppo/ray_trainer.py:945`

```python
loss_mask = batch.batch['info_mask'][:, -response_length:]
batch.batch['loss_mask'] = loss_mask
```

这一步把 `info_mask` 截取 response 部分，变成 actor update 用的 `loss_mask`。

## 11. PPO 数学在代码里的位置

文件：`verl/trainer/ppo/core_algos.py`

### 11.1 GAE

位置：`verl/trainer/ppo/core_algos.py:70`

核心递推：

```python
delta = token_level_rewards[:, t] + gamma * nextvalues - values[:, t]
lastgaelam = delta + gamma * lam * lastgaelam
```

然后：

```python
returns = advantages + values
advantages = masked_whiten(advantages, eos_mask)
```

理解：

1. `token_level_rewards` 只有少数位置非零。
2. GAE 会把未来 reward 信号向前传播。
3. `masked_whiten` 做归一化，提升稳定性。

### 11.2 PPO clipped loss

位置：`verl/trainer/ppo/core_algos.py:163`

核心：

```python
negative_approx_kl = log_prob - old_log_prob
ratio = torch.exp(negative_approx_kl)
pg_losses = -advantages * ratio
pg_losses2 = -advantages * torch.clamp(ratio, 1.0 - cliprange, 1.0 + cliprange)
pg_loss = masked_mean(torch.max(pg_losses, pg_losses2), eos_mask)
```

`eos_mask` 在 actor update 时会被替换为 `loss_mask`，这就是不训练 `<information>` 的最终落点。

### 11.3 value loss

位置：`verl/trainer/ppo/core_algos.py:216`

critic 的目标是拟合 returns：

```python
vf_losses1 = (vpreds - returns)**2
vf_losses2 = (vpredclipped - returns)**2
vf_loss = 0.5 * masked_mean(max(vf_losses1, vf_losses2), eos_mask)
```

### 11.4 KL penalty

位置：`verl/trainer/ppo/core_algos.py:242`

默认 `kl`：

```python
return logprob - ref_logprob
```

然后在 `apply_kl_penalty` 中：

```text
token_level_rewards = token_level_scores - beta * kld
```

## 12. Actor 更新：loss_mask 真正在哪里生效

文件：`verl/workers/actor/dp_actor.py`

### 12.1 `compute_log_prob`

位置：`verl/workers/actor/dp_actor.py:153`

它选取：

```python
select_keys = ['responses', 'input_ids', 'attention_mask', 'position_ids']
```

然后前向模型，返回每个 response token 的 log prob。

训练流程中它被用来计算 `old_log_probs`。

### 12.2 `update_policy`

位置：`verl/workers/actor/dp_actor.py:203`

最关键几行：

```python
response_mask = attention_mask[:, -response_length:]
if self.config.state_masking:
    response_mask = data['loss_mask']
```

也就是说：

```text
state_masking=false -> 所有 response 有效 token 都参与 actor loss
state_masking=true  -> 只让 loss_mask=1 的 token 参与 actor loss
```

然后：

```python
pg_loss, pg_clipfrac, ppo_kl = compute_policy_loss(
    old_log_prob=old_log_prob,
    log_prob=log_prob,
    advantages=advantages,
    eos_mask=response_mask,
    cliprange=clip_ratio,
)
```

这就是从 `generation.py` 的 `responses_with_info_mask` 到 actor loss 的闭环。

## 13. Validation 和预测保存

文件：`verl/trainer/ppo/ray_trainer.py`

### 13.1 `_validate`

位置：`verl/trainer/ppo/ray_trainer.py:468`

当 `do_search=true` 时，验证流程和训练 rollout 基本一致：

1. 从 `val_dataloader` 取 batch。
2. 调 `generation_manager.run_llm_loop`。
3. 调 `val_reward_fn` 得到 answer_correct。
4. 解码 response。
5. 保存 predictions JSON。

保存字段：

```text
id
data_source
prompt
response
ground_truth
```

### 13.2 `scripts/eval.py`

评估脚本只看 predictions JSON，不重新跑模型。

重点函数：

1. `extract_answer`：从 `<answer>...</answer>` 提取最后一个答案。
2. `exact_match_accuracy`：标准化后完全一致。
3. `string_f1_score`：word-level F1。

注意：当前 `extract_answer` 里有一处疑似手误：

```python
if "}" in matches in matches:
```

普通答案一般不触发，但如果你后续维护代码，可以改成：

```python
if "}" in matches:
```

## 14. 当前 resume 训练脚本怎么读

文件：`train.sh`

当前 `train.sh` 的逻辑：

1. 设置 base experiment name。
2. 设置 `SKIP_STEPS=30`、`CKPT_STEP=30`、`TRAIN_BATCH_SIZE=256`。
3. 调 `scripts/slice_parquet_after_steps.py`：

```bash
--global_step $SKIP_STEPS
--batch_size $TRAIN_BATCH_SIZE
```

4. 检查 actor/critic checkpoint 是否存在。
5. 用 actor checkpoint 作为 actor model path。
6. 用 critic checkpoint 作为 critic model path。
7. 继续跑 `verl.trainer.main_ppo`。

### 14.1 为什么要 slice parquet

如果训练中断在 global step 30，而且 dataloader 不保存 iterator 状态，那么直接从 checkpoint 继续训练会从数据开头再消费一遍。

`slice_parquet_after_steps.py` 用：

```python
skip_rows = args.global_step * args.batch_size
sliced = dataframe.iloc[skip_rows:].reset_index(drop=True)
```

跳过已经消耗的样本。

这是一种工程上简单直接的 resume 方案，但前提是：

1. 原训练 `shuffle_train_dataloader=False`。
2. batch size 和之前完全一致。
3. global_step 和实际已完成 optimizer step 对齐。

如果打开 shuffle，这种切片就不再严格等价。

## 15. 关键数据流：一条样本走完整个系统

下面用一条样本串起来。

### 15.1 parquet 行

来自 `musi_search.py`：

```python
{
    "data_source": "musi",
    "prompt": [{"role": "user", "content": "...Question: ..."}],
    "ability": "fact-reasoning",
    "reward_model": {
        "style": "rule",
        "ground_truth": {
            "target": ["Bernalillo County, New Mexico", ...],
            "search_keys": [...]
        }
    },
    "extra_info": {
        "split": "train",
        "index": 123,
        "support_docs": [...]
    }
}
```

### 15.2 Dataset 输出

`RLHFDataset.__getitem__` 后：

```text
input_ids
attention_mask
position_ids
data_source
reward_model
extra_info
index
```

### 15.3 fit 中进入 rollout

`ray_trainer.fit`：

```python
gen_batch = batch.pop(['input_ids', 'attention_mask', 'position_ids'])
final_gen_batch_output = generation_manager.run_llm_loop(...)
```

### 15.4 rollout 后新增字段

`generation.py` 返回：

```text
prompts
responses
responses_with_info_mask
input_ids
attention_mask
info_mask
position_ids
```

### 15.5 reward 后新增字段

`RewardManager` 返回：

```text
token_level_scores
answer_correct
search_key_score
step_scores
```

### 15.6 KL 和 GAE 后新增字段

```text
token_level_rewards
advantages
returns
```

### 15.7 actor update 前新增字段

```text
loss_mask = info_mask[:, -response_length:]
```

这就是一条样本从原始数据到 PPO loss 的完整轨迹。

## 16. 必须理解的三个 mask

### 16.1 `attention_mask`

普通 token 有效性。只区分 pad 和非 pad。

```text
prompt token       1
model response     1
retrieved info     1
pad                0
```

### 16.2 `info_mask`

屏蔽外部 retrieved information。

```text
prompt token       1
model response     1
retrieved info     0
pad                0
```

来源：`generation.py` 中 `responses_with_info_mask`。

### 16.3 `loss_mask`

actor update 真正使用的 response 部分 mask：

```python
loss_mask = info_mask[:, -response_length:]
```

来源：`ray_trainer.py::_create_loss_mask`。

使用：`dp_actor.py::update_policy`。

## 17. 推荐源码精读顺序

### 第一轮：只读主干，不进 worker 细节

1. `train_4.sh`
2. `verl/trainer/main_ppo.py`
3. `verl/trainer/ppo/ray_trainer.py::fit`
4. `search_r1/llm_agent/generation.py::run_llm_loop`
5. `verl/trainer/main_ppo.py::RewardManager`
6. `verl/utils/reward_score/qa_step.py`

目标：能画出训练闭环。

### 第二轮：读数据和检索

1. `scripts/data_process/musi_search.py`
2. `verl/utils/dataset/rl_dataset.py`
3. `search_r1/search/retrieval_server.py`
4. `search_r1/search/index_builder.py`

目标：能解释 `support_docs/search_keys/corpus/index` 的关系。

### 第三轮：读 PPO 细节

1. `verl/trainer/ppo/core_algos.py`
2. `verl/workers/actor/dp_actor.py`
3. `verl/workers/fsdp_workers.py` 中 `compute_log_prob`、`compute_ref_log_prob`、`compute_values` 相关调用

目标：能解释 old log prob、ref log prob、KL、GAE、PPO clipped loss。

### 第四轮：读评估和工程脚本

1. `verl/trainer/ppo/ray_trainer.py::_validate`
2. `scripts/eval_checkpoint.sh`
3. `scripts/eval.py`
4. `eval/retriever_eval.py`
5. 当前 `train.sh` 和 `scripts/slice_parquet_after_steps.py`

目标：能跑通或解释训练后的评估链路。

## 18. 调试建议

如果你之后要真正跑，建议加断点或临时 print 在这些位置。

### 18.1 检查数据字段

位置：`verl/utils/dataset/rl_dataset.py::__getitem__`

看：

```text
row_dict.keys()
row_dict['reward_model']
row_dict['extra_info']
input_ids.shape
```

### 18.2 检查模型动作

位置：`search_r1/llm_agent/generation.py::_postprocess_responses`

看：

```text
responses_str[0]
```

确认模型是否生成合法 `<search>` 或 `<answer>`。

### 18.3 检查检索请求

位置：`search_r1/llm_agent/generation.py::_batch_search`

看：

```text
queries
payload
response.status_code
```

### 18.4 检查 information 格式

位置：`search_r1/llm_agent/generation.py::_passages2string`

看：

```text
format_reference
```

必须包含：

```text
Doc 1<## Title: ... ##> ...
```

否则 `qa_step.extract_content_from_information` 解析会出问题。

### 18.5 检查 reward

位置：`verl/utils/reward_score/qa_step.py::compute_score_f1_steps_plan_with_support_docs`

看：

```text
answer
answer_correct
information_gains
redundancy_penalty
step_scores
search_key_score
final_score
```

### 18.6 检查 mask

位置：`search_r1/llm_agent/generation.py::_compose_final_output`

看：

```text
attention_mask.sum()
info_mask.sum()
```

理论上：

```text
info_mask.sum() <= attention_mask.sum()
```

差值大致就是 `<information>` token 数。

### 18.7 检查 actor loss 是否屏蔽 info

位置：`verl/workers/actor/dp_actor.py::update_policy`

看：

```text
response_mask.sum()
attention_mask[:, -response_length:].sum()
```

如果 `state_masking=true`，前者应该小于等于后者。

## 19. 常见源码疑问

### 19.1 为什么 `_postprocess_responses` 只截到 `</search>` 或 `</answer>`？

因为一次 rollout step 只执行一个环境动作。如果模型生成了 search 后又继续编造 information 或 answer，必须截断，否则会污染环境交互。

### 19.2 为什么 final rollout 不再搜索？

`run_llm_loop` 在 `max_turns` 后还有一次 final generation，并调用：

```python
execute_predictions(..., do_search=False)
```

这相当于告诉模型搜索预算用完了，应该基于已有信息回答。

### 19.3 为什么 step reward 写在 `</search>` 附近而不是 `</information>`？

因为 `<information>` 是外部检索器返回的，不是模型动作。把 reward 写在 search action 结束附近，更直接地把信用分配给 query 生成行为。

### 19.4 为什么 `answer_last_check` 不通过时给 `score=-1`？

这是格式硬约束。模型如果在 `<answer>` 后继续输出内容，说明交互协议不可靠。给负分可以压制格式投机和无效轨迹。

### 19.5 为什么 `search_key_reward` 是 final reward 的一部分，而不是 step reward？

代码里 search key reward 是对整条轨迹所有 queries 与所有 gold key groups 的覆盖程度求平均，不是一轮一轮严格对齐。因此它更像 trajectory-level global signal。

### 19.6 为什么 `support_docs` 测试时为空？

训练时可用 support_docs 做过程监督；测试时主要评估最终答案，不应该依赖 gold supporting docs 给中间步骤奖励，否则会泄漏标注过程信息。

### 19.7 如果检索结果是噪声，会怎样？

模型仍会看到 `<information>`，但 reward 中：

1. 如果噪声和 support_docs 相似度低，information gain 低。
2. 如果重复返回，redundancy penalty 高。
3. 如果最终答案错，answer reward 低。

因此策略会逐渐偏向能搜到有效证据的 query。

## 20. 面试时从代码角度怎么讲

可以用这段：

```text
代码上我把它理解成三层。第一层是环境交互层，LLMGenerationManager 把模型输出截断成 <search> 或 <answer>，search 会请求 FastAPI 检索服务，把返回文档包成 <information> 再拼回上下文。第二层是奖励层，RewardManager 解码完整 trajectory，qa_step.py 根据 answer F1、search key F1、information gain 和 redundancy penalty 构造 token-level reward tensor。第三层是 PPO 层，RayPPOTrainer 计算 old log prob、ref log prob、critic values，扣 KL 后用 GAE 算 advantage，再用 info_mask 生成 loss_mask，确保 actor 只训练自己生成的 token，不训练检索器返回的 information。
```

如果面试官继续问“你最熟悉哪段源码”，优先讲：

1. `generation.py::run_llm_loop`
2. `generation.py::_info_masked_concatenate_with_padding`
3. `qa_step.py::step_information_gain`
4. `main_ppo.py::RewardManager.__call__`
5. `ray_trainer.py::fit`

这五个点串起来，基本就能证明你真的读懂了。

## 21. 自测题

读完源码后，试着不看文档回答这些问题：

1. `responses_with_info_mask` 和 `responses` 有什么区别？
2. `info_mask` 是在哪个函数构造的？
3. `loss_mask` 是在哪里构造的，又在哪里使用？
4. `old_log_probs` 是哪个 worker 计算的？
5. `ref_log_prob` 的作用是什么？
6. `token_level_scores` 和 `token_level_rewards` 的区别是什么？
7. `step_scores` 最终写在 reward tensor 的哪个位置？
8. `search_key_score` 为什么乘以 0.618？
9. `DenseRetriever._batch_search` 的瓶颈可能在哪里？
10. 当前 `train.sh` 为什么要求 `shuffle_train_dataloader=False`？
11. 测试集里没有 `support_docs` 时，step reward 会发生什么？
12. 如果模型输出 `<search>abc</search><answer>def</answer>`，本轮会执行什么？
13. 如果模型输出没有 `<search>` 也没有 `<answer>`，环境会追加什么？
14. 为什么 PPO update 不能让 `<information>` token 参与 loss？
15. 为什么 Search-RL 比普通 RAG 更像一个 RL environment？

## 22. 最终代码理解闭环

你最后要能把项目压缩成这个代码闭环：

```text
musi_search.py 构造带 target/search_keys/support_docs 的 parquet
  -> RLHFDataset tokenizes prompt，并保留 reward_model/extra_info
  -> RayPPOTrainer.fit 取 batch
  -> LLMGenerationManager.run_llm_loop 让 actor 多轮生成 search/answer
  -> retrieval_server 用 E5+FAISS 返回 top-k docs
  -> generation.py 把 docs 包成 information 拼回上下文，同时构造 info_mask
  -> RewardManager 调 qa_step.py 计算 final score 和 step_scores
  -> reward_tensor 在 search token 和 final token 处产生非零奖励
  -> apply_kl_penalty 得到 token_level_rewards
  -> compute_gae_advantage_return 得到 advantages/returns
  -> _create_loss_mask 用 info_mask 屏蔽外部信息
  -> dp_actor.update_policy 用 PPO clipped loss 更新 actor
```

这条链路能讲清楚，你就不是“看过项目”，而是真的能从源码角度理解 StepSearch。

