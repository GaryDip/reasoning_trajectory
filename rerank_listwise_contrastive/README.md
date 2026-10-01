# rerank_listwise_contrastive — 方案 B：真实 top-k 候选池 + listwise 对比学习

对应 `update_doc/0713/0713update.md`"重排结构"讨论里的方案 B。打分模型的结构跟方案 A(`rerank_pairwise_mlp`)
一样——独立对每个候选算 `[emb_score, PCA(Δ)] → MLP → 分数`——**真正不同的是训练数据和训练目标**：
不是"gold vs 一个人工采样的 distractor"这种一对一比较，而是"gold 要在**真实检索出来的整个 top-k
候选池**里胜出"，用 listwise softmax 交叉熵训练，更贴近实际推理时的场景(要在真实召回的一堆候选里
把 gold 排到第一，不是只跟一个挑出来的对手比)。

## 流程(四步，比方案 A 多一步数据构造)

1. **`build_topk_pools.py`** — 纯 BGE，不需要 Llama。对每一跳，用 gold 前置答案展开子问题(跟
   `reader_ceiling_probe`/方案 A 的 eval 一样的 oracle-prior 隔离原则)，跑一次真实 top-k 检索，
   标出 gold 候选在不在池子里、排第几。这一步是方案 A 没有的——现成 `hidden_states/` 数据只有
   "一个 gold + 几个人工采样的 distractor"，不是真实检索池。
   ```bash
   python build_topk_pools.py --split train --retrieve-k 20
   python build_topk_pools.py --split dev --retrieve-k 20
   ```
2. **`extract_pool_hidden_states.py`** — 对池子里**每一个**候选抽 Δ(复用
   `run_retrieval_exp_wavefront.py::batch_last_hidden`，跟 gate 打分用的是同一个底层函数)。**这一步
   需要真实 GPU + Llama checkpoint**，是整个方案 B 里唯一没法在没有 GPU 的环境里跑通真实数据的一步。
   ```bash
   python extract_pool_hidden_states.py --pools data/musique_train_pools.jsonl --layer 31
   ```
3. **`train_listwise_ranker.py`** — 按 `pool_id` 分组，池子里没有 gold 的直接跳过(没有监督信号)，
   其余用 softmax 交叉熵训练(gold 是 k 选 1 的正确类别)。打印 train/dev 的 top-1 accuracy(排第一的
   是不是 gold)，对比"纯余弦排序"基线。
   ```bash
   python train_listwise_ranker.py --train-npz data/musique_train_pools.hidden.npz \
       --dev-npz data/musique_dev_pools.hidden.npz
   ```
4. **`eval_retrieval.py`** — 跟方案 A 的评估脚本结构完全一样(recall@1/@3/MRR，oracle prior 隔离)，
   只是加载这里训出来的 artifacts。同样需要真实 GPU + Llama。
   ```bash
   python eval_retrieval.py --artifacts-dir artifacts --limit 100
   ```
5. **`wavefront_listwise_contrastive.py`** — 跟方案 A 的 `wavefront_pairwise_mlp.py` 结构完全一样：
   跟生产 `run_retrieval_exp_wavefront.py` 同一套评估方法论(真实 vLLM 选段、真实中间/最终答案生成、
   同一套 accumulator 类)，输出格式跟 `gated_rule_a`/`lr_rerank`/beam search 直接可比，唯一区别是
   重排公式换成这里训出来的 listwise MLP 分数。需要真实 GPU + Llama + vLLM。
   ```bash
   python wavefront_listwise_contrastive.py --limit 100
   ```

## 已验证的部分

`build_topk_pools.py` 用真实数据跑通(BGE 检索、gold 覆盖率统计都正常)。`extract_pool_hidden_states.py`/
`train_listwise_ranker.py`/`eval_retrieval.py`/`wavefront_listwise_contrastive.py` 用**假的 Llama
模型 + 假的 vLLM 生成器(真实检索/池子结构/accumulator 逻辑)**验证过整条链路(分组、listwise loss、
accuracy 统计、artifacts 保存/加载、完整三跳+最终答案生成)不报错，真实的 Δ 和最终数字需要在有
GPU 的环境里跑第 2 步之后才有意义。

## 不改动的部分

只读 `hidden_states/`、`traces/`；不碰 `gate/`、`retrieval/` 里任何生产代码。
