# rerank_pairwise_mlp — 方案 A：pairwise 排序 MLP

对应 `update_doc/0713/0713update.md`"重排结构"讨论里的方案 A：不再用现在 `emb_score - λ·abnormal_score`
这个固定加权公式，改成一个真正训出来的模型，输入 `[emb_score, PCA(Δ)]`，用 **pairwise 排序损失**
训练(同一跳的 gold vs 某个 distractor，让模型输出的分数满足 `score(gold) > score(distractor)`)，
而不是"分类：这个候选是不是 gold"。

## 流程(三步)

1. **`build_training_data.py`** — 从现成的 `hidden_states/{train,dev}/` 数据里配对：每条 counterfactual
   (neg) trace 在"错误发生的那一跳"跟它对应的 gold (pos) trace 是天然的一对(correct, wrong)——两者
   `source_id` 相同、`K` 相同、跳数 j 相同，只是这一跳的证据不一样。Δ 直接复用现成 `.npz` 里已经算好的
   (不重新抽)，只需要补算 `emb_score`(BGE 相似度，纯 CPU/GPU 皆可，不需要 Llama)。输出一个 `.npz`。
   ```bash
   python build_training_data.py --split train
   python build_training_data.py --split dev
   ```
2. **`train_pairwise_mlp.py`** — 在补好的特征上训 PCA(64) + 小 MLP(65→32→32→1)，pairwise logistic
   loss。会打印 train/dev 的 pairwise accuracy(模型分数是否把 gold 排在 wrong 前面)，并跟"只用
   emb_score 排序"的基线对比。纯 CPU 也能跑(数据量不大)。
   ```bash
   python train_pairwise_mlp.py --train-npz data/musique_train_pairwise.npz \
       --dev-npz data/musique_dev_pairwise.npz
   ```
3. **`eval_retrieval.py`** — 端到端检索层面评估(recall@1/@3/MRR，不是完整 EM/F1)：每一跳用 **gold
   前置跳答案**填充 `[Answer N]`(隔离"这个 reranker 结构选得准不准"这一个变量，不掺检索链条累积误差，
   参考 `reader_ceiling_probe` 用的同一个"oracle prior"原则)，top-20 BGE 检索 → 对每个候选现抽 Δ
   (复用 gate 那套 last-token hidden state 提取，layer 固定 31，因为训练数据来自哪一层就该用哪一层)
   → 用训好的 MLP 打分排序 → 跟"纯余弦排序"基线对比 recall@1/@3/MRR。这一步需要真实 GPU + Llama
   checkpoint(不需要 vLLM)。
   ```bash
   python eval_retrieval.py --artifacts-dir artifacts --limit 100
   ```
4. **`wavefront_pairwise_mlp.py`** — 跟 `eval_retrieval.py` 不同，这个是**跟生产 `run_retrieval_exp_wavefront.py`
   同一套评估方法论**的版本：真实 vLLM 选段(`prompt_select_passage_with_context`)、真实中间答案生成、
   真实最终答案生成，同一套 accumulator 类(`MetricAccum`/`ChainAccum`/`AnswerAccum` 等)，输出的
   `retrieval_exp_*.json` 是跟 `gated_rule_a`/`lr_rerank`/beam search 的结果**直接可比**的格式
   (recall@1/@3、chain_recall、answer_em/f1 都是同名字段)。跟生产脚本唯一的区别就是重排公式——
   不是 `emb_score - λ·abnormal_score`，是这里训出来的 MLP 分数直接排序。需要真实 GPU + Llama +
   vLLM(比 `eval_retrieval.py` 更重，但结果能直接放进 `update_doc/0713/0713update.md` 的对比表里)。
   ```bash
   python wavefront_pairwise_mlp.py --limit 100
   ```

## 已验证的部分

`build_training_data.py`(真实 BGE 编码 + 真实 hidden_states 数据)和 `train_pairwise_mlp.py`(真实
PyTorch 训练循环)都用真实数据跑通过：loss 下降、pairwise accuracy 上升、产出 `pca.joblib`/
`model.pt`/`meta.json` 格式正确。`eval_retrieval.py`/`wavefront_pairwise_mlp.py` 用假的 Llama 模型 +
假的 vLLM 生成器(真实检索、真实数据加载、真实 accumulator 逻辑)验证过整条链路结构没问题
(`wavefront_pairwise_mlp.py` 验证过完整跑通三跳 + 最终答案生成，输出 JSON 字段跟生产格式一致)，
真实的 recall/EM/F1 数字需要在有 GPU 的环境里跑。

## 不改动的部分

只读 `hidden_states/`、`traces/merged/`；不碰 `gate/`、`retrieval/` 里任何生产代码。
