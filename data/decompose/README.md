# data/decompose — 下游 pipeline 使用的分解文件

`decompose/gt/` 与 `decompose/bart/` 的产出，按用途整理后的统一入口。

## 布局

```
data/decompose/
├── musique/
│   ├── gt/
│   │   ├── train_nl.jsonl
│   │   └── dev_nl.jsonl
│   └── bart/
│       └── dev_nl.jsonl
├── 2wiki/bart/dev_nl.jsonl
└── hotpot/bart/dev_nl.jsonl
```

## 在代码里怎么选

```python
from traces.dataset_loaders import get_decompose_path

get_decompose_path("musique", "train", mode="gt")    # 构造 trace
get_decompose_path("musique", "dev", mode="bart")   # retrieval eval
```

## JSONL 格式

```json
{"id": "2hop__...", "question": "...", "sub_questions": ["...", "[Answer 1] ..."]}
```

源文件与生成脚本见 `decompose/`。
