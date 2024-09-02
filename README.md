# FastIE

<p align="center">
    <a href="https://github.com/xusenlinzy/fastie"><img src="https://img.shields.io/github/license/xusenlinzy/fastie"></a>
    <a href=""><img src="https://img.shields.io/badge/python-3.8+-aff.svg"></a>
    <a href=""><img src="https://img.shields.io/badge/pytorch-%3E=1.12-red?logo=pytorch"></a>
    <a href="https://github.com/xusenlinzy/fastie"><img src="https://img.shields.io/github/last-commit/xusenlinzy/fastie"></a>
    <a href="https://github.com/xusenlinzy/fastie"><img src="https://img.shields.io/github/issues/xusenlinzy/fastie?color=9cc"></a>
    <a href="https://github.com/xusenlinzy/fastie"><img src="https://img.shields.io/github/stars/xusenlinzy/fastie?color=ccf"></a>
    <a href="https://github.com/xusenlinzy/fastie"><img src="https://img.shields.io/badge/langurage-py-brightgreen?style=flat&color=blue"></a>
</p>

此项目为开源**文本分类、实体抽取、关系抽取和事件抽取**模型的训练和推理提供统一的框架，具有以下特性


+ ✨ 支持多种开源文本分类、实体抽取、关系抽取和事件抽取模型


+ 👑 支持百度 [UIE](https://github.com/PaddlePaddle/PaddleNLP) 模型的训练和推理


+ 🚀 统一的训练和推理框架


+ 🎯 集成对抗训练方法，简便易用


## 📢 News 

+ 【2024.8.30】 发布初始版本


---

## 📦 安装


```shell
pip install --upgrade fastie
```

## 🐼 主要模型

### 实体抽取

| 模型             | 论文                                                                                                                                                                            | 备注                                                                                                                                            |
|----------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------|
| softmax        |                                                                                                                                                                               | 全连接层序列标注并使用 `BIO` 解码                                                                                                                          |
| crf            | [Conditional Random Fields: Probabilistic Models for Segmenting and Labeling Sequence Data](https://repository.upenn.edu/cgi/viewcontent.cgi?article=1162&context=cis_papers) | 全连接层+条件随机场，并使用 `BIO` 解码                                                                                                                       |
| cascade-crf    |                                                                                                                                                                               | 先预测实体再预测实体类型                                                                                                                                  |
| span           |                                                                                                                                                                               | 使用两个指针网络预测实体起始位置                                                                                                                              |
| global-pointer |                                                                                                                                                                               | [GlobalPointer：用统一的方式处理嵌套和非嵌套NER](https://spaces.ac.cn/archives/8373)、[Efficient GlobalPointer：少点参数，多点效果](https://spaces.ac.cn/archives/8877) |
| tplinker       | [TPLinker: Single-stage Joint Extraction of Entities and Relations Through Token Pair Linking.](https://aclanthology.org/2020.coling-main.138.pdf)                            | 将实体识别任务转换为表格填充问题                                                                                                                              |
| w2ner          | [Unified Named Entity Recognition as Word-Word Relation Classification.](https://arxiv.org/pdf/2112.10070.pdf)                                                                | 统一解决嵌套实体、不连续实体的抽取问题                                                                                                                           |
| cnn            | [An Embarrassingly Easy but Strong Baseline for Nested Named Entity Recognition.](https://arxiv.org/abs/2208.04534)                                                           | 改进 `W2NER` 方法，采用卷积网络提取实体内部token之间的关系                                                                                                          |


### 关系抽取

| 模型       | 论文                                                                                                                                                 | 备注                                                                  |
|----------|----------------------------------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------|
| casrel   | [A Novel Cascade Binary Tagging Framework for Relational Triple Extraction.](https://aclanthology.org/2020.acl-main.136.pdf)                       | 两阶段关系抽取，先抽取出句子中的主语，再通过指针网络抽取出主语对应的关系和宾语                             |
| tplinker | [TPLinker: Single-stage Joint Extraction of Entities and Relations Through Token Pair Linking.](https://aclanthology.org/2020.coling-main.138.pdf) | 将关系抽取问题转换为主语-宾语的首尾连接问题                                              |
| spn      | [Joint Entity and Relation Extraction with Set Prediction Networks.](http://xxx.itp.ac.cn/pdf/2011.01675v2)                                        | 将关系抽取问题转为为三元组的集合预测问题，采用 `Encoder-Decoder` 框架                        |
| prgc     | [PRGC: Potential Relation and Global Correspondence Based Joint Relational Triple Extraction.](https://aclanthology.org/2021.acl-long.486.pdf)     | 先抽取句子的潜在关系类型，然后对于给定关系抽取出对应的主语-宾语对，最后通过全局对齐模块过滤                      |
| pfn      | [A Partition Filter Network for Joint Entity and Relation Extraction.](https://aclanthology.org/2021.emnlp-main.17.pdf)                            | 采用类似  `LSTM`  的分区过滤机制，将隐藏层信息分成实体识别、关系识别和共享三部分，对与不同的任务利用不同的信息        |
| grte     | [A Novel Global Feature-Oriented Relational Triple Extraction Model based on Table Filling.](https://aclanthology.org/2021.emnlp-main.208.pdf)     | 将关系抽取问题转换为单词对的分类问题，基于全局特征抽取模块循环优化单词对的向量表示                           |
| gplinker |                                                                                                                                                    | [GPLinker：基于GlobalPointer的实体关系联合抽取](https://kexue.fm/archives/8888) |


## 📚 数据

### 实体抽取

将数据集处理成以下 `json` 格式

```json
{
  "text": "结果上周六他们主场0：3惨败给了中游球队瓦拉多利德，近7个多月以来西甲首次输球。", 
  "entities": [
    {
      "id": 0, 
      "entity": "瓦拉多利德", 
      "start_offset": 20, 
      "end_offset": 25, 
      "label": "organization"
    }, 
    {
      "id": 1, 
      "entity": "西甲", 
      "start_offset": 33, 
      "end_offset": 35, 
      "label": "organization"
    }
  ]
}
```

字段含义：

+ `text`: 文本内容

+ `entities`: 该文本所包含的所有实体

    + `id`: 实体 `id`

    + `entity`: 实体名称
  
    + `start_offset`: 实体开始位置

    + `end_offset`: 实体结束位置的下一位

    + `label`: 实体类型


### 关系抽取

将数据集处理成以下 `json` 格式

```json
{
  "text": "查尔斯·阿兰基斯（Charles Aránguiz），1989年4月17日出生于智利圣地亚哥，智利职业足球运动员，司职中场，效力于德国足球甲级联赛勒沃库森足球俱乐部", 
  "spo_list": [
    {
      "predicate": "出生地",
      "object": "圣地亚哥", 
      "subject": "查尔斯·阿兰基斯"
    }, 
    {
      "predicate": "出生日期",
      "object": "1989年4月17日",
      "subject": "查尔斯·阿兰基斯"
    }
  ]
}
```

字段含义：

+ `text`: 文本内容

+ `spo_list`: 该文本所包含的所有关系三元组

    + `subject`: 主体名称

    + `object`: 客体名称
  
    + `predicate`: 主体和客体之间的关系


### 事件抽取

将数据集处理成以下 `json` 格式

```json
{
  "text": "油服巨头哈里伯顿裁员650人 因美国油气开采活动放缓",
  "id": "f2d936214dc2cb1b873a75ee29a30ec9",
  "event_list": [
    {
      "event_type": "组织关系-裁员",
      "trigger": "裁员",
      "trigger_start_index": 8,
      "arguments": [
        {
          "argument_start_index": 0,
          "role": "裁员方",
          "argument": "油服巨头哈里伯顿"
        },
        {
          "argument_start_index": 10,
          "role": "裁员人数",
          "argument": "650人"
        }
      ],
      "class": "组织关系"
    }
  ]
}
```

字段含义：

+ `text`: 文本内容

+ `event_list`: 该文本所包含的所有事件

    + `event_type`: 事件类型

    + `trigger`: 触发词
  
    + `trigger_start_index`: 触发词开始位置

    + `arguments`: 论元
  
        + `role`: 论元角色
      
        + `argument`: 论元名称
      
        + `argument_start_index`: 论元名称开始位置
  
## 🚀 模型训练

### 实体抽取

```shell
python -m fastie.cli examples/named_entity_recognition/global_pointer.yaml
```

训练脚本详见 [named_entity_recognition](./examples/named_entity_recognition)

### 关系抽取

```shell
python -m fastie.cli examples/named_entity_recognition/gplinker.yaml
```

训练脚本详见 [relation_extraction](./examples/relation_extraction)


### 事件抽取

```shell
python -m fastie.cli examples/event_extraction/gplinker.yaml
```

训练脚本详见 [event_extraction](./examples/event_extraction)


## 📊 模型推理

```python
from transformers import AutoModel, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("path_to_model", trust_remote_code=True)
model = AutoModel.from_pretrained("path_to_model", trust_remote_code=True)

print(model.predict(tokenizer, "因肺过度充气，常将肝脏推向下方。"))
```


## 🍭 通用信息抽取

+ [UIE(Universal Information Extraction)](https://arxiv.org/pdf/2203.12277.pdf)：Yaojie Lu等人在ACL-2022中提出了通用信息抽取统一框架 `UIE`。


+ 该框架实现了实体抽取、关系抽取、事件抽取、情感分析等任务的统一建模，并使得不同任务间具备良好的迁移和泛化能力。


+ [PaddleNLP](https://github.com/PaddlePaddle/PaddleNLP)借鉴该论文的方法，基于 `ERNIE 3.0` 知识增强预训练模型，训练并开源了首个中文通用信息抽取模型 `UIE`。


+ 该模型可以支持不限定行业领域和抽取目标的关键信息抽取，实现零样本快速冷启动，并具备优秀的小样本微调能力，快速适配特定的抽取目标。


### 模型训练

模型训练脚本详见 [uie](./examples/uie)

### 模型推理

<details>
<summary>👉 命名实体识别</summary>

```python
from transformers import AutoModel, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("xusenlin/uie-base", trust_remote_code=True)
model = AutoModel.from_pretrained("xusenlin/uie-base", trust_remote_code=True)

schema = ["时间", "选手", "赛事名称"]  # Define the schema for entity extraction
print(model.predict(tokenizer, "2月8日上午北京冬奥会自由式滑雪女子大跳台决赛中中国选手谷爱凌以188.25分获得金牌！", schema=schema))
```

output: 

```json
[
  {
    "时间": [
      {
        "end": 6,
        "probability": 0.98573786,
        "start": 0,
        "text": "2月8日上午"
      }
    ],
    "赛事名称": [
      {
        "end": 23,
        "probability": 0.8503085,
        "start": 6,
        "text": "北京冬奥会自由式滑雪女子大跳台决赛"
      }
    ],
    "选手": [
      {
        "end": 31,
        "probability": 0.8981544,
        "start": 28,
        "text": "谷爱凌"
      }
    ]
  }
]
```
</details>

<details>
<summary>👉 实体关系抽取</summary>

```python
schema = {'竞赛名称': ['主办方', '承办方', '已举办次数']}  # Define the schema for relation extraction
model.set_schema(schema)
print(model.predict(tokenizer, "2022语言与智能技术竞赛由中国中文信息学会和中国计算机学会联合主办，百度公司、中国中文信息学会评测工作委员会和中国计算机学会自然语言处理专委会承办，已连续举办4届，成为全球最热门的中文NLP赛事之一。"))
```

output:

```json
[
  {
    "竞赛名称": [
      {
        "end": 13,
        "probability": 0.78253937,
        "relations": {
          "主办方": [
            {
              "end": 22,
              "probability": 0.8421704,
              "start": 14,
              "text": "中国中文信息学会"
            },
            {
              "end": 30,
              "probability": 0.75807965,
              "start": 23,
              "text": "中国计算机学会"
            }
          ],
          "已举办次数": [
            {
              "end": 82,
              "probability": 0.4671307,
              "start": 80,
              "text": "4届"
            }
          ],
          "承办方": [
            {
              "end": 55,
              "probability": 0.700049,
              "start": 40,
              "text": "中国中文信息学会评测工作委员会"
            },
            {
              "end": 72,
              "probability": 0.61934763,
              "start": 56,
              "text": "中国计算机学会自然语言处理专委会"
            },
            {
              "end": 39,
              "probability": 0.8292698,
              "start": 35,
              "text": "百度公司"
            }
          ]
        },
        "start": 0,
        "text": "2022语言与智能技术竞赛"
      }
    ]
  }
]
```
</details>


<details>
<summary>👉  事件抽取</summary>

```python
schema = {'地震触发词': ['地震强度', '时间', '震中位置', '震源深度']}  # Define the schema for event extraction
model.set_schema(schema)
print(model.predict(tokenizer, "中国地震台网正式测定：5月16日06时08分在云南临沧市凤庆县(北纬24.34度，东经99.98度)发生3.5级地震，震源深度10千米。"))
```

output:

```json
[
  {
    "地震触发词": [
      {
        "end": 58,
        "probability": 0.99774253,
        "relations": {
          "地震强度": [
            {
              "end": 56,
              "probability": 0.9980802,
              "start": 52,
              "text": "3.5级"
            }
          ],
          "时间": [
            {
              "end": 22,
              "probability": 0.98533,
              "start": 11,
              "text": "5月16日06时08分"
            }
          ],
          "震中位置": [
            {
              "end": 50,
              "probability": 0.7874015,
              "start": 23,
              "text": "云南临沧市凤庆县(北纬24.34度，东经99.98度)"
            }
          ],
          "震源深度": [
            {
              "end": 67,
              "probability": 0.9937973,
              "start": 63,
              "text": "10千米"
            }
          ]
        },
        "start": 56,
        "text": "地震"
      }
    ]
  }
]
```
</details>

<details>
<summary>👉 评论观点抽取</summary>

```python
schema = {'评价维度': ['观点词', '情感倾向[正向，负向]']}  # Define the schema for opinion extraction
model.set_schema(schema)
print(model.predict(tokenizer, "店面干净，很清静，服务员服务热情，性价比很高，发现收银台有排队"))
```

output:

```json
[
  {
    "评价维度": [
      {
        "end": 20,
        "probability": 0.98170394,
        "relations": {
          "情感倾向[正向，负向]": [
            {
              "probability": 0.9966142773628235,
              "text": "正向"
            }
          ],
          "观点词": [
            {
              "end": 22,
              "probability": 0.95739645,
              "start": 21,
              "text": "高"
            }
          ]
        },
        "start": 17,
        "text": "性价比"
      },
      {
        "end": 2,
        "probability": 0.9696847,
        "relations": {
          "情感倾向[正向，负向]": [
            {
              "probability": 0.9982153177261353,
              "text": "正向"
            }
          ],
          "观点词": [
            {
              "end": 4,
              "probability": 0.9945317,
              "start": 2,
              "text": "干净"
            }
          ]
        },
        "start": 0,
        "text": "店面"
      }
    ]
  }
]
```
</details>


<details>
<summary>👉 情感分类</summary>


```python
schema = "情感倾向[正向，负向]"  # Define the schema for opinion extraction
model.set_schema(schema)
print(model.predict(tokenizer, "这个产品用起来真的很流畅，我非常喜欢"))
```

output:

```json
[
  {
    "情感倾向[正向，负向]": [
      {
        "probability": 0.9990023970603943,
        "text": "正向"
      }
    ]
  }
]
```
</details>
