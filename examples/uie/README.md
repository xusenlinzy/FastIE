## UIE

## 模型下载

```python
from fastie.models.uie.convert import convert_uie_checkpoint

convert_uie_checkpoint("uie-base", "uie_base_pytorch")
```

### 数据标注

我们推荐使用数据标注平台 `doccano` 进行数据标注，本示例也打通了从标注到训练的通道，即 `doccano` 导出数据后可通过 `doccano.py` 脚本轻松将数据转换为输入模型时需要的形式，实现无缝衔接。

标注完成后，使用下面命令将数据进行处理

```bash
python examples/uie/doccano.py \
     --doccano_file examples/uie/datasets/DuIE/doccano_ext.json \
    --save_dir examples/uie/datasets/DuIE 
```

参数说明：

+ `doccano_file`: 从 `doccano` 导出的数据标注文件


+ `save_dir`: 训练数据的保存目录，默认存储在 `data` 目录下


+ `negative_ratio`: 最大负例比例，该参数只对抽取类型任务有效，适当构造负例可提升模型效果。负例数量和实际的标签数量有关，最大负例数量 = `negative_ratio` * 正例数量。该参数只对训练集有效，默认为 `5`。为了保证评估指标的准确性，验证集和测试集默认构造全负例


+ `splits`: 划分数据集时训练集、验证集所占的比例。默认为 `[0.8, 0.1, 0.1]` 表示按照 `8:1:1` 的比例将数据划分为训练集、验证集和测试集


+ `task_type`: 选择任务类型，可选有抽取和分类两种类型的任务


+ `options`: 指定分类任务的类别标签，该参数只对分类类型任务有效。默认为 `["正向", "负向"]`


+ `prompt_prefix`: 声明分类任务的 `prompt` 前缀信息，该参数只对分类类型任务有效。默认为 "情感倾向"


+ `is_shuffle`: 是否对数据集进行随机打散，默认为 `True`


+ `seed`: 随机种子，默认为 `1000`


+ `separator`: 实体类别/评价维度与分类标签的分隔符，该参数只对实体/评价维度级分类任务有效。默认为 `"##"`


+ `schema_lang`: 选择 `schema` 的语言，可选有 `ch` 和 `en`。默认为 `ch`，英文数据集请选择 `en`


处理之后的数据示例如下

```json
{
  "content": "王国维，字静安，又字伯隅，号观堂",
  "result_list": [
    {
      "text": "观堂",
      "start": 14,
      "end": 16
    }
  ],
  "prompt": "王国维的号"
}
```

### 模型微调

```shell
fastie-cli train uie.yaml
```

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

