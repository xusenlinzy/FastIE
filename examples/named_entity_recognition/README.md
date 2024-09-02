### 主要参数

+ `task_name`: 模型名称，必须在支持模型列表内


+ `model_name_or_path`: 开源模型的文件所在路径


+ `dataset_name`: `huggingface` 数据集名称或本地数据集文件所在路径


+ `train_file`: 训练集文件所在路径


+ `validation_file`: 验证集文件所在路径


+ `preprocessing_num_workers`: 多进程处理数据


+ `num_train_epochs`: 训练轮次


+ `per_device_train_batch_size`: 训练集批量大小


+ `per_device_eval_batch_size`: 验证集批量大小


+ `learning_rate`: 学习率


+ `other_learning_rate`: 差分学习率


+ `output_dir`: 模型保存路径



### 数据处理

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

### 模型训练

```shell
fastie-cli train global_pointer.yaml
```


### 模型评估

```shell
python evaluate.py \
    --eval_file datasets/cmeee/dev.json \
    --model_name_or_path outputs/bert-gp-ner \
    --device cuda:0
```

### 目前支持的模型

```python
from fastie import print_supported_models

print_supported_models("ner")
```
