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


### 模型训练

```shell
fastie-cli train gplinker.yaml
```


### 模型评估

```shell
python evaluate.py \
    --eval_file datasets/duie/dev.json \
    --model_name_or_path outputs/bert-gplinker-relation \
    --device cuda:0
```

### 目前支持的模型

```python
from fastie import print_supported_models

print_supported_models("relation")
```
