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


### 模型训练

```shell
fastie-cli train gplinker.yaml
```


### 模型评估

```shell
python evaluate.py \
    --eval_file datasets/duee/dev.json \
    --model_name_or_path outputs/bert-gplinker-event \
    --device cuda:0
```

### 目前支持的模型

```python
from fastie import print_supported_models

print_supported_models("event")
```
