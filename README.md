# FastIE

<p align="center">
    <a href="https://github.com/xusenlinzy/fastie"><img src="https://img.shields.io/github/license/xusenlinzy/fastie"></a>
    <a href=""><img src="https://img.shields.io/badge/python-3.8+-aff.svg"></a>
    <a href=""><img src="https://img.shields.io/badge/pytorch-%3E=2.0-red?logo=pytorch"></a>
    <a href="https://github.com/xusenlinzy/fastie"><img src="https://img.shields.io/github/last-commit/xusenlinzy/fastie"></a>
    <a href="https://github.com/xusenlinzy/fastie"><img src="https://img.shields.io/github/issues/xusenlinzy/fastie?color=9cc"></a>
    <a href="https://github.com/xusenlinzy/fastie"><img src="https://img.shields.io/github/stars/xusenlinzy/fastie?color=ccf"></a>
    <a href="https://github.com/xusenlinzy/fastie"><img src="https://img.shields.io/badge/langurage-py-brightgreen?style=flat&color=blue"></a>
</p>

此项目为开源**文本分类、实体抽取、关系抽取和事件抽取**[模型](MODELS.md)的训练和推理提供统一的框架，具有以下特性


+ ✨ 支持多种开源文本分类、实体抽取、关系抽取和事件抽取模型


+ 👑 支持百度 [UIE](https://github.com/PaddlePaddle/PaddleNLP) 模型的训练和推理


+ 🚀 统一的训练和推理框架


+ 🎯 集成对抗训练方法，简便易用


## 📢 更新日志 

+ 【2024.8.30】 发布初始版本


---

## 📦 快速安装

源码安装

```shell
git clone https://github.com/xusenlinzy/FastIE.git
pip install -e .
```

docker启动

```shell
docker build -t fastie .

docker compose up -d

docker exec -it fastie bash

```


## 🚀 模型训练

### 实体抽取

```shell
cd examples/named_entity_recognition
fastie-cli train global_pointer.yaml
```

具体参数详见 [named_entity_recognition](./examples/named_entity_recognition)

### 关系抽取

```shell
cd examples/relation_extraction
fastie-cli train gplinker.yaml
```

具体参数详见 [relation_extraction](./examples/relation_extraction)


### 事件抽取

```shell
cd examples/event_extraction
fastie-cli train gplinker.yaml
```

具体参数详见 [event_extraction](./examples/event_extraction)


## 📊 模型推理

本项目实现了对各类模型推理代码的封装，只需要4行代码即可推理！

```python
from transformers import AutoModel, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("path_to_model", trust_remote_code=True)
model = AutoModel.from_pretrained("path_to_model", trust_remote_code=True)

print(model.predict(tokenizer, "因肺过度充气，常将肝脏推向下方。"))
```

一键启动模型接口或DEMO

```shell
# fastie-cli api --model_name_or_path path_to_model --port 9000
fastie-cli demo --model_name_or_path path_to_model --port 9000 --device cuda
```


## 致谢

本项目受益于 [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)，感谢以上诸位作者的付出。
