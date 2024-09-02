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


### 事件抽取

| 模型       | 论文  | 备注                                                                    |
|----------|-----|-----------------------------------------------------------------------|
| gplinker |     | [GPLinker：基于GlobalPointer的事件联合抽取](https://spaces.ac.cn/archives/8926) |
