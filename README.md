# Pretrained Language Model
该repository提供了由华为诺亚方舟实验室开发的最新预训练语言模型及其相关的优化技术。

## 目录结构
* [NEZHA-TensorFlow](https://github.com/huawei-noah/Pretrained-Language-Model/tree/master/NEZHA-TensorFlow) 是预训练的中文语言模型，可在TensorFlow开发的多个中文NLP任务上实现最先进的性能。
* [NEZHA-PyTorch](https://github.com/huawei-noah/Pretrained-Language-Model/tree/master/NEZHA-PyTorch) 是NEZHA的PyTorch版本。
* [NEZHA-Gen-TensorFlow](https://github.com/huawei-noah/Pretrained-Language-Model/tree/master/NEZHA-Gen-TensorFlow) 提供了两种GPT模型。一个是中国古典诗歌生成模型乐府，另一个是中国通用的GPT模型。 
* [TinyBERT](https://github.com/huawei-noah/Pretrained-Language-Model/tree/master/TinyBERT) 是一个压缩的BERT模型，压缩7.5倍，推理时快9.4倍。
* [TinyBERT-MindSpore](https://github.com/huawei-noah/Pretrained-Language-Model/tree/master/TinyBERT-MindSpore) 是TinyBERT的MindSpore版本。
* [DynaBERT](https://github.com/huawei-noah/Pretrained-Language-Model/tree/master/DynaBERT) 是具有自适应宽度和深度的动态BERT模型。
* [BBPE](https://github.com/huawei-noah/Pretrained-Language-Model/tree/master/BBPE) 提供了 byte-level单词表构建工具及其相应的tokenizer。
* [PMLM](https://github.com/huawei-noah/Pretrained-Language-Model/tree/master/PMLM) 是一种改进的预训练语言模型方法。经过训练，无需复杂的two-stream self-attention，PMLM可以看作是XLNet的简单近似。  