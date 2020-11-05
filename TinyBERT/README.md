TinyBERT
======== 
TinyBERT比BERT-base 小7.5倍，而推理速度则快9.4倍，并且在自然语言理解任务中取得了出色的性能。它在预训练和特定任务的学习阶段都进行了新颖的transformer蒸馏。 TinyBERT学习的概述如下所示：
<br />
<br />
<img src="tinybert_overview.png" width="800" height="210"/>
<br />
<br />

有关TinyBERT技术的更多详细信息，请参阅我们的论文：

[TinyBERT: Distilling BERT for Natural Language Understanding](https://arxiv.org/abs/1909.10351)


Release Notes
=============
First version: 2019/11/26

Installation
============
运行以下命令以安装环境 (**using python3**)
```bash
pip install -r requirements.txt
```

通用形式蒸馏 
====================
在通用形式蒸馏中，我们使用原始的BERT库而不进行微调作为teacher，并使用大型文本语料库作为学习数据。通过对来自一般领域的文本进行Transformer蒸馏，我们获得了一个常规的TinyBERT，它为特定于任务的蒸馏提供了良好的初始化。

通用形式蒸馏有两个步骤：(1)生成json格式的语料库； (2)运行transformer蒸馏；

步骤1：使用`pregenerate_training_data.py`生成json格式的语料库

```
 
# ${BERT_BASE_DIR}$ 包含 BERT-base teacher 模型.
 
python pregenerate_training_data.py --train_corpus ${CORPUS_RAW} \ 
                  --bert_model ${BERT_BASE_DIR}$ \
                  --reduce_memory --do_lower_case \
                  --epochs_to_generate 3 \
                  --output_dir ${CORPUS_JSON_DIR}$ 
                             
```

Step 2: 使用 `general_distill.py` 运行通用形式蒸馏
```
 # ${STUDENT_CONFIG_DIR}$ 包含student_model的配置文件.
 
python general_distill.py --pregenerated_data ${CORPUS_JSON}$ \ 
                          --teacher_model ${BERT_BASE}$ \
                          --student_model ${STUDENT_CONFIG_DIR}$ \
                          --reduce_memory --do_lower_case \
                          --train_batch_size 256 \
                          --output_dir ${GENERAL_TINYBERT_DIR}$ 
```


我们在此处提供通用TinyBERT的模型，用户可以跳过通用形式蒸馏。

=================第一个版本可在论文中重现我们的结果 ===========================

[General_TinyBERT(4layer-312dim)](https://drive.google.com/uc?export=download&id=1dDigD7QBv1BmE6pWU71pFYPgovvEqOOj) 

[General_TinyBERT(6layer-768dim)](https://drive.google.com/uc?export=download&id=1wXWR00EHK-Eb7pbyw0VP234i2JTnjJ-x)

=================第二版(2019/11/18)受过更多(book+wiki)的训练，没有`[MASK]`语料 =======

[General_TinyBERT_v2(4layer-312dim)](https://drive.google.com/open?id=1PhI73thKoLU2iliasJmlQXBav3v33-8z)

[General_TinyBERT_v2(6layer-768dim)](https://drive.google.com/open?id=1r2bmEsQe4jUBrzJknnNaBJQDgiRKmQjF)


数据增强
=================
数据增强旨在扩展特定任务的训练集。通过学习更多与任务相关的样本，可以进一步提高student模型的泛化能力。
我们结合了预训练的语言模型BERT和GloVe嵌入来进行单词级替换以增强数据。

使用 `data_augmentation.py`  运行数据增强和扩充数据集， 
`train_aug.tsv` 自动保存到 ${GLUE_DIR/TASK_NAME}$
```

python data_augmentation.py --pretrained_bert_model ${BERT_BASE_DIR}$ \
                            --glove_embs ${GLOVE_EMB}$ \
                            --glue_dir ${GLUE_DIR}$ \  
                            --task_name ${TASK_NAME}$

```
在运行GLUE任务的数据增强之前，您应该下载  [GLUE data](https://gluebenchmark.com/tasks) 
通过运行 [this script](https://gist.github.com/W4ngatang/60c2bdb54d156a41194446737ce03e2e) 并将其解压缩到某个目录GLUE_DIR。
并且TASK_NAME可以是以下之一 CoLA, SST-2, MRPC, STS-B, QQP, MNLI, QNLI, RTE.

特定任务蒸馏
==========================
在特定任务的蒸馏中，我们通过专注于学习特定任务的知识来重新执行提议的transformer蒸馏，以进一步改善TinyBERT。
特定任务的蒸馏包括两个步骤：(1)中间层蒸馏； (2)预测层蒸馏。

Step 1: 使用task_distill.py运行中间层蒸馏。
```

# ${FT_BERT_BASE_DIR}$ 包含 fine-tuned BERT-base model.

python task_distill.py --teacher_model ${FT_BERT_BASE_DIR}$ \
                       --student_model ${GENERAL_TINYBERT_DIR}$ \
                       --data_dir ${TASK_DIR}$ \
                       --task_name ${TASK_NAME}$ \ 
                       --output_dir ${TMP_TINYBERT_DIR}$ \
                       --max_seq_length 128 \
                       --train_batch_size 32 \
                       --num_train_epochs 10 \
                       --aug_train \
                       --do_lower_case  
                         
```


Step 2: 使用task_distill.py运行预测层蒸馏。
```

python task_distill.py --pred_distill  \
                       --teacher_model ${FT_BERT_BASE_DIR}$ \
                       --student_model ${TMP_TINYBERT_DIR}$ \
                       --data_dir ${TASK_DIR}$ \
                       --task_name ${TASK_NAME}$ \
                       --output_dir ${TINYBERT_DIR}$ \
                       --aug_train  \  
                       --do_lower_case \
                       --learning_rate 3e-5  \
                       --num_train_epochs  3  \
                       --eval_step 100 \
                       --max_seq_length 128 \
                       --train_batch_size 32 
                       
```

我们在这里还提供了所有GLUE任务的蒸馏的TinyBERT(4layer-312dim和6layer-768dim)以供评估。
每个任务都有其自己的文件夹，其中已保存了相应的模型。

[TinyBERT(4layer-312dim)](https://drive.google.com/uc?export=download&id=1_sCARNCgOZZFiWTSgNbE7viW_G5vIXYg) 

[TinyBERT(6layer-768dim)](https://drive.google.com/uc?export=download&id=1Vf0ZnMhtZFUE0XoD3hTXc6QtHwKr_PwS)


评估
==========================
运行`task_distill.py`以下命令来评估：

```
${TINYBERT_DIR}$ 包括配置文件，student模型和vocab文件。

python task_distill.py --do_eval \
                       --student_model ${TINYBERT_DIR}$ \
                       --data_dir ${TASK_DIR}$ \
                       --task_name ${TASK_NAME}$ \
                       --output_dir ${OUTPUT_DIR}$ \
                       --do_lower_case \
                       --eval_batch_size 32 \
                       --max_seq_length 128  
                                   
```

To Dos
=========================
* 对TinyBERT的中文任务进行评估。
* Tiny *：使用NEZHA或ALBERT作为TinyBERT学习的teacher。
* 发行更好的通用的TinyBERT。
