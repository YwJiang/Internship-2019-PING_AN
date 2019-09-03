## faq_demo 使用方法

### 框架介绍

faq_demo主要由analysis, retrieval, match，rank等部分组成：

* analysis 部分对用户输入的问题分词，生成 skip-thought 句向量。

* retrieval 部分根据用户输入的问题，elasticsearch 关键词召回和 annoy 语义召回相似问题。

* match 部分比较 retrieval 部分得到的相似问题与用户输入问题的相似评分。bm25, edit distance, jaccard similarity，abcnn 算法计算两个相似句子相似度。
* rank 部分根据 match 部分得到的相似评分，由 Lightgbm 训练得到 score 进行排序，返回前 top_n 问答对。





### 如何使用

1. 关于配置文件

* faq.config 配置文件
  - 若需要修改 skip-thought 生成 embedding 的维度：
    - 需要在 ./src/sentence_embedding/model.py 中修改thought_size，重新训练sentence_embedding 模型，并与 [skip_embedding] 的vec_dim 和 [annoy_search] 的 vec_dim 保持一致
  - 若需要重新训练模型，需修改 faq.config 相应的模型名称和参数。

2. 关于使用

   - 训练数据QA灌库

     - 问答对数据：格式为 question，answer，tab隔开
     - 数据需要放在faq_demo文件夹下

   - 运行make_index.py和make_annoy.py文件

     - ```python
       python ./make_index.py
       python ./make_annoy.py
       ```

   - 运行server.py文件

     - ```python ./src/sever.py``` 

3. 关于修改模型的细节

* 训练 skip-thought 的 sentence_embedding 模型
  * 训练数据 faq.txt 存放 FAQ 问题对，存于 ./src/sentence_embedding/data 文件夹
  
  * 训练sentence_embedding 模型：```python ./src/sentence_embedding/train.py``` 
  
  * 查看 sentence embedding：`python ./src/sentence_embedding/sentence_emb.py`
  
    
  
* 处理相似问题对数据
  * 相似问题对数据：header为sentence1， sentence2，label
  
    
  
* 训练abcnn模型
  * 将相似问题对数据分割为 train 数据和 validation 数据，分别命名为train.csv和dev.csv，放于faq_demo/src/abcnn/input文件夹下
  * 训练abcnn模型：```python ./src/abcnn/abcnn.py``` 
  * 注：abcnn模型会保存在faq_demo/src/abcnn/model文件夹。一个abcnn模型共有4个文件，若训练多次得到不同模型，最好分开存放在不同文件夹下，因为checkpoint文件训练新模型时会被覆盖。
  * 
  
* 训练lightgbm模型
  
  - bm25, edit distance, jaccard similarity, abcnn算法分别计算正负样本数据的语句对相似度，训练lightgbm模型，模型存为lightgbm_train_Model.pkl，放在faq_demo文件夹下





