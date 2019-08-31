## faq_demo 使用方法

### 框架介绍

faq_demo主要由analysis, retrieval, match， rank等部分组成：

- analysis部分将用户输入的问题进行分词，通过skip-thought方法解析为词向量。
- retrieval部分根据用户输入的问题，通过elasticsearch 关键词召回和annoy语义召回分别匹配十五个相似问题，并去除重复。
- match部分比较retrieval部分得到的相似问题与用户输入问题的相似评分。通过bm25, edit distance, jaccard similarity， abcnn方法定义两个相似句子相似度，再通过Lightgbm训练得到综合得分。
- rank部分根据match部分得到的相似评分进行排序，返回前五个评分最高的问答对





### 如何使用

- faq.config配置文件
  - 修改skip-thought生成的embedding的维度：
    - 需要在 faq_demo/src/sentence_embedding/model.py 里修改thought_size，重新训练skip-thought的sentence_embedding模型（参见下文）
    - 需要修改[skip_embedding]的vec_dim
    - 需要修改[annoy_search]的vec_dim
  - 若重新训练了模型，需要修改faq.config相应的模型名称和参数。
- 训练数据QA灌库
  - 问答对数据：按照QA文件格式存放，第一行为question answer用tab隔开，之后的每一行都是相互匹配的问题与答案（tab隔开）
  - 数据需要放在faq_demo文件夹下
- 运行make_index.py和make_annoy.py文件
  - 先make_index.py，再make_annoy.py
  - 需要在faq_demo文件夹路径下运行
- 训练skip-thought的sentence_embedding模型
  - 训练数据是faq.txt，每一行是FAQ的问题对，没有第一行的header，存放在faq_demo/src/sentence_embedding/data文件夹下
  - 运行faq_demo/src/sentence_embedding文件夹下的train.py，在sentence_embedding下直接运行，会读取faq.txt，模型输出在faq_demo/src/sentence_embedding中，命名为skip_best。如果需要修改输出模型名称可以修改train.py中 debug函数的save_loc。
  - 如果需要测试sentence_embedding模型，可以运行faq_demo/src/sentence_embedding中的sentence_emb.py文件。
- 处理相似问题对数据
  - 相似问题对数据：header为sentence1， sentence2，label。若数据全部为正样本，即，问句与其相似问句对，则需要为数据构造负样本，即，问句与不相似问句对。需要平衡的正负样本即最好正负样本各一半，最后加上标签label，1-表示相似，0-表示不相似。
  - 分割数据：40%样本数据用于abcnn模型训练，60%样本数据用于lightgbm模型训练
  - 数据无需放入faq_demo中
- 训练abcnn模型
  - 样本数据处理：将上一步中得到的40%样本数据中相似问题对位置互换，label不变，得到训练数据接在原训练数据后面。这样可以将用于abcnn的训练数据翻倍。
  - 分割数据：将翻倍后的训练数据按照一定比例（6：4）分割为train数据和validation数据，分别命名为train.csv和dev.csv。
  - 两个csv数据文件需要放入faq_demo/src/abcnn/input文件夹下
  - 训练abcnn模型：faq_demo/src/abcnn路径下运行abcnn.py，训练时需要确保：1.） `if __name__ == '__main__':`之后的train代码块可正常运行。2.）test.csv和dev.csv均在faq_demo/src/abcnn/input文件夹下。
  - 注： `if __name__ == '__main__':`之后的predict代码块仅用于开发者测试abcnn模型的结果以及后续使用abcnn预测相似距离有关，与模型训练无关，在训练模型时需要注掉。
  - abcnn模型会自动保存在faq_demo/src/abcnn/model文件夹下。一个abcnn模型共有4个文件，需要匹配好。若训练多次得到不同模型，最好分开存放在不同文件夹下，因为checkpoint文件训练新模型时会被覆盖。
- 训练lightgbm模型
  - 使用bm25, edit distance, jaccard similarity, abcnn分别计算正负样本数据的语句对相似度，保存为矩阵形式，并在最后一列加入label。矩阵的行表示一对语句对，列表示bm25, edit distance, jaccard similarity, abcnn的相似评分以及label。
  - 细节：
    - bm25, edit distance以及jaccard similarity需要在通过faq_demo/src/match.py定义的距离函数在外面写脚本计算距离矩阵。
    - abcnn可以直接faq_demo/src/abcnn路径下运行abcnn.py，需要保证：1.） `if __name__ == '__main__':`之后的predict代码块可正常运行。2.）需要预测的数据格式为sentence1, sentence2，csv文件，在faq_demo/src/abcnn/input文件夹下，保存最后输出的prd数据并在外面存一份，和上面计算的bm25, edit distance以及jaccard similarity距离矩阵合并。
    -  lightgbm模型也需要在外面调参数写脚本训练。
  - 使用相似得分矩阵训练lightgbm模型，模型存为lightgbm_train_Model.pkl。
  - 模型需要放在faq_demo文件夹下
- 运行server.py文件
  - 在terminal中，在faq_demo文件夹的路径下运行`python src/server.py`



