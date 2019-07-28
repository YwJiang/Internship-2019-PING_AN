# Similarity between two sentences
After retrieving several similar questions for one query question, we care about which question is most close to the real query question. We need distance or similarity function to measure how close each question we find to the real query question. The following are three methods we used in faq_demo.

## 1. Okapi BM25 similarity

### 1.1 Formula

BM stands for best matching. It ranks a set of documents which are close to a query.  Given a query Q, containing keywords q1, q2, ..., qn, the BM25 score of a document D is:
$$
score(D, Q) = \sum_{i=1}^nIDF(q_i)\frac{f(q_i, D)*(k_1+1)}{f(q_i, D)+k_1*(1-b+b*\frac{\left|D\right|} {avgdl})}
$$
where $ f(q_i, D) $  is $ q_i 's $ term frequency in the document D,  $\left|D\right|$ is the length of the document D in words, which is the total number of words in document, and avgdl is the average document length in the text collection. $IDF(q_i)$ is the IDF weight of the query term $q_i$, which is usually computed as:

$$ IDF(q_i) = log\frac{N-n(q_i)+0.5}{n(q_i)+0.5}$$ 

where N is the total number of documents in the collection, and $n(q_i)$ is the number of documents containing $q_i$

### 1.2 Range

IDF can be negative and positive. BM25 score could range from negative infinite to the positive infinite. Hence, we can take the normalization when necessary.

* $\frac{1}{1+exp(-score)}$

### 1.3 Code

```python
def bm25_similarity(doc1_list, candits, k=1.5, b=0.75, avgl=12):
    # doc1: string sentence
    # candits: list of strings

    doc1 = doc1_list

    # 统计candits中的单词：频数; 存单个句子的词频
    dic_candits = {} # 每个词在几个句子里出现
    len_candits = len(candits)
    count_dic = [] # 存放每个句子长度
    list_dic_candits = []  # 单个句子的词频字典为元素
    for question in candits: # question是list，元素是tokens
        count_dic.append(len(question))  # 每个单词的数量
        sentence_dic = {}
        for word in question:
            sentence_dic[word] = sentence_dic.get(word, 0) + 1
        for word in set(question): # 为idf计算用，每个单词在几个句子里出现
            dic_candits[word] = dic_candits.get(word, 0) + 1
        list_dic_candits.append(sentence_dic)

    # 计算dic_candits中每个词的idf
    idf = {}
    for word, freq in dic_candits.items():
        idf[word] = math.log(len_candits - freq + 0.5) - math.log(freq + 0.5)

    # define the score
    score_result = []
    for i in range(len(list_dic_candits)):
        score = 0
        for word in doc1:
            idf_word = idf.get(word, 0)
            # print("idf", word, ":", idf_word)
            score += idf_word * (list_dic_candits[i].get(word, 0)/count_dic[i]) * (k + 1) / ((list_dic_candits[i].get(word, 0)/count_dic[i]) + k * (1 - b + b * len(doc1) / avgl)+1)
        score = 1.0 / (1 + math.exp(-score))
        score_result.append(score)
    return score_result


```



## 2. Edit distance

### 2.1 Formula

Edit distance would quantify how dissimilar two strings are to one another by counting the minimum number of operations required to transform one string into the other. There are three types of edit operations:

* Insertion
* Deletion
* Substitution

Assume we have word "ivan1" and word "ivan2".  First, we create a two dimension matrix, with the dimension of (m+1)*(n+1), where m is the number of letters of word 1 and n is the number of letters of word 2. The first column is a list of numbers from 0 to m, and the first row is a list of numbers from 0 to n. 

Second, we calculate the elements of the matrix by the following rules:

*  temp variable records whether two strings are equal to each other, temp = 0 if str[i] = str[j] and temp = 1 otherwise.
* d[i-1, j] + 1 represents the insertion manipulation
* d[i, j-1] +1 represents the deletion manipulation
* d[i-1, j-1] + temp represents the substitution manipulation

For each element from left top to right down, we calculate the d[i, j] according to the d[i-1, j-1] by choosing the minimum of {d[i-1, j] + 1, d[i, j-1] +1, d[i-1, j-1] + temp}

Finally, the right down corner value is the distance between two words. The larger the value is, the more different these two words are. The similarity would be normalized by $ 1- \frac{matrix(right down corner)}{max(word1.len, word2.len)}$, which is uniform to our common sense: 0 means these two words are not similar to each other and 1 means these two words are the same.

### 2.2 Range

Edit distance has its normalization method, which allows the edit distance range from 0 to 1. The larger the edit distance value is, the further the words are different from each other.

### 2.3 Code

```python 
def edit_similarity(v1, v2):
    # vec1, vec2: vector
    # 0-1 the bigger, the closer
    if len(v1) == 0:
        return 1 - len(v2) / max(len(v1), len(v2))
    if len(v2) == 0:
        return 1 - len(v1) / max(len(v1), len(v2))
    matrix = np.zeros((len(v1) + 1, len(v2) + 1))
    matrix[0, :] = range(0, len(v2) + 1)  # first row
    matrix[:, 0] = range(0, len(v1) + 1)  # first column

    for i in range(1, len(v1) + 1):
        for j in range(1, len(v2) + 1):
            temp = 0 if v1[i - 1] == v2[j - 1] else 1
            matrix[i, j] = min(matrix[i - 1, j] + 1, matrix[i, j - 1] + 1, matrix[i - 1, j - 1] + temp)
    return 1 - matrix[len(v1), len(v2)] / max(len(v1), len(v2))

```



## 3. Jaccard similarity

### 3.1 Formula

It measures the similarity between finite sample sets by calculating the intersection of two sets divided by the size of the union of the sets.

$$ J(A, B)  = \frac{ \left |A \bigcap B\right|}{ \left |A \bigcup B\right|} $$

dis(A, B) = 1-J(A, B)

### 3.2 Range

It will be range from 0 to 1, the larger the value is, the further the two sets are from each other.

### 3.3 Code

```python 
def jaccard_similarity(list1, list2):
    intersection_res = list(set(list1).intersection(set(list2)))
    union_res = list(set(list1).union(set(list2)))
    sim = len(intersection_res) * 1.0 / (len(union_res) + 1e-9)
    return sim
```

