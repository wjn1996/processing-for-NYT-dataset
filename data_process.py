#
#数据集加载与预处理
#（1）NYT数据集
#
#
#

import re
import numpy as np
import json
import ast
import nltk
from configure import FLAGS


dataset_dir = './' # 原始数据集的根目录
dataser_save = './data_save/' # 预处理后保存的二进制文件


# 字符串预处理
def clean_str(text):
    text = text.lower()
    # Clean the text
    text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"that's", "that is ", text)
    text = re.sub(r"there's", "there is ", text)
    text = re.sub(r"it's", "it is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "can not ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r",", " ", text)
    text = re.sub(r"\.", " ", text)
    text = re.sub(r"!", " ! ", text)
    text = re.sub(r"\/", " ", text)
    text = re.sub(r"\^", " ^ ", text)
    text = re.sub(r"\+", " + ", text)
    text = re.sub(r"\-", " - ", text)
    text = re.sub(r"\=", " = ", text)
    text = re.sub(r"'", " ", text)
    text = re.sub(r"(\d+)(k)", r"\g<1>000", text)
    text = re.sub(r":", " : ", text)
    text = re.sub(r" e g ", " eg ", text)
    text = re.sub(r" b g ", " bg ", text)
    text = re.sub(r" u s ", " american ", text)
    text = re.sub(r"\0s", "0", text)
    text = re.sub(r" 9 11 ", "911", text)
    text = re.sub(r"e - mail", "email", text)
    text = re.sub(r"j k", "jk", text)
    text = re.sub(r"\s{2,}", " ", text)
    text = re.sub(r"<br >", "", text)
    text = re.sub(r"<br>", "", text)
    text = re.sub(r"<br />", "", text)

    return text.strip()


def openEntity2mid():
    '''
    额外实体文件
    读取entity2mid文件，获取每个实体及其对应的编号
    :return:
    '''
    with open(dataset_dir + 'entity2mid.json', 'r', encoding = 'utf-8') as fr:
        entity2id = fr.readlines()[0]
        entity2id = json.loads(entity2id)
    print(len(entity2id)) # 69040个实体
    return entity2id

def openEntityType():
    '''
    额外实体文件
    (1)读取entity_type文件，获取每个实体对应的所有类型
    (2)获取所有类型（一级类型和多级类型）
    :return:
    '''
    with open(dataset_dir + 'entity_type.json', 'r', encoding = 'utf-8') as fr:
        entityType = fr.readlines()[0]
        entityType = json.loads(entityType)
    one_layer_type = dict()
    multi_layer_type = dict()
    for i in dict(entityType).keys():
        types = entityType[i]
        for j in types:
            if j in multi_layer_type.keys():
                multi_layer_type[j] += 1
            else:
                multi_layer_type[j] = 0
            type_root = j.split("/")[1]
            if type_root in one_layer_type.keys():
                one_layer_type[type_root] += 1
            else:
                one_layer_type[type_root] = 0
    print('len(one_layer_type)=', len(one_layer_type))
    print('len(multi_layer_type)=', len(multi_layer_type))
    return entityType

def openWord2vec():
    '''
    读取词向量文件，生成词编号及每个词的向量

    :return:
    '''
    word2vec_str = ''
    with open(dataset_dir + 'word_vec.json', 'r', encoding = 'utf-8') as fr:
        num = 0
        while True:
            num += 1
            # print(num)
            readstr = fr.read(1024 * 1024)
            if not readstr:
                break
            word2vec_str += readstr
    word2vec_str = word2vec_str[1:-1].replace("}, {\"word\"", "}<SPLIT>{\"word\"").split('<SPLIT>')
    word2vec = dict()
    word2id = dict()
    word2id['null'] = 0
    word2id['<None>'] = 1
    word2vec[0] = [0.] * 50
    word2vec[1] = np.random.randn(50)
    for i in word2vec_str:
        word_vec = json.loads(i)
        word = word_vec['word'].replace(' ', '_').lower()
        vec = word_vec['vec']
        word2id[word] = len(word2id)
        word2vec[word2id[word]] = vec
    return word2id, word2vec

def openRelation():
    with open('rel2id.json', 'r', encoding = 'utf-8') as fr:
        rel2id = fr.read()
    rel2id = json.loads(rel2id)
    return rel2id

def openSampleJson(kind, one_layer_type, entity2type):
    '''
    读取数据集文件（kind=train或test）
    :return:
    '''
    datasets = ''
    with open(dataset_dir + kind + '.json', 'r', encoding = 'utf-8') as fr:
        num = 0
        while True:
            num += 1
            # print(num)
            readstr = fr.read(1024*1024)
            if not readstr:
                break
            datasets += readstr
    datasets = datasets[1:-1].replace("}, {\"head\"", "}<SPLIT>{\"head\"").split("<SPLIT>")
    # datasets = ast.literal_eval(datasets)
    print('len(datasets)=', len(datasets))
    # print('datasets[2]=', datasets[0])
    # word2id, word2vec = openWord2vec()
    # rel2id = openRelation()
    def load_json(sample, pos):
        '''
        加载每个样本的两个实体及对应的类型、实体词语
        引入外部变量用于保存所有实体及其编号，和每个实体对应的所有类型
        返回当前样本的实体词语
        :param sample:
        :param pos:
        :return:
        '''
        nonlocal one_layer_type
        # nonlocal multi_layer_type
        nonlocal entity2type
        entity = sample[pos]
        entity_type = entity['type']
        type_set = set()
        for j in entity_type.split(','):
            # if j not in multi_layer_type.keys():
            #     multi_layer_type[j] = len(multi_layer_type)
            root = j.split('/')[1]
            type_set.add(root)
            if root not in one_layer_type.keys():
                one_layer_type[root] = len(one_layer_type)
        entity_word = entity['word'].replace(' ', '_').lower()
        if entity_word not in entity2type:
            entity2type[entity_word] = list(type_set)
        return entity_word


    #加载所有训练集，并进行bag打包
    bags = dict()
    for ei, i in enumerate(datasets):
        bag = dict()
        sample = json.loads(i)
        head_word = load_json(sample, 'head') #读取每个样本的两个实体及对应的类型，加载出所有类型并进行编号，同时保存每个实体对应的所有类型
        tail_word = load_json(sample, 'tail') #读取每个样本的两个实体及对应的类型，加载出所有类型并进行编号，同时保存每个实体对应的所有类型
        relation = sample['relation']
        sentence = sample['sentence']
        # print(sentence)
        key = head_word + '/-/' + tail_word + '/-/' + relation
        # print(key)
        if key not in bags.keys():
            bags[key] = [sentence]
        else:
            sents = bags[key]
            sents.append(sentence)
            bags[key] = sents

    print(one_layer_type)
    print('所有实体类型 len(one_layer_type)=', max(one_layer_type.values()))
    # print('所有实体类型 len(multi_layer_type)=', max(multi_layer_type.values()))
    type_size = dict()
    for i in entity2type.keys():
        size = len(entity2type[i])
        if size not in type_size.keys():
            type_size[size] = 0
        else:
            type_size[size] += 1
    print('类型数对应的实体数量:', type_size)

    sample_bags = []
    # 记录不同尺寸包对应的数量，例如bag_size[5]表示包内样布数量少于5的包的个数
    # bag_size = {'5':0, '15':0, '25':0, '35':0, '45':0, '55':0, '65':0, '75':0, '85':0, '86':0}
    for i in bags.keys():
        bag = dict()
        head = i.split('/-/')[0]
        tail = i.split('/-/')[1]
        relation = i.split('/-/')[2]
        bag['head'] = head.lower()
        bag['tail'] = tail.lower()
        bag['relation'] = relation
        bag['sentence'] = bags[i]
        # size = len(bags[i])
        # if size > 85:
        #     bag_size['86'] += 1
        # elif size > 75:
        #     bag_size['85'] += 1
        # elif size > 65:
        #     bag_size['75'] += 1
        # elif size > 55:
        #     bag_size['65'] += 1
        # elif size > 45:
        #     bag_size['55'] += 1
        # elif size > 35:
        #     bag_size['45'] += 1
        # elif size > 25:
        #     bag_size['35'] += 1
        # elif size > 15:
        #     bag_size['25'] += 1
        # elif size > 5:
        #     bag_size['15'] += 1
        # else:
        #     bag_size['5'] += 1
        sample_bags.append(bag)
    print("=========")
    # print(sample_bags[0:100])
    print("=========")
    print('包数量 len(sample_bags)=', len(sample_bags))
    # print('不同尺寸区间的包对应的数量: ', bag_size)

    return one_layer_type, entity2type, sample_bags


def process_sample(sample_bags, word2id, word2vec, rel2id, one_layer_type, entity2type, max_len):
    '''
    该部分根据之前生成的word2id，rel2id以及打包后的数据集，为每个句子更改为word id序列，并生成对应的position
    同时生成每个实体对的type矩阵

    type矩阵：
    例如假设有五个类型，对应编号为[0,1,2,3,4]，实体对(A,B)中实体分别有类型[2,4]和[0,1,4]，则该矩阵为
    [[ 0 0 1 0 1 ]
     [ 0 0 1 0 1 ]
     [ 0 0 0 0 0 ]
     [ 0 0 0 0 0 ]
     [ 0 0 1 0 1 ]
     在计算注意力时候，为0的部分则权重自动为0，为1的部分则参与计算，例如最后实体A的权重可以为[0, 0, 0.7, 0 ,0.3]
     实体B的权重可以为[0.1, 0.7, 0, 0, 0.2]

    :param sample_bags: 已打包的原始数据集（包含原始类型，原始句子， 需要转换为对应的编号序列）
    格式：[{'head':'', 'tail': '', 'relation': '', 'sentence': ['', '', ...]}, ...]
    :return:
    '''
    max_sentence_length = 0

    def suppleWord2vec(entity):
        '''
        针对数据集中出现的词组型实体，若其不在现有的word2vec中，则将对应的所有单词的词向量进行平均，
        作为该词组的词向量，并保存在对应的word2id和word2vec
        :return:
        '''
        nonlocal word2id
        nonlocal word2vec

        if entity not in word2id.keys():
            words = entity.split('_')
            vec = []
            if len(words) > 1:
                for i in words:
                    if i not in word2id.keys():
                        word2id[i] = len(word2id)
                        word2vec[word2id[i]] = np.random.randn(50)
                    vec.append(word2vec[word2id[i]])
                vec = np.average(vec, axis=0)
                word2id[entity] = len(word2id)
                word2vec[word2id[entity]] = vec
            else:
                word2id[entity] = len(word2id)
                word2vec[word2id[entity]] = np.random.randn(50)

    sample_bags_new = []
    for ei, i in enumerate(sample_bags):
        if (ei + 1) % 1000 == 0:
            print('has finished:', ei + 1, '( total:', len(sample_bags), ')')
        sample_bag_new = dict()
        ####关系数值化
        relation = i['relation']
        relation_id = rel2id[relation] #每个包对应实体关系的编号
        ####句子/位置数值化
        sentences = i['sentence']
        head = i['head']
        head_s = head.replace(' ', '_').lower()
        tail = i['tail']
        tail_s = tail.replace(' ', '_').lower()
        suppleWord2vec(head_s)
        suppleWord2vec(tail_s)
        sentence_wids = [] #每个包中所有句子中单词对应的编号，并根据max_len进行padding
        sentence_poss1 = [] #每个包中所有句子中单词相对于实体1的位置，并根据max_len进行padding
        sentence_poss2 = []  # 每个包中所有句子中单词相对于实2体的位置，并根据max_len进行padding
        for j in sentences:
            # sentence = clean_str(j)
            sentence = j.lower()
            sentence_s = sentence.replace(head, head_s).replace(tail, tail_s)
            tokens_s = nltk.word_tokenize(sentence_s)
            # if len(tokens_s) > FLAGS.max_sentence_length:
            #     continue
            # tokens_s = sentence_s.split(' ')
            head_pos = 0
            tail_pos = 0
            h_pos = []
            t_pos = []
            for k in range(len(tokens_s)):
                if tokens_s[k] == head_s:
                    head_pos = k
                if tokens_s[k] == tail_s:
                    tail_pos = k
            # if head_pos == 0 or tail_pos == 0:
            #     print('entity=', head_s, '(', head, ')', ',', tail_s, '(', tail, ')', ',token_s=', tokens_s, '\nsentence_s=', sentence_s)
            # print('head_pos=', head_pos, ';tail_pos=', tail_pos, '\ntokens_s=', tokens_s)
            for k in range(len(tokens_s)):
                h_p = [abs(k - head_pos) + 1]
                t_p = [abs(k - tail_pos) + 1]
                # if h_p == 1:
                #     h_p = h_p * len(head_s.split('_'))
                # if t_p == 1:
                #     t_p = t_p * len(tail_s.split('_'))
                h_pos += h_p
                t_pos += t_p
            # h_pos += [0] * (max_len - len(h_pos))
            # t_pos += [0] * (max_len - len(t_pos))

            sentence_wid = []
            for k in tokens_s:
                if k in word2id.keys():
                    sentence_wid.append(word2id[k])
                else:
                    sentence_wid.append(word2id['<None>'])
            if max_sentence_length < len(sentence_wid):
                max_sentence_length = len(sentence_wid)
            if max_len <= len(sentence_wid):
                sentence_wid = sentence_wid[0:max_len]
                h_pos = h_pos[0:max_len]
                t_pos = t_pos[0:max_len]
            else:
                sentence_wid += [0] * (max_len - len(sentence_wid))  # padding
                h_pos += [0] * (max_len - len(h_pos)) # padding
                t_pos += [0] * (max_len - len(t_pos)) # padding

            sentence_poss1.append(h_pos)
            sentence_poss2.append(t_pos)
            sentence_wids.append(sentence_wid)
        ####实体对类矩阵
        # T = [[0.]*(len(one_layer_type))]*(len(one_layer_type))
        # for j in entity2type[head]:
        #     for k in entity2type[tail]:
        #         T[one_layer_type[j]][one_layer_type[k]] = 1.
        ###实体对类编号
        T = []
        for j in entity2type[head]:
            for k in entity2type[tail]:
                T.append([one_layer_type[j], one_layer_type[k]])
        sample_bag_new['head'] = word2id[head_s]
        sample_bag_new['tail'] = word2id[tail_s]
        sample_bag_new['sentence'] = sentence_wids
        sample_bag_new['position_head'] = sentence_poss1
        sample_bag_new['position_tail'] = sentence_poss2
        sample_bag_new['T'] = T
        sample_bag_new['relation'] = relation_id
        sample_bags_new.append(sample_bag_new)
    print('len(sample_bags_new)=', len(sample_bags_new))
    # print(sample_bags_new[0:20])
    print('最长：', max_sentence_length)
    return sample_bags_new, word2id, word2vec


one_layer_type = dict() #一级实体类型（只取第一个根）
one_layer_type['none'] = 0 #设置未知类型
# multi_layer_type = dict() #多级实体类型（保留整个路径）
entity2type = dict() #保存训练集中出现的所有实体及对应的所有类型（字符串数组）
print('开始预处理NYT数据集...')
print('获取词向量')
word2id, word2vec = openWord2vec()
rel2id = openRelation()
print('完成生成word2id和wordvec\n开始获取训练语料和测试语料...')
one_layer_type, entity2type, train_sample_bags = openSampleJson('train', one_layer_type, entity2type)
print('完成训练语料读取')
one_layer_type, entity2type, test_sample_bags = openSampleJson('test', one_layer_type, entity2type)
print('完成测试语料读取\n开始对训练语料进行数值化...')
# #保存变量值，供data_loader使用
# np.savez('words', word2id = word2id, word2vec = word2vec)
# np.savez('relations', rel2id = rel2id)
# np.savez('entitys', one_layer_type = one_layer_type, entity2type = entity2type)
# np.savez('samples', train_set = train_sample_bags, test_set = test_sample_bags)
train_sample_bags_new, word2id, word2vec = process_sample(train_sample_bags, word2id, word2vec, rel2id, one_layer_type, entity2type, max_len=120)
print('完成训练语料数值化处理\n开始对测试语料进行数值化...')
test_sample_bags_new, word2id, word2vec = process_sample(test_sample_bags, word2id, word2vec, rel2id, one_layer_type, entity2type, max_len=120)
print('完成测试语料数值化处理\n正在保存...')
np.savez('datasets3', train_set = train_sample_bags_new, test_set = test_sample_bags_new, rel2id = rel2id,
         word2id = word2id, word2vec = word2vec, one_layer_type = one_layer_type, entity2type = entity2type)
print('已完成所有数据预处理，请查看指定目录下文件.')

#####格式说明#####
'''
train_set/test_set:字典数据（每一个字典代表一个包）
[
    {'head': <word_id>, 'tail': <word_id>, 'sentence': [[<word_id>, ...], [...], ...],
    'position_head': [[4, 3, 2, 1, 1, 2, ...], [...], [...]],
    'position_tail': [[4, 3, 2, 1, 1, 2, ...], [...], [...]],
    'T': [[0., 0., 1., ...], ...]
    'relation': <rel2id>},
    ...
]
rel2id:字典
{'<relation_name>': <relation_id>}
word2id:字典
{'<word_name>': <word_id>}
word2vec:字典
{'word_id': <vector_50_dim>}
one_layer_type:字典
{'entity_type_name', <id>}
'''
