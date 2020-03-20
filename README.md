# processing-for-NYT-dataset
This is the python procedures for preprocessing New York Times dataset which userd for distant supervision relation extraction. We also have a brief statistic on this dataset, we think it suffer from noisy and long-tail problems.

We have a long time for relation extraction research. In order to leverage the NYT dataset for experiment, we provide the processing procedures for combining some information of sentences and entities' types.

NYT dataset can download in [NYT（New York Times）Dataset for Distant Supervision Relation Extraction](https://download.csdn.net/download/qq_36426650/12258744)

# Outfile

run the py file as 
```
python3 data_process.py
```

You will get one npz file 'datasets.npz' after fiew minutes. Also, if you wang to know the dataset statistics, you can run 
```
pyhton3 statistic.py
```
And we list here:
- train_data

&emsp;&emsp;len(datasets)= 172448

&emsp;&emsp;bags number: len(sample_bags)= 280592

- test_data

&emsp;&emsp;len(datasets)= 172448

&emsp;&emsp;bags number: len(sample_bags)= 96867


# dataset stucture:

```
train_set/test_set:
[
    {'head': <word_id>, 'tail': <word_id>, 'sentence': [[<word_id>, ...], [...], ...],
    'position_head': [[4, 3, 2, 1, 1, 2, ...], [...], [...]],
    'position_tail': [[4, 3, 2, 1, 1, 2, ...], [...], [...]],
    'T': [[0., 0., 1., ...], ...]
    'relation': <rel2id>},
    ...
]
rel2id:
{'<relation_name>': <relation_id>}
word2id:
{'<word_name>': <word_id>}
word2vec:
{'word_id': <vector_50_dim>}
one_layer_type:
{'entity_type_name', <id>}
```
 
In train_set/test_set:
- head: denotes the head entity, the value is word2id;
- tail: denotes the tail entity, the value is word2id;
- sentence: denotes some corresponding expressed instances with the same entity pair (head,tail);
- position_head: denotes the relative position of the each word in one instance corresponding head entity;
- position_tail: denotes the relative position of the each word in one instance corresponding tail entity;
- T: denotes the entity pair type matrix, You can completely ignore;
- relation: the relation label index.
