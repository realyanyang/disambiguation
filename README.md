# Name Disambiguation

在 *OAG-WhoIsWho 赛道二*中取得验证集0.86299，测试集0.96405的成绩，分享本次比赛中的代码。

比赛[链接](https://www.biendata.com/competition/aminer2019_2/)

## 环境

- Ubuntu 1804
- python3.6
- 所使用到的python包及版本见requirements.txt

## 运行前目录结构

```
disambiguation/
├── channel2_v2.py
├── data2
│   ├── cna_data
│   └── train
├── final_dir
│   ├── data
│   ├── name.different.2.modified.json
│   └── name.different.modified.json
├── final_subpipe.py
├── glove.840B.300d.txt
├── README.md
├── requirements.txt
├── stack_model.py
├── train_model.py
├── train_triplet_model.py
├── triplet_model.py
└── utils.py
```

以上，data2目录存放训练数据，其中的cna_data目录存放cna_valid_pub.json、whole_author_profile.json、cna_valid_unass_competition.json、whole_author_profile_pub.json、valid_example_evaluation_continuous.json；train目录存放train_author.json、train_pub.json。final_dir目录主要存放测试数据以及结果，其中的data目录存放cna_test_pub.json 、cna_test_unass_competition.json、test_example_evaluation_continuous.json；name.different.modified.json以及name.different.2.modified.json是在预测过程中，分数较低的一些作者，我们检查发现其中一些作者名字没有匹配正确，故手动建立了部分对应关系。 glove.840B.300d.txt下载[链接](https://nlp.stanford.edu/projects/glove/)。

## 运行步骤

- 预处理

	`python3 channel2_v2.py`
  
- 训练triplet_model

	`python3 train_triplet_model.py`

- 训练预测模型
	
	`python3 train_model.py`
	
- 预测

	`python3 final_subpipe.py`
	
  
  
  `final_subpipe.py`文件中632行，`save_time(models[i])`for循环，建议通过手动更改i值跑多个py文件，可以节省时间。没有在代码中直接开多进程，是因为有的模型使用多进程预测，会发生冲突。

**Note:**  本代码中模型训练数据与当时使用的有所不同，因为当时进行了很多尝试，划分出了很多不同的训练数据，但只保留了最后划分时的代码，也已经遗忘当时划分数据的准则，故只是用同一份训练数据（大家可能奇怪有两个模型参数完全一样，即test-2-sm-191127-withsetinfo-sample11.pkl和sm-191127-withsetinfo-11.pkl，但当时训练时的训练数据其实是不同的），所以可能会有达不到提交时分数的可能性。



大家有兴趣可以下载当时训练好的模型

| 名称                      | 链接                                                         | MD5                              |
| ------------------------- | ------------------------------------------------------------ | -------------------------------- |
| trained-models.zip        | [百度网盘](https://pan.baidu.com/s/1Ne9Z3Q7eHFRcCT_FE09RiA)，[Google Drive](https://drive.google.com/open?id=1TAYUzBE9Lj7tLXGu3U5RhNHRrzYGlz8w) | 8b553624dfe6ca1d49dea0144f1e9aab |
| tm.title.1.checkpoint.pth | [百度网盘](https://pan.baidu.com/s/197crXIRolO1mS4PKedmYIA)，[Google Drive](https://drive.google.com/open?id=1eyUrF6wdNmUAJj8_jxGbIWwg7jkJLhgF) | 8609612c0665eef1dd044c5c83ef8416 |

**NEW:**  新增visualization.ipynb，一些可视化内容

**NEW:**  新增ppt
