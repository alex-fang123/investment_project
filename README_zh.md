**其他语言版本：[English](README.md),[中文](README_zh.md).**

注意事项：
1. 请按顺序运行part1, part2, part3，因为后两个文件会用到前面文件产生的中间变量，而因为这些变量较大，我们没有足够多的钱来支付每月的git lfs，因此temp文件夹下空着
2. 我们的第三部分文件是real-main-part3.ipynb，文件目录中的main-part3.ipynb是我们的一个失败的基于transformer的机器学习模型（已弃用）
3. 文件目录中，img是我们在ipynb文件中用到的方便说明的图片文件，papers是我们用到的一些论文，source_data是我们的原始数据，temp是我们的中间变量，最后的main-part1, main-part2, real-main-part3是我们的三个部分的主要文件。output文件夹是我们用于报告的变量集合，temp保存了一些中间变量，models主要保存的是失败的模型文件（原本用于part 3，现已弃用）

# 投资学2小组作业

## 1. 项目介绍

课堂小组作业，具体要求见根目录下Project_&_Presentation.pdf

### 1.1 Source_data

Source_data文件夹为原始数据，数据时间均为2000-01-01至2024-01-01.本项目使用的数据是中国数据

三个Fama-French五因子模型数据来源于**CSMAR数据库**，分别为日度、周度和月度，各文件夹下包含数据字段说明等信息

无风险利率为One-year lump-sum deposit（一年期存款），日度数据，来自**CSMAR数据库**

市场收益率采用**万得全A指数**，原始数据为日度数据，数据源为**万得数据库**

Balance Sheet中包含了公司总资产（用于计算投资水平风险因子）和公司总所有者权益（用于计算账面市值比因子），来自**CSMAR**

所有Daily stock price returns和monthly stock price returns均来自**CSMAR**
，日度数据，受限于CSMAR一次只能导出不超过5年的数据，因此每5年搞一个文件夹装，月度可以一次性导出

### 1.2 Part 1

这一部分主要是复现了Fama-French五因子模型，其中五个因子的构建方式参考了Fama和French（2016）的文章，具体构建方式见[fama and french (2016)](papers/Fama-French%20A%20five-factor%20asset%20pricing%20model.pdf)

此外，在这一部分中，我们还做了GRS检验，具体方式和论文原文保持一致

### 1.3 Part 2

这一部分主要是验证January Anomaly和Monday Anomaly的存在性并做相应的回测，其中January Anomaly我们采用月度数据，后者采用日度数据

### 1.3 Part 3

这一部分中，我们寻找了一个因子：毛利/总资产。我们首先验证了这确实可以作为一个风险因子，同时我们针对不同时间尺度构建了投资组合并做相应的回测。

# 2. 怎么部署

首先确保安装了git，然后依次执行以下命令

```shell
git clone https://github.com/alex-fang123/investment_project.git
cd investment_project
```

