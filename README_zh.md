**其他语言版本：[English](README.md),[中文](README_zh.md).**
# 投资学2小组作业

## 1. 项目介绍
课堂小组作业，具体要求见根目录下Project_&_Presentation.pdf

### 1.1 Source_data

Source_data文件夹为原始数据，数据时间均为2000-01-01至2024-01-01.本项目使用的数据是中国数据

三个Fama-French五因子模型数据来源于**CSMAR数据库**，分别为日度、周度和月度，各文件夹下包含数据字段说明等信息

无风险利率为One-year lump-sum deposit（一年期存款），日度数据，来自**CSMAR数据库**

市场收益率采用**万得全A指数**，原始数据为日度数据，数据源为**万得数据库**

Balance Sheet中包含了公司总资产（用于计算投资水平风险因子）和公司总所有者权益（用于计算账面市值比因子），来自**CSMAR**

所有Daily stock price returns和monthly stock price returns均来自**CSMAR**，日度数据，受限于CSMAR一次只能导出不超过5年的数据，因此每5年搞一个文件夹装，月度可以一次性导出

# 2. 怎么部署
首先确保安装了git和git lfs，然后依次执行以下命令

```shell
git clone https://github.com/alex-fang123/investment_project.git
cd investment_project
git lfs fetch
git lfs checkout
```
