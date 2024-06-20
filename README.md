**Read this in other languages:[English](README.md),[中文](README_zh.md).**

**Caution!!!**:the English version of README.md is the same as the Chinese version of README_zh.md, and it is translated from the Chinese version by ChatGPT.

Note:
1. Please run part1, part2, and part3 in order, as the latter two files will use intermediate variables generated by the previous files. Due to the large size of these variables and our limited budget for monthly git lfs payments, the temp folder is empty.

2. Our third part file is real-main-part3.ipynb. The file named main-part3.ipynb in the directory is a failed machine learning model based on transformers.

3. In the directory:
   - `./img` contains images used for illustration in the ipynb files.
   - `./papers` contains some of the papers we referenced.
   - `./source_data` contains our raw data.
   - `./temp` is for our intermediate variables.
   - `main-part1.ipynb`, `main-part2.ipynb`, and `real-main-part3.ipynb` are the main files for our three parts.
   - `./output` folder contains the variable sets used for our report.
   - `./temp` stores some intermediate variables.
   - `./models` primarily holds failed model files (originally intended for part 3 but now discarded).

# Investment 2 Studies Group Assignment
## 1. Project Introduction
This is a classroom group assignment. For specific requirements, please refer to Project_&_Presentation.pdf in the root directory.

### 1.1 Source_data
The Source_data folder contains the raw data, covering the period from January 1, 2000, to January 1, 2024. The data used in this project is from China.

The three Fama-French five-factor model datasets are sourced from the CSMAR database, available in daily, weekly, and monthly frequencies. Each folder includes data field descriptions and other information.

The risk-free rate is the One-year lump-sum deposit, available as daily data from the CSMAR database.

The market return rate is based on the Wind All A Index, with the raw data being daily data sourced from the Wind database.

The Balance Sheet includes total company assets (used to calculate the investment-level risk factor) and total shareholders' equity (used to calculate the book-to-market factor), sourced from CSMAR.
All daily stock price returns and monthly stock price returns are sourced from CSMAR. Due to CSMAR's limitation of exporting no more than five years of data at a time, daily data is organized in five-year intervals, while monthly data can be exported in one go.

## 1.2 Part 1
This part primarily replicates the Fama-French five-factor model. The construction of the five factors is based on the methods described in Fama and French (2016). For detailed construction methods, refer to Fama and French (2016).

Additionally, we conducted the GRS test in this section, following the same methodology as the original paper.

## 1.3 Part 2
This part focuses on verifying the existence of the January Anomaly and the Monday Anomaly and conducting corresponding backtests. For the January Anomaly, we used monthly data, while for the Monday Anomaly, we used daily data.

## 1.3 Part 3
In this section, we identified a factor: Gross Profit/Total Assets. We first verified that this can indeed be used as a risk factor, and then we constructed investment portfolios based on different time scales and conducted corresponding backtests.

# 2. How to Deploy
First, ensure that git is installed, then execute the following commands in sequence:

```shell
git clone https://github.com/alex-fang123/investment_project.git
cd investment_project
```