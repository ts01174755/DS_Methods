# 【SHOPEE CHALLENGE 2020】 - User Spending Prediction 第五名做法分享

[![Shopee challenge 2020 - I'm the best coder](https://imgur.dcard.tw/sXkKnDd.jpg "Shopee challenge 2020 - I'm the best coder")](https://imgur.dcard.tw/sXkKnDd.jpg "Shopee challenge 2020 - I'm the best coder")

> 本次比賽主辦方給了兩個月的準備期，但是**實際比賽時長只給兩個小時(當場出題目)**，非常考驗 Coder 的基本功:
> 1. Data processing
> 2. EDA
> 3. Feature engineering & Domain Knowledge
> 4. Data resampling
> 5. Module Selection
> ###### 加分項:Sampling、Leakage 洞察、環境選擇

## 宣傳一下...
Kaggle: [Shopee challenge 2020 - User Spending Prediction](https://www.kaggle.com/c/iamthebestcoderopen2020/overview "Shopee challenge 2020")

隊名:台灣梯度下降第一品牌

隊員:

Ethan - [Github](https://github.com/CubatLin "Github")

我 - [Github](https://github.com/ts01174755 "Github")


:trophy:以往成績：

- 《2019 autumn E.Sun Bank AI open competition- credit card fraud detection》:

  2nd of final selected 20 teams in business solutions competition.
  
  Led team of 5 to achieve top 1%(15/1366) in F1 score predict competition.

- 《2020 Shopee Code League- series of Kaggle competition in Asia-Pacific(open category)》

  15th in Sentiment Analysis(NLP task).
  
  23th in Product Title Translation(NLP task, BLEU score: 40.5).
  
  23th in Marketing Analytics(Recommendation forecasting task).
  
  47th in Order Brushing(accuracy:0.9779).
  
  67th, score 0.81 accuracy in Product Detection(Computer Vision task).

:point_right:本次排名....
- 《I'm the Best Coder! Challenge 2020(open category)》

  5th in User Spending Prediction.

  ![Shopee challenge 2020 - 5th](https://github.com/ts01174755/Competition/blob/main/Shopee%20challenge%202020%20-%20Flow.jpg)

## 賽前準備工作 - Pipeline白板
Pipeline 白板很重要，它可以幫助我們在面臨未知問題時能節省大量工作，我們拿Shopee Challenge 2019 - Marketing Analytics 的資料做為假想資料，並考慮各種特殊情況下產出Pipeline白板。
我們 Pipeline 白板的流程大概長這樣:
```flow
st=>start: 環境建置 Colab
Data_input=>inputoutput: 資料輸入
Data_processing=>subroutine: 資料處理

Data_mainornot=>condition: 資料主檔?
Data_main=>operation: 資料主檔
Data_descript=>operation: 資料描述檔
feature_engin_grouping=>operation: Grouping
feature_engin_encoding=>operation: Encoding
feature_entropy=>condition: 高資訊含量特徵?

EDA=>operation: EDA觀察特徵分布
feature_diff=>condition: 特徵分布差異大?
Data_train=>inputoutput: 模型資料拆分(Train,Validation)
Data_balance=>condition: 資料平衡?
Module_select=>subroutine: XGB
Data_resampling=>operation: Resampling
ed=>end: Answer submit

st->Data_input->Data_processing->Data_mainornot
Data_mainornot(yes)->Data_main
Data_mainornot(no)->Data_descript->feature_entropy
feature_entropy(yes)->feature_engin_grouping->feature_engin_encoding(left)->Data_main
feature_entropy(no)->feature_engin_encoding(left)->Data_main
Data_main->EDA->feature_diff
feature_diff(no)->Data_processing
feature_diff(yes)->Data_balance
Data_balance(yes)->Data_train->Module_select->ed
Data_balance(no)->Data_resampling->Data_train
```

### 資料處理
分辨主檔是很重要的，通常拿Submit檔案當主檔(有目標Y值，也有主要key值)，其他當描述檔，接下來的工作就很簡單了:**想辦法把資料描述檔塞進主檔裡**(注意mapping過程Key值不能一對多)。
這時就考驗平常處理資料的功夫了，不外乎就幾種:
1. 時間序列與數字的轉換。
2. 文字資料的處理
3. 資料與資料間的mapping。
4. 資料缺失值的合理填充、刪除。
5. 高基數資料的轉換

> 其實當有一個資料描述檔Mapping到主檔後，就可以直接跑模型了，就算還有其他描述檔還沒有Mapping到主檔也沒關係:laughing:。

### Feature engineering
特徵工程直接決定了模型的表現能力，但兩個小時能做的其實也不多，以下列出幾個這次有用到的:
- Encoding: One-Hot encoding, Target encoding, mean-encoding
- Grouping: K-means, DBSCAN
- Dimensionality reduction: PCA,
- rescaling: min-max rescale, 統計標準化

> Grouping 是一定要做的，現在不做，後面的資料重採樣也得做。

### EDA
這一步其實就可以略窺模型的能力了，特徵做得越好，目標特徵(Y值)分離度越大，模型表現越好。

> 用 EDA 評估 Feature 就可以知道做出來的變數是好變數還是渣渣。

### Data resampling
這一步決定的模型Robust的能力，做得好模型就有好的泛化能力。
資料重採樣有兩個重點:
1. 分類任務中，標籤的數量要保持"平衡"(這部分見仁見智，但是我從來沒有搞懂"平衡"的定義...:joy:)。
2. 抽到的樣本要具有代表性(如果Grouping做不好，抽樣的代表性就差，模型效能直接炸開:boom:)。

### Module Selection
唯一支持 XGBoost!!

不知道的 Boosting 原理的童鞋們，請參考周志華教授所撰寫的西瓜書。

如果要我用一句話來表示Boosting，它就是個 **可以把 "渣渣"模型變成 "我就是屌"模型(周杰倫?)...**的算法。

## 心得
打過幾次比賽後開始體認到Machine Learning是一個完整的資料分析框架，已經成為做Data的必備技能：清晰的資料處理邏輯、運用Feature engineering & Domain Knowledge製作變數、變數好壞評估、模型選擇...等。

**以上技能是必備，更要滿足輕量、快速部署、算法效率/效能高要求...等要求**，如何達到？刷些leetcode，然後做出自己的Pipeline吧！

面對資料如同面對疫情一樣，**超前部署**是一定要做的事情了。

## 聯絡方式

