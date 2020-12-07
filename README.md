# DS---Methods
#【SHOPEE CHALLENGE 2020】 - User Spending Prediction 第五名做法分享

[![Shopee challenge 2020 - I'm the best coder](https://imgur.dcard.tw/sXkKnDd.jpg "Shopee challenge 2020 - I'm the best coder")](https://imgur.dcard.tw/sXkKnDd.jpg "Shopee challenge 2020 - I'm the best coder")

> 本次比賽主辦方給了兩個月的準備期，但是**實際比賽時長只給兩個小時(當場出題目)**，非常考驗 Coder 的基本功:
1. Data processing
2. EDA
3. Feature engineering & Domain Knowledge
4. Data resampling
5. Module Selection
###### 加分項:Sampling、Leakage 洞察、環境選擇

## 參賽資訊
Kaggle: [Shopee challenge 2020 - User Spending Prediction](https://www.kaggle.com/c/iamthebestcoderopen2020/overview "Shopee challenge 2020")
隊名:台灣梯度下降第一品牌
團隊成績：...

## 比賽結果

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

EDA=>operation: 特徵分布
feature_diff=>condition: 特徵分布差異大?
Data_train=>inputoutput: 模型資料拆分(Train,Validation)
Data_balance=>condition: 資料平衡?
Module_select=>subroutine: XGB
Data_resampling=>operation: 資料重採樣
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
分辨主檔是很重要的，通常拿Submit檔案當主檔(有目標Y值，也有主要key-X值)，其他當描述檔，接下來的工作就很簡單了:**想辦法把資料描述檔塞進主檔裡**(注意mapping過程不能一對多)。
這時就考驗平常處理資料的功夫了，不外乎就幾種:
1. 時間序列與數字的轉換。
2. 資料與資料間的mapping。
3. 資料缺失值的合理填充、刪除。
4. 高基數資料的轉換

> 其實當有主檔後，就可以直接跑模型了，就算主檔還沒完善也沒關係:laughing:。

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
2. 抽到的樣本要具有代表性(意思就是說 Grouping 做不好的話就炸開了:boom:)。

### Module Selection
唯一支持 XGBoost!!
不知道的 Boosting 原理的童鞋們，請參考周志華教授所撰寫的西瓜書。
如果要我用一句話來表示Boosting，它就是個 **可以把 "渣渣"模型變成 "我就是屌"模型(周杰倫?)...**

## 參賽過程

## 心得
經過這次比賽，體認到Pipeline白板好重要...平常沒事多經營Github好重要...管理自己的代碼庫好重要...，比賽當天所遇到的所有問題通通往Github上找，實在是沒有多少時間上網查資料。
也在這次備賽中學到很多新的方法，感謝同隊隊友，也感謝網路上的大神們。

## 聯絡方式
我的Github - [WPU](https://github.com/ts01174755 "Github")
