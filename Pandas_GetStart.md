

table of contents:
- [Pandas入門](#pandas--)
  * [數據結構：](#-----)
    + [Series](#series)
      - [基礎操作：](#-----)
    + [DataFrame](#dataframe)
      - [基礎操作：](#------1)
      - [可輸入的數據結構：](#---------)
    + [索引 Object](#---object)
      - [常用：](#---)
      - [基本操作：](#-----)
  * [基本功能](#----)
    + [丟棄行/列數據 - drop](#----------drop)
    + [索引、選取＆過濾](#--------)
    + [算術運算＆數據對齊](#---------)
    + [函數應用和映射](#-------)
    + [排序和排名](#-----)
  * [描述性統計](#-----)
    + [相關係數和斜方差](#--------)
    + [唯一值、值計數及成員資格](#------------)
  * [處理缺失值](#-----)
  * [層次化索引](#-----)
    + [Groupby](#groupby)
  * [Other](#other)

# Pandas入門

## 數據結構：

------

### Series

​		由一組 Numpy 類型的數據以及由 Index 作為標籤所組成的數據結構。

```
In[1]: obj = pd.Series([4,7,-5,3])
In[2]: obj
Out[2]: 
    0    4
    1    7
    2   -5
    3    3
    dtype: int64

In[3]: obj2 = pd.Series([4,7,-5,3], index=['a','b','c','d'])
In[4]: obj2
Out[4]: 
    a    4
    b    7
    c   -5
    d    3
    dtype: int64
```

#### 基礎操作：

- 取值

  ```
  In[1]: obj2 = pd.Series([4,7,-5,3], index=['a','b','c','d'])
  In[2]: obj2['a']
  In[3]: obj2[['a','b','c']]
  In[4]: obj2[[Ture,False,Ture,False]]
  ```

- 檔案轉換

  ```
  In[1]: sdata = {'a':1, 'b':2, 'c':3}
  In[2]: obj3 = pd.Series(sdata)
  ```


- 命名：

  ```
  In[1]: obj2 = pd.Series([4,7,-5,3], index=['a','b','c','d'])
  In[2]: obj2.name = 'Pandas 入門'
  In[e]: obj2.index.name = 'Index_name'
  ```


### DataFrame

​	可以被視為由 Series 組成的字典，而 DataFram 的數據以一個或多個二維模塊存放。

**!! DataFrame axis=0：指的是沿著 column（**

```
In[1]: data={'a':[1,2,3], 'b':[4,5,6], 'c':[7,8,9]}
In[2]: frame = pd.DataFrame(data, columns = ['a', 'b', 'c', 'd'], index['one','two','three'])
```

#### 	基礎操作：

- 缺失值：

  ```
  In[3]: DataFrame(data, columns = ['a', 'b', 'c', 'd'], index['one','two','three'])
  Out[3]: 
             a  b  c    d
      one    1  4  7  NaN
      two    2  5  8  NaN
      three  3  6  9  NaN
  ```

- 索引：

  ```
  In[4]: frame.ix['two']
  Out[4]: 
      b      6
      a      3
      c      9
      d    NaN
      Name: 2, dtype: object
  ```

- 附值：

  ```
  In[5]: frame['d']=5
  Out[5]:
             a  b  c  d
      one    1  4  7  5
      two    2  5  8  5
      three  3  6  9  5	

  In[6]: val = pd.Series([-1.2,-1.5], index=['three','one'])
  In[7]: frame['d'] = val
  In[8]: frame
  Out[8]: 
             a  b  c    d
      one    1  4  7 -1.5
      two    2  5  8  NaN
      three  3  6  9 -1.2
  ```

- 命名：

  ```
  In[4]: frame.index.name = 'Data_index'
  In[5]: frame.columns.name = 'Data_col_name'
  In[6]: frame
  Out[6]:
      Data_col_name    a    b    c
      Data_index                  
      one            1.0  4.0  7.0
      two            2.0  5.0  8.0
      three          3.0  6.0  9.0
      four           NaN  NaN  NaN
  ```

- 取值：（以 numpy 二維 ndarray 形式返回）

  ```
  In[7]: frame.valuse
  Out[7]:
      array([[ 1.,  4.,  7.],
             [ 2.,  5.,  8.],
             [ 3.,  6.,  9.],
             [nan, nan, nan]])
  ```

- 布林判斷：

  ```
  In[8]: frame['c'].isnull()
  Out[8]: 
      Data_index
      one      False
      two      False
      three    False
      four      True
      Name: c, dtype: bool

  In[9]: frame['b'] > 5
  Out[9]: 
      Data_index
      one      False
      two      False
      three     True
      four     False
      Name: b, dtype: bool
  ```

- 刪除：

  ```
  In[10]: del frame['c']
  ```

- 轉置：

  ```
  In[11]: frame.T
  ```
#### 可輸入的數據結構：

  - 列表、元祖

  - 字典或 Series 的列表

  - 由 Series 組成的字典

  - 由字典組成的字典：由此印證 DataFrame 是由具有 Key 值得 Series 組成。

    ```
    In[1]: pop = {'a':{'one':1,'two':2,'three':3},
                  'b':{'one':4,'two':5,'three':6},
                  'c':{'one':7,'two':8,'three':9}}
    In[2]: frame = pd.DataFrame(pop, index = ['one','two','three','four'])
    In[3]: frame
    Out[3]:
                 a    b    c
        one    1.0  4.0  7.0
        two    2.0  5.0  8.0
        three  3.0  6.0  9.0
        four   NaN  NaN  NaN
    ```

  - Numpy 的 ndarry

  - Numpy 的結構化/記錄數組

  - Numpy 的 MaskedArray

  - DataFrame


### 索引 Object

​	當 Pandas 構建 Series、DataFrame 時，所用到的任何數組或其他序列的標籤都會轉換成一個 Index，並且 Index 是不可修改的。

```
In[1]: obj = pd.Series(range(3), index=['a','b','c'])
In[2]: index = obj.index
In[3]: index
Out[3]: Index(['a', 'b', 'c'], dtype='object')
```

#### 常用：

- Index：最一般的 Index。
- Int64Index：針對整數的特殊 Index。
- Multilndex：層次化索引。
- DatetimeIndex：儲存奈秒即時間戳。
- PeriodIndex：針對 Period 的特殊 Index。

#### 基本操作：

- append：
- diff：計算差集。
- intersection：計算交集。
- union：計算聯集。
- isin：計算一個指示各值是否都包含在參數集合中的 Boolean 數組。
- delete：刪除索引處 i 的元素，並得到新的 Index。
- drop：刪除傳入的值，並得到新的 Index。
- insert：將元素插入到索引處 i 並得到新的 Index。
- is_monotonic：當個元素均大於等於前一個元素時，返回 True。
- is_unique：當 Index 沒有重複值時，返回 True。
- unique：計算 Index 中唯一值的數組。

------

## 基本功能

### 丟棄行/列數據 - drop

​	由於需要執行一些數據整理和集合邏輯，故 drop 返回的是新物件。

- Series

  ```
  In[1]: obj = pd.Series(np.arange(5),index=['a','b','c','d','e'])
  In[2]: obj.drop('c')
  Out[2]: 
      a    0
      b    1
      d    3
      e    4
      dtype: int64
  ```

- DataFrame

  - axis =：

    0：沿著軸 0（column）進行迭代操作 -> 選取特定 column，迭代操作特定 Index 數據；

    1：沿著軸 1（Index）進行迭代操作 -> 選取特定 Index ，迭代操作特定 Column數據。

  ```
  In[1]: obj = pd.DataFrame(np.arange(16).reshape((4,4)),index=['a','b','c','d'],
                     columns=['col1','col2','col3','col4'])
  In[2]: obj.drop('a',axis=0)
  In[3]: obj.drop('col1',axis=1)
  ```

### 索引、選取＆過濾

```
In[1]: obj = pd.DataFrame(np.arange(16).reshape((4,4)),index=['a','b','c','d'],
                   columns=['col1','col2','col3','col4'])
```

- 一般方法
  - 選取

    - Series

      ```
      In[2]: obj['col']['b']		# obj['col1'] 是 Series，故而可以用 Index 索引。
      ```

    - **DataFrame（不支持）**

      ```
      In[3]: obj[['col2','col3']]['b']		# Error -> obj[['col2','col3']] 是DataFrame，不支持 Index 索引。
      ```

  - 切片

    - Series

      ```
      In[4]: obj['col2']['b':'d']	
      In[5]: obj['col2'][obj['col2']>5] 		#布林切片
      ```

    - **DataFrame（不支持）**

      ```
      In[6]: obj[['col2','col3']]['b':'d']		# Error -> obj[['col2','col3']] 是DataFrame，不支持 Index 切片。
      ```

- Pandas-Object.ix[ row_index, cols ]

  ```
  In[6]: obj.ix[ [2,3,4] , ['col2','col3'] ]
  In[7]: obj.ix[ obj['col1'] % 2 == 0 , ['col2','col3'] ]
  ```

- Pandas-Object.reindex()：將舊索引序列對應的數據，改為由新索引對應的數據， Pandas Object ( e.g Series, DataFrame ) 皆適用
  - index =：新索引序列。

  - columns =：新欄位序列。

  - fill_value =：int, 補缺值。

  - method =：'ffill' or 'pad' - 向前填充值 ; 'bfill' or 'backfill' - 向後填充值

  - limit =：向前或向後最大的填充量

  - level =：在 MultiIndex 的指定級別上匹配簡單索引，否則選取其子集。

  - copy =：預設為 True：複製；False：不複製。

    ```
    In[8]: obj2 = pd.Series([0,1,2,3], index=['b','a','c','d'])
    In[9]: obj2.reindex(index = ['a','b','c','d','e'], fill_value=0)
    Out[9]: 
        a    1
        b    0
        c    2
        d    3
        e    0
        dtype: int64
    
    In[10]: obj3 = pd.Series(['blue','purple','yello'], index=[0,2,4])
    In[11]: obj3.reindex(index = range(6), method='ffill')
    Out[11]:
    0      blue
    1      blue
    2    purple
    3    purple
    4     yello
    5     yello
    dtype: object
    ```

  - Pandas-Object.xs()：**根據標籤**選取單列或單行，返回一個 Series。

  - Pandas-Object.irow() / .icol()：**根據整數位置**選取單列或單行，返回一個 Series。

  - Pandas-Object.iloc[ ] / .loc[ ]：根據行標籤和列標籤返回 Series 或 DataFrame。

    ```
    In[12]: obj.loc[:,'col1']		# .loc[] 必須使用 Index, Columns 原始名稱。
    In[13]: obj.iloc[:,1]				# .iloc[] 支援用數字代替 Index, Columns 原始名稱（從 0 開始）
    ```

### 算術運算＆數據對齊

​	核心概念：當有兩個 Series, DataFrame 資料要進行「加、減、成、除、附值、取代、替換...等操作」，Series,DataFrame 會將有相同的 Index 的資料進行運算，Index 不重疊得部分則引入缺失值NA。

**!! Series 要與 Numpy 資料結構的物件進行「加、減、成、除、附值、取代、替換...等操作」，要確保兩者的長度(Index)是一樣的。**

- 一般加／減／乘／除法：

  - Series：

    ```
    In[1]: obj = pd.Series([4,7,-5,3])
    In[2]: obj2 = pd.Series([1,2,3,4,5)
    In[3]: obj = obj + obj2
    ```

  - DataFrame：

    ```
    In[4]: obj = pd.DataFrame(np.arange(9).reshape((3,3)),index=['a','b','c'],
                       columns=['col1','col2','col3'])
    In[5]: obj2 = pd.DataFrame(np.arange(12).reshape((4,3)),index=['b','c','d','e'],
                       columns=['col1','col3','col4'])
    In[6]: obj + obj2
    Out[6]:
           col1  col2  col3  col4
        a   NaN   NaN   NaN   NaN
        b   3.0   NaN   6.0   NaN
        c   9.0   NaN  12.0   NaN
        d   NaN   NaN   NaN   NaN
        e   NaN   NaN   NaN   NaN
    ```

- Pandas-LeftObject.OperatorFunc_( Pandas.RightObject , Parameter )：Pandas-LeftObject 為主，將 Pandas-RightObject 值經過運算併入 Pandas-LeftObject 。

  - OperatorFunc_()
    - .add()：加法
    - .sub()：減法
    - .div()：除法
    - .mul：乘法

    ```
    In[7]: obj.add(obj2, fill_value=0)		# 其中一方有缺失值得補0，雙方皆無值則引入NA值
    Out[7]:
           col1  col2  col3  col4
        a   0.0   1.0   2.0   NaN
        b   3.0   4.0   6.0   2.0
        c   9.0   7.0  12.0   5.0
        d   6.0   NaN   7.0   8.0
        e   9.0   NaN  10.0  11.0
    ```

- 批量運算：

  - 行運算

    ```
    In[8]: obj4 = obj.iloc[0,:]
    In[9]: obj - obj4
    Out[9]:   
            col1  col2  col3	|
        a     0     0     0		|
        b     3     3     3		|
        c     6     6     6		|
                              V
    ```

  - 列運算：

    ```
    In[10]: obj3 = obj.iloc[:,0]
    In[11]: obj.sub(obj3, axis=0)		# 迭代操作不同 Column
    Out[11]: 
           col1  col2  col3
        a     0     1     2
        b     0     1     2
        c     0     1     2
        -------------------->
    ```

- 布林判斷

  ```
  In[1]: obj2 = pd.Series([4,7,-5,3], index=['a','b','c','d'])
  In[2]: obj2 > 2 			# 數值判斷
  In[3]: 'a' in obj2		# True,標籤判斷
  In[4]: 'e' in obj2 		# False,標籤判斷
  ```

### 函數應用和映射

- Pandas-Object.apply( func\_(x) , axis= 0 or 1 )：輸入 Pandas-Object 數據至函式 Func\_(x)，並依據迭代方向進行計算。
  - axis = 0：沿著 Column 迭代計算func\_(Pandas-Object[ column ])
  - axis = 1：沿著 Index 迭代計算func\_(Pandas-Object[ Index ])

```
In[1]: obj = pd.DataFrame(np.arange(12).reshape((4,3)),index=['b','c','d','e'],
                   columns=['col1','col3','col4'])
                   
In[2]: obj.apply(lambda x:x.max() - x.min() , axis=0)
Out[2]: 
    col1    9
    col3    9
    col4    9
    dtype: int64
In[3]: obj.apply(lambda x:x.max() - x.min() , axis=1)
Out[3]: 
    b    2
    c    2
    d    2
    e    2
    dtype: int64
```

### 排序和排名

- Pandas-Object.sort_index()：對索引進行排序。

  - aixs = 0：沿著 Column 方向迭代，進行排序操作。（對每個 Column 數值排序 Index）
  - aixs = 1：沿著 Index 方向迭代，進行排序操作。（對每個 Index 數值排序 Column）
  - by = 'Column_name'：排序 Pandas-Object[Column_name] 的值，以產生新的 Index 序列，結果適用於全部數據。
  - by = [ [ 'col1' , 'col2' ] ]：依給定 Colmun 的順序， Pandas-Object[Column_name] 的值，以產生新的 Index 序列，結果適用於全部數據。
  - ascending = False：降序。
  - ascending = True：升序。

  ```
  In[1]: obj = pd.DataFrame(np.arange(12).reshape((4,3)),index=[1,3,2,4],
                     columns=['col3','col1','col4'])
  In[2]: obj.sort_index()
  Out[2]:
         col3  col1  col4
      1     0     1     2
      2     6     7     8
      3     3     4     5
      4     9    10    11
  In[3]: obj.sort_index(axis=1)
  Out[3]: 
         col1  col3  col4
      1     1     0     2
      3     4     3     5
      2     7     6     8
      4    10     9    11
  ```

- Pandas-Series.Order()：對 Series 其值進行排序，保留原始 Index。

- Pandas-Series.rank()：

  - Series：其值進行排序，依據順序給定 Index。

    Method = 'average'：默認：相同數值時，使用平均排名。

    Method = 'min'：使用最小排名。

    Method = 'max'：使用最大排名。

    Method = 'first'：按照原始數據出現的順序進行排名。

  - DataFrame：對不同分組的內部的資料，給其分組內的排名（不排序）。

    ```
    In[1]: obj = pd.DataFrame(np.array([1,7,3,9,4,6,3,4,5,1,3,4]).reshape((4,3)),index=[1,3,2,4],
                       columns=['col3','col1','col4'])
    In[2]: obj.rank(axis=0)
    Out[2]: 
           col3  col1  col4	|
        1   1.5   4.0   1.0	|
        3   4.0   2.5   4.0	|
        2   3.0   2.5   3.0	|
        4   1.5   1.0   2.0	|
        										V
    In[3]: obj.rank(axis=1)
    Out[3]: 
           col3  col1  col4
        1   1.0   3.0   2.0
        3   3.0   1.0   2.0
        2   1.0   2.0   3.0
        4   1.0   2.0   3.0
        ------------------->
    ```

## 描述性統計

### 相關係數和斜方差

### 唯一值、值計數及成員資格

## 處理缺失值

- Series

  ```
  In[1]: obj = pd.Series({'a':1, 'b':2, 'c':3}, index=['b','c','d'])
  In[2]: obj
  Out[2]: 
      b    2.0
      c    3.0
      d    NaN
      dtype: float64
  obj.isnull()	# 檢測缺失值
  obj.notnull()	# 檢測非缺失值
  obj.fillna(0)	# 對缺失值引入數值 0 
  obj.dropna()	# 刪除缺失值
  ```
  
- DataFrame

  ```
  In[3]: data = pd.DataFrame([[1,2,3],[4,np.nan,np.nan],[np.nan,np.nan,np.nan],[np.nan,5,6]])
  In[4]: data[4]=np.nan
  In[5]: data.dropna()			# 丟棄任何含有缺失值的行資料
  In[6]: data.dropna(how='all')	# 丟棄全為NA的行資料
  In[7]: data.dropna(thresh=2)		# 回傳至少有 2 個非缺失值的行資料。
  In[8]: data.fillna({0:0.1,2:0.5})	#	填補特定 Column 的缺失值
  Out[8]:
           0    1    2   4
      0  1.0  2.0  3.0 NaN
      1  4.0  NaN  0.5 NaN
      2  0.1  NaN  0.5 NaN
      3  0.1  5.0  6.0 NaN
  In[9]: data.fillna(method='ffill', limit=2)		# 向前填充缺失值，最多填充兩個。
  ```

## 層次化索引

​	透過在一個軸上創立多個索引級別，使在低維度處理高維度數據 --> 雙層字典的概念。

```
In[1]: NpArray = np.random.randn(10)
In[2]: NpIndex = [['a','a','a','b','b','b','c','c','d','d'],
           [1,2,3,1,2,3,1,2,2,3]]
In[3]: data = pd.Series(NpArray,index = NpIndex)
In[4]: data
Out[4]: 
    a  1    0.227849
       2   -0.083232
       3    1.406042
    b  1    1.109092
       2    0.793754
       3    0.547979
    c  1   -0.434295
       2    0.322415
    d  2    1.553198
       3   -2.295112
    dtype: float64
```

- Series：

  Pandas-SeriesObj[ 'Key1']

  Pandas-SeriesObj[ 'Key1', 'Key2' ]

  - 選取

    ```
    In[5]: data['b':'c']
    Out[5]:
        b  1    1.109092
           2    0.793754
           3    0.547979
        c  1   -0.434295
           2    0.322415
        dtype: float64
    In[6]: data[:,1]
    Out[6]: 
        a    0.227849
        b    1.109092
        c   -0.434295
        dtype: float64
    ```

  - 升維：

    ```
    In[7]: data.unstack()
    Out[7]: 
                  1         2         3
        a  0.227849 -0.083232  1.406042
        b  1.109092  0.793754  0.547979
        c -0.434295  0.322415       NaN
        d       NaN  1.553198 -2.295112
    ```

- DataFrame：

  ```
  In[9]:	NpArray_2D = np.arange(12).reshape((4,3))
  				Index_2D = [['a','a','b','b'],[1,2,1,2]]
  				Columns_2D = [['Ohio','Ohio','Colorado'],['G','R','G']]
  In[10]:	data_2D = pd.DataFrame(NpArray_2D,index = Index_2D, columns = Columns_2D)
  In[11]:	data_2D
  Out[11]:
          Ohio     Colorado
             G   R        G
      a 1    0   1        2
        2    3   4        5
      b 1    6   7        8
        2    9  10       11
  ```
   - 命名：

      ```
      In[11]:	data_2D.index.names = ['Key1','Key2']
      In[12]:	data_2D.columns.names = ['State','Color']
      In[13]:	data_2D
      Out[13]:
          State     Ohio     Colorado
          Color        G   R        G
          Key1 Key2                  
          a    1       0   1        2
               2       3   4        5
          b    1       6   7        8
               2       9  10       11
      ```
      
   - Key **順序互換**：

      ```
      In[14]:	data_2D.swaplevel('Key1','Key2')
      ```

   - Key 排序：

      Pandas-DataFrameObj.sort_index( axis = 0 or 1 )：對每個軸進行 Index  排序，然後迭代。
      
      ```
      In[15]:	data_2D.swaplevel('Key1','Key2').sort_index(0)	# 對每個 Column 排序 Index
      ```
      
      Pandas-DataFrameObj.sortlevel( <Index_Level> )：對第幾層 Index  排序：
      
    ```
      In[16]:	data_2D.swaplevel('Key1','Key2').sortlevel(0)		＃ 對 Index level 1 排序
    ```
  
   - 降維：
  
      ```
      In[17]: data_2D.stack()
      Out[17]: 
                 Colorado  Ohio
          a 1 G       2.0     0
              R       NaN     1
            2 G       5.0     3
              R       NaN     4
        b 1 G       8.0     6
              R       NaN     7
            2 G      11.0     9
              R       NaN    10
      ```
  

### Groupby

- Pandas-DataFrameObj.groupby()：

- 沿著軸方向統計：

  ```
  In[18]:	data_2D.sum(axis=0, level='Key2')
  Out[18]: 
      State Ohio     Colorado
      Color    G   R        G
      Key2                   
      1        6   8       10
      2       12  14       16
  ```

- 列索引轉行索引：

  Pandas-DataFrameObj.set_index( [ 'col1', 'col2'] ,drop = True or False)：將所選的列轉成行索引。
  
  - set_index() - Parameter：
  
    drop = True：將轉換的列資料刪除。
  
    drop = False：保留轉換的列資料。
  
  ```
  In[1]:	Data_Dict = {'a':range(7),'b':range(7,0,-1),
                       'c':['one','one','one','two','two','two','two'],
                       'd':[0,1,3,0,1,2,3]}
  In[2]:	Data_Dict2Frame = pd.DataFrame(Data_Dict)
  In[3]:	Data_Dict2Frame
  Out[3]: 
         a  b    c  d
      0  0  7  one  0
      1  1  6  one  1
      2  2  5  one  3
      3  3  4  two  0
      4  4  3  two  1
      5  5  2  two  2
      6  6  1  two  3
  In[4]: Data_Dict2Frame.set_index(['c','d'])	
  Out[4]: 
            a  b
     c   d      
     one 0  0  7
         1  1  6
         3  2  5
     two 0  3  4
         1  4  3
         2  5  2
         3  6  1
  ```

## Other
