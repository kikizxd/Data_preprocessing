# 常见的数据预处理--python篇 #
做过数据分析的孩子一般都知道：数据预处理很重要，大概会占用整个分析过程50％到80％的时间，良好的数据预处理会让建模结果达到事半功倍的效果。本文简单介绍python中一些常见的数据预处理，包括**数据加载**、**缺失值处理**、**异常值处理**、**描述性变量转换为数值型**、**训练集测试集划分**、**数据规范化**。
## 1、 加载数据 ##
### 1.1 数据读取 ###
数据格式有很多，介绍常见的csv,txt,excel以及数据库mysql中的文件读取
```
import pandas as pd
data = pd.read_csv(r'../filename.csv')	#读取csv文件
data = pd.read_table(r'../filename.txt')	#读取txt文件
data = pd.read_excel(r'../filename.xlsx')  #读取excel文件

#  获取数据库中的数据
import pymysql
conn = pymysql.connect(host='localhost',user='root',passwd='123456',db='mydb',charset='utf8')	#连接数据库，注意修改成要连的数据库信息
cur = conn.cursor()	#创建游标
cur.execute("select * from train_data limit 100")	#train_data是要读取的数据名
data = cur.fetchall()	#获取数据
cols = cur.description	#获取列名
conn.commit()	#执行
cur.close()	#关闭游标
conn.close()	#关闭数据库连接
col = []
for i in cols:
	col.append(i[0])
data = list(map(list,data))
data = pd.DataFrame(data,columns=col)

```
### 1.2 CSV文件合并 ###

实际数据可能分布在一个个的小的csv或者txt文档，而建模分析时可能需要读取所有数据，这时呢，需要将一个个小的文档合并到一个文件中
```
#合并多个csv文件成一个文件
import glob

#合并
def hebing():
    csv_list = glob.glob('*.csv') #查看同文件夹下的csv文件数
    print(u'共发现%s个CSV文件'% len(csv_list))
    print(u'正在处理............')
    for i in csv_list: #循环读取同文件夹下的csv文件
        fr = open(i,'rb').read()
        with open('result.csv','ab') as f: #将结果保存为result.csv
            f.write(fr)
    print(u'合并完毕！')

#去重    
def quchong(file):
        df = pd.read_csv(file,header=0)    
        datalist = df.drop_duplicates()    
        datalist.to_csv(file) 
        
if __name__ == '__main__':    
    hebing()    
    quchong("result.csv.csv") 

```
### 1.3 CSV文件拆分 ###
对于一些数据量比较大的文件，想直接读取或者打开比较困难，介绍一个可以拆分数据的方法吧，方便查看数据样式以及读取部分数据
```
##csv比较大，打不开，将其切分成一个个小文件，看数据形式
f = open('NEW_Data.csv','r') #打开大文件
i = 0 #设置计数器

#这里1234567表示文件行数，如果不知道行数可用每行长度等其他条件来判断
while i<1234567 : 
    with open('newfile'+str(i),'w') as f1:
        for j in range(0,10000) : #这里设置每个子文件的大小
            if i < 1234567: #这里判断是否已结束，否则最后可能报错
                f1.writelines(f.readline())
                i = i+1
            else:
                break   
```
### 1.4 数据查看 ###
在进行数据分析前呢，可以查看一下数据的总体情况，从宏观上了解数据
```
data.head() #显示前五行数据
data.tail() #显示末尾五行数据
data.info() #查看各字段的信息
data.shape #查看数据集有几行几列,data.shape[0]是行数,data.shape[1]是列数
data.describe() #查看数据的大体情况，均值，最值，分位数值...
data.columns.tolist()   #得到列名的list
```
## 2、缺失值 ##
现实获取的数据经常存在缺失，不完整的情况**（能有数据就不错了，还想完整！！！）**，为了更好的分析，一般会对这些缺失数据进行识别和处理

### 2.1 缺失值查看 ###
```
print(data.isnull().sum())  #统计每列有几个缺失值
missing_col = data.columns[data.isnull().any()].tolist() #找出存在缺失值的列

import numpy as np
#统计每个变量的缺失值占比
def CountNA(data):
    cols = data.columns.tolist()    #cols为data的所有列名
    n_df = data.shape[0]    #n_df为数据的行数
    for col in cols:
        missing = np.count_nonzero(data[col].isnull().values)  #col列中存在的缺失值个数
        mis_perc = float(missing) / n_df * 100
        print("{col}的缺失比例是{miss}%".format(col=col,miss=mis_perc))
```

### 2.2 缺失值处理 ###
面对缺失值，一般有三种处理方法：不处理、删除以及填充
#### 2.2.1 不处理 ####
有的算法（贝叶斯、xgboost、神经网络等）对缺失值不敏感，或者有些字段对结果分析作用不大，此时就没必要费时费力去处理缺失值啦 **=。=**
#### 2.2.2 删除 ####
在数据量比较大时候或者一条记录中多个字段缺失，不方便填补的时候可以选择删除缺失值
```
data.dropna(axis=0,how="any",inplace=True)  #axis=0代表'行','any'代表任何空值行,若是'all'则代表所有值都为空时，才删除该行
data.dropna(axis=0,inplace=True)  #删除带有空值的行
data.dropna(axis=1,inplace=True)  #删除带有空值的列
```
#### 2.2.3 填充 ####
数据量较少时候，以最可能的值来插补缺失值比删除全部不完全样本所产生的信息丢失要少
##### 2.2.3.1 固定值填充 #####
```
data = data.fillna(0)   #缺失值全部用0插补
data['col_name'] = data['col_name'].fillna('UNKNOWN')  #某列缺失值用固定值插补
```
##### 2.2.3.2 出现最频繁值填充 #####

即众数插补，离散/连续数据都行，适用于名义变量，如性别
```
freq_port = data.col_name.dropna().mode()[0]  # mode返回出现最多的数据,col_name为列名
data['col_name'] = data['col_name'].fillna(freq_port)   #采用出现最频繁的值插补
```
##### 2.2.3.3 中位数/均值插补 #####
```
data['col_name'].fillna(data['col_name'].dropna().median(),inplace=True)  #中位数插补，适用于偏态分布或者有离群点的分布
data['col_name'].fillna(data['col_name'].dropna().mean(),inplace=True)    #均值插补，适用于正态分布
```
##### 2.2.3.4 用前后数据填充 #####
```
data['col_name'] = data['col_name'].fillna(method='pad')    #用前一个数据填充
data['col_name'] = data['col_name'].fillna(method='bfill')  #用后一个数据填充
```     
##### 2.2.3.5 拉格朗日插值法 #####

一般针对有序的数据，如带有时间列的数据集,且缺失值为连续型数值小批量数据
```
from scipy.interpolate import lagrange
#自定义列向量插值函数,s为列向量,n为被插值的位置,k为取前后的数据个数，默认5
def ployinterp_columns(s, n, k=5):
    y = s[list(range(n-k,n)) + list(range(n+1,n+1+k))]  #取数
    y = y[y.notnull()]  #剔除空值
    return lagrange(y.index, list(y))(n)    #插值并返回插值结果

#逐个元素判断是否需要插值
for i in data.columns:
    for j in range(len(data)):
        if (data[i].isnull())[j]:   #如果为空即插值
            data[i][j] = ployinterp_columns(data[i],j)
```            
##### 2.2.3.6 其它插补方法 #####

最近邻插补、回归方法、牛顿插值法、随机森林填充等。

## 3、异常值 ##
异常值是指样本中的个别值，其数值明显偏离它所属样本的其余观测值。异常值有时是记录错误或者其它情况导致的错误数据，有时是代表少数情况的正常值
### 3.1 异常值识别 ###
#### 3.1.1 描述性统计法 ####
```
#与业务或者基本认知不符的数据,如年龄为负
neg_list = ['col_name_1','col_name_2','col_name_3']
for item in neg_list:
    neg_item = data[item] < 0
    print(item + '小于0的有' + str(neg_item.sum())+'个')
    
#删除小于0的记录
for item in neg_list:
    data = data[(data[item]>=0)]
```
#### 3.1.2 三西格玛法 ####
当数据服从正态分布时，99.7%的数值应该位于距离均值3个标准差之内的距离，P(|x−μ|>3σ)≤0.003
```
#当数值超出这个距离，可以认为它是异常值
for item in neg_list:
    data[item + '_zscore'] = (data[item] - data[item].mean()) / data[item].std()
    z_abnormal = abs(data[item + '_zscore']) > 3
    print(item + '中有' + str(z_abnormal.sum())+'个异常值')
```
#### 3.1.3 箱型图 ####
```
#IQR(差值) = U(上四分位数) - L(下四分位数)
#上界 = U + 1.5IQR
#下界 = L-1.5IQR
for item in neg_list:
    IQR = data[item].quantile(0.75) - data[item].quantile(0.25)
    q_abnormal_L = data[item] < data[item].quantile(0.25) - 1.5*IQR
    q_abnormal_U = data[item] > data[item].quantile(0.75) + 1.5*IQR
    print(item + '中有' + str(q_abnormal_L.sum() + q_abnormal_U.sum())+'个异常值')
```
#### 3.1.4 其它 ####

基于聚类方法检测、基于密度的离群点检测、基于近邻度的离群点检测等。

### 3.2 异常值处理 ###
对于异常值，可以删除，可以不处理，也可以视作缺失值进行处理。

## 4、描述性变量转换为数值型 ##
大部分机器学习算法要求输入的数据必须是数字，不能是字符串，这就要求将数据中的描述性变量（如性别）转换为数值型数据
```
#寻找描述变量，并将其存储到cat_vars这个list中去
cat_vars = []
print('\n描述变量有:')
cols = data.columns.tolist()
for col in cols:
    if data[col].dtype == 'object':
        print(col)
        cat_vars.append(col)


##若变量是有序的##     
print('\n开始转换描述变量...')
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
#将描述变量自动转换为数值型变量，并将转换后的数据附加到原始数据上
for col in cat_vars:
    tran = le.fit_transform(data[col].tolist())
    tran_df = pd.DataFrame(tran,columns=['num_'+col])
    print('{col}经过转化为{num_col}'.format(col=col,num_col='num_'+col))
    data = pd.concat([data, tran_df], axis=1)
    del data[col]	#删除原来的列


##若变量是无序变量## 
#值得注意的是one-hot可能引发维度爆炸
for col in cat_vars:
    onehot_tran = pd.get_dummies(data.col)
    data = data.join(onehot_tran)	#将one-hot后的数据添加到data中
    del data[col]	#删除原来的列
```
## 5、训练测试集划分 ##
实际在建模前大多需要对数据进行训练集和测试集划分，此处介绍两种划分方式
#### 法一、直接调用train_test_split函数 ####
```
from sklearn.model_selection import train_test_split
X = data.drop('目标列',1)	#X是特征列
y = data['目标列']	#y是目标列
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3)	
```
#### 法二：随机抽样 ####
```
#随机选数据作为测试集
test_data = data.sample(frac=0.3,replace=False,random_state=123,axis=0)
#frac是抽取30%的数据，replace是否为有放回抽样，取replace=True时为有放回抽样，axis=0是抽取行、为1时抽取列
#在data中除去test_data，剩余数据为训练集
train_data = (data.append(test_data)).drop_duplicates(keep=False)
X_train = train_data.drop('目标列',1)
X_test = test_data.drop('目标列',1)
y_train = train_data['目标列']
y_test = test_data['目标列']
```
## 6、数据规范化 ##

数据的标准化（normalization）是将数据按比例缩放，使之落入一个小的特定区间。在某些比较和评价的指标处理中经常会用到，去除数据的单位限制，将其转化为无量纲的纯数值，便于不同单位或量级的指标能够进行比较和加权。
一些需要数据规范化的算法：LR、SVM、KNN、KMeans、GBDT、AdaBoost、神经网络等
### 6.1 最小最大规范化 ###
对原始数据进行线性变换，变换到[0,1]区间。计算公式为：
x* = (x-x.min)/(x.max-x.min)
```
from sklearn.preprocessing import MinMaxScaler

x_scaler = MinMaxScaler()
y_scaler = MinMaxScaler()

#特征归一化
x_train_sca = x_scaler.fit_transform(X_train)
x_test_sca = x_scaler.transform(X_test)
y_train_sca = y_scaler.fit_transform(pd.DataFrame(y_train))
```
### 6.2 零均值规范化 ###
对原始数据进行线性变换，经过处理的数据的均值为0，标准差为1。计算方式是将特征值减去均值，除以标准差。计算公式为：x* = (x-x.mean)/σ
```
from sklearn.preprocessing import StandardScaler

#一般把train和test集放在一起做标准化，或者在train集上做标准化后，用同样的标准化器去标准化test集
scaler = StandardScaler()
train = scaler.fit_transform(train)
test = scaler.transform(test)
```
**啦啦啦，终于写完了，吐血ING**
