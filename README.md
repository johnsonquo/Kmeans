# Kmeans
Kmeans演算法

* Model:
   * data : 輸入資料(np.array型態)
   * k : 選取中心數 
   * maxIterTimes : 最大迭帶次數限制

* Method:
   * Fit() : 開始訓練
    
* Attribute:
   * classifyList : 分類結果
   * centerList : 資料中心
    
    


```python
import numpy as np
import random
from scipy.stats import mode
import matplotlib.pyplot as plt
import matplotlib.cm as cm
```

**KMEANS CLASS**


```python
class Kmeans:
    
    def __init__(self, data, target, k, maxIterTimes = 30):
        self.data = data
        self.target = target
        self.k = k
        self.size = len(self.data)
        self.centerList = self.SetInitialCenter()
        self.preCenterList = self.centerList
        self.classifyResultList = np.full(self.size, np.nan)
        self.maxIterTimes = maxIterTimes
        self.dim = len(self.data[0])
    
    ##計算距離
    def Distance(self, x, y):
        return ((x-y)**2).sum()**0.5
    
    ##更新中心點
    def UpdateCenter(self):
        self.preCenterList = self.centerList.copy()
        for centerIndex in range(self.k):
            self.centerList[centerIndex] = np.mean(self.data[np.where(self.classifyResultList == centerIndex)], axis = 0)
    
    ##將資料分群
    def ClassifyPoint(self):
        for dataIndex in range(self.size):
            dataPoint = self.data[dataIndex]
            distanceList = np.array([])
            for centerIndex in range(self.k):
                centerPoint = self.centerList[centerIndex]
                distance = self.Distance(dataPoint, centerPoint) 
                distanceList = np.append(distanceList, distance)
            classifyResult = np.where(distanceList == min(distanceList))[0]
            ##防止距離相同的error
            if (len(classifyResult) != 1):
                classifyResult = np.array(classifyResult[0])
            self.classifyResultList[dataIndex] = classifyResult
            
    ##設定初始權重
    def SetInitialCenter(self):
        randomIndex = random.sample(range(self.size),self.k)
        center = self.data[randomIndex]
        return center
    
    ##計算精確度
    def Accuracy(self):
        classList = np.unique(self.target)
        classifyCorrectNum = 0
        for classType in classList:
            someClassIndex = np.where(self.target==classType)##找出真實結果在同是第n群的index列表
            expectType = mode(self.classifyResultList[someClassIndex])[0][0]##找出在這群index中預期分類的眾數
            classifyCorrectNum += np.size(np.where(self.classifyResultList[someClassIndex] == expectType),1) ##加總該群預期正確的個數
        return classifyCorrectNum/len(self.target)##計算精確度並回傳
    ##
    def plot(self):
        plt.subplot(224)
        plt.plot([0,1],[0,4])
    ##執行
    def Fit(self):
        iterTimes = 0
        while(True):
            self.ClassifyPoint()
            self.UpdateCenter()
            iterTimes += 1
            ##終止條件
            if( (iterTimes > self.maxIterTimes) & (self.centerList == self.preCenterList).any() ):
                break
            
            
        
```

**FCM CLASS**


```python
class FCMeans(Kmeans):
    
    def __init__(self, data, target, k, m, maxIterTimes = 30):
        super().__init__(data, target, k, maxIterTimes)
        self.m = m
        self.w = self.SetInitialWeight()
        
    def UpdateCenter(self):
        for centerIndex in range(self.k):
            sum_of_wx = np.zeros(self.dim)
            sum_of_w = 0
            for dataIndex in range(self.size):
                sum_of_wx += self.w[centerIndex, dataIndex]**self.m * self.data[dataIndex]
                sum_of_w += self.w[centerIndex, dataIndex]**self.m
            self.centerList[centerIndex] = sum_of_wx/sum_of_w
    
    def UpdateWeight(self):
        for dataIndex in range(self.size):
            for centerIndex_i in range(self.k):
                sum_of_distance_ratio = 0
                for centerIndex_k in range(self.k):
                    sum_of_distance_ratio += (self.Distance(self.centerList[centerIndex_i], self.data[dataIndex])/(self.Distance(self.centerList[centerIndex_k], self.data[dataIndex])))**(2/(self.m-1))
                self.w[centerIndex_i, dataIndex] = 1/sum_of_distance_ratio
            ##更新分類  
            classifyResult = np.argmax(self.w[:, dataIndex])
            """##防止距離相同的error
            if (len(classifyResult) != 1):
                classifyResult = np.array(classifyResult[0])"""
            self.classifyResultList[dataIndex] = classifyResult
        
    def SetInitialWeight(self):
        weightList = np.array([])
        for i in range(self.size):
            weight = np.random.uniform(0,(1/(self.k-1)),self.k-1)
            weight = np.append(weight, 1-sum(weight))
            weightList = np.append(weightList, weight, axis = 0)
        weightList = weightList.reshape((self.size,self.k))
        return np.transpose(weightList)
    
    def Fit(self):
        iterTimes = 0
        while(True):
            self.UpdateCenter()
            self.UpdateWeight()
            iterTimes += 1
            ##終止條件
            if( (iterTimes > self.maxIterTimes) and (self.centerList == self.preCenterList).any() ):
                break
        
        
```

**讀取資料**


```python
from sklearn import datasets
iris = datasets.load_iris()
X = iris.data[:, :]  # we only take the first two features.
y = iris.target
```

**Kmeans用法**


```python
model = Kmeans(X, y, k = 3, maxIterTimes = 30)
model.Fit()
model.Accuracy()
```




    0.8866666666666667




```python
model.centerList
```




    array([[5.88360656, 2.74098361, 4.38852459, 1.43442623],
           [5.006     , 3.428     , 1.462     , 0.246     ],
           [6.85384615, 3.07692308, 5.71538462, 2.05384615]])



**FCM用法**


```python
model = FCMeans(X, y, k = 3, m=2, maxIterTimes = 30)
model.Fit()
model.Accuracy()
```




    0.8933333333333333




```python
model.centerList
```




    array([[5.88892958, 2.76106836, 4.36394758, 1.39731295],
           [6.77500786, 3.05238129, 5.64677759, 2.05354504],
           [5.00396595, 3.414089  , 1.48281531, 0.25354622]])



**兩種模型比較**


```python
KM_accuracyCounter = 0
FCM_accuracyCounter = 0
for i in range(100):
    KMmodel = Kmeans(X, y, k = 3, maxIterTimes = 30)
    KMmodel.Fit()
    KM_accuracyCounter += KMmodel.Accuracy()
    
    FCMmodel = FCMeans(X, y, k = 3, m=2, maxIterTimes = 30)
    FCMmodel.Fit()
    FCM_accuracyCounter += FCMmodel.Accuracy()
    
KM_accuracyRatio = KM_accuracyCounter/100
FCM_accuracyRatio = FCM_accuracyCounter/100
print("K-means 100次平均精確度:", KM_accuracyRatio)
print("FCM 100次平均精確度:", FCM_accuracyRatio)
```

    K-means 100次平均精確度: 0.8815333333333336
    FCM 100次平均精確度: 0.8933333333333319
    

**畫圖**


```python
plt.figure(figsize=(8,24))
colors = cm.rainbow(np.linspace(0, 1, model.k))
for subplotPosition,i  in zip(range(311,314), range(1,4)):
    plt.subplot(subplotPosition)
    for classIndex, c, in zip(range(model.k), colors):   
        data = model.data[np.where(model.classifyResultList == classIndex)]
        plt.scatter(data[:,0],data[:,i], s = 30, c = c, marker = "x")

```

    'c' argument looks like a single numeric RGB or RGBA sequence, which should be avoided as value-mapping will have precedence in case its length matches with 'x' & 'y'.  Please use a 2-D array with a single row if you really want to specify the same RGB or RGBA value for all points.
    'c' argument looks like a single numeric RGB or RGBA sequence, which should be avoided as value-mapping will have precedence in case its length matches with 'x' & 'y'.  Please use a 2-D array with a single row if you really want to specify the same RGB or RGBA value for all points.
    'c' argument looks like a single numeric RGB or RGBA sequence, which should be avoided as value-mapping will have precedence in case its length matches with 'x' & 'y'.  Please use a 2-D array with a single row if you really want to specify the same RGB or RGBA value for all points.
    'c' argument looks like a single numeric RGB or RGBA sequence, which should be avoided as value-mapping will have precedence in case its length matches with 'x' & 'y'.  Please use a 2-D array with a single row if you really want to specify the same RGB or RGBA value for all points.
    'c' argument looks like a single numeric RGB or RGBA sequence, which should be avoided as value-mapping will have precedence in case its length matches with 'x' & 'y'.  Please use a 2-D array with a single row if you really want to specify the same RGB or RGBA value for all points.
    'c' argument looks like a single numeric RGB or RGBA sequence, which should be avoided as value-mapping will have precedence in case its length matches with 'x' & 'y'.  Please use a 2-D array with a single row if you really want to specify the same RGB or RGBA value for all points.
    'c' argument looks like a single numeric RGB or RGBA sequence, which should be avoided as value-mapping will have precedence in case its length matches with 'x' & 'y'.  Please use a 2-D array with a single row if you really want to specify the same RGB or RGBA value for all points.
    'c' argument looks like a single numeric RGB or RGBA sequence, which should be avoided as value-mapping will have precedence in case its length matches with 'x' & 'y'.  Please use a 2-D array with a single row if you really want to specify the same RGB or RGBA value for all points.
    'c' argument looks like a single numeric RGB or RGBA sequence, which should be avoided as value-mapping will have precedence in case its length matches with 'x' & 'y'.  Please use a 2-D array with a single row if you really want to specify the same RGB or RGBA value for all points.
    


![png](https://github.com/johnsonquo/Kmeans/tree/master/pic//output_16_1.png)


**參考資料**  

[Kmeans參考](https://medium.com/@chih.sheng.huang821/%E6%A9%9F%E5%99%A8%E5%AD%B8%E7%BF%92-%E9%9B%86%E7%BE%A4%E5%88%86%E6%9E%90-k-means-clustering-e608a7fe1b43)  

[FCM參考](https://blog.csdn.net/zjsghww/article/details/50922168)
