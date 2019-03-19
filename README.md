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
    
    

* 使用範例:
```python

from sklearn import datasets
iris = datasets.load_iris()
X = iris.data[:, :]  # we only take the first two features.
y = iris.target

model = Kmeans(X,3)
model.Fit()
model.centerList
```
![](https://github.com/johnsonquo/Kmeans/blob/master/pic/%E5%AF%A6%E4%BD%9C.PNG)  
