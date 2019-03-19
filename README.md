# Kmeans
Kmeans演算法

```python
from sklearn import datasets
iris = datasets.load_iris()
X = iris.data[:, :]  # we only take the first two features.
y = iris.target

model = Kmeans(X,3)
model.Fit()
model.centerList
```
![](https://github.com/johnsonquo/Kmeans/edit/master/pic/實作.gif)  
