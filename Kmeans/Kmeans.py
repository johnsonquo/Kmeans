#!/usr/bin/env python
# coding: utf-8

import numpy as np
import random

class Kmeans:
    
    def __init__(self,data, k, maxIterTimes = 30):
        self.data = data
        self.k = k
        self.size = len(self.data)
        self.centerList = self.SetInitialCenter()
        self.preCenterList = self.centerList
        self.classifyResultList = np.full(self.size, np.nan)
        self.maxIterTimes = maxIterTimes
    
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
    
    ##執行
    def Fit(self):
        iterTimes = 0
        while(True):
            self.ClassifyPoint()
            self.UpdateCenter()
            iterTimes += 1
            ##終止條件
            if( (iterTimes < self.maxIterTimes) & (self.centerList != self.preCenterList).any() ):
                break
            
            
