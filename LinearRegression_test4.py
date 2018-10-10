#!/usr/bin/python3
# -*- coding:utf-8 -*-

# @Time      :  2018/10/10 11:00
# @Auther    :  WangYang
# @Email     :  evilwangyang@126.com
# @Project   :  LinearRegression
# @File      :  LinearRegression_test4.py
# @Software  :  PyCharm Community Edition

# ********************************************************* 
import numpy as np
import matplotlib.pyplot as plt

def loadDataSet(fileName):
	numFeat = len(open(fileName).readline().split('\t')) - 1
	dataMat = []
	labelMat = []
	fr = open(fileName)
	for line in fr.readlines():
		lineArr = []
		curLine = line.strip().split('\t')
		for i in range(numFeat):
			lineArr.append(float(curLine[i]))
		dataMat.append(lineArr)
		labelMat.append(float(curLine[-1]))
	return dataMat,labelMat

def ridgeRegres(xMat,yMat,lam=0.2):
	xTx = xMat.T*xMat
	denom = xTx + np.eye(np.shape(xMat)[1]) * lam
	if np.linalg.det(denom) == 0.0:
		print('This matrix is singular, cannot do inverse')
		return
	ws = denom.I * (xMat.T*yMat)
	return ws

def ridgeTest(xArr,yArr):
	xMat = np.mat(xArr)
	yMat = np.mat(yArr).T
	yMean = np.mean(yMat,0)
	yMat = yMat - yMean
	xMeans = np.mean(xMat,0)
	xVar = np.var(xMat,0)
	xMat = (xMat - xMeans)/xVar
	numTestPts = 30
	wMat = np.zeros((numTestPts,np.shape(xMat)[1]))
	for i in range(numTestPts):
		ws = ridgeRegres(xMat,yMat,np.exp(i-10))
		wMat[i,:] = ws.T
	return wMat

if __name__ == '__main__':
	abX,abY = loadDataSet('abalone.txt')
	ridgeWeights = ridgeTest(abX,abY)

	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.plot(ridgeWeights)
	plt.show()