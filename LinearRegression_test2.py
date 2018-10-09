#!/usr/bin/python3
# -*- coding:utf-8 -*-

# @Time      :  2018/10/9 19:33
# @Auther    :  WangYang
# @Email     :  evilwangyang@126.com
# @Project   :  LinearRegression
# @File      :  LinearRegression_test2.py
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

def lwlr(testPoint,xArr,yArr,k=1.0):
	xMat = np.mat(xArr)
	yMat = np.mat(yArr).T
	m = np.shape(xMat)[0]
	weights = np.mat(np.eye(m))
	for j in range(m):
		diffMat = testPoint - xMat[j,:]
		weights[j,j] = np.exp(diffMat*diffMat.T/(-2.0*k**2))
	xTx = xMat.T * (weights * xMat)
	if np.linalg.det(xTx) == 0.0:
		print('This matrix is singular, cannot do inverse')
		return
	ws = xTx.I * (xMat.T * (weights * yMat))
	return testPoint * ws

def lwlrTest(testArr,xArr,yArr,k=1.0):
	m = np.shape(testArr)[0]
	yHat = np.zeros(m)
	for i in range(m):
		yHat[i] = lwlr(testArr[i],xArr,yArr,k)
	return yHat

if __name__ == '__main__':
	xArr,yArr = loadDataSet('ex0.txt')

	print(yArr[0])
	print(lwlr(xArr[0],xArr,yArr,1.0))
	print(lwlr(xArr[0],xArr,yArr,0.001))

	yHat = lwlrTest(xArr,xArr,yArr,0.003)

	xMat = np.mat(xArr)
	srtInd = xMat[:,1].argsort(0)
	xSort = xMat[srtInd][:,0,:]

	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.plot(xSort[:,1],yHat[srtInd])
	ax.scatter(xMat[:,1].flatten().A[0],np.mat(yArr).T.flatten().A[0],s=2,c='red')
	plt.show()