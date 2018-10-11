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

def rssError(yArr,yHatArr):
	return ((yArr-yHatArr)**2).sum()

def regularize(xMat):
	inMat = xMat.copy()
	inMeans = np.mean(inMat,0)
	inVar = np.var(inMat,0)
	inMat = (inMat - inMeans)/inVar
	return inMat

def stageWise(xArr,yArr,eps=0.01,numIt=100):
	xMat = np.mat(xArr)
	yMat = np.mat(yArr).T
	yMean = np.mean(yMat,0)
	yMat = yMat - yMean
	xMat = regularize(xMat)
	m,n = np.shape(xMat)
	returnMat = np.zeros((numIt,n))
	ws = np.zeros((n,1))
	wsTest = ws.copy()
	wsMax = ws.copy()
	for i in range(numIt):
		print(ws.T)
		lowestError = np.inf
		for j in range(n):
			for sign in [-1,1]:
				wsTest = ws.copy()
				wsTest[j] += eps*sign
				yTest = xMat*wsTest
				rssE = rssError(yMat.A,yTest.A)
				if rssE < lowestError:
					lowestError = rssE
					wsMax = wsTest
		ws = wsMax.copy()
		returnMat[i,:] = ws.T
	return returnMat

def standRegres(xArr,yArr):
	xMat = np.mat(xArr)
	yMat = np.mat(yArr).T
	xTx = xMat.T*xMat
	if np.linalg.det(xTx) == 0.0:
		print('This matrix is singular, cannot do inverse')
		return
	ws = xTx.I * (xMat.T*yMat)
	return ws

if __name__ == '__main__':
	# abX,abY = loadDataSet('abalone.txt')
	# ridgeWeights = ridgeTest(abX,abY)
	#
	# fig = plt.figure()
	# ax = fig.add_subplot(111)
	# ax.plot(ridgeWeights)
	# plt.show()

	xArr,yArr = loadDataSet('abalone.txt')
	# print(stageWise(xArr,yArr,0.01,200))

	print(stageWise(xArr,yArr,0.001,5000))

	xMat = np.mat(xArr)
	yMat = np.mat(yArr).T
	xMat = regularize(xMat)
	yMean = np.mean(yMat,0)
	yMat = yMat - yMean
	weights = standRegres(xMat,yMat.T)
	print(weights.T)
