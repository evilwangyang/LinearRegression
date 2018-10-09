#!/usr/bin/python3
# -*- coding:utf-8 -*-

# @Time      :  2018/10/9 22:14
# @Auther    :  WangYang
# @Email     :  evilwangyang@126.com
# @Project   :  LinearRegression
# @File      :  LinearRegression_test3.py
# @Software  :  PyCharm Community Edition

# ********************************************************* 
import numpy as np

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

def standRegres(xArr,yArr):
	xMat = np.mat(xArr)
	yMat = np.mat(yArr).T
	xTx = xMat.T*xMat
	if np.linalg.det(xTx) == 0.0:
		print('This matrix is singular, cannot do inverse')
		return
	ws = xTx.I * (xMat.T*yMat)
	return ws

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

def rssError(yArr,yHatArr):
	return ((yArr-yHatArr)**2).sum()

if __name__ == '__main__':
	abX,abY = loadDataSet('abalone.txt')
	yHat01 = lwlrTest(abX[0:99],abX[0:99],abY[0:99],0.1)
	yHat1 = lwlrTest(abX[0:99], abX[0:99], abY[0:99], 1)
	yHat10 = lwlrTest(abX[0:99], abX[0:99], abY[0:99], 10)

	print(rssError(abY[0:99],yHat01.T))
	print(rssError(abY[0:99], yHat1.T))
	print(rssError(abY[0:99], yHat10.T))

	yHat01 = lwlrTest(abX[100:199],abX[0:99],abY[0:99],0.1)
	yHat1 = lwlrTest(abX[100:199], abX[0:99], abY[0:99], 1)
	yHat10 = lwlrTest(abX[100:199], abX[0:99], abY[0:99], 10)

	print(rssError(abY[100:199],yHat01.T))
	print(rssError(abY[100:199], yHat1.T))
	print(rssError(abY[100:199], yHat10.T))

	ws = standRegres(abX[0:99],abY[0:99])
	yHat = np.mat(abX[100:199])*ws
	print(rssError(abY[100:199],yHat.T.A))
