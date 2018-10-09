#!/usr/bin/python3
# -*- coding:utf-8 -*-

# @Time      :  2018/10/8 10:09
# @Auther    :  WangYang
# @Email     :  evilwangyang@126.com
# @Project   :  LinearRegression
# @File      :  LinearRegression_test1.py
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
	xArr,yArr = loadDataSet('ex0.txt')

	print(xArr[0:2])

	ws = standRegres(xArr,yArr)
	print(ws)

	xMat = np.mat(xArr)
	yMat = np.mat(yArr)
	yHat = xMat * ws

	print(np.corrcoef(yHat.T,yMat))

	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.scatter(xMat[:,1].flatten().A[0],yMat.T[:,0].flatten().A[0])

	xCopy = xMat.copy()
	xCopy.sort(0)
	yHat = xCopy*ws
	ax.plot(xCopy[:,1],yHat)

	plt.show()

