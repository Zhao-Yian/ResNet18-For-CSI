# -*- coding: utf-8 -*-
"""
Created on Thu Sep 27 14:56:57 2018

@author: Administrator
"""

import matplotlib.pyplot as plt
from matplotlib import animation
import os
import pickle
import cmath
#import math
from math import *
import scipy.stats as scita
import scipy.signal as signal
import scipy.io as scio
from scipy.signal import butter
import numpy as np
from scipy.spatial.distance import euclidean
from numpy import linalg as la
import scipy.io as sio
from keras.preprocessing.sequence import pad_sequences


winLen = 30






def db(x):
#变换到log空间
	if (x==0):
		ans=0
	else:
		ans=20*(np.log10(x))
	return ans

def butterworth_II(file,Fc):
#去噪
	N  = 2    # Filter order
	Wn = 2*(np.pi)*Fc # Cutoff frequency
	B, A = butter(N, Wn, output='ba')
	ret = signal.filtfilt(B,A,file)
	return ret

def relative_phase(tmp1,tmp2):
#计算相对相位
	tmp=tmp1*np.conjugate(tmp2)
	tmp_1=(tmp.real)/(abs(tmp))
	ret=np.arccos(tmp_1)

	return (ret)


def file_data(filename):
#解析原始数据包
	l=[]
	with open(filename, "rb") as f:
		while 1:
			try:
				flag = 1
				k = pickle.load(f)
				for i in range(90):
					if(abs(k[i]) <= 0):
						flag = 0
				if flag == 0:
					continue
				l.append(k[0:90])
			except Exception as e:
				break
	a = np.array(l).T
	return(a)

def csi_amplitude(file):
	[row,col]=file.shape
	newFile = np.zeros((row,col))
	for i in range(row):
		for j in range(col):
			newFile[i,j]=db(abs(file[i,j]))
			#print (file[i,j])
	ret=np.array(newFile)
	return (ret.real)

def csi_relative_phase(file):
#计算数据的相对相位
	file=file.reshape(3,30,-1)
	[row,col,other]=file.shape
	csi_ant1=file[0]
	csi_ant2=file[1]
	csi_ant3=file[2]
	rephase1_2=[]
	rephase1_3=[]
	rephase3_2=[]
	rephase_all=[]

	for i in range(col):
		tmp1_2=relative_phase(csi_ant1[i],csi_ant2[i])

		tmp1_3=relative_phase(csi_ant1[i],csi_ant3[i])
		tmp3_2=relative_phase(csi_ant3[i],csi_ant2[i])
		rephase1_2.append(tmp1_2)
		rephase1_3.append(tmp1_3)
		rephase3_2.append(tmp3_2)

	for j in range(30):
		rephase_all.append(rephase1_2[j])
	for j in range(30):
		rephase_all.append(rephase1_3[j])
	for j in range(30):
		rephase_all.append(rephase3_2[j])

	ret=np.array(rephase_all)
	return ret.real
 
def preprocessingPhase(phases):
    """
    将相位进行线性变换
    index是 -28 到 28 根据 IEEE 802.11n 协议
    返回变换后的相位
    """
    phases=np.reshape(phases,(-1,3*30))
    #print(phases.shape)
    index =list( range(-28,0,2)) + [-1, 1] + list(range(3,28, 2)) + [28]
    tphases=np.array(np.zeros(phases.shape))
    for m in range(N):
        for l in range(10):
            clear = True
            base = 0
            tphases[m][0] = phases[m][0]

            for i in range(1, 30):
                if phases[m][i] - phases[m][i-1] > pi:
                    base += 1
                    clear = False
                elif phases[m][i] - phases[m][i-1] < -pi:
                    base -= 1
                    clear = False
                tphases[m][i] = phases[m][i] - 2 * pi * base

            if clear == True:
                break
            else:
                for i in range(30):
                    phases[m][i] = tphases[m][i] - (tphases[m][29] - tphases[m][0])* 1.0 /(28 - (-28)) * (index[i])- 1.0 / 30 * sum([tphases[m][j] for j in range(30)])
      
                
    return phases
def relative_conjugate(tmp1,tmp2):
#计算相对chufa
	#tmp=tmp1/np.conjugate(tmp2)
	tmp=tmp1/tmp2
	#tmp_1=(tmp.real)/(abs(tmp))
	#ret=np.arccos(tmp_1)
    
	return (tmp)
def csi_relative_conjugate(file):
#计算数据的相对相位
	file=file.reshape(3,30,-1)
	[row,col,other]=file.shape
	csi_ant1=file[0]
	csi_ant2=file[1]
	csi_ant3=file[2]
	rephase1_2=[]
	rephase1_3=[]
	rephase3_2=[]
	rephase_all=[]

	for i in range(col):
		tmp1_2=relative_conjugate(csi_ant1[i],csi_ant2[i])

		tmp1_3=relative_conjugate(csi_ant1[i],csi_ant3[i])
		tmp3_2=relative_conjugate(csi_ant3[i],csi_ant2[i])
		rephase1_2.append(tmp1_2)
		rephase1_3.append(tmp1_3)
		rephase3_2.append(tmp3_2)
	for j in range(30):
		rephase_all.append(rephase1_2[j])
	#for j in range(30):
	#	rephase_all.append(rephase1_3[j])
	#for j in range(30):
	#	rephase_all.append(rephase3_2[j])

	ret=np.array(rephase_all)
	return ret    
def roadDetect(input):
	csiMatrix=[]
	ans=input
	for i in range(input.shape[0]):
		csiList=[]
		for j in range(0,input.shape[1],3):	
			csiStd=input[i][j]-np.mean(input[i][j:j+2])
			csiList.append(csiStd)
		csiMatrix.append(csiList)
	ans=np.array(csiMatrix)
	return ans
 
def srAlgorithm(input):
## 循环找动态变量和面积最大量
	step=0.005
	step1=0.001
	finaList=[]
	ansList,orderList,ansStd = densityDetect(input,step)
	if (len(ansList)>1):
		while (((ansList[0])!=np.max(ansList)) or (ansStd[0])!=np.min(ansStd) ):
			step=step+step1
			ansList,orderList,ansStd=densityDetect(input,step)
			
	else:
		while ((ansStd[0])!=np.min(ansStd)):
			step=step-step1
			ansList,orderList,ansStd=densityDetect(input,step)
	if len(orderList)>=1:
		finaNum=(orderList[0])
	else:
		finaNum=orderList
	return finaNum,step



def densityDetect(input,step):
	input=np.abs(input)
	ansLen=[]
	ansStd=[]
	tmp1=[]
	tmp=[]
	orderList=[]
	tmpList=np.argsort(input)
	tt=0
	j=0
	while((np.min(input)+(tt+1)*step)<=np.max(input)):	
		while ((input[tmpList[j]]>(np.min(input)+(tt)*step)) and (input[tmpList[j]]<=(np.min(input)+(tt+1)*step)) or input[tmpList[j]]==np.min(input) ) :
			tmp.append(input[tmpList[j]])
			tmp1.append(tmpList[j])
			j=j+1
		ansLen.append(len(tmp))
		orderList.append(tmp1)
		ansStd.append(np.std(tmp))
		tmp=[]
		tmp1=[]
		tt=tt+1
	return ansLen,orderList,ansStd
 
def solve_data(input_path):
    load_data = sio.loadmat(input_path)
    #print(load_data)
    raw_data= load_data['csi']  #维度N*3*90(30)???
    #print(raw_data)
    #print(raw_data.shape)
    N = raw_data.shape[0]
    raw_data=np.reshape(raw_data,[N*30*3,1])
    raw_data=np.reshape(raw_data,(3*30,-1),order="F")  #维度90*N
    print(raw_data.shape)
    return raw_data


    
def classification(input):
#根据标签list转化为标签矩阵（one-hot编码）
	labelOrder=set()
	labelList=[]
	i=0
	for label in input:
		label=label.split('_')[0]
		labelList.append(label)
		labelOrder.add(label)
	labelOrder=list(labelOrder)
	labelCol=len(labelOrder)
	labelMatrix = []
	for label in labelList:
		l = [0] * labelCol
		labelNum = labelOrder.index(label)
		l[labelNum] = 1
		labelMatrix.append(l)
	labelMatrix = np.array(labelMatrix, dtype=object)
	return labelMatrix,labelOrder
    
if __name__=='__main__'	:
      path="C:/data/mat"
      labelFina=[]
      choseMatrix=[]
      num=[]
      files_all= os.listdir(path)
      for file in files_all:
          filepath=os.path.join(path, file)
          #print(filepath)
          t=os.listdir(filepath)
          for files in t:
              if files:
                  print(files)
                  ans=solve_data(filepath+'/'+files)
                  #amplitudes=csi_amplitude(ans)
                  
                  divide=csi_relative_conjugate(ans)
                  
                  #csiPhase=phases
                  amplitudes=csi_amplitude(divide)
                  csiAmplitude=butterworth_II(amplitudes,0.03)
                  plotNum=[1]
                  csiStdmatrix=roadDetect(csiAmplitude)
                  print(csiStdmatrix.shape)
                  numList=[]
                  for jj in plotNum: 
                     finaNum,step=srAlgorithm(csiStdmatrix[jj])
                     numList=list(set(numList)^set(finaNum))
                  #compare=list(range(len(csiStdmatrix[plotNum])))
                  #finaNum1=list(set(compare)^set(finaNum))
                  # print(len(csiAmplitude[plotNum]),len(finaNum))
                  numList=3*np.sort(numList)           
                  csiAmplitude[:,numList]=0
                 
                  csichose=csiAmplitude
                  choseMatrix.extend((csichose).tolist())
                  labels=30*[file]
                  labelFina.extend(labels)
      labelMatrix,labelOrder=classification(labelFina)
      temp=0
      for i in range(len(choseMatrix)):
          if temp<len(choseMatrix[i]):
              temp=len(choseMatrix[i])
      for i in range(len(choseMatrix)):
          if temp!=len(choseMatrix[i]):
              num=temp-len(choseMatrix[i])   
              for j in range(num):
                    choseMatrix[i].append(0.0)
      result=np.array(choseMatrix)  
     
                    

      #for i in range(len(num)): 
          
      #choseMatrix_narry=np.array(choseMatrix)
      
      #result=pad_sequences(choseMatrix, maxlen=None, dtype='int32',padding='post', truncating='pre', value=0.)
      shape=result.shape
      result=result.reshape((-1))
      result[np.isnan(result)]=0
      result=result.reshape(shape)
      np.save('train//'+"not_predict.npy",result)
      np.save('train//'+"volAlabelshaochu.npy",labelMatrix)
      plt.show()



	
	

	
	
	
	
	
	
