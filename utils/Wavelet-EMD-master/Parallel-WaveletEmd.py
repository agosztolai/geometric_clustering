#Import OpenCv library
#from cv2 import *
import cv2
import time
#import cv
import cv2.cv as cv
import matplotlib.pyplot as plt                     #importing library for plotting
import pywt
import numpy as np 
import math
import os
from scipy import signal
from pywt import wavedecn, waverecn
#from joblib import Parallel, delayed
import threading

### HISTOGRAM FUNCTION #########################################################
def calcHistogram(src):
	# Convert to HSV
	hsv = cv.CreateImage(cv.GetSize(src), 8, 3)
	cv.CvtColor(src, hsv, cv.CV_BGR2HSV)

	# Extract the H and S planes
	size = cv.GetSize(src)
	h_plane = cv.CreateMat(size[1], size[0], cv.CV_8UC1)
	s_plane = cv.CreateMat(size[1], size[0], cv.CV_8UC1)
	cv.Split(hsv, h_plane, s_plane, None, None)
	planes = [h_plane, s_plane]

	#Define numer of bins
	h_bins = 30
	s_bins = 32

	#Define histogram size
	hist_size = [h_bins, s_bins]

	# hue varies from 0 (~0 deg red) to 180 (~360 deg red again */
	h_ranges = [0, 180]

	# saturation varies from 0 (black-gray-white) to 255 (pure spectrum color)
	s_ranges = [0, 255]

	ranges = [h_ranges, s_ranges]

	#Create histogram
	hist = cv.CreateHist([h_bins, s_bins], cv.CV_HIST_ARRAY, ranges, 1)

	#Calc histogram
	cv.CalcHist([cv.GetImage(i) for i in planes], hist)

	cv.NormalizeHist(hist, 1.0)

	#Return histogram
	return hist

### EARTH MOVERS ############################################################
def calcEM(hist1,hist2,h_bins,s_bins):
	print hist1
	return 0
	#Define number of rows
	numRows = h_bins*s_bins

	sig1 = cv.CreateMat(numRows, 3, cv.CV_32FC1)
	sig2 = cv.CreateMat(numRows, 3, cv.CV_32FC1)    

	for h in range(hist1_bins):
		for s in range(s_bins): 
			bin_val = cv.QueryHistValue_2D(hist1, h, s)
			cv.Set2D(sig1, h*s_bins+s, 0, cv.Scalar(bin_val))
			cv.Set2D(sig1, h*s_bins+s, 1, cv.Scalar(h))
			cv.Set2D(sig1, h*s_bins+s, 2, cv.Scalar(s))

			bin_val = cv.QueryHistValue_2D(hist2, h, s)
			cv.Set2D(sig2, h*s_bins+s, 0, cv.Scalar(bin_val))
			cv.Set2D(sig2, h*s_bins+s, 1, cv.Scalar(h))
			cv.Set2D(sig2, h*s_bins+s, 2, cv.Scalar(s))
	print cv.GetSize(sig1)
	sig3 = np.asarray(sig1)
	sig4 = np.asarray(sig2)
	#wavelet = pywt.ContinuousWavelet('gaus1') 
	#This is the important line were the OpenCV EM algorithm is called
	#coef, freqs=pywt.cwt(sig3-sig4,np.arange(1,129),wavelet)
	#coef, freqs=signal.cwt(sig3-sig4, signal.ricker, np.arange(1,129))
	#coeffs = wavedecn(sig3-sig4, 'db1')
	#coeffs = pywt.dwtn(sig3-sig4,'db1')
	#print coeffs
	'''
	sum = 0
	for x in range(1,129):
		for y in range(0,256):
			sum = sum+(math.pow(2,x*-2)*abs(coef[x-1][y]))  #Calculating EMD in wavelet domain
	'''
	return cv.CalcEMD2(sig1,sig2,cv.CV_DIST_L2)

### MAIN ########################################################################
#if __name__=="__main__":
#Load image 1
#src1 = cv.LoadImage("image1.jpg")
#src1 = cv2.imread("image1.jpg", 0)
#src2 = cv2.imread("image3.jpg", 0)

temp2 = []
for filename in os.listdir("image.orig"):
	img = cv2.imread("image.orig/"+ filename,0)                        #reads an input image
#    histr = cv2.calcHist([img],[0],None,[256],[0,256])          #frequency of pixels in range 0 to 255
	temp2.append(img)

#for N in [32, 64, 96, 128,150,200]: #Number of Bins in a Histogram
timeTaken = {}

def func(arg,x): 
	time_start = time.time()
	#x = temp2[0]
	for src in temp2[1:1000]:
		#time_start = time.time()
		histg1 = cv2.calcHist([src],[0],None,[arg],[0,256]) 
		histg2 = cv2.calcHist([x],[0],None,[arg],[0,256]) 
		
		a = np.zeros((arg, 2))
		b = np.zeros((arg, 2))

		for i in range(arg):
			a[i][1] = histg1[i][0]
			a[i][0] = 1
			b[i][1] = histg2[i][0]
			b[i][0] = 1

		a64 = cv.fromarray(a)
		a32 = cv.CreateMat(a64.rows, a64.cols, cv.CV_32FC1)
		cv.Convert(a64, a32)

		b64 = cv.fromarray(b)
		b32 = cv.CreateMat(b64.rows, b64.cols, cv.CV_32FC1)
		cv.Convert(b64, b32)
		#time_end = time.time()
		cv.CalcEMD2(a32,b32,cv.CV_DIST_L2)
		#print(time_end - time_start)
		#print()
		'''  Uncomment to Execute Wavelet Emd and Comment Wavelet Emd
		#time_start = time.time()
		#histg1 = cv2.calcHist([src1],[0],None,[N],[0,256]) 
		#histg2 = cv2.calcHist([src2],[0],None,[N],[0,256]) 
		
		histg2 = np.squeeze(histg2)
		histg1 = np.squeeze(histg1)
		lst = pywt.families(short=True)
		wavelet = pywt.ContinuousWavelet('gaus1')
		coef, freqs=pywt.cwt(histg2 - histg1, np.arange(1,10), wavelet)
		sum = 0

		for x in range(1,10):
			for y in range(N):
				sum = sum+(math.pow(2,x*-2)*abs(coef[x-1][y]))  #Calculating EMD in wavelet domain
		'''
	time_end = time.time()
	timeTaken[arg] = (time_end-time_start)    
		#return (time_end - time_start)
	#break

	'''
	sig1 = cv.CreateMat(32, 1, cv.CV_32FC1)

	print(type(histg1))
	x = cv.fromarray(histg1)

	#sig1 = np.asarray(histg1)
	#sig2 = np.asarray(histg2)
	print cv.CalcEMD2(cv.fromarray(histg1),cv.fromarray(histg2),cv.CV_DIST_L2)
	'''
	

	#histg1 = np.squeeze(histg1)
	#histg2 = np.squeeze(histg2)
	#print histg1[0]
	#print histg1


	# Convert from numpy array to CV_32FC1 Mat
	#Load image 1
	#src2 = cv.LoadImage("image2.jpg")
	'''
	# Get histograms
	histSrc1= calcHistogram(src1)
	histSrc2= calcHistogram(src2)

	# Compare histograms using earth mover's
	histComp = calcEM(histSrc1,histSrc2,30,32)

	#Print solution
	print(histComp)
	'''

arg_instances = [32, 64, 96, 128,150,200]
thread_list = []

for i in arg_instances:
	t = threading.Thread(target=func,args=(i,temp2[0]))
	thread_list.append(t)

# Starts threads
for thread in thread_list:
	thread.start()

#waits till all the threads finish
for thread in thread_list:
	thread.join()

for i in arg_instances:
	print timeTaken[i]
