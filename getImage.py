# My first whack at using computer vision to analyze a frame of honeybees.
# reference:https://opencv-python-tutroals.readthedocs.org/en/latest/py_tutorials/py_imgproc/py_thresholding/py_thresholding.html#thresholding
#smoothing:https://opencv-python-tutroals.readthedocs.org/en/latest/py_tutorials/py_imgproc/py_filtering/py_filtering.html#filtering
import cv2
import numpy as np
from matplotlib import pyplot as plt

class BeeAnalyze:
	def __init__(self,fileName):
		self.img = cv2.imread(fileName)
	def crop(self):
		pass
	def resize(self):
		pass
	def smooth(self):
		pass
	def show(self,title='image'):
		cv2.imshow(title,self.img)
		cv2.waitKey(0)
		cv2.destroyAllWindows()

	def simpleThresh(self):
		pass
	def adaptiveThresh(self):
		pass
	def otsuBinary(self):
		pass
	def houghCircle(self):
		pass


def simpleThreshold(fileName):
	import cv2
	import numpy as np
	from matplotlib import pyplot as plt

	img = cv2.imread(fileName,0)
	ret,thresh1 = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
	ret,thresh2 = cv2.threshold(img,127,255,cv2.THRESH_BINARY_INV)
	ret,thresh3 = cv2.threshold(img,127,255,cv2.THRESH_TRUNC)
	ret,thresh4 = cv2.threshold(img,127,255,cv2.THRESH_TOZERO)
	ret,thresh5 = cv2.threshold(img,127,255,cv2.THRESH_TOZERO_INV)

	titles = ['Original Image','BINARY','BINARY_INV','TRUNC','TOZERO','TOZERO_INV']
	images = [img, thresh1, thresh2, thresh3, thresh4, thresh5]

	for i in xrange(6):
		plt.subplot(2,3,i+1),plt.imshow(images[i],'gray')
		plt.title(titles[i])
		plt.xticks([]),plt.yticks([])

	plt.show()

def adaptiveThreshold(fileName):
	import cv2
	import numpy as np
	from matplotlib import pyplot as plt

	img = cv2.imread(fileName,0)
	img = cv2.medianBlur(img,5)

	ret,th1 = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
	th2 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
				cv2.THRESH_BINARY,11,2)
	th3 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
				cv2.THRESH_BINARY,11,2)

	titles = ['Original Image', 'Global Thresholding (v = 127)',
				'Adaptive Mean Thresholding', 'Adaptive Gaussian Thresholding']
	images = [img, th1, th2, th3]

	for i in xrange(4):
		plt.subplot(2,2,i+1),plt.imshow(images[i],'gray')
		plt.title(titles[i])
		plt.xticks([]),plt.yticks([])
	plt.show()

def otsuBinary(fileName):
	import cv2
	import numpy as np
	from matplotlib import pyplot as plt

	img = cv2.imread(fileName,0)

	# global thresholding
	ret1,th1 = cv2.threshold(img,127,255,cv2.THRESH_BINARY)

	# Otsu's thresholding
	ret2,th2 = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

	# Otsu's thresholding after Gaussian filtering
	blur = cv2.GaussianBlur(img,(5,5),0)
	ret3,th3 = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

	# plot all the images and their histograms
	images = [img, 0, th1,
			  img, 0, th2,
			  blur, 0, th3]
	titles = ['Original Noisy Image','Histogram','Global Thresholding (v=127)',
			  'Original Noisy Image','Histogram',"Otsu's Thresholding",
			  'Gaussian filtered Image','Histogram',"Otsu's Thresholding"]

	for i in xrange(3):
		plt.subplot(3,3,i*3+1),plt.imshow(images[i*3],'gray')
		plt.title(titles[i*3]), plt.xticks([]), plt.yticks([])
		plt.subplot(3,3,i*3+2),plt.hist(images[i*3].ravel(),256)
		plt.title(titles[i*3+1]), plt.xticks([]), plt.yticks([])
		plt.subplot(3,3,i*3+3),plt.imshow(images[i*3+2],'gray')
		plt.title(titles[i*3+2]), plt.xticks([]), plt.yticks([])
	plt.show()

def smoothImage(fileName):
	import cv2
	import numpy as np
	from matplotlib import pyplot as plt

	img = cv2.imread(fileName)

	kernel = np.ones((5,5),np.float32)/25
	dst = cv2.filter2D(img,-1,kernel)

	plt.subplot(121),plt.imshow(img),plt.title('Original')
	plt.xticks([]), plt.yticks([])
	plt.subplot(122),plt.imshow(dst),plt.title('Averaging')
	plt.xticks([]), plt.yticks([])
	plt.show()

def houghCircle(fileName):
	"""source:https://opencv-python-tutroals.readthedocs.org/en/latest/py_tutorials/py_imgproc/py_houghcircles/py_houghcircles.html#hough-circles"""
	import cv2
	import cv2.cv as cv
	import numpy as np

	# image processing stuff
	img = cv2.imread(fileName,0)
	img = cv2.medianBlur(img,5)
	img = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
				cv2.THRESH_BINARY,11,2)

	# resize if necessary
	height, width = img.shape[:2]
	img = cv2.resize(img,(int(width/3), int(height/3)), interpolation = cv2.INTER_CUBIC)

	# img = cv2.medianBlur(img,5)
	cimg = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
	# circles = cv2.HoughCircles(img,cv2.HOUGH_GRADIENT,1,20, param1=50,param2=30,minRadius=0,maxRadius=0)
	circles = cv2.HoughCircles(img,cv.CV_HOUGH_GRADIENT, 1, 2,param1=50,param2=80,minRadius=0,maxRadius=0)
	print len(circles[0,:])," circles detected.\r\n"
	circles = np.uint16(np.around(circles))
	for i in circles[0,:]:
		# draw the outer circle
		# cv2.circle(cimg,(i[0],i[1]),i[2],(0,255,0),2)
		# draw the center of the circle
		cv2.circle(cimg,(i[0],i[1]),1,(0,0,255),3)

	# height, width = cimg.shape[:2]
	# cimg = cv2.resize(cimg,(int(width*2), int(height*2)), interpolation = cv2.INTER_CUBIC)
	print cimg.shape[:2]
	cv2.imshow('detected circles',cimg)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

fileName = 'bees.jpg'
# fileName = 'houghCircles2.jpg'
# houghCircle(fileName)