# import utilities
from __future__ import division
import json
import os
import time

# import API packages
from tornado.ioloop import IOLoop
import tornado.web
from tornado import gen
import concurrent
from concurrent.futures import ThreadPoolExecutor
from tornado.concurrent import run_on_executor
# from tornado.httpclient import AsyncHTTPClient
# from tornado.web import url
#import tornado.ioloop

# import opencv and numpy
import cv2
import numpy as np



# initialize global constants
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)),"output")
INPUT_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)),"input")

opencv_file_location = cv2.__file__
important_index = opencv_file_location.split('/').index('opencv3') + 2
FRONTAL_FACE_HAAR_LOCATION = os.path.join("/".join(opencv_file_location.split('/')[:important_index]),'share/OpenCV/haarcascades/haarcascade_frontalface_alt.xml')

def canny_grab(image):
	"""
	Run Canny Edge Detection on input image and find the bounding box that surrounds the edges

	input - cv2 Image
	output - grayscale mask for the edges, the top left corner of the bounding box, and the bottom right of the bounding box
	"""
	canny_out = cv2.Canny(image,0,image.flatten().mean())
	y,x = canny_out.nonzero()
	top_left = x.min(), y.min()
	bot_right = x.max(), y.max()
	return canny_out, top_left, bot_right


def grab_cut(image, top_left, bot_right):
	"""
	Utililize openCv's foreground detection algorithm

	input - cv2 image, top left and bottom right of a bounding box to focus on
	output - cv2 image of the estimated foreground
	"""
	mask = np.zeros(image.shape[:2],np.uint8)
	background = np.zeros((1,65),np.float64)
	foreground = np.zeros((1,65),np.float64)
	roi = (top_left[0],top_left[1],bot_right[0],bot_right[1])
	cv2.grabCut(image, mask, roi, background, foreground, 5, cv2.GC_INIT_WITH_RECT)
	new_mask = np.where((mask==2)|(mask==0),0,1).astype('uint8')
	image = image*new_mask[:,:,np.newaxis]
	return image


def bg_removal(orig, grabbed):
	"""
	Remove additional background pixels after the initial grabcut

	input - cv2 image (the original image before the grabcut), the image after the grabcut
	output - a grayscale mask of background pixels
	"""
	kernel = np.ones((3,3),np.uint8)
	mean,std = cv2.meanStdDev(cv2.cvtColor(orig,cv2.COLOR_BGR2HLS), cv2.inRange(grabbed,0,0))
	min_thresh = mean - std
	max_thresh = mean + std
	grab_bg = cv2.inRange(cv2.cvtColor(grabbed,cv2.COLOR_BGR2HLS),min_thresh,max_thresh)
	dilated_bg = cv2.morphologyEx(grab_bg, cv2.MORPH_OPEN, kernel)
	return dilated_bg


def watershed(grabbed):
	"""
	Run the watershed algorithm to try and find connected components

	input - cv2 image
	output - cv2 image with the marked connected components
	"""
	gray = cv2.cvtColor(grabbed,cv2.COLOR_BGR2GRAY)
	_, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

	kernel = np.ones((3,3),np.uint8)
	opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations=3)

	background = cv2.dilate(opening, kernel, iterations=2)

	dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
	_, foreground = cv2.threshold(dist_transform, 0.3*dist_transform.max(), 255, 0)

	foreground = np.uint8(foreground) 
	unknown = cv2.subtract(background,foreground)

	_, ccs = cv2.connectedComponents(foreground)
	ccs = ccs + 1
	ccs[unknown==255] = 0
	ccs = cv2.watershed(grabbed,ccs)
	return ccs


def get_skin_hair_mean_std(image, K = 2):
	"""
	Identify the skin pixel mean and standard deviation from a face image using Kmeans
	
	input - cv2 image of a human face, the number of clusters (K) for the Kmeans to reduce down to
	output - 
	"""
	data = image.reshape((-1,3))
	data = np.float32(data)

	criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
	K = K

	ret,label,center=cv2.kmeans(data, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

	center = np.uint8(center)
	result = center[label.flatten()]
	result_2 = result.reshape((image.shape))

	one_count = np.sum(label)
	zero_count = label.size - one_count

	skin_label = 1 if one_count > zero_count else 0
	hair_label = 1 - skin_label

	skin_BGR = center[skin_label]
	hair_BGR = center[hair_label]
	skin_mean,skin_std = cv2.meanStdDev(cv2.cvtColor(image, cv2.COLOR_BGR2HSV), mask = cv2.inRange(result_2,skin_BGR,skin_BGR))
	hair_mean,hair_std = cv2.meanStdDev(cv2.cvtColor(image, cv2.COLOR_BGR2HSV), mask = cv2.inRange(result_2,hair_BGR,hair_BGR))

	return skin_mean, skin_std, hair_mean, hair_std


def subtract_skin(orig, face):
	"""
	Subtract the skin from an image given a facial crop
	
	input - cv2 image
	output - a grayscale mask of the skin in the image
	""" 
	skin_mean, skin_std, _, _ = get_skin_hair_mean_std(face,2)
	kernel = np.ones((3,3),np.uint8)
	min_thresh = skin_mean - (skin_std*2)
	max_thresh = skin_mean + (skin_std*2)
	min_thresh[2] = 0
	max_thresh[2] = 255
	grab_skin = cv2.inRange(cv2.cvtColor(orig,cv2.COLOR_BGR2HSV),min_thresh,max_thresh)
	dilated_skin = cv2.morphologyEx(grab_skin, cv2.MORPH_OPEN, kernel)
	return dilated_skin


def subtract_hair(orig, face): 
	"""
	Subtract the hair from an image given a facial crop

	input - cv2 image
	output - a grayscale mask of the hair in the image
	""" 
	_, _, hair_mean, hair_std = get_skin_hair_mean_std(face,2)
	kernel = np.ones((3,3),np.uint8)
	min_thresh = hair_mean - (hair_std*2)
	max_thresh = hair_mean + (hair_std*2)
	grab_hair = cv2.inRange(cv2.cvtColor(orig,cv2.COLOR_BGR2HSV),min_thresh,max_thresh)
	dilated_hair = cv2.morphologyEx(grab_hair, cv2.MORPH_OPEN, kernel)
	return dilated_hair


def find_face(image):
	"""
	Subtract the skin from an image

	input - cv2 image
	output - a grayscale image of the face
	""" 
	img2 = image.copy()
	gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
	face_cascade = cv2.CascadeClassifier(FRONTAL_FACE_HAAR_LOCATION)
	faces = face_cascade.detectMultiScale(gray, 1.1, 5)

	the_face = None

	if len(faces) > 0:
	    face_imgs = list()
	    for (x,y,w,h) in faces:
	        face_imgs.append(img2[y:y+h, x:x+w])

	    face_sizes = np.array([face.size for face in face_imgs])
	    the_face = face_imgs[face_sizes.argmax()]

	return the_face

# executor class that allows us to identify the number of threads after API initialization
class Executor(concurrent.futures.ThreadPoolExecutor):
	def __init__(self, threads):
		self.threads = threads
		self.executor = ThreadPoolExecutor(max_workers=self.threads)

# primary request handling class
class MainHandler(tornado.web.RequestHandler):
	@run_on_executor
	def segment_image(self, image_path):
		"""
		Segment the prominent pieces of clothing out of an image

		input - a path to a local image
		output - the image path, if it was successfully analyzed, the start, finish, and total analysis time 
		"""
		try:
			start_time = time.time()

			# load image
			img = cv2.imread(image_path)
			# run canny edge detection
			canny_img, tl, br = canny_grab(img)
			roi = img[tl[1]:br[1], tl[0]:br[0]]
			
			i = 0
			the_face = None
			# the face isn't always found on the first iteration, so we search 5 times
			while (the_face is None) and (i < 5):
				i += 1
				grab = grab_cut(img, tl, br)
				the_face = find_face(grab)

			# remove any left over background in the image
			bg_mask = bg_removal(img, grab)
			bg_removed = cv2.subtract(grab, cv2.cvtColor(bg_mask,cv2.COLOR_GRAY2BGR))
	    	
	    	# if there is a face, we subtract the skin and the hair
			if the_face is not None:
				subtracted_skin_mask = subtract_skin(grab,the_face)
				subtracted_hair_mask = subtract_hair(grab,the_face)
				skin_removed = cv2.subtract(bg_removed, cv2.cvtColor(subtracted_skin_mask,cv2.COLOR_GRAY2BGR))
				hair_removed = cv2.subtract(skin_removed, cv2.cvtColor(subtracted_hair_mask,cv2.COLOR_GRAY2BGR))
				grab = hair_removed.copy()
			else: # could be improved by using kmeans to try and identify the skin when a face isn't found
				grab = bg_removed.copy()
	    
	    	# run the watershed algorithm to find connected components
			grab = cv2.GaussianBlur(grab, (15,15),0)
			watershed_out = watershed(grab)

			# subtract everything out but the foreground watershed piece 
			final_piece = cv2.bitwise_and(img, cv2.cvtColor(cv2.inRange(watershed_out,1,1),cv2.COLOR_GRAY2BGR))
			
			# write the piece to the output folder
			output_filename = os.path.join(OUTPUT_DIR,image_path.split('/')[-1])
			cv2.imwrite(output_filename, final_piece)
			
			finish_time = time.time()
			
			return {'path':image_path,
					'was_successful': True,
					'start_time':start_time,
					'finish_time':finish_time,
					'analysis_time':finish_time-start_time}

		except Exception, e:
			return {'path':image_path,
					'was_successful': False,
					'start_time': None,
					'finish_time': None,
					'analysis_time': None,
					'exception': e}


	@gen.coroutine
	def post(self):
		start_time = time.time()
		try:
			# get data from POST request
			data = tornado.escape.json_decode(self.request.body)
			num_of_threads = data['num_threads']
		except Exception, e:
			self.write({400: "ERROR: Request didn't have the data expected"})

		# initialize executor pool with the number of threads provided in the request
		self.executor = Executor(num_of_threads).executor
		
		# identify files that have certain image extensions in the input directory
		valid_images = [os.path.join(INPUT_DIR,fname) for fname in os.listdir(INPUT_DIR) if fname.split('.')[-1] in ['jpg','jpeg','png']]
		
		if len(valid_images) > 0:
			# execute the segmentation in parallel
			res = yield {i: self.segment_image(img_path) for i,img_path in enumerate(valid_images)}
			
			total_analysis_time = 0
			total_images_analyzed = 0
			for i,image_result in res.iteritems():
				if image_result['was_successful']:
					total_images_analyzed += 1
					total_analysis_time += image_result['analysis_time']

			end_time = time.time()
			total_analysis_time_ms = int(round((end_time - start_time) * 100))
			avg_analysis_time_ms = int(round(total_analysis_time_ms/total_images_analyzed))
			response = {'output_dir':OUTPUT_DIR, 
						'total_processed':total_images_analyzed, 
						'total_processing_time_ms': total_analysis_time_ms,
						'average_processing_time_ms': avg_analysis_time_ms}
			
			self.write(response)
		else:
			self.write({500: "ERROR: No valid images found in input directory"})

	@gen.coroutine
	def get(self):
			res = {400:"ERROR: Please make a POST request"}
			self.write(res)

def main():
	return tornado.web.Application([(r"/cut", MainHandler)])
    

if __name__ == "__main__":
	app = main()
	app.listen(9999)
	tornado.ioloop.IOLoop.current().start()