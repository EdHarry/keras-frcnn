from __future__ import division
import os
from PIL import Image, ImageDraw
import numpy as np
import sys
import pickle
#from optparse import OptionParser
import time
from keras_frcnn import config
from keras import backend as K
from keras.layers import Input
from keras.models import Model
from keras_frcnn import roi_helpers_PIL as roi_helpers
from keras_frcnn import data_augment

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")
from MovieGenerator.MovieGenerator import positions_and_radii

def Detect(movie):
	SubMovieDim = 128

	sys.setrecursionlimit(40000)

	# parser = OptionParser()

	# parser.add_option("-p", "--path", dest="test_path", help="Path to test data.")
	# parser.add_option("-n", "--num_rois", dest="num_rois",
	# 				help="Number of ROIs per iteration. Higher means more memory use.", default=32)
	# parser.add_option("--config_filename", dest="config_filename", help=
	# 				"Location to read the metadata related to the training (generated when training).",
	# 				default="config.pickle")
	# parser.add_option("--network", dest="network", help="Base network to use. Supports vgg or resnet50.", default='resnet50')

	# (options, args) = parser.parse_args()

	# # if not options.test_path:   # if filename is not given
	# # 	parser.error('Error: path to test data must be specified. Pass --path to command line')


	# config_output_filename = options.config_filename
	config_output_filename = "config.pickle"

	with open(config_output_filename, 'rb') as f_in:
		C = pickle.load(f_in)

	if C.network == 'resnet50':
		import keras_frcnn.resnet as nn
	elif C.network == 'vgg':
		import keras_frcnn.vgg as nn

	# turn off any data augmentation at test time
	C.use_horizontal_flips = False
	C.use_vertical_flips = False
	C.rot_90 = False

	#img_path = options.test_path

	def format_img_size(img, C):
		""" formats the image size based on config """
		img_min_side = float(C.im_size)
		(height, width) = img.shape
			
		if width <= height:
			ratio = img_min_side/width
			new_height = int(ratio * height)
			new_width = int(img_min_side)
		else:
			ratio = img_min_side/height
			new_width = int(ratio * width)
			new_height = int(img_min_side)
		img = Image.fromarray(img)
		img = img.resize((new_width, new_height), Image.ANTIALIAS)
		img = np.array(img)

		#img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
		return img, ratio	

	def format_img_channels(img, C):
		img = img.astype(np.float32)
		img -= np.mean(img.flatten())
		img /= np.std(img.flatten())
		img = img[:, :, np.newaxis]

		img = np.transpose(img, (2, 0, 1))
		img = np.expand_dims(img, axis=0)

	        #""" formats the image channels based on config """
		#img = img[:, :, (2, 1, 0)]
		#img = img.astype(np.float32)
		#img[:, :, 0] -= C.img_channel_mean[0]
		#img[:, :, 1] -= C.img_channel_mean[1]
		#img[:, :, 2] -= C.img_channel_mean[2]
		#img /= C.img_scaling_factor
		#img = np.transpose(img, (2, 0, 1))
		#img = np.expand_dims(img, axis=0)
		return img

	def format_img(img, C):
		""" formats an image for model prediction based on config """
		img, ratio = format_img_size(img, C)
		img = format_img_channels(img, C)
		return img, ratio

	# Method to transform the coordinates of the bounding box to its original size
	def get_real_coordinates(ratio, x1, y1, x2, y2):

		# real_x1 = int(round(x1 // ratio))
		# real_y1 = int(round(y1 // ratio))
		# real_x2 = int(round(x2 // ratio))
		# real_y2 = int(round(y2 // ratio))
		real_x1 = x1 / ratio
		real_y1 = y1 / ratio
		real_x2 = x2 / ratio
		real_y2 = y2 / ratio

		return (real_x1, real_y1, real_x2 ,real_y2)

	class_mapping = C.class_mapping

	if 'bg' not in class_mapping:
		class_mapping['bg'] = len(class_mapping)

	class_mapping = {v: k for k, v in class_mapping.items()}
	#print(class_mapping)
	class_to_color = {class_mapping[v]: np.random.randint(0, 255, 3) for v in class_mapping}
	#C.num_rois = int(options.num_rois)
	C.num_rois = 32

	if C.network == 'resnet50':
		num_features = 1024
	elif C.network == 'vgg':
		num_features = 512

	if K.image_dim_ordering() == 'th':
		input_shape_img = (3, None, None)
		input_shape_features = (num_features, None, None)
	else:
		input_shape_img = (None, None, 1)
		input_shape_features = (None, None, num_features)


	img_input = Input(shape=input_shape_img)
	roi_input = Input(shape=(C.num_rois, 4))
	feature_map_input = Input(shape=input_shape_features)

	# define the base network (resnet here, can be VGG, Inception, etc)
	shared_layers = nn.nn_base(img_input, trainable=True)

	# define the RPN, built on the base layers
	num_anchors = len(C.anchor_box_scales) * len(C.anchor_box_ratios)
	rpn_layers = nn.rpn(shared_layers, num_anchors)

	classifier = nn.classifier(feature_map_input, roi_input, C.num_rois, nb_classes=len(class_mapping), trainable=True)

	model_rpn = Model(img_input, rpn_layers)
	model_classifier_only = Model([feature_map_input, roi_input], classifier)

	model_classifier = Model([feature_map_input, roi_input], classifier)

	print('Loading weights from {}'.format(C.model_path))
	model_rpn.load_weights(C.model_path, by_name=True)
	model_classifier.load_weights(C.model_path, by_name=True)

	model_rpn.compile(optimizer='sgd', loss='mse')
	model_classifier.compile(optimizer='sgd', loss='mse')

	all_imgs = []

	classes = {}

	bbox_threshold = 0.8

	#visualise = True

	#for idx, img_name in enumerate(sorted(os.listdir(img_path))):
	#	if not img_name.lower().endswith(('.bmp', '.jpeg', '.jpg', '.png', '.tif', '.tiff')):
	#		continue

	def AddPosAndRadFromCornerCoords(coords, pnr=None):
		(x0, y0, x1, y1) = coords
		centre = (np.array([x0+x1, y0+y1]) / 2)[np.newaxis, :]
		rad = np.array([max(x1-x0, y1-y0)]) / 2
		if pnr is None:
			pnr = positions_and_radii(centre, rad)
		else:
			pnr.positions = np.concatenate((pnr.positions, centre))
			pnr.radii = np.concatenate((pnr.radii, rad))

		return pnr

	print("---\n---")

	rois = []
	rows, cols, frames = movie.shape
	assert(rows >= SubMovieDim)
	assert(cols >= SubMovieDim)

	deltaSub = int(SubMovieDim / 2)
	
	for idx in range(movie.shape[-1]):
		st = time.time()
		#_, img = data_augment.augment(n=np.random.randint(low=1, high=5))
		img_frame = movie[:, :, idx]

		bboxes = {}
		probs = {}
		subImgRange = np.array([0, SubMovieDim, 0, SubMovieDim])

		done_y = False
		while not done_y:
			done_y = subImgRange[3] == cols
			done_x = False
			subImgRange[:2] = np.array([0, SubMovieDim])
			while not done_x:
				done_x = subImgRange[1] == rows
				img = img_frame[subImgRange[2]:subImgRange[3], subImgRange[0]:subImgRange[1]]

				X, ratio = format_img(img, C)
				X = np.transpose(X, (0, 2, 3, 1))

				# tmp = Image.fromarray((img * 255).astype(np.uint8))
				# img = Image.new('RGBA', tmp.size)
				# img.paste(tmp)
				# del tmp
				# draw = ImageDraw.Draw(img)
				#img = (img * 255).astype(np.uint8)
				#img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
				
				[Y1, Y2, F] = model_rpn.predict(X)
				R = roi_helpers.rpn_to_roi(Y1, Y2, C, K.image_dim_ordering(), overlap_thresh=0.7)

				R[:, 2] -= R[:, 0]
				R[:, 3] -= R[:, 1]

				for jk in range(R.shape[0]//C.num_rois + 1):
					ROIs = np.expand_dims(R[C.num_rois*jk:C.num_rois*(jk+1), :], axis=0)
					if ROIs.shape[1] == 0:
						break

					if jk == R.shape[0]//C.num_rois:
						#pad R
						curr_shape = ROIs.shape
						target_shape = (curr_shape[0],C.num_rois,curr_shape[2])
						ROIs_padded = np.zeros(target_shape).astype(ROIs.dtype)
						ROIs_padded[:, :curr_shape[1], :] = ROIs
						ROIs_padded[0, curr_shape[1]:, :] = ROIs[0, 0, :]
						ROIs = ROIs_padded

					[P_cls, P_regr] = model_classifier_only.predict([F, ROIs])

					for ii in range(P_cls.shape[1]):
						if np.max(P_cls[0, ii, :]) < bbox_threshold or np.argmax(P_cls[0, ii, :]) == (P_cls.shape[2] - 1):
							continue

						cls_name = class_mapping[np.argmax(P_cls[0, ii, :])]

						if cls_name not in bboxes:
							bboxes[cls_name] = []
							probs[cls_name] = []

						(x, y, w, h) = ROIs[0, ii, :]

						cls_num = np.argmax(P_cls[0, ii, :])
						try:
							(tx, ty, tw, th) = P_regr[0, ii, 4*cls_num:4*(cls_num+1)]
							tx /= C.classifier_regr_std[0]
							ty /= C.classifier_regr_std[1]
							tw /= C.classifier_regr_std[2]
							th /= C.classifier_regr_std[3]
							x, y, w, h = roi_helpers.apply_regr(x, y, w, h, tx, ty, tw, th)
						except:
							pass
						
						tmpBox = np.array([C.rpn_stride*x, C.rpn_stride*y, C.rpn_stride*(x+w), C.rpn_stride*(y+h)]).astype(np.float32)

						if np.all(tmpBox >= 0) and np.all(tmpBox < (SubMovieDim * ratio)):

							tmpBox += (np.array([subImgRange[0], subImgRange[2], subImgRange[0], subImgRange[2]]) * ratio)

							bboxes[cls_name].append(tmpBox.tolist())
							probs[cls_name].append(np.max(P_cls[0, ii, :]))

				if not done_x:
					delta = min(deltaSub, rows-subImgRange[1])
					subImgRange += np.array([delta, delta, 0, 0])

			if not done_y:
				delta = min(deltaSub, cols-subImgRange[3])
				subImgRange += np.array([0, 0, delta, delta])

		#all_dets = []

		det = None
		for key in bboxes:
			bbox = np.array(bboxes[key])

			new_boxes, new_probs = roi_helpers.non_max_suppression_fast(bbox, np.array(probs[key]), overlap_thresh=0.2)
			for jk in range(new_boxes.shape[0]):
				(x1, y1, x2, y2) = new_boxes[jk,:]

				(real_x1, real_y1, real_x2, real_y2) = get_real_coordinates(ratio, x1, y1, x2, y2)
				det = AddPosAndRadFromCornerCoords((real_x1, real_y1, real_x2, real_y2), det)

				#draw.rectangle(xy=[real_x1, real_y1, real_x2, real_y2], outline='red')
	    		#cv2.rectangle(img,(real_x1, real_y1), (real_x2, real_y2), (int(class_to_color[key][0]), int(class_to_color[key][1]), int(class_to_color[key][2])),2)

				#textLabel = '{}: {}'.format(key,int(100*new_probs[jk]))
				#all_dets.append((key,100*new_probs[jk]))

				#(retval,baseLine) = cv2.getTextSize(textLabel,cv2.FONT_HERSHEY_COMPLEX,1,1)
				#textOrg = (real_x1, real_y1-0)

				#cv2.rectangle(img, (textOrg[0] - 5, textOrg[1]+baseLine - 5), (textOrg[0]+retval[0] + 5, textOrg[1]-retval[1] - 5), (0, 0, 0), 2)
				#cv2.rectangle(img, (textOrg[0] - 5,textOrg[1]+baseLine - 5), (textOrg[0]+retval[0] + 5, textOrg[1]-retval[1] - 5), (255, 255, 255), -1)
				#cv2.putText(img, textLabel, textOrg, cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 0), 1)

		if det is None:
			det = positions_and_radii(-1 * np.ones((1, 1)), np.array([-1]))

		rois.append(det)
		sys.stdout.write("Detecting: {0:.2f}%, Elapsed time/frame = {1:.2f}s".format(100.0 * (idx+1) / movie.shape[-1], time.time() - st) + '\r')
		sys.stdout.flush()

	print("Detecting: {0:.2f}%, Elapsed time/frame = {1:.2f}s".format(100.0 * (idx+1) / movie.shape[-1], time.time() - st) + '\r')
	print("---")
	return rois
		#print('Elapsed time = {}'.format(time.time() - st))
		#print(all_dets)
		#cv2.imshow('img', img)
		#cv2.waitKey(0)
		#cv2.imwrite('./results_imgs/{}.png'.format(idx),img)
		#img.save('./results_imgs/{}.png'.format(idx))

def Detect_old(movie):
	sys.setrecursionlimit(40000)

	# parser = OptionParser()

	# parser.add_option("-p", "--path", dest="test_path", help="Path to test data.")
	# parser.add_option("-n", "--num_rois", dest="num_rois",
	# 				help="Number of ROIs per iteration. Higher means more memory use.", default=32)
	# parser.add_option("--config_filename", dest="config_filename", help=
	# 				"Location to read the metadata related to the training (generated when training).",
	# 				default="config.pickle")
	# parser.add_option("--network", dest="network", help="Base network to use. Supports vgg or resnet50.", default='resnet50')

	# (options, args) = parser.parse_args()

	# # if not options.test_path:   # if filename is not given
	# # 	parser.error('Error: path to test data must be specified. Pass --path to command line')


	# config_output_filename = options.config_filename
	config_output_filename = "config.pickle"

	with open(config_output_filename, 'rb') as f_in:
		C = pickle.load(f_in)

	if C.network == 'resnet50':
		import keras_frcnn.resnet as nn
	elif C.network == 'vgg':
		import keras_frcnn.vgg as nn

	# turn off any data augmentation at test time
	C.use_horizontal_flips = False
	C.use_vertical_flips = False
	C.rot_90 = False

	#img_path = options.test_path

	def format_img_size(img, C):
		""" formats the image size based on config """
		img_min_side = float(C.im_size)
		(height, width) = img.shape
			
		if width <= height:
			ratio = img_min_side/width
			new_height = int(ratio * height)
			new_width = int(img_min_side)
		else:
			ratio = img_min_side/height
			new_width = int(ratio * width)
			new_height = int(img_min_side)
		img = Image.fromarray(img)
		img = img.resize((new_width, new_height), Image.ANTIALIAS)
		img = np.array(img)

		#img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
		return img, ratio	

	def format_img_channels(img, C):
		img = img.astype(np.float32)
		img -= np.mean(img.flatten())
		img /= np.std(img.flatten())
		img = img[:, :, np.newaxis]

		img = np.transpose(img, (2, 0, 1))
		img = np.expand_dims(img, axis=0)

	        #""" formats the image channels based on config """
		#img = img[:, :, (2, 1, 0)]
		#img = img.astype(np.float32)
		#img[:, :, 0] -= C.img_channel_mean[0]
		#img[:, :, 1] -= C.img_channel_mean[1]
		#img[:, :, 2] -= C.img_channel_mean[2]
		#img /= C.img_scaling_factor
		#img = np.transpose(img, (2, 0, 1))
		#img = np.expand_dims(img, axis=0)
		return img

	def format_img(img, C):
		""" formats an image for model prediction based on config """
		img, ratio = format_img_size(img, C)
		img = format_img_channels(img, C)
		return img, ratio

	# Method to transform the coordinates of the bounding box to its original size
	def get_real_coordinates(ratio, x1, y1, x2, y2):

		# real_x1 = int(round(x1 // ratio))
		# real_y1 = int(round(y1 // ratio))
		# real_x2 = int(round(x2 // ratio))
		# real_y2 = int(round(y2 // ratio))
		real_x1 = x1 / ratio
		real_y1 = y1 / ratio
		real_x2 = x2 / ratio
		real_y2 = y2 / ratio

		return (real_x1, real_y1, real_x2 ,real_y2)

	class_mapping = C.class_mapping

	if 'bg' not in class_mapping:
		class_mapping['bg'] = len(class_mapping)

	class_mapping = {v: k for k, v in class_mapping.items()}
	#print(class_mapping)
	class_to_color = {class_mapping[v]: np.random.randint(0, 255, 3) for v in class_mapping}
	#C.num_rois = int(options.num_rois)
	C.num_rois = 32

	if C.network == 'resnet50':
		num_features = 1024
	elif C.network == 'vgg':
		num_features = 512

	if K.image_dim_ordering() == 'th':
		input_shape_img = (3, None, None)
		input_shape_features = (num_features, None, None)
	else:
		input_shape_img = (None, None, 1)
		input_shape_features = (None, None, num_features)


	img_input = Input(shape=input_shape_img)
	roi_input = Input(shape=(C.num_rois, 4))
	feature_map_input = Input(shape=input_shape_features)

	# define the base network (resnet here, can be VGG, Inception, etc)
	shared_layers = nn.nn_base(img_input, trainable=True)

	# define the RPN, built on the base layers
	num_anchors = len(C.anchor_box_scales) * len(C.anchor_box_ratios)
	rpn_layers = nn.rpn(shared_layers, num_anchors)

	classifier = nn.classifier(feature_map_input, roi_input, C.num_rois, nb_classes=len(class_mapping), trainable=True)

	model_rpn = Model(img_input, rpn_layers)
	model_classifier_only = Model([feature_map_input, roi_input], classifier)

	model_classifier = Model([feature_map_input, roi_input], classifier)

	print('Loading weights from {}'.format(C.model_path))
	model_rpn.load_weights(C.model_path, by_name=True)
	model_classifier.load_weights(C.model_path, by_name=True)

	model_rpn.compile(optimizer='sgd', loss='mse')
	model_classifier.compile(optimizer='sgd', loss='mse')

	all_imgs = []

	classes = {}

	bbox_threshold = 0.8

	#visualise = True

	#for idx, img_name in enumerate(sorted(os.listdir(img_path))):
	#	if not img_name.lower().endswith(('.bmp', '.jpeg', '.jpg', '.png', '.tif', '.tiff')):
	#		continue

	def AddPosAndRadFromCornerCoords(coords, pnr=None):
		(x0, y0, x1, y1) = coords
		centre = (np.array([x0+x1, y0+y1]) / 2)[np.newaxis, :]
		rad = np.array([max(x1-x0, y1-y0)]) / 2
		if pnr is None:
			pnr = positions_and_radii(centre, rad)
		else:
			pnr.positions = np.concatenate((pnr.positions, centre))
			pnr.radii = np.concatenate((pnr.radii, rad))

		return pnr

	print("---\n---")

	rois = []
	for idx in range(movie.shape[-1]):
		st = time.time()
		#_, img = data_augment.augment(n=np.random.randint(low=1, high=5))
		img = movie[:, :, idx]

		X, ratio = format_img(img, C)
		X = np.transpose(X, (0, 2, 3, 1))

		# tmp = Image.fromarray((img * 255).astype(np.uint8))
		# img = Image.new('RGBA', tmp.size)
		# img.paste(tmp)
		# del tmp
		# draw = ImageDraw.Draw(img)
		#img = (img * 255).astype(np.uint8)
		#img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
		
		[Y1, Y2, F] = model_rpn.predict(X)
		R = roi_helpers.rpn_to_roi(Y1, Y2, C, K.image_dim_ordering(), overlap_thresh=0.7)

		R[:, 2] -= R[:, 0]
		R[:, 3] -= R[:, 1]

		bboxes = {}
		probs = {}

		for jk in range(R.shape[0]//C.num_rois + 1):
			ROIs = np.expand_dims(R[C.num_rois*jk:C.num_rois*(jk+1), :], axis=0)
			if ROIs.shape[1] == 0:
				break

			if jk == R.shape[0]//C.num_rois:
				#pad R
				curr_shape = ROIs.shape
				target_shape = (curr_shape[0],C.num_rois,curr_shape[2])
				ROIs_padded = np.zeros(target_shape).astype(ROIs.dtype)
				ROIs_padded[:, :curr_shape[1], :] = ROIs
				ROIs_padded[0, curr_shape[1]:, :] = ROIs[0, 0, :]
				ROIs = ROIs_padded

			[P_cls, P_regr] = model_classifier_only.predict([F, ROIs])

			for ii in range(P_cls.shape[1]):
				if np.max(P_cls[0, ii, :]) < bbox_threshold or np.argmax(P_cls[0, ii, :]) == (P_cls.shape[2] - 1):
					continue

				cls_name = class_mapping[np.argmax(P_cls[0, ii, :])]

				if cls_name not in bboxes:
					bboxes[cls_name] = []
					probs[cls_name] = []

				(x, y, w, h) = ROIs[0, ii, :]

				cls_num = np.argmax(P_cls[0, ii, :])
				try:
					(tx, ty, tw, th) = P_regr[0, ii, 4*cls_num:4*(cls_num+1)]
					tx /= C.classifier_regr_std[0]
					ty /= C.classifier_regr_std[1]
					tw /= C.classifier_regr_std[2]
					th /= C.classifier_regr_std[3]
					x, y, w, h = roi_helpers.apply_regr(x, y, w, h, tx, ty, tw, th)
				except:
					pass
				bboxes[cls_name].append([C.rpn_stride*x, C.rpn_stride*y, C.rpn_stride*(x+w), C.rpn_stride*(y+h)])
				probs[cls_name].append(np.max(P_cls[0, ii, :]))

		#all_dets = []

		det = None
		for key in bboxes:
			bbox = np.array(bboxes[key])

			new_boxes, new_probs = roi_helpers.non_max_suppression_fast(bbox, np.array(probs[key]), overlap_thresh=0.2)
			for jk in range(new_boxes.shape[0]):
				(x1, y1, x2, y2) = new_boxes[jk,:]

				(real_x1, real_y1, real_x2, real_y2) = get_real_coordinates(ratio, x1, y1, x2, y2)
				det = AddPosAndRadFromCornerCoords((real_x1, real_y1, real_x2, real_y2), det)

				#draw.rectangle(xy=[real_x1, real_y1, real_x2, real_y2], outline='red')
	    		#cv2.rectangle(img,(real_x1, real_y1), (real_x2, real_y2), (int(class_to_color[key][0]), int(class_to_color[key][1]), int(class_to_color[key][2])),2)

				#textLabel = '{}: {}'.format(key,int(100*new_probs[jk]))
				#all_dets.append((key,100*new_probs[jk]))

				#(retval,baseLine) = cv2.getTextSize(textLabel,cv2.FONT_HERSHEY_COMPLEX,1,1)
				#textOrg = (real_x1, real_y1-0)

				#cv2.rectangle(img, (textOrg[0] - 5, textOrg[1]+baseLine - 5), (textOrg[0]+retval[0] + 5, textOrg[1]-retval[1] - 5), (0, 0, 0), 2)
				#cv2.rectangle(img, (textOrg[0] - 5,textOrg[1]+baseLine - 5), (textOrg[0]+retval[0] + 5, textOrg[1]-retval[1] - 5), (255, 255, 255), -1)
				#cv2.putText(img, textLabel, textOrg, cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 0), 1)

		if det is None:
			det = positions_and_radii(-1 * np.ones((1, 1)), np.array([-1]))

		rois.append(det)
		sys.stdout.write("Detecting: {0:.2f}%, Elapsed time/frame = {1:.2f}s".format(100.0 * (idx+1) / movie.shape[-1], time.time() - st) + '\r')
		sys.stdout.flush()

	print("Detecting: {0:.2f}%, Elapsed time/frame = {1:.2f}s".format(100.0 * (idx+1) / movie.shape[-1], time.time() - st) + '\r')
	print("---")
	return rois
		#print('Elapsed time = {}'.format(time.time() - st))
		#print(all_dets)
		#cv2.imshow('img', img)
		#cv2.waitKey(0)
		#cv2.imwrite('./results_imgs/{}.png'.format(idx),img)
		#img.save('./results_imgs/{}.png'.format(idx))