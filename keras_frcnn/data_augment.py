from PIL import Image, ImageDraw
import numpy as np

GenImSize = 128

def MakeCellImage(x, y, r, i):
    im = Image.new(mode='F', size=(GenImSize, GenImSize))
    draw = ImageDraw.Draw(im)
    draw.ellipse(xy=[x-r, y-r, x+r, y+r], fill='White')
    im = np.array(im).astype(np.float32)
    im *= (i / 255.0)
    #im += (np.random.randn(im.shape[0], im.shape[1]) * 0.1) + 0.2
    #im[im < 0] = 0
    #im[im > 1] = 1
    return im

def MakeRandomCellImage(n):
	rois = []
	im = np.zeros(shape=(GenImSize, GenImSize))
	for i in range(n):
		radius = np.random.randint(low=-5, high=10) + 10
		intensity = (np.random.randn() * 0.1) + 0.5
		intensity = max(min(intensity, 1.0), 0.0)
		position = np.random.randint(low=radius, high=GenImSize-radius, size=2)
		im += MakeCellImage(position[0], position[1], radius, intensity)
		rois.append(np.array([position[0] - radius, position[0] + radius, position[1] - radius, position[1] + radius, intensity]))
	
	# im_max = max(im.flatten())
	# im_min = min(im.flatten())
	# im = (im - im_min) / (im_max - im_min)
	im += (np.random.randn(im.shape[0], im.shape[1]) * 0.1) + 0.2
	im[im < 0] = 0
	im[im > 1] = 1
	return im, rois

def augment(n=1):
	im_data = {}
	im, rois = MakeRandomCellImage(n)
	im_data['width'] = GenImSize
	im_data['height'] = GenImSize
	
	im_data['bboxes'] = [{'class': 'cell', 'x1': int(box[0]), 'x2': int(box[1]), 'y1': int(box[2]), 'y2': int(box[3])} for box in rois]

	return im_data, im

#def augment(img_data, config, augment=True):
	assert 'filepath' in img_data
	assert 'bboxes' in img_data
	assert 'width' in img_data
	assert 'height' in img_data

	img_data_aug = copy.deepcopy(img_data)

	img = cv2.imread(img_data_aug['filepath'])

	if augment:
		rows, cols = img.shape[:2]

		if config.use_horizontal_flips and np.random.randint(0, 2) == 0:
			img = cv2.flip(img, 1)
			for bbox in img_data_aug['bboxes']:
				x1 = bbox['x1']
				x2 = bbox['x2']
				bbox['x2'] = cols - x1
				bbox['x1'] = cols - x2

		if config.use_vertical_flips and np.random.randint(0, 2) == 0:
			img = cv2.flip(img, 0)
			for bbox in img_data_aug['bboxes']:
				y1 = bbox['y1']
				y2 = bbox['y2']
				bbox['y2'] = rows - y1
				bbox['y1'] = rows - y2

		if config.rot_90:
			angle = np.random.choice([0,90,180,270],1)[0]
			if angle == 270:
				img = np.transpose(img, (1,0,2))
				img = cv2.flip(img, 0)
			elif angle == 180:
				img = cv2.flip(img, -1)
			elif angle == 90:
				img = np.transpose(img, (1,0,2))
				img = cv2.flip(img, 1)
			elif angle == 0:
				pass

			for bbox in img_data_aug['bboxes']:
				x1 = bbox['x1']
				x2 = bbox['x2']
				y1 = bbox['y1']
				y2 = bbox['y2']
				if angle == 270:
					bbox['x1'] = y1
					bbox['x2'] = y2
					bbox['y1'] = cols - x2
					bbox['y2'] = cols - x1
				elif angle == 180:
					bbox['x2'] = cols - x1
					bbox['x1'] = cols - x2
					bbox['y2'] = rows - y1
					bbox['y1'] = rows - y2
				elif angle == 90:
					bbox['x1'] = rows - y2
					bbox['x2'] = rows - y1
					bbox['y1'] = x1
					bbox['y2'] = x2        
				elif angle == 0:
					pass

	img_data_aug['width'] = img.shape[1]
	img_data_aug['height'] = img.shape[0]
	return img_data_aug, img
