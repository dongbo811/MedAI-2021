import os
import cv2
import numpy as np
from tqdm import tqdm
root = r'E:\master\com\Polyp_NORA\worker\res_deal1'
models = ['SUB1','SUB4','SUB5']
save = r'E:\master\com\Polyp_NORA\worker\res_deal1\KP_OPT6'
path = r'E:\master\com\Polyp_NORA\worker\res_deal1\SUB5\KP/'
img_list = os.listdir(path)

for img in tqdm(img_list):
	result_list = []
	for model in models:
		KP_path = os.path.join(root, model, 'KP')
		# kps = os.listdir(KP_path)
		# for kp in kps:
		# 	image_path = os.path.join(KP_path, kp)
			# print(image_path + img)
		image = cv2.imread(os.path.join(KP_path, img), 0)
		cv2.normalize(image, image, 0, 255, cv2.NORM_MINMAX)
		ret, image = cv2.threshold(image, 0.5, 1, cv2.THRESH_BINARY)

		result_list.append(image)

	height, width = result_list[0].shape
	vote_mask = np.zeros((height, width))
	for h in range(height):
		for w in range(width):
			record = np.zeros((1, 2))
			for n in range(len(result_list)):
				mask = result_list[n]
				pixel = mask[h, w]
				# if pixel ==1:
				# 	# print('pix:',pixel)
				record[0, pixel] += 1

			label = record.argmax()
			# print(label)
			vote_mask[h, w] = label

	save_path = os.path.join(save, img)
	cv2.imwrite(save_path, vote_mask*255)
