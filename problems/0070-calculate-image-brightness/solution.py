import numpy as np
def calculate_brightness(img):
	# Write your code here
	if len(img) == 0:
		return -1
	try:
		img = np.array(img)

	except ValueError as e:
		return -1

	if img.max() > 255 or img.min() < 0:
		return -1

	return round(img.mean())
