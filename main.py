import math
import easygui as eg
import cv2 as cv
import numpy as np


def imshow(img, delay=None, winname="output"):
	cv.imshow(winname, img)
	cv.waitKey(delay)


def find_field(p, c):
	px, py = p
	cx, cy = c

	if px > cx:
		return 1 if py > cy else 2
	return 4 if py > cy else 3


def find_angle(p, c):
	field = find_field(p, c)
	px, py = p
	cx, cy = c

	x = abs(cx - px)
	y = abs(cy - py)

	degree = math.degrees(math.atan(x / y))

	if field == 1:
		return degree
	elif field == 2:
		return 180 - degree
	elif field == 3:
		return 180 + degree
	elif field == 4:
		return 360 - degree


def make_thicker(length, img, center, thickness, angle, space):
	r = int((length // space + 1) * space) if length % space > space // 2 else int((length // space) * space)
	cv.ellipse(img=img, center=center, axes=(r, r), thickness=thickness, angle=0,
			   startAngle=angle - 2, endAngle=angle + 2, color=(0, 0, 0))


def main(f, img):
	filename = f.split("/")[-1]
	# img_gray = cv.imread(f, 0)
	img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
	img_gray = cv.resize(img_gray, (1024, 1024), interpolation=cv.INTER_AREA)

	img_blur = cv.blur(img_gray, (31, 31))

	scale = 15
	dim = (int(img_blur.shape[0] / scale), int(img_blur.shape[1] / scale))
	img_px = cv.resize(img_blur, dim)

	dim = (int(img_px.shape[0] * scale), int(img_px.shape[1] * scale))
	img_px = cv.resize(img_px, dim, interpolation=cv.INTER_AREA)

	magic = np.ones(img_px.shape, dtype=np.uint8) * 255

	center_of_magic = (magic.shape[0] // 2, magic.shape[1] // 2)

	space = 20

	for radius in range(0, max(magic.shape), space):
		cv.circle(magic, center_of_magic, radius, (0, 0, 0), 4)

	for row in range(scale // 2, magic.shape[0], scale):
		for col in range(scale // 2, magic.shape[1], scale):
			piece = img_px[row - scale // 2:row + scale // 2 + 1, col - scale // 2:col + scale // 2 + 1]
			mean_val = int(np.sum(piece) // scale ** 2)
			length = math.sqrt((row - center_of_magic[0]) ** 2 + (col - center_of_magic[1]) ** 2)
			angle = find_angle((row, col), center_of_magic)

			# make_thicker(length, magic, center_of_magic, 15 - mean_val // space, angle)

			if mean_val < 50:
				make_thicker(length, magic, center_of_magic, 16, angle, space)
			elif mean_val < 100:
				make_thicker(length, magic, center_of_magic, 13, angle, space)
			elif mean_val < 150:
				make_thicker(length, magic, center_of_magic, 10, angle, space)
			elif mean_val < 200:
				make_thicker(length, magic, center_of_magic, 7, angle, space)

	mask = np.zeros(img_px.shape, dtype=np.uint8)
	cv.circle(mask, center_of_magic, center_of_magic[0], (255, 255, 255), -1)

	result = cv.bitwise_not(cv.bitwise_and(mask, cv.bitwise_not(magic)))
	imshow(result, winname="result", delay=1)
	print("./output/" + filename)
	cv.imwrite("./output/result_" + filename, result)


def crop_gui(_):
	radius = cv.getTrackbarPos('radius', frameName)
	x = cv.getTrackbarPos('x', frameName)
	y = cv.getTrackbarPos('y', frameName)

	shadow_img = inp.copy()
	shadow_img = change_brightness(shadow_img)

	mask = np.zeros(inp.shape, dtype=np.uint8)
	cv.circle(mask, (x, y), radius, (255, 255, 255), -1)

	bright = inp.copy()

	bright = cv.bitwise_and(bright, mask)
	shadow = cv.bitwise_and(shadow_img, cv.bitwise_not(mask))

	result = cv.bitwise_or(bright, shadow)
	cv.imshow(frameName, result)

	main(f, bright[x-radius:x+radius, y-radius:y+radius])


def change_brightness(img, value=-100):
	hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
	h, s, v = cv.split(hsv)
	v = cv.add(v, value)
	v[v > 255] = 255
	v[v < 0] = 0
	final_hsv = cv.merge((h, s, v))
	img = cv.cvtColor(final_hsv, cv.COLOR_HSV2BGR)
	return img


if __name__ == '__main__':

	f = eg.fileopenbox()
	inp = cv.imread(f)
	width, heigth, _ = inp.shape

	font = cv.FONT_HERSHEY_SIMPLEX
	frameName = 'Crop_GUI'
	cv.namedWindow(frameName)

	cv.createTrackbar('x', frameName, width // 2, width, crop_gui)
	cv.createTrackbar('y', frameName, heigth // 2, heigth, crop_gui)
	cv.createTrackbar('radius', frameName, min(width, heigth) // 2, min(width, heigth) // 2, crop_gui)

	if cv.waitKey(0) == ord('q'):  # to avoid auto close situation
		cv.destroyAllWindows()
