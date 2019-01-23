import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig

# display several images in one row if flag is 1
# if flag is 0, you must call plt.show() yourself
# arguments must be image variable and image name one after the other:
# displayimages(title, flag, var_image1, 'first image', var_image_2, 'second image')
def displayimages(title, flag, *images):

	# the number of images to plot is half the number of *images arguments
	number = len(images) >> 1

	# window
	plt.figure().canvas.set_window_title(title)

	# show images in one row
	for count in range(number):
		plt.subplot(1, number, count + 1)
		plt.imshow(images[count << 1])
		plt.title(images[(count << 1) + 1])
		plt.xticks([])
		plt.yticks([])

	# display all
	flag and plt.show()

# take a pyplot colour image and make it greyscale
def col2gry(colour, power):

	# check power
	if power <= 0:
		print 'In function col2gry'
		print 'MathError: power must not be negative.'
		print '-----'
		raise SystemExit

	# generate greyscale image using norm-like math
	grey = colour
	norm = ((colour[:, :, 0] ** power + colour[:, :, 1] ** power + colour[:, :, 2] ** power) / 3.0) ** (1.0 / power)
	grey[:, :, 0] = grey[:, :, 1] = grey[:, :, 2] = norm
	return grey

# display an 8-bit greyscale image and plot its histogram alongside
# img is the image to display and plot a histogram of
# ptitle is the title of the histogram plot
# flag indicates whether the function should open the window for you or not
# if not, you must call the plt.show() yourself
def plotgry8(ptitle, flag, img):

	# window
	plt.figure().canvas.set_window_title(ptitle)

	# first display the image
	plt.subplot(1, 2, 1)
	plt.imshow(img)
	plt.title('8-bit image')
	plt.xticks([])
	plt.yticks([])

	# now generate a histogram and display it
	plt.subplot(1, 2, 2)
	hist, bin_edges = np.histogram(img[:, :, 0].ravel(), bins = np.linspace(0, 256, 257))
	plt.bar(bin_edges[: -1], hist, width = 1)
	plt.grid(True)
	plt.xlabel('pixel intensity')
	plt.ylabel('number of pixels')
	plt.title('histogram')
	flag and plt.show()

# perform two-dimensional convolution of 8-bit greyscale image with a mask
def imconv(img, mask):

	# get the convolved image
	slope = np.zeros(img.shape)
	slope[:, :, 0] = sig.convolve2d(img[:, :, 0], mask, mode = 'same', boundary = 'symm')
	slope[:, :, 2] = slope[:, :, 1] = slope[:, :, 0]

	# scale the output
	slope -= np.amin(slope)
	slope *= 255.0 / np.amax(slope)
	slope = np.uint8(slope)

	# return the scaled output
	return slope

# sharpen an 8-bit image using Laplacian
def imsharpen(img, order, scale):

	# check order
	if order not in range(1, 3):
		print 'In function imsharpen'
		print 'ValueError: order out of bounds.'
		print '-----'
		raise SystemExit

	# choose Laplacian
	if order == 1:
		mask = np.array([[0, 1, 0],
                         [1, -4, 1],
                         [0, 1, 0]])
	else:
		mask = np.array([[1, 1, 1],
                         [1, -8, 1],
                         [1, 1, 1]])

	# sharpen using Laplacian
	edges = np.zeros(img.shape)
	edges[:, :, 0] = sig.convolve2d(img[:, :, 0], mask, mode = 'same', boundary = 'symm')
	edges[:, :, 1] = sig.convolve2d(img[:, :, 1], mask, mode = 'same', boundary = 'symm')
	edges[:, :, 2] = sig.convolve2d(img[:, :, 2], mask, mode = 'same', boundary = 'symm')
	fine = np.uint8(np.clip(img - scale * edges, 0, 255))

	# the sharpened image is output
	return fine
