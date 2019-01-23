#!/usr/bin/env python

import cv2
import numpy as np
import sys
import pywt

np.set_printoptions(threshold = np.nan, linewidth = np.inf, formatter = {'float': lambda x: '{0:5.2f}'.format(x)})

################################################################################

'''
	check whether argument is a power of 2
	this is a simple trick to make the check
	any power of 2 is of the form N = 1000...0 in binary
	N - 1 = 111...1
	so, N & (N - 1) = 0
	else, 'N' is not a power of 2
	this method returns True for N = 0
	it is necessary to make another check when using it
'''
def is_power_of_2(N):
	return N & (N - 1) == 0

################################################################################

'''
	generate a dictionary whose columns are Haar Wavelets
	to make the dictionary overcomplete, all possible shifts are used
	a complete dictionary does not remove much noise
'''
def generate_Haar_dictionary(dimension):

	'''
		'dimension' must be validated before it is passed to this function
		so no validation will be performed here
		it should be a perfect square of a power of 2 and be >= 64
		the number of columns ('size') required is as per the formula
		size = 3 + (dimension + 1) * log2(dimension) - 2 * dimension
	'''
	logarithm = 0
	number = dimension
	while number != 1:
		number >>= 1
		logarithm += 1
	size = 3 + (dimension + 1) * logarithm - 2 * dimension
	dictionary = np.zeros([dimension, size])

	'''
		create a Haar Wavelet object
		this is used only to make the process of getting phi and psi easy
	'''
	wave = pywt.Wavelet('haar')

	'''
		traverse the columns of 'dictionary' starting from the second column
		each column will be a wavelet
	'''
	level = 1 # used to generate wavelets
	column = 1
	while column < size:

		'''
			'level' is the level of the wavelet
			len(psi) = len(phi) = 2 ** level + 2
			two elements are zeros padded at the start and end, so not required
			number of ways in which sequence of length '2 ** level' can be placed in vector of length 'dimension'
			will be 'dimension - 2 ** level + 1'
		'''
		phi, psi, x = wave.wavefun(level)
		wavelet_length = 1 << level
		n_bases = dimension - wavelet_length + 1

		# decides where to start placing the psi vector from
		start = 0

		'''
			place the psi vectors in increasing positions
			for example, if psi = [1, -1]
			place psi in the first two rows
			then place it in the second and third rows and so on
		'''
		for count in range(n_bases):

			'''
				the psi vector is padded with zeros at the beginning and the end, as mentioned previously
				array slicing is used to remove them, because they are not required
				that square rooting is to normalize psi so that it is of unit energy
			'''
			dictionary[start : start + wavelet_length, column] = np.sqrt(1.0 / wavelet_length) * psi[1 : -1]

			# move to the next column, and advance the location of next wavelet
			column += 1
			start += 1

		# all vectors of 'level' have been used; go to the next
		level += 1

	# fill the first column of 'dictionary' with 'phi' of the last step
	dictionary[:, 0] = np.sqrt(1.0 / wavelet_length) * phi[1 : -1]

	return dictionary

################################################################################

# return a three-dimensional array of all possible square patches of given size
def separate_image_into_blocks(image, square_side):

	# how many patches can be placed on the image
	rows, columns = image.shape
	horizontal = rows - square_side + 1
	vertical = columns - square_side + 1
	n_patches = horizontal * vertical

	# separate the image into said patches
	patches = np.zeros([n_patches, square_side, square_side])
	count = 0
	for x in range(horizontal):
		for y in range(vertical):
			patches[count] = image[x : x + square_side, y : y + square_side]
			count += 1

	return patches


################################################################################

if __name__ == '__main__':

	# check arguments
	if len(sys.argv) != 3:
		print 'usage:'
		print '\tpython reconstruction.py <input image> <length of square block>'
		raise SystemExit

	# second argument is the block size to be used
	square_side = int(sys.argv[2])
	if square_side < 2 or not is_power_of_2(square_side):
		raise ValueError('Size of blocks must be an integral power of 2 and be >= 8.')

	# first argument is image file
	source_image = sys.argv[1]
	noisy = cv2.imread(source_image, 0)
	if noisy is None:
		raise IOError('Image file \'%s\' not found.' % source_image)

	# size of blocks must be less than that of image
	rows, columns = noisy.shape
	if rows < square_side:
		raise ValueError('Length of block exceeds length of image.')
	if columns < square_side:
		raise ValueError('Width of block exceeds width of image.')
	cv2.imwrite('noisy_image.png', noisy[square_side - 1 : -square_side + 1, square_side - 1 : -square_side + 1])

	# obtain the dictionary of Haar Wavelets
	print 'generating overcomplete dictionary of Haar Wavelets'
	dimension = square_side * square_side
	dictionary = generate_Haar_dictionary(dimension)
	print dictionary.shape, 'dictionary created\n'

	# separate the image into scaled square patches of 'square_side'
	print 'separating image into (%d, %d) blocks' % (square_side, square_side)
	noisy = noisy.astype(float) / 255
	patches = separate_image_into_blocks(noisy, square_side)
	n_patches = len(patches)
	print n_patches, 'blocks created\n'

	# store sparsely represented vectors here
	processed_patches = np.zeros(patches.shape)

	# process each square patch and represent it sparsely using sparse vector 'alpha'
	for count in range(n_patches):

		print 'processing block #%d / %d' % (count, n_patches - 1)

		'''
			open up the patch into a one-dimensional vector
			if any pixels are missing, replace them by the average of the surrounding pixels
		'''
		v = np.ravel(patches[count])
		alpha = np.dot(np.linalg.pinv(dictionary), v)
		threshold = 0.03
		alpha[abs(alpha) < threshold] = 0
		alpha[alpha < 0] += threshold
		alpha[alpha > 0] -= threshold

		# sparsely represented vector is used to reconstruct the patch
		w = np.dot(dictionary, alpha)
		processed_patches[count] = w.reshape([square_side, square_side])

	# everything is ready, so restore the image using the reconstructed patches
	destination = np.zeros([rows, columns])
	horizontal = rows - square_side + 1
	vertical = columns - square_side + 1
	count = 0
	for x in range(horizontal):
		for y in range(vertical):
			destination[x : x + square_side, y : y + square_side] += processed_patches[count]
			count += 1
	destination = np.uint8(np.clip(255 * destination / dimension, 0, 255))
	cv2.imwrite('noisy_image_denoised.png', destination[square_side - 1 : -square_side + 1, square_side - 1 : -square_side + 1])
