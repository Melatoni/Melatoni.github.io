# CS180 (CS280A): Project 1 starter Python code

# these are just some suggested libraries
# instead of scikit-image you could use matplotlib and opencv to read, write, and display images

import numpy as np
import skimage as sk
import skimage.io as skio

# NCC (Normalized Cross-Correlation) function
def ncc(a, b):
    a = a - a.mean(axis=0)
    b = b - b.mean(axis=0)
    return np.sum(((a / np.linalg.norm(a)) * (b / np.linalg.norm(b))))

# NCC-based image alignment function
def nccAlign(a, b, t):
    min_ncc = -1
    ivalue = np.linspace(-t, t, 2 * t, dtype=int)
    jvalue = np.linspace(-t, t, 2 * t, dtype=int)

    for i in ivalue:
        for j in jvalue:
            nccDiff = ncc(a, np.roll(b, [i, j], axis=(0, 1)))
            if nccDiff > min_ncc:
                min_ncc = nccDiff
                output = [i, j]
    return output

# name of the input file
imname = 'data/tobolsk.jpg'

# read in the image
im = skio.imread(imname)

# convert to double (might want to do this later on to save memory)
im = sk.img_as_float(im)

# compute the height of each part (just 1/3 of total)
height = np.floor(im.shape[0] / 3.0).astype(int)

# separate color channels
b = im[:height]
g = im[height: 2*height]
r = im[2*height: 3*height]

# align the images using NCC alignment
alignGtoB = nccAlign(b, g, 15)
alignRtoB = nccAlign(b, r, 15)

# apply the alignment shifts
g_aligned = np.roll(g, alignGtoB, axis=(0, 1))
r_aligned = np.roll(r, alignRtoB, axis=(0, 1))

# create a color image
im_out = np.dstack([r_aligned, g_aligned, b])

# save the image
fname = 'processed_image/tobolsk.jpg'
skio.imsave(fname, im_out)

# display the image
skio.imshow(im_out)
skio.show()
