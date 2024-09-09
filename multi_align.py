import numpy as np
import skimage.io as skio
import skimage.transform as skt
from skimage import img_as_float
from multiprocessing import Pool

# NCC (Normalized Cross-Correlation) function
def ncc(a, b):
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0
    a = a - np.mean(a)
    b = b - np.mean(b)
    return np.sum(((a / norm_a) * (b / norm_b)))

# Optimized NCC-based image alignment function
def nccAlign(a, b, t):
    min_ncc = -1
    output = [0, 0]  # Initialize output

    ivalue = range(-t, t + 1)
    jvalue = range(-t, t + 1)

    for i in ivalue:
        for j in jvalue:
            shifted_b = np.roll(b, [i, j], axis=(0, 1))
            nccDiff = ncc(a, shifted_b)
            if nccDiff > min_ncc:
                min_ncc = nccDiff
                output = [i, j]

    return output

# Image Pyramid-based alignment with improved accuracy and efficiency
def pyramidAlign(a, b, max_levels=5, scale=0.5, t=20):
    pyramid_a = [a]
    pyramid_b = [b]

    for level in range(1, max_levels):
        pyramid_a.append(skt.rescale(pyramid_a[-1], scale, channel_axis=None, anti_aliasing=True))
        pyramid_b.append(skt.rescale(pyramid_b[-1], scale, channel_axis=None, anti_aliasing=True))

    shift = [0, 0]  # Initial shift is zero
    for level in reversed(range(max_levels)):
        a_layer = pyramid_a[level]
        b_layer = pyramid_b[level]
        current_shift = nccAlign(a_layer, b_layer, t)
        shift[0] = current_shift[0] * (1 / scale) + shift[0]
        shift[1] = current_shift[1] * (1 / scale) + shift[1]
        scale *= 2  # Adjust scale for higher resolution layers

    return [int(shift[0]), int(shift[1])]

def automatic_cropping(img, threshold=0.05):
    std_dev = np.std(img, axis=2)
    mask = std_dev < threshold
    coords = np.column_stack(np.where(mask))
    if coords.size == 0:
        return img
    top_left = coords.min(axis=0)
    bottom_right = coords.max(axis=0)
    cropped_img = img[top_left[0]:bottom_right[0], top_left[1]:bottom_right[1]]
    return cropped_img

def align_channels(args):
    channel_a, channel_b = args
    return pyramidAlign(channel_a, channel_b)

# Main processing function
def process_image(imname):
    im = skio.imread(imname)
    im = img_as_float(im)

    height = np.floor(im.shape[0] / 3.0).astype(int)
    b = im[:height]
    g = im[height: 2*height]
    r = im[2*height: 3*height]

    # Use multiprocessing for faster alignment
    with Pool(processes=2) as pool:
        alignments = pool.map(align_channels, [(b, g), (b, r)])
    
    # Print (x, y) displacement vectors
    print(f"Alignment for G channel: {alignments[0]}")
    print(f"Alignment for R channel: {alignments[1]}")

    g_aligned = np.roll(g, alignments[0], axis=(0, 1))
    r_aligned = np.roll(r, alignments[1], axis=(0, 1))

    im_out = np.dstack([r_aligned, g_aligned, b])
    im_out = automatic_cropping(im_out)

    fname = 'data/melons.tif'
    skio.imsave(fname, im_out)

    skio.imshow(im_out)
    skio.show()

# This part ensures that multiprocessing works correctly
if __name__ == "__main__":
    process_image('data/melons.tif')

