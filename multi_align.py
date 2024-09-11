import numpy as np
import skimage.io as skio
import skimage.transform as skt
from skimage import img_as_float
from multiprocessing import Pool
from skimage import color

# NCC (Normalized Cross-Correlation) function
def ncc(a, b):
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0
    a = a - np.mean(a)
    b = b - np.mean(b)
    return np.sum((a / norm_a) * (b / norm_b))

# Optimized NCC-based image alignment function
def nccAlign(a, b, t):
    min_ncc = -1
    output = [0, 0]  # Initialize output

    # Crop central region of the image to avoid border influence
    h, w = a.shape
    crop_h, crop_w = int(0.05 * h), int(0.05 * w)
    a_cropped = a[crop_h:-crop_h, crop_w:-crop_w]
    b_cropped = b[crop_h:-crop_h, crop_w:-crop_w]
    
    ivalue = range(-t, t + 1)
    jvalue = range(-t, t + 1)

    for i in ivalue:
        for j in jvalue:
            shifted_b = np.roll(b_cropped, [i, j], axis=(0, 1))
            ncc_diff = ncc(a_cropped, shifted_b)
            if ncc_diff > min_ncc:
                min_ncc = ncc_diff
                output = [i, j]

    return output

# Image Pyramid-based alignment with improved accuracy and efficiency
def pyramidAlign(a, b, max_levels=3, scale=0.5, t=15):
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

    return [int(shift[0]), int(shift[1])]

# Automatic cropping based on color consistency
def automatic_cropping(img, threshold=0.05):
    diff_b = np.abs(img[:,:,0] - img[:,:,1])  # Difference between blue and green channels
    diff_r = np.abs(img[:,:,2] - img[:,:,1])  # Difference between red and green channels
    total_diff = diff_b + diff_r

    mask = total_diff > threshold
    coords = np.column_stack(np.where(mask))
    if coords.size == 0:
        return img
    top_left = coords.min(axis=0)
    bottom_right = coords.max(axis=0)
    cropped_img = img[top_left[0]:bottom_right[0], top_left[1]:bottom_right[1]]
    return cropped_img

# Automatic contrast adjustment using cumulative histogram equalization
def automatic_contrast(img):
    img = (img * 255).astype(np.uint8)  # Convert image to uint8 for histogram equalization
    for i in range(3):  # Apply to each channel
        channel = img[:, :, i]
        # Compute histogram
        hist, _ = np.histogram(channel.flatten(), bins=256, range=[0, 256])
        # Compute cumulative histogram
        cdf = hist.cumsum()
        cdf_normalized = cdf / cdf[-1]  # Normalize CDF
        # Use linear interpolation of CDF to find new pixel values
        img[:, :, i] = np.interp(channel.flatten(), np.arange(256), cdf_normalized * 255).reshape(channel.shape)
    return img / 255.0  # Convert back to float in [0, 1]

# Automatic white balance adjustment
def automatic_white_balance(img):
    img = img_as_float(img)
    
    # Compute the average color of the image
    avg_color = np.mean(img, axis=(0, 1))
    
    # Define a target gray point (e.g., [0.5, 0.5, 0.5])
    gray_point = np.array([0.5, 0.5, 0.5])
    
    # Compute the scaling factors to adjust each color channel
    scaling = gray_point / avg_color
    
    # Apply the scaling factors to the entire image
    balanced_img = img * scaling[None, None, :]
    
    # Clip the pixel values to be within the valid range [0, 1]
    balanced_img = np.clip(balanced_img, 0, 1)
    
    return balanced_img

def align_channels(args):
    channel_a, channel_b = args
    return pyramidAlign(channel_a, channel_b)

# Save the output image in both TIFF and JPEG formats
def save_output(im_out, fname_tif, fname_jpg):
    # Convert float64 image to uint8
    im_out_uint8 = (im_out * 255).astype(np.uint8)
    
    # Save as TIFF and JPEG
    skio.imsave(fname_tif, im_out_uint8)
    skio.imsave(fname_jpg, im_out_uint8)

# Main processing function
def process_image(imname):
    im = skio.imread(imname)
    im = img_as_float(im)

    # Split the image into thirds (B, G, R)
    height = np.floor(im.shape[0] / 3.0).astype(int)
    b = im[:height]
    g = im[height: 2*height]
    r = im[2*height: 3*height]

    # Use multiprocessing to speed up alignment
    with Pool(processes=4) as pool:
        alignments = pool.map(align_channels, [(b, g), (b, r)])
    
    # Output the displacement for Green and Red channels
    print(f"Alignment for G channel: {alignments[0]}")
    print(f"Alignment for R channel: {alignments[1]}")

    # Apply alignment shifts
    g_aligned = np.roll(g, alignments[0], axis=(0, 1))
    r_aligned = np.roll(r, alignments[1], axis=(0, 1))

    # Combine the aligned channels
    im_out = np.dstack([r_aligned, g_aligned, b])

    # Apply automatic cropping, contrast, and white balance
    im_out = automatic_cropping(im_out)
    im_out = automatic_contrast(im_out)
    im_out = automatic_white_balance(im_out)

    # Save the output as both TIFF and JPEG
    save_output(im_out, 'processed_image/emir.tif', 'processed_image/emir.jpg')

    # Display the final image
    skio.imshow(im_out)
    skio.show()

# Ensure multiprocessing works correctly
if __name__ == "__main__":
    process_image('data/emir.tif')
