
import cv2
import skimage.io
import skimage.color
import skvideo.measure
import skimage.metrics
import skimage.restoration
import skimage.transform
from skimage import img_as_ubyte
from sklearn.cluster import KMeans

import scipy
from scipy import ndimage
import imageio
scipy.ndimage.imread = imageio.imread


import cpbd
# fix for old skvideo version
import numpy as np
np.int = int  # for newer np versions
import PIL
from PIL import Image


def imresize(image, factor, interp="nearest", mode=None):
    """
    resize an image with a specified resizing factor, this factor can also be
    the target shape of the resized image specified as tuple.
    """
    interp_methods = {
        "nearest": PIL.Image.NEAREST,
        "bicubic": PIL.Image.BICUBIC,
        "bilinear": PIL.Image.BILINEAR,
    }
    assert interp in interp_methods

    if type(factor) != tuple:
        new_shape = (int(factor * image.shape[0]), int(factor * image.shape[1]))
    else:
        assert len(factor) == 2
        new_shape = factor

    h, w = new_shape
    return np.array(
        Image.fromarray(image, mode=mode).resize(
            (w, h), resample=interp_methods[interp.lower()]
        )
    )
scipy.misc.imresize = imresize
# fix end


def extract_niqe(gray):
    """extract niqe score for a given image

    Args:
        gray (gray): grayscale image

    Returns:
        dict:predicted niqe score
    """
    niqe = skvideo.measure.niqe(gray)
    return float(niqe[0])


def color_fulness_features(image_rgb):
    """
    calculates color fullness

    re-implementated by Serge Molina

    References
    ----------
    - Hasler, David, and Sabine E. Suesstrunk. "Measuring colorfulness in natural images."
      In: Human vision and electronic imaging VIII. Vol. 5007. International Society for Optics and Photonics, 2003.
    """
    if len(image_rgb.shape) != 3:
        return -1

    rg = (image_rgb[:, :, 0] - image_rgb[:, :, 1]).ravel()
    yb = (image_rgb[:, :, 0] / 2 + image_rgb[:, :, 1] / 2 - image_rgb[:, :, 2]).ravel()

    rg_std = np.std(rg)
    yb_std = np.std(yb)

    rg_mean = np.mean(rg)
    yb_mean = np.mean(yb)

    trigo_len_std = np.sqrt(rg_std ** 2 + yb_std ** 2)
    neutral_dist = np.sqrt(rg_mean ** 2 + yb_mean ** 2)

    return float(trigo_len_std + 0.3 * neutral_dist)


def calc_tone_features(image, gray=False):
    """
    calculate tone feature,

    re-implemented by Serge Molina

    References
    ----------
    - T. O. Aydın, A. Smolic, and M. Gross. "Automated aesthetic analysis of photographic images".
      In: IEEE transactions on visualization and computer graphics 21.1 (2015), pp. 31–42.
    """
    if not gray:
        image_gray = skimage.color.rgb2gray(image)
    else:
        image_gray = image

    image_1d = image_gray.ravel()
    percentile05_value = np.percentile(image_1d, 5)
    percentile95_value = np.percentile(image_1d, 95)

    percentile30_value = np.percentile(image_1d, 30)
    percentile70_value = np.percentile(image_1d, 70)

    u = 0.05
    o = 0.05

    c_u = min(u, percentile95_value - percentile70_value) / u
    c_o = min(o, percentile30_value - percentile05_value) / o

    return c_u * c_o * (percentile95_value - percentile05_value)

def calc_contrast_features(frame):
    """
    calculates contrast based on histogram equalization,

    based on julan zebelein's master thesis
    """
    frame = img_as_ubyte(frame)
    hist, bins = np.histogram(frame.flatten(), 1024, [0, 1024])
    cdf = hist.cumsum()
    cdf_normalized = cdf * hist.max() / cdf.max()

    cdf_m = np.ma.masked_equal(cdf, 0)
    cdf_m = (cdf_m - cdf_m.min()) * 1024 / (cdf_m.max() - cdf_m.min())
    cdf = np.ma.filled(cdf_m, 0).astype("uint8")
    img2 = cdf[frame]

    hist2, bins = np.histogram(img2.flatten(), 1024, [0, 1024])
    cdf2 = hist2.cumsum()
    cdf2_normalized = cdf2 * hist2.max() / cdf2.max()

    sumAverageDifCDF = 0
    for x in range(256):
        histValue = cdf_normalized[x]
        perfectHistValue = cdf2_normalized[x]

        histValuePercent = (100 * histValue) / perfectHistValue
        difPercent = abs(histValuePercent - 100)

        sumAverageDifCDF += difPercent

    avgDif = 100 - sumAverageDifCDF / len(cdf_normalized)
    return float(avgDif)


def calc_fft_features(frame, debug=False):
    """
    calculates fft feature,

    based on julan zebelein's master thesis

    References
    ----------
    - I. Katsavounidis et al. "Native resolution detection of video sequences".
      In: Annual Technical Conference and Exhibition, SMPTE 2015. SMPTE. 2015, pp. 1–20.
    """

    def radial_profile(data, center):
        y, x = np.indices((data.shape))
        r = np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2)
        r = r.astype(np.int)

        tbin = np.bincount(r.ravel(), data.ravel())
        nr = np.bincount(r.ravel())
        radialprofile = tbin / nr
        return radialprofile

    # start video
    file_width = int(frame.shape[1])
    file_height = int(frame.shape[0])
    frame = np.uint8(frame)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    prefinal = cv2.resize(gray, (file_width, file_height))
    # final = cv2.GaussianBlur(prefinal,(5,5),0)
    final = cv2.bilateralFilter(prefinal, 9, 75, 75)

    f = np.fft.fft2(final)
    fshift = np.fft.fftshift(f)

    magnitude_spectrum = 20 * np.log(0.00000001 + np.abs(fshift))

    file_height, file_width = magnitude_spectrum.shape
    CurrentCenter = (file_width / 2, file_height / 2)

    # calculate the azimuthally averaged 1D power spectrum
    psf1D = radial_profile(magnitude_spectrum, CurrentCenter)
    lowFreqInd = int((len(psf1D) / 2))

    psf1D_onlyHighFreq = psf1D[lowFreqInd:]
    sum_of_high_frequencies = sum(psf1D_onlyHighFreq)

    return float(sum_of_high_frequencies)


def calc_saturation_features(frame, debug=True):
    """
    calculates saturation of a given image,

    re-implemented by Serge Molina

    References
    ----------
    - T. O. Aydın, A. Smolic, and M. Gross. "Automated aesthetic analysis of photographic images".
      In: IEEE transactions on visualization and computer graphics 21.1 2015, pp. 31–42.""
    """
    file_width = int(frame.shape[1])
    file_height = int(frame.shape[0])
    frame = np.uint8(frame)

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    average_hsvValue = hsv[:, :, 1].sum() / (file_width * file_height)
    averageSaturationCurrentFrame = (average_hsvValue * 100) / 256

    return float(averageSaturationCurrentFrame)


def calc_blur_features(frame, debug=False):
    """
    estimates blurriness using Laplacian filter,

    based on julian zebelein's master thesis
    """

    def variance_of_laplacian(image):
        # compute the Laplacian of the image and then return the focus
        # measure, which is simply the variance of the Laplacian
        return cv2.Laplacian(image, cv2.CV_64F, ksize=5).var()

    file_width = int(frame.shape[1])
    file_height = int(frame.shape[0])
    frame = np.uint8(frame)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    prefinal = cv2.resize(gray, (file_width, file_height))
    # final = cv2.GaussianBlur(prefinal,(5,5),0)
    final = cv2.bilateralFilter(prefinal, 9, 75, 75)
    fm = variance_of_laplacian(final)
    return float(fm)

def calc_noise(frame):
    """
    calcualtes noise std based on skimage.restoration.estimate_sigma
    "Robust wavelet-based estimator of the (Gaussian) noise standard deviation."

    Returns
    -------
    mean value of all channels for std value of noise assuming that the noise has a Gaussian distribution
    """
    return float(skimage.restoration.estimate_sigma(frame, average_sigmas=True, channel_axis=2))


def calc_dominant_color(image):
    """
    estimates the dominant color of an image

    the estimation is done by Kmeans clustering (8 clusters/colors) of the image

    Args:
        image ([3D]): image array in RGB

    Returns:
        [r,g,b]: rgb value of most dominant color
    """
    # rescale image
    c, r = image.shape[:2]
    out_r = 120
    image = skimage.transform.resize(
        image,
            (int(out_r*float(c)/r), out_r)
    )
    pixels = image.reshape((-1, 3))

    km = KMeans(n_clusters=8)
    km.fit(pixels)
    colors = km.cluster_centers_

    labels = km.labels_
    hist = {}
    for i in labels:
        hist[i] = hist.get(i, 0) + 1

    max_label = max(hist, key=lambda x: hist[x])

    return [int(x*255) for x in colors[max_label]]


def calc_si(image_gray):
    sobx = ndimage.sobel(image_gray, axis=0)
    soby = ndimage.sobel(image_gray, axis=1)
    value = np.hypot(sobx, soby).std()
    return float(value)


def calc_cpbd(gray):
    """
    see: https://github.com/0x64746b/python-cpbd
    cite
    N. D. Narvekar and L. J. Karam, "A No-Reference Image Blur Metric Based on the Cumulative Probability of Blur Detection (CPBD)," in IEEE Transactions on Image Processing, vol. 20, no. 9, pp. 2678-2683, Sept. 2011.
    N. D. Narvekar and L. J. Karam, "An Improved No-Reference Sharpness Metric Based on the Probability of Blur Detection," International Workshop on Video Processing and Quality Metrics for Consumer Electronics (VPQM), January 2010, http://www.vpqm.org (pdf)
    N. D. Narvekar and L. J. Karam, "A no-reference perceptual image sharpness metric based on a cumulative probability of blur detection," 2009 International Workshop on Quality of Multimedia Experience, San Diego, CA, 2009, pp. 87-91.

    this feature has been also used in https://dl.acm.org/doi/pdf/10.1145/3423328.3423501
    """
    return float(cpbd.compute(gray))


def calc_blur_strength(gray):
    """cite: Frederique Crete, Thierry Dolmiere, Patricia Ladret, and Marina Nicolas “The blur effect: perception and estimation with a new no-reference perceptual blur metric” Proc. SPIE 6492, Human Vision and Electronic Imaging XII, 64920I (2007) https://hal.archives-ouvertes.fr/hal-00232709 DOI:10.1117/12.702790

    """
    return float(skimage.measure.blur_effect(gray))


def extract_features(imagefilename):
    img = skimage.io.imread(imagefilename)
    if len(img.shape) == 2: # in this case the image itself is gray, so convert it to a "colored" aka 3 channel image
        img = skimage.color.gray2rgb(img)

    try:
        # check datatype here of img, and also of gray
        gray = skimage.color.rgb2gray(img)
        features = {
            "image": imagefilename,
            "niqe": extract_niqe(gray),
            "color_fulness": color_fulness_features(img),
            "tone": calc_tone_features(gray, gray=True),
            "blur": calc_blur_features(img),
            "saturation": calc_saturation_features(img),
            "fft": calc_fft_features(img),
            "si": calc_si(gray),
            "contrast": calc_contrast_features(img),
            "noise": calc_noise(img),
            "dominant_color": calc_dominant_color(img),
            "cpbd": calc_cpbd(gray),
            "blur_stength": calc_blur_strength(gray)
        }
        return features
    except:
        print(img.shape)
        print(f"[error] image: {imagefilename}")
        return {}