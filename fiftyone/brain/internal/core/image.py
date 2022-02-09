"""
Core methods for working with images.

| Copyright 2017-2020, Voxel51, Inc.
| `voxel51.com <https://voxel51.com/>`_
|
"""
import cv2
import skimage.metrics as skm

import eta.core.image as etai


def compute_quality(img, method=None):
    """Computes the quality of the image using the specified method.

    Quality is returned on a ``[0, 100]`` scale, where ``0 = low quality`` and
    ``100 = highest quality``.

    Args:
        img: the image
        method (None): the image quality method. Supported values are
            ``("laplacian-stdev", "median-psnr", "median-ssim")``. The default
            is ``"laplacian-stdev"``

    Returns:
        the image quality score, in ``[0, 100]``
    """
    method_lower = method.lower() if method else "laplacian-stdev"

    if method_lower == "laplacian-stdev":
        stdev = stdev_of_laplacian(img)
        return min(stdev, 100.0)

    if method_lower == "median-psnr":
        #
        # @todo improve this? currently we assume that PSNR = [30dB, 50dB] is
        # a typical range for 8bit images
        #
        # @todo handle non 8-bit images
        #
        psnr = psnr_wrt_median(img)
        return 5.0 * min(max(0, psnr - 30.0), 20.0)

    if method_lower == "median-ssim":
        ssim = ssim_wrt_median(img)
        return 50.0 * (1.0 + ssim)

    raise ValueError("Unsupported `method = %s`" % method)


def stdev_of_laplacian(img, kernel_size=3):
    """Computes the standard deviation of the Laplacian of the given image.

    References:
        https://www.pyimagesearch.com/2015/09/07/blur-detection-with-opencv
        https://ieeexplore.ieee.org/document/903548
        http://isp-utb.github.io/seminario/papers/Pattern_Recognition_Pertuz_2013.pdf

    Args:
        img: an image
        kernel_size (3): the kernel size to use

    Returns:
        the standard deviation of the Laplacian filtered version of the image
    """
    return cv2.Laplacian(img, cv2.CV_32F, ksize=kernel_size).std()


def psnr_wrt_median(img, kernel_size=3):
    """Computes the PSNR, in dB, of the image with respect to a median-filtered
    version of the image.

    Args:
        img: an image
        kernel_size (3): the median kernel size to use

    Returns:
        the PSNR
    """
    img_median = cv2.medianBlur(img, ksize=kernel_size)
    return cv2.PSNR(img, img_median)


def ssim_wrt_median(img, kernel_size=3):
    """Computes the SSIM of the image with respect to a median-filtered version
    of the image.

    Args:
        img: an image
        kernel_size (3): the median kernel size to use

    Returns:
        the SSIM, in ``[-1, 1]``
    """
    img_median = cv2.medianBlur(img, ksize=kernel_size)
    multichannel = etai.is_color(img)
    return skm.structural_similarity(
        img, img_median, multichannel=multichannel
    )
