"""
Image hashing methods.

| Copyright 2017-2024, Voxel51, Inc.
| `voxel51.com <https://voxel51.com/>`_
|
"""

import numpy as np
import eta.core.image as etai
import scipy


def compute_image_hash(image_path, method="phash", hash_size=8):
    """
    Computes a hash of the input image.

    Args:
        image_path: Input image path.
        method: The hashing method to use. Supported values are
            "ahash", "phash", "dhash", and "whash".
        hash_size: Size of the hash (default is 8x8).

    Returns:
        A 1D NumPy array representing the hash.
    """
    image = etai.read(image_path)
    if method == "ahash":
        return ahash(image, hash_size=hash_size)
    elif method == "phash":
        return phash(image, hash_size=hash_size)
    elif method == "dhash":
        return dhash(image, hash_size=hash_size)
    elif method == "whash":
        return whash(image, hash_size=hash_size)
    else:
        raise ValueError("Unsupported hashing method '%s'" % method)


def ahash(image, hash_size=8):
    """
    Computes the average hash (aHash) of an image.

    Args:
        image: Input image as a NumPy array.
        hash_size: Size of the hash (default is 8x8).

    Returns:
        A 1D NumPy array representing the hash.
    """
    # Step 1: Convert to grayscale
    gray = etai.rgb_to_gray(image)

    # Step 2: Resize to hash_size x hash_size
    resized = etai.resize(gray, hash_size, hash_size)

    # Step 3: Compute the mean pixel value
    mean = resized.mean()

    # Step 4: Create the binary hash
    binary_hash = (resized >= mean).astype(np.uint8)

    # Step 5: Flatten the hash to 1D
    flat_hash = binary_hash.flatten()

    return flat_hash


def phash(image, hash_size=8):
    """
    Computes the perceptual hash (pHash) of an image.

    Args:
        image: Input image as a NumPy array.
        hash_size: Size of the hash (default is 8x8).

    Returns:
        A 1D NumPy array representing the hash.
    """
    # Step 1: Convert to grayscale
    gray = etai.rgb_to_gray(image)

    # Step 2: Resize to hash_size x hash_size
    resized = etai.resize(gray, hash_size, hash_size)

    # Step 3: Compute the Discrete Cosine Transform (DCT)
    dct = scipy.fft.dct(resized, norm="ortho")

    # Step 4: Extract the top-left hash_size x hash_size values
    dct = dct[:hash_size, :hash_size]

    # Step 5: Compute the median of the top-left values
    median = np.median(dct)

    # Step 6: Create the binary hash
    binary_hash = (dct >= median).astype(np.uint8)

    # Step 7: Flatten the hash to 1D
    flat_hash = binary_hash.flatten()

    return flat_hash


def dhash(image, hash_size=8):
    """
    Compute the dHash for the input image.

    :param image: Input image to hash (as a NumPy array).
    :param hash_size: Size of the hash (default 8x8).
    :return: The dHash value of the image as a 64-bit integer.
    """
    # Convert the image to grayscale
    gray = etai.rgb_to_gray(image)

    # Resize the image to (hash_size + 1, hash_size)
    resized = etai.resize(gray, hash_size + 1, hash_size)

    # Compute the differences between adjacent pixels
    diff = resized[:, 1:] > resized[:, :-1]

    # Convert the difference image to a binary array
    binary_array = diff.flatten().astype(int)

    return binary_array


def whash(image, hash_size=8):
    """
    Computes the wavelet hash (wHash) of an image.

    Args:
        image: Input image as a NumPy array.
        hash_size: Size of the hash (default is 8x8).

    Returns:
        A 1D NumPy array representing the hash.
    """
    import pywt

    # Step 1: Convert to grayscale
    gray = etai.rgb_to_gray(image)

    # Step 2: Resize to hash_size x hash_size
    resized = etai.resize(gray, hash_size, hash_size)

    # Step 3: Compute the wavelet transform
    coeffs = pywt.dwt2(resized, "haar")
    cA, (cH, cV, cD) = coeffs

    # Step 4: Extract the approximation coefficients
    cA = cA.flatten()

    # Step 5: Compute the mean of the approximation coefficients
    mean = cA.mean()

    # Step 6: Create the binary hash
    binary_hash = (cA >= mean).astype(np.uint8)

    return binary_hash


def hamming_distance(hash1, hash2):
    """
    Computes the Hamming distance between two hashes.

    Args:
        hash1: First hash as a 1D NumPy array.
        hash2: Second hash as a 1D NumPy array.

    Returns:
        The Hamming distance (integer).
    """
    return np.count_nonzero(hash1 != hash2)
