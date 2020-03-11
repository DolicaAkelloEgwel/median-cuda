import scipy.ndimage as scipy_ndimage


def scipy_median_filter(data, size):
    for idx in range(0, data.shape[0]):
        data[idx] = scipy_ndimage.median_filter(data[idx], size, mode="mirror")
