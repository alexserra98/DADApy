# metrics.pyx
import numpy as np
cimport numpy as cnp

cpdef double angular_distance(cnp.ndarray[double, ndim=1] v1, cnp.ndarray[double, ndim=1] v2):
    np.import_array()
    cdef double dot_product = np.dot(v1, v2)
    cdef double norm_product = np.linalg.norm(v1) * np.linalg.norm(v2)
    cdef double cosine_similarity = dot_product / norm_product

    # Clip to [-1, 1] and assert within bounds in Python space
    cosine_similarity = max(min(cosine_similarity, 1), -1)
    assert -1 <= cosine_similarity <= 1

    cdef double angle = np.arccos(cosine_similarity)
    return angle / np.pi
