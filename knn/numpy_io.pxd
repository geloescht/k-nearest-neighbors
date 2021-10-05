cimport numpy

ctypedef (int, int) npy_version

cdef numpy.ndarray _array_from_buffer(const unsigned char[::1] raw)
cdef bytes _array_to_buffer(numpy.ndarray array, npy_version version=*)
