cimport numpy
import numpy
import struct
import ast

cdef bytes MAGIC_PREFIX = b'\x93NUMPY'
cdef int MAGIC_LEN = len(MAGIC_PREFIX) + 2
cdef int ARRAY_ALIGN = 64 # plausible values are powers of 2 between 16 and 4096

_header_size_info = {
    (1, 0): ('<H', struct.calcsize('<H'), 'latin1'),
    (2, 0): ('<I', struct.calcsize('<I'), 'latin1'),
    (3, 0): ('<I', struct.calcsize('<I'), 'utf8'),
}

cdef bytes magic(npy_version version):
    if version[0] < 0 or version[0] > 255:
        raise ValueError("major version must be 0 <= major < 256")
    if version[1] < 0 or version[1] > 255:
        raise ValueError("minor version must be 0 <= minor < 256")
    return MAGIC_PREFIX + bytes([version[0], version[1]])

cdef void _check_version(npy_version version):
    if (version[0] != -1 or version[1] != -1) and (version[0] < 1 or version[0] > 3 or version[1] != 0):
        msg = "we only support format version (1,0), (2,0), and (3,0), not %s"
        raise ValueError(msg % (version,))

cdef npy_version read_magic(const unsigned char[::1] raw):
    cdef const unsigned char[::1] magic_str = raw[:MAGIC_LEN]
    for i in range(0, MAGIC_LEN-2):
        if MAGIC_PREFIX[i] != magic_str[i]:
            msg = "the magic string is not correct; expected %r, got %r"
            raise ValueError(msg % (MAGIC_PREFIX, magic_str[:-2]))
    cdef int major = magic_str[-2]
    cdef int minor = magic_str[-1]
    return major, minor

cdef object descr_to_dtype(descr):
    if isinstance(descr, str):
        # No padding removal needed
        return numpy.dtype(descr)
    elif isinstance(descr, tuple):
        # subtype, will always have a shape descr[1]
        dt = descr_to_dtype(descr[0])
        return numpy.dtype((dt, descr[1]))

    titles = []
    names = []
    formats = []
    offsets = []
    offset = 0
    for field in descr:
        if len(field) == 2:
            name, descr_str = field
            dt = descr_to_dtype(descr_str)
        else:
            name, descr_str, shape = field
            dt = numpy.dtype((descr_to_dtype(descr_str), shape))

        # Ignore padding bytes, which will be void bytes with '' as name
        # Once support for blank names is removed, only "if name == ''" needed)
        is_pad = (name == '' and dt.type is numpy.void and dt.names is None)
        if not is_pad:
            title, name = name if isinstance(name, tuple) else (None, name)
            titles.append(title)
            names.append(name)
            formats.append(dt)
            offsets.append(offset)
        offset += dt.itemsize

    return numpy.dtype({'names': names, 'formats': formats, 'titles': titles,
                        'offsets': offsets, 'itemsize': offset})

cdef safe_eval = ast.literal_eval
cdef calcsize = struct.calcsize
cdef unpack = struct.unpack
cdef pack = struct.pack
cdef allowed_header_keys = ['descr', 'fortran_order', 'shape']

cdef _read_array_header(const unsigned char[::1] raw, npy_version version):
    #hinfo = _header_size_info.get(version)
    #if hinfo is None:
    #    raise ValueError("Invalid version {!r}".format(version))
    #hlength_type, hlength_length, encoding = hinfo
    if version[0] < 1 or version[0] > 3 or version[1] != 0:
         raise ValueError("Invalid version {!r}".format(version))
    
    cdef unsigned int hlength_length
    if version[0] == 1:
        hlength_type, hlength_length, encoding = '<H', 2, 'latin1'
    elif version[0] == 2:
        hlength_type, hlength_length, encoding = '<I', 4, 'latin1'
    else: # version 3, prevents unbound local checks
        hlength_type, hlength_length, encoding = '<I', 4, 'utf8'
    
    cdef const unsigned char[::1] hlength_str = raw[0:hlength_length]
    cdef unsigned int header_length = unpack(hlength_type, hlength_str)[0]
    
    cdef const unsigned char[::1] raw_header = raw[hlength_length:hlength_length+header_length]
    header = str(raw_header, encoding)
    
    #print(header)
    
    try:
        d = safe_eval(header)
    except SyntaxError as e:
        msg = "Cannot parse header: {!r}\nException: {!r}"
        raise ValueError(msg.format(header, e))
    if not isinstance(d, dict):
        msg = "Header is not a dictionary: {!r}"
        raise ValueError(msg.format(d))
    keys = sorted(d.keys())
    if keys != allowed_header_keys:
        msg = "Header does not contain the correct keys: {!r}"
        raise ValueError(msg.format(keys))
    
    cdef shape = d['shape']
    
    # Sanity-check the values.
    if not isinstance(shape, tuple):
        msg = "shape is not valid: {!r}"
        raise ValueError(msg.format(shape))
    for x in shape:
        if not isinstance(x, int):
            msg = "shape is not valid: {!r}"
            raise ValueError(msg.format(shape))
    
    cdef fortran_order = d['fortran_order']
    
    if not isinstance(fortran_order, bool):
        msg = "fortran_order is not a valid bool: {!r}"
        raise ValueError(msg.format(fortran_order))
    
    try:
        dtype = descr_to_dtype(d['descr'])
    except TypeError:
        msg = "descr is not a valid dtype descriptor: {!r}"
        raise ValueError(msg.format(d['descr']))

    return shape, fortran_order, dtype, raw[hlength_length+header_length:]

cdef int64 = numpy.int64
cdef frombuffer = numpy.frombuffer

cdef numpy.ndarray _array_from_buffer(const unsigned char[::1] raw):
    cdef npy_version version = read_magic(raw)
    _check_version(version)
    raw = raw[MAGIC_LEN:]
    shape, fortran_order, dtype, raw = _read_array_header(raw, version)
    
    count = 1
    for dim in shape:
        count *= dim
    
    if dtype.hasobject:
        raise NotImplementedError("Object arrays cannot be loaded")
    
    array = frombuffer(raw, dtype=dtype, count=count)
    if fortran_order:
        array.shape = shape[::-1]
        array = array.transpose()
    else:
        array.shape = shape
    
    return array

def array_from_buffer(buf):
    return _array_from_buffer(memoryview(buf))

cdef dtype_to_descr(numpy.dtype dtype):
    if (<object>dtype).names is not None:
        # This is a record array. The .descr is fine.  XXX: parts of the
        # record array with an empty name, like padding bytes, still get
        # fiddled with. This needs to be fixed in the C implementation of
        # dtype().
        return dtype.descr
    else:
        return dtype.str

cdef dict header_data_from_array_1_0(numpy.ndarray array):
    d = {'shape': (<object> array).shape}
    if array.flags.c_contiguous:
        d['fortran_order'] = False
    elif array.flags.f_contiguous:
        d['fortran_order'] = True
    else:
        # Totally non-contiguous data. We will have to make it C-contiguous
        # before writing. Note that we need to test for C_CONTIGUOUS first
        # because a 1-D array is both C_CONTIGUOUS and F_CONTIGUOUS.
        d['fortran_order'] = False

    d['descr'] = dtype_to_descr(array.dtype)
    return d

cdef structerror = struct.error

cdef bytes _wrap_header(str header, npy_version version):
    cdef unsigned int hlength_length
    if version[0] == 1:
        fmt, hlength_length, encoding = '<H', 2, 'latin1'
    elif version[0] == 2:
        fmt, hlength_length, encoding = '<I', 4, 'latin1'
    else: # version 3, prevents unbound local checks
        fmt, hlength_length, encoding = '<I', 4, 'utf8'
    
    cdef bytes header_bytes = header.encode(encoding)
    cdef unsigned int hlen = len(header_bytes) + 1
    cdef unsigned int padlen = ARRAY_ALIGN - ((MAGIC_LEN + hlength_length + hlen) % ARRAY_ALIGN)
    try:
        header_prefix = magic(version) + pack(fmt, hlen + padlen)
    except structerror:
        msg = "Header length {} too big for version={}".format(hlen, version)
        raise ValueError(msg)

    # Pad the header with spaces and a final newline such that the magic
    # string, the header-length short and the header are aligned on a
    # ARRAY_ALIGN byte boundary.  This supports memory mapping of dtypes
    # aligned up to ARRAY_ALIGN on systems like Linux where mmap()
    # offset must be page-aligned (i.e. the beginning of the file).
    return header_prefix + header_bytes + b' '*padlen + b'\n'

cdef bytes _wrap_header_guess_version(str header):
    try:
        return _wrap_header(header, (1, 0))
    except ValueError:
        pass

    try:
        return _wrap_header(header, (2, 0))
    except UnicodeEncodeError:
        pass

    return _wrap_header(header, (3, 0))

cdef bytes _array_header_to_bytes(dict d, npy_version version=(-1,-1)):
    cdef list header = ["{"]
    # Need to use repr here, since we eval these when reading
    header.append("'%s': %s, " % ('descr', repr(d['descr'])))
    header.append("'%s': %s, " % ('fortran_order', repr(d['fortran_order'])))
    header.append("'%s': %s, " % ('shape', repr(d['shape'])))
    header.append("}")
    
    cdef str str_header = "".join(header)
    #header = _filter_header(header) # filtering is only for Python 2 / 3 compatibility. We aim only at Python 3
    if version[0] == -1 and version[1] == -1:
        return _wrap_header_guess_version(str_header)
    else:
        return _wrap_header(str_header, version)

cdef bytes _array_to_buffer(numpy.ndarray array, npy_version version=(-1,-1)):
    _check_version(version)
    cdef bytes header = _array_header_to_bytes(header_data_from_array_1_0(array), version)
    return header + array.tobytes(order='A')

def array_to_buffer(array, version=None):
    if version is None:
        return _array_to_buffer(array)
    else:
        return _array_to_buffer(array, version)
