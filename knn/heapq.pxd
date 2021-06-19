cimport cython

ctypedef fused heap_item:
    cython.double

ctypedef fused aux_item:
    cython.long

ctypedef fused pair:
    (cython.double, cython.long)

ctypedef enum min_heap_enum: min_heap
ctypedef enum max_heap_enum: max_heap
ctypedef enum min_max_heap_enum: min_max_heap
ctypedef enum max_min_heap_enum: max_min_heap

ctypedef fused heap_type:
    min_heap_enum
    max_heap_enum
    min_max_heap_enum
    max_min_heap_enum

ctypedef fused simple_heap_type:
    min_heap_enum
    max_heap_enum

ctypedef fused dual_heap_type:
    min_max_heap_enum
    max_min_heap_enum

cdef heappush(heap_item[::1] heap, heap_item item, aux_item[::1] aux, aux_item aux_item, heap_type ht)
cdef heappop(heap_item[::1] heap, aux_item[::1] aux, heap_type ht)
cdef heapreplace(heap_item[::1] heap, heap_item item, aux_item[::1] aux, aux_item aux_item, heap_type ht)
cdef heappushpop(heap_item[::1] heap, heap_item item, aux_item[::1] aux, aux_item aux_item, heap_type ht)
cdef heapify(heap_item[::1] heap, aux_item[::1] aux, heap_type ht)
