cimport numpy as np
from knn.distance_metrics cimport _euclidean, _euclidean_pairwise, _manhattan, _manhattan_pairwise, _hamming,\
    _hamming_pairwise
cimport knn.heapq as heapq

ctypedef (double, long) heap_pair
cdef class DynamicHeap:
    cdef public np.ndarray heap
    cdef np.ndarray aux
    cdef public unsigned int size
    cdef public unsigned int max_size
    cdef char is_heap
    
    cdef double[::1] heap_view
    cdef long[::1] aux_view

    cdef inline heap_pair peek(self)
    cdef inline heap_pair pop(self)
    cdef inline void push(self, double heap_item, long aux_item)
    cdef inline heap_pair replace(self, double heap_item, long aux_item)

cdef class Stack:
    cdef public np.ndarray stack
    cdef public np.ndarray aux
    cdef public long stack_ptr # full ascending stack
    
    cdef double[::1] stack_view
    cdef long[::1] aux_view
    
    cdef inline int is_empty(self)
    cdef inline heap_pair peek(self)
    cdef inline heap_pair pop(self)
    cdef inline void push(self, double heap_item, long aux_item)


# Types That Hold Metric Functions
ctypedef double (*metric_func)(double[::1], double[::1])
ctypedef double[:, ::1] (*pairwise_metric_fun)(double[:, ::1], double[:, ::1])

cdef class BallTree:
    # Search Data
    cdef double[:, ::1] data_view
    cdef long[::1] data_inds_view
    cdef np.ndarray data
    cdef np.ndarray data_inds

    # Query Data
    cdef double[:,::1] query_data_view

    # Tree Data
    cdef np.ndarray node_data_inds
    cdef np.ndarray node_radius
    cdef np.ndarray node_is_leaf
    cdef np.ndarray node_center
    cdef public long[:, ::1] node_data_inds_view
    cdef public double[::1] node_radius_view
    cdef public unsigned char[::1] node_is_leaf_view
    cdef public double[:, ::1] node_center_view

    # Tree Shape
    cdef public int leaf_size
    cdef public int node_count
    cdef public int tree_height

    # Query Results
    cdef public np.ndarray heap
    cdef double[:, ::1] heap_view
    cdef public np.ndarray heap_inds
    cdef long[:, ::1] heap_inds_view
    cdef public long nodes_visited

    # Query Metrics
    cdef metric_func metric
    cdef pairwise_metric_fun pair_metric
    
    cdef _build(self, long node_index, long node_data_start, long node_data_end)
    cdef int _hoare_partition(self, long pivot, long low, long high, double[::1] projected_data)
    cdef int _query(self, int query_vect_ind, double dist_to_cent, int curr_node, double[::1] query_data)
    cdef double _heap_peek_head(self, int level)
    cdef int _heap_pop_push(self, int level, double value, int index)

cdef class Cursor:
    cdef BallTree tree
    cdef double[::1] center
    cdef public DynamicHeap node_heap
    cdef public Stack node_stack
    cdef public DynamicHeap elements
    
    cdef heap_pair _next(self)
    cdef inline heap_pair _pop_node(self)
    cdef inline heap_pair _peek_node(self)
    
