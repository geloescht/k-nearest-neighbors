import cython
import numpy as np
cimport numpy as np
import math
from .distance_metrics cimport _euclidean, _euclidean_pairwise, _manhattan, _manhattan_pairwise, _hamming,\
    _hamming_pairwise
cimport knn.heapq as heapq

# comment out this line if you want a debug build
DEF __debug__ = 0

cdef class DynamicHeap:
    def __init__(self, x, y = None):
        if isinstance(x, int):
            self.size = 0
            self.heap = np.full((x, ), np.inf, order='C')
            self.aux = np.zeros((x, ), dtype=int, order='C')
            self.heap_view = memoryview(self.heap)
            self.aux_view = memoryview(self.aux)
            self.is_heap = 0
        else:
            assert(len(x) == len(y))
            self.size = len(x)
            self.heap = np.copy(x)
            self.aux = np.copy(y)
            self.heap_view = memoryview(self.heap)
            self.aux_view = memoryview(self.aux)
            heapq.heapify(self.heap_view, self.aux_view, heapq.min_heap)
            self.is_heap = 1

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    cdef inline heap_pair peek(self):
        if self.size <= 0:
            raise IndexError("Trying to peek on empty DynamicHeap")
        return (self.heap_view[0], self.aux_view[0])

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    cdef inline heap_pair pop(self):
        if self.size <= 0:
            raise IndexError("Trying to pop from empty DynamicHeap")
        if not self.is_heap:
            heapq.heapify(self.heap_view[:self.size], self.aux_view[:self.size], heapq.min_heap)
            self.is_heap = 1
        self.size -= 1
        return heapq.heappop(self.heap_view[:self.size+1], self.aux_view[:self.size+1], heapq.min_heap)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    cdef inline void push(self, double heap_item, long aux_item):
        self.size += 1
        self.max_size = max(self.size, self.max_size)
        if self.size > self.heap_view.shape[0]:
            if __debug__:
                print("Warning: Resizing DynamicHeap from", self.heap.shape[0], "to", self.heap.shape[0]*2)
            self.heap.resize((self.heap.shape[0]*2,), refcheck=False)
            self.aux.resize((self.heap.shape[0]*2,), refcheck=False)
            self.heap_view = memoryview(self.heap)
            self.aux_view = memoryview(self.aux)
        if self.is_heap:
            heapq.heappush(self.heap_view[:self.size], heap_item, self.aux_view[:self.size], aux_item, heapq.min_heap)
        else:
            if heap_item < self.heap_view[0]:
                self.heap_view[self.size-1], self.aux_view[self.size-1] = self.heap_view[0], self.aux_view[0]
                self.heap_view[0], self.aux_view[0] = heap_item, aux_item
            else:
                self.heap_view[self.size-1], self.aux_view[self.size-1] = heap_item, aux_item

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    cdef inline heap_pair replace(self, double heap_item, long aux_item):
        # FIXME deal with self.is_heap == 0
        return heapq.heapreplace(self.heap_view[:self.size], heap_item, self.aux_view[:self.size], aux_item, heapq.min_heap)

cdef class Stack:
    def __init__(self, x, y = None):
        cdef double[::1] stack
        cdef long[::1] aux

        if isinstance(x, int):
            self.stack_ptr = -1
            self.stack = np.full((x, ), np.inf, order='C')
            self.aux = np.zeros((x, ), dtype=int, order='C')
            self.stack_view = memoryview(self.stack)
            self.aux_view = memoryview(self.aux)
        else:
            assert(len(x) == len(y))
            self.stack_ptr = len(x)-1
            self.stack = np.copy(x)
            self.aux = np.copy(y)
            self.stack_view = memoryview(self.stack)
            self.aux_view = memoryview(self.aux)

    cdef inline int is_empty(self):
        return self.stack_ptr < 0

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    cdef inline heap_pair peek(self):
        if self.stack_ptr < 0:
            raise IndexError("Trying to peek on empty Stack")
        return (self.stack_view[self.stack_ptr], self.aux_view[self.stack_ptr])

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    cdef inline heap_pair pop(self):
        if self.stack_ptr < 0:
            raise IndexError("Trying to pop from empty Stack")
        cdef heap_pair ret = (self.stack_view[self.stack_ptr], self.aux_view[self.stack_ptr])
        self.stack_ptr -= 1
        return ret

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    cdef inline void push(self, double heap_item, long aux_item):
        if self.stack_ptr >= self.stack_view.shape[0]:
            raise IndexError("Trying to pop from empty Stack")
        self.stack_ptr += 1
        self.stack_view[self.stack_ptr], self.aux_view[self.stack_ptr] = heap_item, aux_item

cdef class BallTree:
    """ A Ball Tree used for nearest neighbor searches.

    """
    def __init__(self, data, leaf_size, metric="euclidean"):
        """ Creates A BallTree instance.

        This does not construct the tree.

        The search data must be numpy arrays of type np.float64 (float_).

        Args:
            data (ndarray): A 2D array of vectors being searched through.
            leaf_size (int): The number of vectors contained in the leaves of the Ball Tree.
            metric (string): The distance metric used by the tree.
        """

        # Data
        self.data = np.asarray(data, dtype=np.float, order='C')
        self.data_view = memoryview(self.data)
        self.data_inds = np.arange(data.shape[0], dtype=np.int)
        self.data_inds_view = memoryview(self.data_inds)

        # Tree Shape
        self.leaf_size = leaf_size
        leaf_count = self.data.shape[0] / leaf_size
        self.tree_height = math.ceil(np.log2(leaf_count)) + 1
        self.node_count = int(2 ** self.tree_height) - 1

        # Node Data
        self.node_data_inds = np.zeros((self.node_count, 2), dtype=np.int, order='C')
        self.node_radius = np.zeros(self.node_count, order='C')
        self.node_is_leaf = np.zeros(self.node_count, dtype=np.uint8, order='C')
        self.node_center = np.zeros((self.node_count, data.shape[1]), order='C')
        self.node_data_inds_view = memoryview(self.node_data_inds)
        self.node_radius_view = memoryview(self.node_radius)
        self.node_is_leaf_view = memoryview(self.node_is_leaf)
        self.node_center_view = memoryview(self.node_center)

        # Metric Selection
        if metric == "manhattan":
            self.metric = _manhattan
            self.pair_metric = _manhattan_pairwise
        elif metric == "hamming":
            self.metric = _hamming
            self.pair_metric = _hamming_pairwise
        else:
            self.metric = _euclidean
            self.pair_metric = _euclidean_pairwise

    def build_tree(self):
        """ Python visible method to build the Ball Tree.

        """
        self._build(0, 0, self.data.shape[0]-1)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    cdef _build(self, long node_index, long node_data_start, long node_data_end):
        """ Cython(Not Visible From Python) method that recursively builds the Ball Tree.
        
        Args:
            node_index: The level of the current node currently being worked on.
            node_data_start: The begging index of the current nodes data.
            node_data_end: The ending index of the current nodes data.

        Returns:
            (None) marks the end of a recursive path, meaning a leaf was created

        """

        ##########################
        # Current Node Is A Leaf #
        ##########################
        if (node_data_end-node_data_start+1) <= self.leaf_size:

            self.node_center[node_index] = np.mean(self.data[self.data_inds[node_data_start:node_data_end+1]], axis=0)

            self.node_radius[node_index] = np.max(self.pair_metric(self.data[self.data_inds[node_data_start:node_data_end+1]],
                                                            self.node_center[node_index,  :][np.newaxis, :]))

            self.node_data_inds_view[node_index, 0] = node_data_start
            self.node_data_inds_view[node_index, 1] = node_data_end

            self.node_is_leaf_view[node_index] = True
            return None

        #################################
        # Current Node Is Internal Node #
        #################################

        # Select Random Point -  x0
        rand_index = np.random.choice(node_data_end-node_data_start+1, 1, replace=False)
        rand_point = self.data[self.data_inds[rand_index], :]

        # Find Point Farthest Away From x0 - x1
        distances = self.pair_metric(self.data[self.data_inds[node_data_start:node_data_end+1]], rand_point)
        ind_of_max_dist = np.argmax(distances)
        max_vector_1 = self.data[ind_of_max_dist]

        # Find Point Farthest Away From x1 - x2
        distances = self.pair_metric(self.data[self.data_inds[node_data_start:node_data_end+1]], max_vector_1[np.newaxis, :])
        ind_of_max_dist = np.argmax(distances)
        max_vector_2 = self.data[ind_of_max_dist]

        # Project Data On Vector Between x1 and x2
        proj_data = np.dot(self.data[self.data_inds[node_data_start:node_data_end+1]], max_vector_1-max_vector_2)
        cdef double[::1] proj_data_view = memoryview(proj_data)
        cdef long proj_data_size = proj_data.size//2

        # Find Median Of Projected Data
        cdef long median = np.partition(proj_data, proj_data_size)[proj_data_size]

        # Split Data Around Median Using Hoare Partitioning
        low = node_data_start
        high = node_data_end
        pivot = median
        self._hoare_partition(pivot, low, high, proj_data_view)

        # Create Balls
        center = np.mean(self.data[self.data_inds[node_data_start:node_data_end+1]], axis=0)
        radius = np.max(self.pair_metric(self.data[self.data_inds[node_data_start:node_data_end+1]], center[np.newaxis, :]))

        self.node_data_inds_view[node_index, 0] = node_data_start
        self.node_data_inds_view[node_index, 1] = node_data_end

        self.node_radius[node_index] = radius
        self.node_center[node_index] = center

        self.node_is_leaf[node_index] = False

        # Build Children Balls
        cdef long left_index = 2 * node_index + 1
        cdef long right_index = left_index + 1
        self._build(left_index, node_data_start,  node_data_start+proj_data_size-1 )
        self._build(right_index, node_data_start+proj_data_size,   node_data_end)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    cdef int _hoare_partition(self, long pivot, long low, long high, double[::1] projected_data):
        """ Cython(Not Visible From Python) method that performs Hoare partitioning.
        
        All Numbers greater than the pivot is to the left of the pivot, and everything greater to the right.
        
        This method takes the current node's data after being projected and uses this data to perform partitioning. It
        will also update the current balls associated data's indices.
        
        Args:
            pivot: The value that is used to partition the array.
            low: The beginning index of the current balls data.
            high: The ending index of the current balls data
            projected_data: The nodes data after being projected.

        Returns:
            The data index were the partitioning stopped.

        """

        cdef long i_data_inds = low - 1
        cdef long j_data_inds = high + 1
        cdef long i_projected = -1
        cdef long j_projected = projected_data.shape[0]

        while True:

            # Scan From Left To Find Value Greater Than Pivot
            condition = True
            while condition:
                i_data_inds += 1
                i_projected += 1
                condition = projected_data[i_projected] < pivot

            # Scan From Right To Find Value Less Than Pivot
            condition = True
            while condition:
                j_data_inds -= 1
                j_projected -= 1
                condition = projected_data[j_projected] > pivot

            # Time To End Algorithm
            if (i_data_inds >= j_data_inds):
                return j_data_inds

            # Swap Values
            projected_data[i_projected], projected_data[j_projected] = projected_data[j_projected], projected_data[i_projected]
            self.data_inds_view[i_data_inds], self.data_inds_view[j_data_inds] = self.data_inds_view[j_data_inds], self.data_inds_view[i_data_inds]


    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    def query(self, query_data, k):
        """ Python visible method to query the Ball Tree.

        This will assign self.heap_inds with the indices of the NN for each query vector. Each row off object.heap_inds
        represents the NN of one training vector.

        The NNs are not sorted by distance. If you need distance information access object.heap.

        Args:
            query_data (ndarray): A 2D array of query vectors.
            k (int): The number of NNs to find.

        """

        cdef int i
        cdef double[::1] query_vector, initial_center
        cdef int numb_query_vectors = query_data.shape[0]
        cdef double dist

        self.heap = np.full((query_data.shape[0], k), np.inf, order='C')
        self.heap_view = memoryview(self.heap)
        self.heap_inds = np.zeros((query_data.shape[0], k), dtype=np.int, order='C')
        self.heap_inds_view = memoryview(self.heap_inds)
        self.query_data_view = memoryview(query_data)

        initial_center = self.node_center_view[0]
        for i in range(0, numb_query_vectors):
            query_vector = self.query_data_view[i]
            dist = self.metric(initial_center, query_vector)
            self._query(i, dist, 0, query_vector)


    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    cdef int _query(self, int query_vect_ind, double dist_to_cent, int curr_node, double[::1] query_data):
        """ Cython(Not Visible From Python) method that searches the ball tree.
        
        This method updates the heap data structures created in query().
        
        Args:
            query_vect_ind (int): The index of the query vector.
            dist_to_cent (double): The distance the query vector is from the center of the current ball.
            curr_node (int): The index of the current ball/node.
            query_data (ndarray): The current query vector.

        Returns:
            (int) Zero to represent the end of the recursion.
        """

        cdef int i, child1, child2, lower_index, upper_index, curr_index
        cdef double child1_dist, child2_dist, dist
        cdef double[::1] curr_vect, child1_center, child2_center

        # Prune The Ball
        if dist_to_cent - self.node_radius_view[curr_node] > self._heap_peek_head(query_vect_ind):
            return 0

        # Currently A Leaf Node
        if self.node_is_leaf_view[curr_node]:
            lower_index = self.node_data_inds_view[curr_node][0]
            upper_index = self.node_data_inds_view[curr_node][1] + 1
            for i in range(lower_index, upper_index):
                curr_index = self.data_inds_view[i]
                curr_vect = self.data_view[curr_index]
                dist = self.metric(curr_vect, query_data)
                if dist < self._heap_peek_head(query_vect_ind):
                    self._heap_pop_push(query_vect_ind, dist, self.data_inds_view[i])

        # Not A Leaf So Explore Children
        else:
            child1 = 2 * curr_node + 1
            child2 = child1 + 1

            child1_center = self.node_center_view[child1]
            child2_center = self.node_center_view[child2]

            child1_dist = self.metric(child1_center, query_data)
            child2_dist = self.metric(child2_center, query_data)

            if child1_dist <= child2_dist:
                self._query(query_vect_ind, child1_dist, child1, query_data)
                self._query(query_vect_ind, child2_dist, child2, query_data)
            else:
                self._query(query_vect_ind, child2_dist, child2, query_data)
                self._query(query_vect_ind, child1_dist, child1, query_data)

        return 0


    ####################
    # Max Heap Methods #
    ####################

    @cython.initializedcheck(False)
    @cython.boundscheck(False)
    cdef inline double _heap_peek_head(self, int level):
        """ Cython(Not Visible From Python) method that gets the top element in the max heap for the specified query
            vector.
            
        This just retrieves the value it does not remove the head.
        
        Args:
            level (int): The index of the current query vector.

        Returns:
            (double) The distance NN of the current query vector that is farthest away.
    
        """
        return self.heap_view[level, 0]

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    # Pop Current Top Element In Heap And Push New Value Into Heap
    cdef inline int _heap_pop_push(self, int level, double value, int index):
        heapq.heapreplace(self.heap_view[level], value, self.heap_inds_view[level], index, heapq.max_heap)


cdef class Cursor:
    def __init__(self, tree, query):
        self.tree = tree
        self.center = query
        self.node_heap = DynamicHeap((tree.node_count+1) // 4)
        self.node_stack = Stack(tree.tree_height)
        dist = self.tree.metric(self.center, tree.node_center_view[0]) - tree.node_radius_view[0]
        self.node_heap.push(dist, 0)
        self.elements = DynamicHeap((tree.node_count+1) // 4 * tree.leaf_size)

    def next(self):
        ret = self._next()
        if ret[0] == -1:
            raise StopIteration
        return ret

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    cdef inline heap_pair _peek_node(self):
        if not self.node_stack.is_empty():
            return self.node_stack.peek()
        else:
            return self.node_heap.peek()

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    cdef inline heap_pair _pop_node(self):
        if not self.node_stack.is_empty():
            return self.node_stack.pop()
        else:
            return self.node_heap.pop()

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    cdef heap_pair _next(self):
        cdef long next_node, left_child, right_child, index, lower_index, upper_index
        cdef double next_dist, left_dist, right_dist, dist, radius
        cdef double[::1] left_center, right_center
        cdef heap_pair ret

        if self.node_heap.size <= 0 and self.node_stack.is_empty():
            if self.elements.size > 0:
                return self.elements.pop()
            else:
                return (-1, -1)

        radius = self.elements.peek()[0] if self.elements.size > 0 else np.inf

        while self.node_heap.size > 0 or not self.node_stack.is_empty():
            if self.node_stack.is_empty():
                next_dist, next_node = self.node_heap.peek()
                # the closest node does still not overlap the closest element
                if next_dist > radius:
                    return self.elements.pop()

            self._pop_node()

            if not self.tree.node_is_leaf_view[next_node]:
                # inner node; push its children
                left_child = 2 * next_node + 1
                right_child = left_child + 1
                assert(right_child < self.tree.node_count)

                left_center = self.tree.node_center_view[left_child]
                left_center_dist = self.tree.metric(self.center, left_center)
                left_dist = max(next_dist, left_center_dist - self.tree.node_radius_view[left_child])

                right_center = self.tree.node_center_view[right_child]
                right_center_dist = self.tree.metric(self.center, right_center)
                right_dist = max(next_dist, right_center_dist - self.tree.node_radius_view[right_child])

                # swap so that left_child is the one that is closer
                if right_center_dist < left_center_dist:
                     left_center_dist, right_center_dist = right_center_dist, left_center_dist
                     left_dist, right_dist = right_dist, left_dist
                     left_child, right_child = right_child, left_child

                # push farther child first, closer second
                if right_dist < radius:
                    self.node_stack.push(right_dist, right_child)
                else:
                    self.node_heap.push(right_dist, right_child)

                if left_dist < radius:
                    self.node_stack.push(left_dist, left_child)
                else:
                    self.node_heap.push(left_dist, left_child)

            else: # leaf
                lower_index = self.tree.node_data_inds_view[next_node][0]
                upper_index = self.tree.node_data_inds_view[next_node][1] + 1
                for i in range(lower_index, upper_index):
                    index = self.tree.data_inds_view[i]
                    dist = self.tree.metric(self.center, self.tree.data_view[index])
                    self.elements.push(dist, index)
                radius = self.elements.peek()[0]

            # weed the stack
            while not self.node_stack.is_empty():
                next_dist, next_node = self.node_stack.peek()
                if next_dist < radius:
                    break
                else:
                    self.node_stack.pop()
                    self.node_heap.push(next_dist, next_node)

        if self.elements.size > 0:
            return self.elements.pop()
        else:
            return (-1, -1)
