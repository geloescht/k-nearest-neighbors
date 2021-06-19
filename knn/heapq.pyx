cimport cython

__all__ = ['heappush', 'heappop', 'heapify', 'heapreplace', 'merge', 'heappushpop']

cdef minheappush(heap_item[::1] heap, heap_item item, aux_item[::1] aux, aux_item aux_item):
    return heappush(heap, item, aux, aux_item, min_heap)

cdef maxheappush(heap_item[::1] heap, heap_item item, aux_item[::1] aux, aux_item aux_item):
    return heappush(heap, item, aux, aux_item, max_heap)

@cython.boundscheck(False)
@cython.initializedcheck(False)
cdef heappush(heap_item[::1] heap, heap_item item, aux_item[::1] aux, aux_item aux_item, heap_type ht):
    """Push item onto heap, maintaining the heap invariant."""
    heap[-1] = item
    aux[-1]  = aux_item
    _siftdown(heap, aux, 0, heap.shape[0]-1, ht)

cdef minheappop(heap_item[::1] heap, aux_item[::1] aux):
    return heappop(heap, aux, min_heap)

cdef maxheappop(heap_item[::1] heap, aux_item[::1] aux):
    return heappop(heap, aux, max_heap)

@cython.initializedcheck(False)
cdef heappop(heap_item[::1] heap, aux_item[::1] aux, heap_type ht):
    """Pop the smallest item off the heap, maintaining the heap invariant."""
    lastelt = (heap[-1], aux[-1])    # raises appropriate IndexError if heap is empty
    if heap.shape[0] > 1:
        returnitem = (heap[0], aux[0])
        heap[0], aux[0] = lastelt
        _siftup(heap[:-1], aux[:-1], 0, ht)
        return returnitem
    return lastelt

@cython.wraparound(False)
@cython.initializedcheck(False)
cdef heapreplace(heap_item[::1] heap, heap_item item, aux_item[::1] aux, aux_item aux_item, heap_type ht):
    """Pop and return the current smallest value, and add the new item.

    This is more efficient than heappop() followed by heappush(), and can be
    more appropriate when using a fixed-size heap.  Note that the value
    returned may be larger than item!  That constrains reasonable uses of
    this routine unless written as part of a conditional replacement:

        if item > heap[0]:
            item = heapreplace(heap, item)
    """
    returnitem = (heap[0], aux[0])    # raises appropriate IndexError if heap is empty
    heap[0], aux[0] = item, aux_item
    _siftup(heap, aux, 0, ht)
    return returnitem

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
cdef heappushpop(heap_item[::1] heap, heap_item item, aux_item[::1] aux, aux_item aux_item, heap_type ht):
    """Fast version of a heappush followed by a heappop."""
    if heap.shape[0] > 0 and heap[0] < item:
        item, heap[0] = heap[0], item
        aux_item, aux[0]  = aux[0], aux_item
        _siftup(heap, aux, 0, ht)
    return item, aux_item

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
cdef heapify(heap_item[::1] heap, aux_item[::1] aux, heap_type ht):
    """Transform list into a heap, in-place, in O(len(x)) time."""
    cdef long n = heap.shape[0]
    # Transform bottom-up.  The largest index there's any point to looking at
    # is the largest with a child index in-range, so must have 2*i + 1 < n,
    # or i < (n-1)/2.  If n is even = 2*j, this is (2*j-1)/2 = j-1/2 so
    # j-1 is the largest, which is n//2 - 1.  If n is odd = 2*j+1, this is
    # (2*j+1-1)/2 = j so j-1 is the largest, and that's again n//2-1.
    for i in reversed(range(n//2)):
       _siftup(heap, aux, i, ht)

# 'heap' is a heap at all indices >= startpos, except possibly for pos.  pos
# is the index of a leaf with a possibly out-of-order value.  Restore the
# heap invariant.
cdef void _siftdown(heap_item[::1] heap, aux_item[::1] aux, long startpos, long pos, heap_type ht) nogil:
    if heap_type is min_max_heap_enum or heap_type is max_min_heap_enum:
        _mm_siftdown(heap, aux, startpos, pos, ht)
    else:
        _simple_siftdown(heap, aux, startpos, pos, ht)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
cdef inline void _simple_siftdown(heap_item[::1] heap, aux_item[::1] aux, long startpos, long pos, simple_heap_type ht) nogil:
    cdef heap_item newitem = heap[pos]
    cdef aux_item newauxitem = aux[pos]
    # Follow the path to the root, moving parents down until finding a place
    # newitem fits.
    cdef long parentpos
    cdef heap_item parent
    cdef aux_item auxparent
    while pos > startpos:
        parentpos = (pos - 1) >> 1
        parent, auxparent = heap[parentpos], aux[parentpos]
        if (simple_heap_type is min_heap_enum and newitem < parent) or (simple_heap_type is max_heap_enum and newitem > parent):
            heap[pos], aux[pos] = parent, auxparent
            pos = parentpos
            continue
        break
    heap[pos], aux[pos] = newitem, newauxitem

# The child indices of heap index pos are already heaps, and we want to make
# a heap at index pos too.  We do this by bubbling the smaller child of
# pos up (and so on with that child's children, etc) until hitting a leaf,
# then using _siftdown to move the oddball originally at index pos into place.
#
# We *could* break out of the loop as soon as we find a pos where newitem <=
# both its children, but turns out that's not a good idea, and despite that
# many books write the algorithm that way.  During a heap pop, the last array
# element is sifted in, and that tends to be large, so that comparing it
# against values starting from the root usually doesn't pay (= usually doesn't
# get us out of the loop early).  See Knuth, Volume 3, where this is
# explained and quantified in an exercise.
#
# Cutting the # of comparisons is important, since these routines have no
# way to extract "the priority" from an array element, so that intelligence
# is likely to be hiding in custom comparison methods, or in array elements
# storing (priority, record) tuples.  Comparisons are thus potentially
# expensive.
#
# On random arrays of length 1000, making this change cut the number of
# comparisons made by heapify() a little, and those made by exhaustive
# heappop() a lot, in accord with theory.  Here are typical results from 3
# runs (3 just to demonstrate how small the variance is):
#
# Compares needed by heapify     Compares needed by 1000 heappops
# --------------------------     --------------------------------
# 1837 cut to 1663               14996 cut to 8680
# 1855 cut to 1659               14966 cut to 8678
# 1847 cut to 1660               15024 cut to 8703
#
# Building the heap by using heappush() 1000 times instead required
# 2198, 2148, and 2219 compares:  heapify() is more efficient, when
# you can use it.
#
# The total compares needed by list.sort() on the same lists were 8627,
# 8627, and 8632 (this should be compared to the sum of heapify() and
# heappop() compares):  list.sort() is (unsurprisingly!) more efficient
# for sorting.

cdef void _siftup(heap_item[::1] heap, aux_item[::1] aux, long pos, heap_type ht) nogil:
    if heap_type is min_max_heap_enum or heap_type is max_min_heap_enum:
        _mm_siftup(heap, aux, pos, ht)
    else:
        _simple_siftup(heap, aux, pos, ht)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
cdef inline void _simple_siftup(heap_item[::1] heap, aux_item[::1] aux, long pos, simple_heap_type ht) nogil:
    cdef long endpos = heap.shape[0]
    cdef long startpos = pos
    cdef heap_item newitem = heap[pos]
    cdef aux_item newauxitem = aux[pos]
    # Bubble up the smaller child until hitting a leaf.
    cdef long childpos = 2*pos + 1    # leftmost child position
    cdef long rightpos
    while childpos < endpos:
        # Set childpos to index of smaller child.
        rightpos = childpos + 1
        if rightpos < endpos and ((simple_heap_type is min_heap_enum and not heap[childpos] < heap[rightpos]) or
                                  (simple_heap_type is max_heap_enum and not heap[childpos] > heap[rightpos])):
            childpos = rightpos
        # Move the smaller child up.
        heap[pos], aux[pos] = heap[childpos], aux[childpos]
        pos = childpos
        childpos = 2*pos + 1
    # The leaf at pos is empty now.  Put newitem there, and bubble it up
    # to its final resting place (by sifting its parents down).
    heap[pos], aux[pos] = newitem, newauxitem
    _simple_siftdown(heap, aux, startpos, pos, ht)

cdef extern int __builtin_clzl (unsigned long) nogil

cdef inline int _heaplevel(long i) nogil:
    return __builtin_clzl(1ul) ^ __builtin_clzl(i+1)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
cdef inline void _mm_siftdown(heap_item[::1] heap, aux_item[::1] aux, long startpos, long pos, dual_heap_type ht) nogil:
    cdef int level = _heaplevel(pos)
    cdef long parent = max(0, (pos-1) // 2)
    # print("MM siftdown: pos", pos, "startpos", startpos, "level", level, "parent", parent)
    if (dual_heap_type is min_max_heap_enum and level % 2 == 0) or (dual_heap_type is max_min_heap_enum and level % 2 == 1):  # min level
        if pos > 0 and heap[pos] > heap[parent]: # FIXME maybe?
            heap[pos], heap[parent] = heap[parent], heap[pos]
            aux[pos], aux[parent] = aux[parent], aux[pos]
            # print("Min level swap parent")
            _mm_siftdown_impl(heap, aux, startpos, parent, max_min_heap)
        else:
            # print("Min level")
            _mm_siftdown_impl(heap, aux, startpos, pos, min_max_heap)
    else:  # max level
        if pos > 0 and heap[pos] < heap[parent]: # FIXME maybe?
            heap[pos], heap[parent] = heap[parent], heap[pos]
            aux[pos], aux[parent] = aux[parent], aux[pos]
            _mm_siftdown_impl(heap, aux, startpos, parent, min_max_heap)
        else:
            _mm_siftdown_impl(heap, aux, startpos, pos, max_min_heap)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
cdef inline void _mm_siftdown_impl(heap_item[::1] heap, aux_item[::1] aux, long startpos, long pos, dual_heap_type ht) nogil:
    cdef long grandparent
    startpos = max(2, startpos)
    # print("Impl: type", "min" if (dual_heap_type is min_max_heap_enum) else "max")
    while pos > startpos:
        grandparent = (pos-3) // 4
        if (dual_heap_type is min_max_heap_enum and heap[pos] < heap[grandparent]) or (dual_heap_type is max_min_heap_enum and heap[pos] > heap[grandparent]):
            heap[pos], heap[grandparent] = heap[grandparent], heap[pos]
            aux[pos],  aux[grandparent]  = aux[grandparent], aux[pos]
            pos = grandparent
        else:
            return


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
cdef inline void _mm_siftup(heap_item[::1] heap, aux_item[::1] aux, long pos, dual_heap_type ht) nogil:
    cdef int level = _heaplevel(pos)
    # print("MM siftup: pos", pos, "level", level)
    if ((dual_heap_type is min_max_heap_enum and level % 2 == 0) or
        (dual_heap_type is max_min_heap_enum and level % 2 == 1)):  # primary level
        _mm_siftup_impl(heap, aux, pos, min_max_heap)
    else: # secondary level
        _mm_siftup_impl(heap, aux, pos, max_min_heap)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
cdef inline long smallest_grandchild(heap_item[::1] heap, long pos, long size) nogil:
    cdef long a = pos*4+3
    cdef long b, c, d
    b, c, d = a+1, a+2, a+3
    cdef long x = a + (heap[b] < heap[a])
    cdef long y = c + (heap[d] < heap[c])
    return x if heap[x] < heap[y] else y

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
cdef inline long largest_grandchild(heap_item[::1] heap, long pos, long size) nogil:
    cdef long a = pos*4+3
    cdef long b, c, d
    b, c, d = a+1, a+2, a+3
    cdef long x = a + (heap[b] > heap[a])
    cdef long y = c + (heap[d] > heap[c])
    return x if heap[x] > heap[y] else y

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
@cython.cdivision(True)
cdef public inline void _mm_siftup_impl(heap_item[::1] heap, aux_item[::1] aux, long pos, dual_heap_type ht) nogil:
    cdef unsigned char child
    cdef long size = heap.shape[0]
    cdef long j, m
    
    while size > pos * 4 + 7:  # pos has four grandchildren
    #while False:
        m = smallest_grandchild(heap, pos, size) if dual_heap_type is min_max_heap_enum else largest_grandchild(heap, pos, size)
        
        if ((dual_heap_type is min_max_heap_enum and heap[m] < heap[pos]) or
            (dual_heap_type is max_min_heap_enum and heap[m] > heap[pos])):
            heap[pos], heap[m] = heap[m], heap[pos]
            aux[pos], aux[m] = aux[m], aux[pos]
            
            if ((dual_heap_type is min_max_heap_enum and heap[m] > heap[(m-1) // 2]) or
                (dual_heap_type is max_min_heap_enum and heap[m] < heap[(m-1) // 2])):
                heap[m], heap[(m-1)//2] = heap[(m-1)//2], heap[m]
                aux[m], aux[(m-1)//2] = aux[(m-1)//2], aux[m]
            
            pos = m
        else:
            return
    
    while size > pos * 2 + 1:  # pos has children
        m = pos * 2 + 1
        if pos * 2 + 2 < size and ((dual_heap_type is min_max_heap_enum and heap[pos*2+2] < heap[m]) or
                                   (dual_heap_type is max_min_heap_enum and heap[pos*2+2] > heap[m])):
            m = pos * 2 + 2
        
        child = 1
        
        for j in range(pos*4+3, min(pos*4+7, size)):
            if ((dual_heap_type is min_max_heap_enum and heap[j] < heap[m]) or
                (dual_heap_type is max_min_heap_enum and heap[j] > heap[m])):
                m = j
                child = 0
        if child:
        #best_grandkid = smallest_grandkid(heap, pos) if dual_heap_type is min_max_heap_enum else largest_grandkid(heap, pos)
        #if ((dual_heap_type is min_max_heap_enum and heap[best_grandkid] < heap[m]) or
        #    (dual_heap_type is max_min_heap_enum and heap[best_grandkid] > heap[m])):
        #    m = best_grandkid
        #    
            if ((dual_heap_type is min_max_heap_enum and heap[m] < heap[pos]) or
                (dual_heap_type is max_min_heap_enum and heap[m] > heap[pos])):
                heap[pos], heap[m] = heap[m], heap[pos]
                aux[pos], aux[m] = aux[m], aux[pos]
        else:
            if ((dual_heap_type is min_max_heap_enum and heap[m] < heap[pos]) or
                (dual_heap_type is max_min_heap_enum and heap[m] > heap[pos])):
                heap[m], heap[pos] = heap[pos], heap[m]
                aux[m], aux[pos] = aux[pos], aux[m]
                
                if ((dual_heap_type is min_max_heap_enum and heap[m] > heap[(m-1) // 2]) or
                    (dual_heap_type is max_min_heap_enum and heap[m] < heap[(m-1) // 2])):
                    heap[m], heap[(m-1)//2] = heap[(m-1)//2], heap[m]
                    aux[m], aux[(m-1)//2] = aux[(m-1)//2], aux[m]
                
                pos = m
                continue
        break
        


if __name__ == "__main__":

    import doctest # pragma: no cover
    # print(doctest.testmod()) # pragma: no cover

