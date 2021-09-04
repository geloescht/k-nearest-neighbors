cimport cython
import numpy as np
cimport numpy as np
from libc.stdio cimport printf
from libc.string cimport strcpy, strlen
from io import BytesIO
from .ball_tree cimport BallTree, Cursor
from .ball_tree import BallTree, Cursor
from cpython.object cimport PyObject
from cpython.ref cimport Py_INCREF, Py_DECREF

# comment out this line if you want a debug build
DEF __debug__ = 0

cdef extern from "sqlite3ext.h":
    ctypedef struct sqlite3
    ctypedef struct sqlite3_context
    ctypedef struct sqlite3_value
    ctypedef struct sqlite3_stmt
    ctypedef unsigned long long sqlite3_uint64
    ctypedef signed long long sqlite3_int64
    
    ctypedef struct sqlite3_index_constraint:
         int iColumn              # Column constrained.  -1 for ROWID
         unsigned char op         # Constraint operator
         unsigned char usable     # True if this constraint is usable
         int iTermOffset          # Used internally - xBestIndex should ignore
    
    ctypedef struct sqlite3_index_orderby:
        int iColumn              # Column number
        unsigned char desc       # True for DESC.  False for ASC.
    
    ctypedef struct sqlite3_index_constraint_usage:
        int argvIndex           # if >0, constraint is part of argv to xFilter
        unsigned char omit      # Do not code a test for this constraint
    
    ctypedef struct sqlite3_index_info:
        # Inputs
        int nConstraint           # Number of entries in aConstraint
        sqlite3_index_constraint *aConstraint            # Table of WHERE clause iinfo[0].aConstraint
        int nOrderBy              # Number of terms in the ORDER BY clause
        sqlite3_index_orderby *aOrderBy               # The ORDER BY clause
        # Outputs
        sqlite3_index_constraint_usage *aConstraintUsage
        int idxNum                # Number used to identify the index
        char *idxStr              # String, possibly obtained from sqlite3_malloc
        int needToFreeIdxStr      # Free idxStr using sqlite3_free() if true
        int orderByConsumed       # True if output is already ordered
        double estimatedCost           # Estimated cost of using this index
        # Fields below are only available in SQLite 3.8.2 and later
        sqlite3_int64 estimatedRows    # Estimated number of rows returned
        # Fields below are only available in SQLite 3.9.0 and later
        int idxFlags              # Mask of SQLITE_INDEX_SCAN_* flags
        # Fields below are only available in SQLite 3.10.0 and later
        sqlite3_uint64 colUsed    # Input: Mask of columns used by statement
    
    ctypedef struct sqlite3_vtab:
        const sqlite3_module *pModule
        int nRef
        char *zErrMsg
    
    ctypedef struct sqlite3_vtab_cursor:
        sqlite3_vtab *pVtab
    
    ctypedef struct sqlite3_module:
        int iVersion
        int (*xCreate)(sqlite3*, void *pAux,
                   int argc, const char *const*argv,
                   sqlite3_vtab **ppVTab, char**)
        int (*xConnect)(sqlite3*, void *pAux,
                   int argc, const char *const*argv,
                   sqlite3_vtab **ppVTab, char**)
        int (*xBestIndex)(sqlite3_vtab *pVTab, sqlite3_index_info*)
        int (*xDisconnect)(sqlite3_vtab *pVTab)
        int (*xDestroy)(sqlite3_vtab *pVTab)
        int (*xOpen)(sqlite3_vtab *pVTab, sqlite3_vtab_cursor **ppCursor)
        int (*xClose)(sqlite3_vtab_cursor*)
        int (*xFilter)(sqlite3_vtab_cursor*, int idxNum, const char *idxStr,
                    int argc, sqlite3_value **argv)
        int (*xNext)(sqlite3_vtab_cursor*)
        int (*xEof)(sqlite3_vtab_cursor*)
        int (*xColumn)(sqlite3_vtab_cursor*, sqlite3_context*, int)
        int (*xRowid)(sqlite3_vtab_cursor*, sqlite3_int64 *pRowid)
        int (*xUpdate)(sqlite3_vtab *, int, sqlite3_value **, sqlite3_int64 *)
        int (*xBegin)(sqlite3_vtab *pVTab)
        int (*xSync)(sqlite3_vtab *pVTab)
        int (*xCommit)(sqlite3_vtab *pVTab)
        int (*xRollback)(sqlite3_vtab *pVTab)
        int (*xFindFunction)(sqlite3_vtab *pVtab, int nArg, const char *zName,
                           void (**pxFunc)(sqlite3_context*,int,sqlite3_value**),
                           void **ppArg)
        int (*xRename)(sqlite3_vtab *pVtab, const char *zNew)
        int (*xSavepoint)(sqlite3_vtab *pVTab, int)
        int (*xRelease)(sqlite3_vtab *pVTab, int)
        int (*xRollbackTo)(sqlite3_vtab *pVTab, int)
    
    ctypedef struct sqlite3_api_routines:
        int  (*create_function)(sqlite3*,const char*,int,int,void*,
                          void (*xFunc)(sqlite3_context*,int,sqlite3_value**),
                          void (*xStep)(sqlite3_context*,int,sqlite3_value**),
                          void (*xFinal)(sqlite3_context*)) nogil
        int (*create_module)(sqlite3 *, const char *, const sqlite3_module *, void *) nogil
        void (*result_int)(sqlite3_context*,int) nogil
        void (*result_int64)(sqlite3_context*, sqlite3_int64) nogil
        void (*result_double)(sqlite3_context*,double) nogil
        void (*result_blob)(sqlite3_context*, const void*, int, void(*)(void*)) nogil
        int  (*value_type)(sqlite3_value*) nogil
        int  (*value_bytes)(sqlite3_value*) nogil
        sqlite3_int64 (*value_int64)(sqlite3_value*) nogil
        sqlite3_int64 (*column_int64)(sqlite3_stmt*, int) nogil
        int (*column_bytes)(sqlite3_stmt*, int) nogil
        const void* (*column_blob)(sqlite3_stmt*, int) nogil
        const void* (*value_blob)(sqlite3_value*) nogil
        sqlite3_value* (*column_value)(sqlite3_stmt*, int) nogil
        double (*value_double)(sqlite3_value*) nogil
        int (*declare_vtab)(sqlite3 *db, const char *zCreateTable) nogil
        void * (*malloc)(int) nogil
        void (*free)(void*) nogil
        int (*prepare_v2)(sqlite3 *, const char *zSql, int nByte, sqlite3_stmt **ppStmt, const char **pzTail) nogil
        int (*step)(sqlite3_stmt*) nogil
        int (*finalize)(sqlite3_stmt *pStmt) nogil
        const char* (*errmsg)(sqlite3*) nogil
    
    cdef int SQLITE_UTF8
    cdef int SQLITE_INNOCUOUS
    cdef int SQLITE_DETERMINISTIC
    cdef int SQLITE_NULL
    cdef int SQLITE_BLOB
    cdef int SQLITE_OK
    cdef int SQLITE_ERROR
    cdef int SQLITE_NOMEM
    cdef int SQLITE_CONSTRAINT
    cdef int SQLITE_CONFIG_LOG
    cdef int SQLITE_ROW
    cdef int SQLITE_TRANSIENT
    
    cdef unsigned char SQLITE_INDEX_CONSTRAINT_EQ
    cdef unsigned char SQLITE_INDEX_CONSTRAINT_GT
    cdef unsigned char SQLITE_INDEX_CONSTRAINT_LE
    cdef unsigned char SQLITE_INDEX_CONSTRAINT_LT
    cdef unsigned char SQLITE_INDEX_CONSTRAINT_GE
    cdef unsigned char SQLITE_INDEX_CONSTRAINT_MATCH
    cdef unsigned char SQLITE_INDEX_CONSTRAINT_LIKE
    cdef unsigned char SQLITE_INDEX_CONSTRAINT_GLOB
    cdef unsigned char SQLITE_INDEX_CONSTRAINT_REGEXP
    cdef unsigned char SQLITE_INDEX_CONSTRAINT_NE
    cdef unsigned char SQLITE_INDEX_CONSTRAINT_ISNOT
    cdef unsigned char SQLITE_INDEX_CONSTRAINT_ISNOTNULL
    cdef unsigned char SQLITE_INDEX_CONSTRAINT_ISNULL
    cdef unsigned char SQLITE_INDEX_CONSTRAINT_IS
    cdef unsigned char SQLITE_INDEX_CONSTRAINT_FUNCTION
    cdef unsigned char SQLITE_INDEX_SCAN_UNIQUE

cdef const sqlite3_api_routines *sqlite

cdef class BallTreeAttrs:
    cdef public np.ndarray rowids
    cdef public sqlite3_int64[::1] rowids_view
    cdef public np.ndarray data
    cdef public double[:, ::1] data_view
    cdef public BallTree tree
    cdef public str table_field

ctypedef struct balltree_vtab:
    sqlite3_vtab sqlite_attrs
    PyObject* balltree_attrs # custom attributes must be PyObject*, not object, so they can safely pass through pure C code

cdef class CursorAttrs:
    cdef public np.ndarray query
    cdef public Cursor cursor

ctypedef struct balltree_vtab_cursor:
    sqlite3_vtab_cursor sqlite_attrs
    PyObject* balltree_attrs # custom attributes must be PyObject*, not object, so they can safely pass through pure C code
    int strategy # see definition of strategies below
    double current_distance
    sqlite3_int64 current_id
    sqlite3_int64 current_i
    sqlite3_int64 min_id
    sqlite3_int64 max_id
    double max_distance
    unsigned char flags # see definition of flags below

cdef int id_field = 0
cdef int distance_field = 1
cdef int rank_field = 2
cdef int query_field = 3

cdef int nn_strategy = 0        # nearest neighbor
cdef int id_strategy = 1        # bisect and scan IDs order

cdef unsigned char scan_backward_flag =  1 # scan in descending order if set
cdef unsigned char inclusive_min_flag =  2 # search for or scan until a minimum ID if set
cdef unsigned char exclusive_min_flag =  4 # same, but exclude min_id
cdef unsigned char inclusive_max_flag =  8 # search for or scan until a maximum ID if set
cdef unsigned char exclusive_max_flag = 16 # same, but exclude max_id
cdef unsigned char single_row_flag    = 32 # return only a single row (used with equality constraint)
cdef unsigned char min_flags          = inclusive_min_flag | exclusive_min_flag
cdef unsigned char max_flags          = inclusive_max_flag | exclusive_max_flag
cdef unsigned char min_or_max_flags   = min_flags | max_flags

cdef np.ndarray load_array_from_blob(sqlite3_value* value):
    if sqlite.value_type(value) != SQLITE_BLOB:
        return None
    cdef int size = sqlite.value_bytes(value)
    cdef const unsigned char[::1] raw = <const unsigned char[:size:1]>sqlite.value_blob(value)
    return np.load(BytesIO(raw))

cdef bytes encode_array_as_buffer(np.ndarray array):
    f = BytesIO()
    np.save(f, array)
    return f.getvalue()

cdef double distance_from_blobs(
      sqlite3_value* blob_a,
      sqlite3_value* blob_b
    ):
    a = load_array_from_blob(blob_a)
    b = load_array_from_blob(blob_b)
    if a is None or b is None:
        return -1
    
    return float(np.linalg.norm((a - b).flatten()))

cdef int balltree_vtab_create(sqlite3* db, void* pAux, int argc, const char* const* argv, sqlite3_vtab **ppVTab, char** pzErr) with gil:
    if argc != 4:
        return SQLITE_ERROR
    
    cdef balltree_vtab* new_vtab
    cdef int rc = SQLITE_OK
    cdef BallTreeAttrs attrs = BallTreeAttrs()
    
    attrs.table_field = argv[3].decode('UTF-8')
    
    try:
        table, field = attrs.table_field.strip('"').split('.')
    except ValueError:
        # TODO error message
        return SQLITE_ERROR
    
    # count all rows with field in table
    count_sql = f'SELECT COUNT("{field}") FROM "{table}"'
    
    cdef sqlite3_stmt *count_statement
    rc = sqlite.prepare_v2(db, count_sql.encode('UTF-8'), -1, &count_statement, NULL)
    if rc != SQLITE_OK:
        msg = sqlite.errmsg(db);
        print("Prepare error", rc, msg.decode('UTF-8'))
        return rc
    
    rc = sqlite.step(count_statement)
    if rc != SQLITE_ROW:
        return rc
    
    count = sqlite.column_int64(count_statement, 0)
    rc = sqlite.finalize(count_statement)
    if rc != SQLITE_OK:
        return rc
    
    # read in all rows
    data = None
    sql = f'SELECT rowid, "{field}" FROM "{table}" WHERE "{field}" IS NOT NULL'
    i = 0
    
    cdef sqlite3_stmt *statement
    rc = sqlite.prepare_v2(db, sql.encode('UTF-8'), -1, &statement, NULL)
    if rc != SQLITE_OK:
        msg = sqlite.errmsg(db);
        print("Prepare error", rc, msg.decode('UTF-8'))
        return rc
    
    while sqlite.step(statement) == SQLITE_ROW:
        array = load_array_from_blob(sqlite.column_value(statement, 1))
        id = sqlite.column_int64(statement, 0)
        if data is None:
            data = np.zeros_like(array, shape=(count, array.flatten().shape[0]), order='C')
            attrs.data = data
            attrs.data_view = memoryview(data)
            attrs.rowids = np.zeros(shape=(count, ), dtype=np.int64, order='C')
            attrs.rowids_view = memoryview(attrs.rowids)
        data[i] = array.flatten()
        attrs.rowids[i] = id
        i += 1
    
    # data is filled, now build the tree
    attrs.tree = BallTree(data, 3)
    attrs.tree.build_tree()
    
    # declare the vtab's schema
    rc = sqlite.declare_vtab(db, "CREATE TABLE ignored (id INTEGER, distance REAL, rank INTEGER, query HIDDEN BLOB)")
    if rc != SQLITE_OK:
        return rc
    
    # allocate the vtab structure
    new_vtab = <balltree_vtab*>sqlite.malloc(sizeof(balltree_vtab))
    if new_vtab == NULL:
        return SQLITE_NOMEM
    
    # add our custom attributes to the vtab structure
    new_vtab.balltree_attrs = <PyObject*>attrs
    # increment the reference count for our custom attributes so they persist through C code until the vtab is destroyed
    Py_INCREF(attrs)
    # pass it to sqlite
    ppVTab[0] = <sqlite3_vtab*>new_vtab
    
    if __debug__:
        print("Done!")
    
    return SQLITE_OK

cdef int balltree_vtab_connect(sqlite3* db, void *pAux, int argc, const char *const*argv, sqlite3_vtab **ppVTab, char** pzErr) with gil:
    if __debug__:
        print("Connecting balltree")
    return balltree_vtab_create(db, pAux, argc, argv, ppVTab, pzErr)

cdef int balltree_vtab_disconnect(sqlite3_vtab *pVTab) with gil:
    if __debug__:
        print("Disconnecting balltree")
    cdef balltree_vtab* vtab = <balltree_vtab*>pVTab
    # decrement the reference count to our custom attributes so they get freed when no more *ython code references it
    Py_DECREF(<object>vtab.balltree_attrs)
    sqlite.free(vtab)
    return SQLITE_OK

cdef int balltree_vtab_destroy(sqlite3_vtab *pVTab) with gil:
    return balltree_vtab_disconnect(pVTab)

cdef int balltree_vtab_best_index(sqlite3_vtab *pVTab, sqlite3_index_info* iinfo) with gil:
    if __debug__:
        print("Best index called")
    cdef balltree_vtab* vtab = <balltree_vtab*>pVTab
    cdef BallTreeAttrs attrs = <BallTreeAttrs>vtab.balltree_attrs
    cdef long max_rows = attrs.rowids.shape[0]
    cdef double[2] cost_per_row = [200, 100]
    cdef double[2] fixed_cost = [50000, 10000]
    cdef long[2] rows = [max_rows, max_rows]
    cdef char flags = 0
    cdef int min_arg, max_arg
    iinfo[0].estimatedCost = 100000
    
    # plan for this function:
    # - estimate how many rows would be returned
    #   - equality cuts rows by max_rows (to 1 if no other constraints apply)
    #   - inequalities cut rows by by half
    # - estimate cost for nearest neighbor and bisect and scan IDs
    #   - each usable constraint modifies estimated cost for each method
    #   - ordering by ascending distance cuts cost for nearest neighbor
    #   - ordering by ID cuts cost for bisect and scan
    # - choose better method and return values accordingly
    #   - return cost and estimated rows
    #   - set idxNum to better method
    #   - for bisect and scan
    #     - set filterArgs to minimum and maximum or equality
    #     - set idxStr to information whether minimum and maximum are inclusive or exclusive
    
    cdef int queryFound = -1
    cdef int filterArgc = 2
    cdef int column
    cdef unsigned char operator
    
    for i_constraint in range(iinfo[0].nConstraint):
        if __debug__:
            print("Constraint", i_constraint,
                  ": column", iinfo[0].aConstraint[i_constraint].iColumn,
                  " operator", iinfo[0].aConstraint[i_constraint].op,
                  " usable", iinfo[0].aConstraint[i_constraint].usable)
        
        if not iinfo[0].aConstraint[i_constraint].usable:
            continue
        
        column, operator = iinfo[0].aConstraint[i_constraint].iColumn, iinfo[0].aConstraint[i_constraint].op
        
        if column == id_field:
            # constraints on id only affect the bisect and scan strategy
            # nearest neighbor still has to look at every row unless there
            # are other constraints
            if operator == SQLITE_INDEX_CONSTRAINT_EQ:
                rows[id_strategy] = 1
            else:
                rows[id_strategy] //= 2
            # assume we have to bisect if there is a constraint on id
            # that might not actually be true if we can scan forward until a max
            # or backward until a min. But this is just for the query planner
            # so let's keep it unless we notice a suboptimal strategy being chosen.
            fixed_cost[id_strategy] = 50000
             
        elif column == distance_field:
            # constraints on distance only affect nearest neighbor
            if operator == SQLITE_INDEX_CONSTRAINT_LT or operator == SQLITE_INDEX_CONSTRAINT_LE:
                # assume that only a minority of rows will be found if distance is bounded
                rows[nn_strategy] //= 4

        elif column == query_field:
            if operator == SQLITE_INDEX_CONSTRAINT_EQ:
                iinfo[0].aConstraintUsage[i_constraint].argvIndex = 1 # query is always the first argument
                iinfo[0].aConstraintUsage[i_constraint].omit = 1
                queryFound = i_constraint
            else:
                return SQLITE_ERROR # can't use query in any other way than equality
        else:
            return SQLITE_ERROR
    
    if queryFound == -1:
        # query is a needed parameter, so don't use a plan without it under any circumstances
        return SQLITE_CONSTRAINT
    
    for i_order in range(iinfo[0].nOrderBy):
        if __debug__:
            print("Order", i_order, ": column", iinfo[0].aOrderBy[i_order].iColumn, " desc", iinfo[0].aOrderBy[i_order].desc)
        
        if iinfo[0].aOrderBy[i_order].iColumn == query_field:
            return SQLITE_ERROR # query is strictly an input column, ordering by query doesn't mean anything
        
        if i_order == 0:
            if iinfo[0].aOrderBy[i_order].iColumn == distance_field and not iinfo[0].aOrderBy[i_order].desc:
                rows[nn_strategy] //= 2 # assume not all rows will be consumed if we order by distance ascending
                cost_per_row[nn_strategy] /= 2
            elif iinfo[0].aOrderBy[i_order].iColumn == id_field:
                cost_per_row[id_strategy] /= 2
    
    nn_cost = fixed_cost[nn_strategy] + rows[nn_strategy]*cost_per_row[nn_strategy]
    id_cost = fixed_cost[id_strategy] + rows[id_strategy]*cost_per_row[id_strategy]
    
    if __debug__:
        print(f"NN cost {nn_cost} ({fixed_cost[nn_strategy]} fixed + {rows[nn_strategy]} rows * {cost_per_row[nn_strategy]} per row)")
        print(f"ID cost {id_cost} ({fixed_cost[id_strategy]} fixed + {rows[id_strategy]} rows * {cost_per_row[id_strategy]} per row)")
    
    if nn_cost < id_cost:
        # nearest neighbor is better strategy
        iinfo[0].idxNum = nn_strategy
        iinfo[0].estimatedCost = nn_cost
        
        # check if we can satisfy the order clause
        if iinfo[0].nOrderBy == 1:
            if iinfo[0].aOrderBy[0].iColumn == distance_field and not iinfo[0].aOrderBy[i_order].desc:
                iinfo[0].orderByConsumed = 1
        
        # check if a constraint allows us to early-out
        for i_constraint in range(iinfo[0].nConstraint):
            if not iinfo[0].aConstraint[i_constraint].usable:
                continue
            
            column, operator = iinfo[0].aConstraint[i_constraint].iColumn, iinfo[0].aConstraint[i_constraint].op
            if column == distance_field:
                if operator == SQLITE_INDEX_CONSTRAINT_LT or operator == SQLITE_INDEX_CONSTRAINT_LE:
                    iinfo[0].aConstraintUsage[i_constraint].argvIndex = 2
                    iinfo[0].aConstraintUsage[i_constraint].omit = 1
                    flags = inclusive_max_flag if operator == SQLITE_INDEX_CONSTRAINT_LE else exclusive_max_flag
    else:
        # bisect and scan is better strategy
        iinfo[0].idxNum = id_strategy
        iinfo[0].estimatedCost = id_cost
        # search constraints to see if we can bound id and save that info in flags, min_arg, max_arg
        for i_constraint in range(iinfo[0].nConstraint):
            if not iinfo[0].aConstraint[i_constraint].usable:
                continue
            
            column, operator = iinfo[0].aConstraint[i_constraint].iColumn, iinfo[0].aConstraint[i_constraint].op
            
            if column == id_field:
                if operator == SQLITE_INDEX_CONSTRAINT_EQ:
                    flags = inclusive_min_flag | single_row_flag
                    min_arg = i_constraint
                elif operator == SQLITE_INDEX_CONSTRAINT_GE:
                    flags |= inclusive_min_flag
                    min_arg = i_constraint
                elif operator == SQLITE_INDEX_CONSTRAINT_GT:
                    flags |= exclusive_min_flag
                    min_arg = i_constraint
                elif operator == SQLITE_INDEX_CONSTRAINT_LE:
                    flags |= inclusive_max_flag
                    max_arg = i_constraint
                elif operator == SQLITE_INDEX_CONSTRAINT_LT:
                    flags |= exclusive_max_flag
                    max_arg = i_constraint
        
        # check if we can satisfy the order clause by scanning forward or backward
        if iinfo[0].nOrderBy == 1:
            if iinfo[0].aOrderBy[0].iColumn == id_field:
                iinfo[0].orderByConsumed = 1
                flags |= scan_backward_flag * iinfo[0].aOrderBy[0].desc
        
        argi = 2
        
        # pass the bounds we found in the first step as parametres to the filter method
        if flags & (inclusive_min_flag | exclusive_min_flag):
            iinfo[0].aConstraintUsage[min_arg].argvIndex = argi
            iinfo[0].aConstraintUsage[min_arg].omit = 1
            argi += 1
        if flags & (inclusive_max_flag | exclusive_max_flag):
            iinfo[0].aConstraintUsage[max_arg].argvIndex = argi
            iinfo[0].aConstraintUsage[max_arg].omit = 1
    
    iinfo[0].idxStr = <char *>sqlite.malloc(sizeof(char)*2)
    iinfo[0].idxStr[0] = flags
    iinfo[0].idxStr[1] = 0
    iinfo[0].needToFreeIdxStr = 1
    
    if __debug__:
        print(f"Final cost {iinfo[0].estimatedCost} orderByConsumed {iinfo[0].orderByConsumed} flags {flags}")
    
    return SQLITE_OK

####################### GENERAL CURSOR METHODS ###########################

cdef int balltree_vtab_open(sqlite3_vtab *pVTab, sqlite3_vtab_cursor **ppCursor) with gil:
    if __debug__:
        print("Open called")
    
    cdef balltree_vtab_cursor* new_cursor = <balltree_vtab_cursor*>sqlite.malloc(sizeof(balltree_vtab_cursor))
    if new_cursor == NULL:
        return SQLITE_NOMEM
    
    cdef CursorAttrs attrs = CursorAttrs()
    new_cursor.current_distance = -1
    new_cursor.current_id = -1
    new_cursor.current_i = -1
    new_cursor.balltree_attrs = <PyObject*>attrs
    ppCursor[0] = <sqlite3_vtab_cursor*>new_cursor
    Py_INCREF(attrs)
    return SQLITE_OK

cdef int balltree_vtab_filter(sqlite3_vtab_cursor* sqlite_cursor, int idxNum, const char *idxStr, int argc, sqlite3_value **argv) with gil:
    cdef balltree_vtab_cursor* cursor = <balltree_vtab_cursor*>sqlite_cursor
    cdef CursorAttrs attrs = <CursorAttrs>cursor.balltree_attrs
    cdef balltree_vtab* vtab = <balltree_vtab*>sqlite_cursor.pVtab
    cdef BallTreeAttrs vtab_attrs = <BallTreeAttrs>vtab.balltree_attrs
    
    if __debug__:
        print("Filter called: index no", idxNum, " argc", argc)
    
    if argc < 1:
        return SQLITE_ERROR
    # try to parse the query
    attrs.query = load_array_from_blob(argv[0])
    if attrs.query is None:
        return SQLITE_ERROR
    attrs.query = attrs.query.flatten()
    
    cursor.strategy = idxNum
    cursor.flags = idxStr[0]
    
    if idxNum == nn_strategy:
        return balltree_vtab_filter_tree(sqlite_cursor, idxNum, idxStr, argc, argv)
    elif idxNum == id_strategy:
        return balltree_vtab_filter_id(sqlite_cursor, idxNum, idxStr, argc, argv)
    else:
        return SQLITE_ERROR

cdef int balltree_vtab_close(sqlite3_vtab_cursor* sqlite_cursor):
    cdef balltree_vtab_cursor* cursor = <balltree_vtab_cursor*>sqlite_cursor
    if cursor.balltree_attrs != NULL:
        # not sure why this is crashing (probably double free)
        #Py_DECREF(<object>cursor.balltree_attrs)
        pass
    sqlite.free(cursor)
    return SQLITE_OK

cdef int balltree_vtab_next(sqlite3_vtab_cursor* sqlite_cursor) with gil:
    cdef balltree_vtab_cursor* cursor = <balltree_vtab_cursor*>sqlite_cursor
    
    if cursor.strategy == nn_strategy:
        return balltree_vtab_next_tree(sqlite_cursor)
    elif cursor.strategy == id_strategy:
        return balltree_vtab_next_id(sqlite_cursor)
    else:
        return SQLITE_ERROR

cdef int balltree_vtab_advance_to_eof(balltree_vtab_cursor* cursor) nogil:
    cursor.current_distance = -1
    cursor.current_id = -1
    cursor.current_i = -1
    return SQLITE_OK

cdef int balltree_vtab_eof(sqlite3_vtab_cursor* sqlite_cursor) with gil:
    cdef balltree_vtab_cursor* cursor = <balltree_vtab_cursor*>sqlite_cursor
    return cursor.current_i == -1

cdef int balltree_vtab_column(sqlite3_vtab_cursor *sqlite_cursor, sqlite3_context *db, int i_column) with gil:
    cdef balltree_vtab_cursor* cursor = <balltree_vtab_cursor*>sqlite_cursor
    cdef CursorAttrs attrs = <CursorAttrs>cursor.balltree_attrs
    cdef balltree_vtab* vtab = <balltree_vtab*>sqlite_cursor.pVtab
    cdef BallTreeAttrs vtab_attrs = <BallTreeAttrs>vtab.balltree_attrs
    
    if i_column == id_field:
        sqlite.result_int64(db, vtab_attrs.rowids_view[cursor.current_id])
    elif i_column == distance_field:
        sqlite.result_double(db, cursor.current_distance)
    elif i_column == query_field:
        sqlite.result_int(db, 0) # FIXME read from cursor
    else:
        return SQLITE_ERROR
    return SQLITE_OK

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
cdef int balltree_vtab_rowid(sqlite3_vtab_cursor *sqlite_cursor, sqlite3_int64 *pRowid) with gil:
    cdef balltree_vtab_cursor* cursor = <balltree_vtab_cursor*>sqlite_cursor
    cdef CursorAttrs attrs = <CursorAttrs>cursor.balltree_attrs
    cdef balltree_vtab* vtab = <balltree_vtab*>sqlite_cursor.pVtab
    cdef BallTreeAttrs vtab_attrs = <BallTreeAttrs>vtab.balltree_attrs
    if cursor.current_id >= vtab_attrs.rowids_view.shape[0]:
        return SQLITE_ERROR 
    pRowid[0] = vtab_attrs.rowids_view[cursor.current_id]
    return SQLITE_OK


######################## NEAREST NEIGHBOR CURSOR METHODS ###########################

cdef int balltree_vtab_filter_tree(sqlite3_vtab_cursor* sqlite_cursor, int idxNum, const char *idxStr, int argc, sqlite3_value **argv) with gil:
    cdef balltree_vtab_cursor* cursor = <balltree_vtab_cursor*>sqlite_cursor
    cdef CursorAttrs attrs = <CursorAttrs>cursor.balltree_attrs
    cdef balltree_vtab* vtab = <balltree_vtab*>sqlite_cursor.pVtab
    cdef BallTreeAttrs vtab_attrs = <BallTreeAttrs>vtab.balltree_attrs
    
    attrs.cursor = Cursor(vtab_attrs.tree, attrs.query)
    
    if argc >= 2:
        cursor.max_distance = sqlite.value_double(argv[1])
    
    return balltree_vtab_next_tree(sqlite_cursor)


cdef int balltree_vtab_next_tree(sqlite3_vtab_cursor* sqlite_cursor) with gil:
    cdef balltree_vtab_cursor* cursor = <balltree_vtab_cursor*>sqlite_cursor
    cdef CursorAttrs attrs = <CursorAttrs>cursor.balltree_attrs
    try:
        cursor.current_distance, cursor.current_id = attrs.cursor.next()
        cursor.current_i += 1
        if   cursor.flags & inclusive_max_flag and cursor.current_distance >  cursor.max_distance:
            return balltree_vtab_advance_to_eof(cursor)
        elif cursor.flags & exclusive_max_flag and cursor.current_distance >= cursor.max_distance:
            return balltree_vtab_advance_to_eof(cursor)
    except StopIteration:
        balltree_vtab_advance_to_eof(cursor)
    return SQLITE_OK

########################### BISECT & SCAN CURSOR METHODS #############################

# in a sorted list of values named haystack, 
# returns the index of the first element with a value >= needle 
# OR len(haystack) if all values < needle
cdef long bisect(sqlite3_int64[::1] haystack, sqlite3_int64 needle):
    cdef unsigned long low, mid, high
    
    low = 0
    high = haystack.shape[0]-1
    
    if needle <= haystack[low]:
        return 0
    if needle > haystack[high]:
        return haystack.shape[0]
    
    while high-low > 1:
      mid = low + (high-low) // 2
      if needle == haystack[mid]:
          return mid
      elif needle > haystack[mid]:
          low = mid
      else:
          high = mid
    
    return high

cdef int balltree_vtab_filter_id(sqlite3_vtab_cursor* sqlite_cursor, int idxNum, const char *idxStr, int argc, sqlite3_value **argv) with gil:
    cdef balltree_vtab_cursor* cursor = <balltree_vtab_cursor*>sqlite_cursor
    cdef CursorAttrs attrs = <CursorAttrs>cursor.balltree_attrs
    cdef balltree_vtab* vtab = <balltree_vtab*>sqlite_cursor.pVtab
    cdef BallTreeAttrs vtab_attrs = <BallTreeAttrs>vtab.balltree_attrs
    cdef int argi = 1
    cdef sqlite3_int64 id_value
    
    if cursor.flags & inclusive_min_flag or cursor.flags & exclusive_min_flag:
        if argi >= argc:
            return SQLITE_ERROR
        cursor.min_id = sqlite.value_int64(argv[argi])
        argi += 1
    if cursor.flags & inclusive_max_flag or cursor.flags & exclusive_max_flag:
        if argi >= argc:
            return SQLITE_ERROR
        cursor.max_id = sqlite.value_int64(argv[argi])
        argi += 1
    
    # if scan should start in the middle of the table, bisect to find starting point
    if cursor.flags & scan_backward_flag and cursor.flags & max_flags:
        cursor.current_id = bisect(vtab_attrs.rowids_view, cursor.max_id)
        # if we didn't find the maximum in the table start at the last element
        if cursor.current_id >= vtab_attrs.rowids_view.shape[0]:
            cursor.current_id -= 1
    elif not cursor.flags & scan_backward_flag and cursor.flags & min_flags:
        cursor.current_id = bisect(vtab_attrs.rowids_view, cursor.min_id)
        # if we went past the end of the table, we cannot scan forward even more
        if cursor.current_id >= vtab_attrs.rowids_view.shape[0]:
            return balltree_vtab_advance_to_eof(cursor)
    else:
        # otherwise just start at the beginning or end of table
        cursor.current_id = 0 if not cursor.flags & scan_backward_flag else len(attrs.rowids_view) - 1
    
    # now check if we need to adjust the found starting point because bisect either
    # went past the desired element or found an exclusive maximum / minimum
    id_value = vtab_attrs.rowids_view[cursor.current_id]
    if cursor.flags & scan_backward_flag:
        if ((cursor.flags & inclusive_max_flag and id_value >  cursor.max_id) or
            (cursor.flags & exclusive_max_flag and id_value >= cursor.max_id)):
            cursor.current_id -= 1
    else:
        if cursor.flags & exclusive_min_flag and id_value == cursor.min_id:
            cursor.current_id += 1
    
    # if our adjustment made us go out of the table bounds, return eof
    if cursor.current_id < 0 or cursor.current_id >= vtab_attrs.rowids_view.shape[0]:
        return balltree_vtab_advance_to_eof(cursor)
    
    # if we are on an eq constraint and the value is not actually equal, return eof
    if cursor.flags & single_row_flag and cursor.flags & inclusive_min_flag:
        id_value = vtab_attrs.rowids_view[cursor.current_id]
        if id_value != cursor.min_id:
            return balltree_vtab_advance_to_eof(cursor)
    
    cursor.current_i = 0
    cursor.current_distance = np.linalg.norm((vtab_attrs.data[cursor.current_id] - attrs.query))
    
    return SQLITE_OK

cdef int balltree_vtab_next_id(sqlite3_vtab_cursor* sqlite_cursor) with gil:
    cdef balltree_vtab_cursor* cursor = <balltree_vtab_cursor*>sqlite_cursor
    cdef CursorAttrs attrs = <CursorAttrs>cursor.balltree_attrs
    cdef balltree_vtab* vtab = <balltree_vtab*>sqlite_cursor.pVtab
    cdef BallTreeAttrs vtab_attrs = <BallTreeAttrs>vtab.balltree_attrs
    cdef sqlite3_int64 i, id
    
    if cursor.flags & single_row_flag:
        return balltree_vtab_advance_to_eof(cursor)
    
    # try to advance cursor
    i = cursor.current_id + (-1 if cursor.flags & scan_backward_flag else 1)
    if i < 0 or i >= vtab_attrs.rowids_view.shape[0]:
        return balltree_vtab_advance_to_eof(cursor)
    
    id = vtab_attrs.rowids_view[i]
    
    if cursor.flags & scan_backward_flag:
        if cursor.flags & inclusive_min_flag and id < cursor.min_id:
            return balltree_vtab_advance_to_eof(cursor)
        if cursor.flags & exclusive_min_flag and id <= cursor.min_id:
            return balltree_vtab_advance_to_eof(cursor)
    else:
        if cursor.flags & inclusive_max_flag and id > cursor.max_id:
            return balltree_vtab_advance_to_eof(cursor)
        if cursor.flags & exclusive_max_flag and id >= cursor.max_id:
            return balltree_vtab_advance_to_eof(cursor)
    
    cursor.current_id = i
    cursor.current_distance = np.linalg.norm((vtab_attrs.data_view[cursor.current_id] - attrs.query))
    cursor.current_i += 1
    
    return SQLITE_OK

cdef void numpydist(
      sqlite3_context *context,
      int argc,
      sqlite3_value **argv
    ) with gil:
    assert argc == 2
    
    a = load_array_from_blob(argv[0])
    b = load_array_from_blob(argv[1])
    if a is None or b is None:
        return
    
    dist = np.linalg.norm((a - b).flatten())
    
    sqlite.result_double(context, float(dist))

cdef void numpymult(
      sqlite3_context *context,
      int argc,
      sqlite3_value **argv
    ) with gil:
    assert argc == 2
    
    a = load_array_from_blob(argv[0])
    b = load_array_from_blob(argv[1])
    if a is None or b is None:
        return
    
    result = a * b
    
    cdef const unsigned char[::1] result_buffer = memoryview(encode_array_as_buffer(result))
    sqlite.result_blob(context, &(result_buffer[0]), len(result_buffer), <void (*)(void *)>SQLITE_TRANSIENT)

cdef void numpyallgt(
      sqlite3_context *context,
      int argc,
      sqlite3_value **argv
    ) with gil:
    assert argc == 2
    
    a = load_array_from_blob(argv[0])
    b = load_array_from_blob(argv[1])
    if a is None or b is None:
        return
    
    sqlite.result_int(context, 1 if (a > b).all() else 0)

cdef void numpyanygt(
      sqlite3_context *context,
      int argc,
      sqlite3_value **argv
    ) with gil:
    assert argc == 2
    
    a = load_array_from_blob(argv[0])
    b = load_array_from_blob(argv[1])
    if a is None or b is None:
        return
    
    sqlite.result_int(context, 1 if (a > b).any() else 0)

cdef void numpyalllt(
      sqlite3_context *context,
      int argc,
      sqlite3_value **argv
    ) with gil:
    assert argc == 2
    
    a = load_array_from_blob(argv[0])
    b = load_array_from_blob(argv[1])
    if a is None or b is None:
        return
    
    sqlite.result_int(context, 1 if (a < b).all() else 0)

cdef void numpyanylt(
      sqlite3_context *context,
      int argc,
      sqlite3_value **argv
    ) with gil:
    assert argc == 2
    
    a = load_array_from_blob(argv[0])
    b = load_array_from_blob(argv[1])
    if a is None or b is None:
        return
    
    sqlite.result_int(context, 1 if (a < b).any() else 0)

firsttime = True

cdef sqlite3_module balltree_module = [
    0, # version
    balltree_vtab_create, balltree_vtab_connect,
    balltree_vtab_best_index,
    balltree_vtab_disconnect, balltree_vtab_destroy,
    balltree_vtab_open, balltree_vtab_close,
    balltree_vtab_filter,
    balltree_vtab_next, balltree_vtab_eof,
    balltree_vtab_column, balltree_vtab_rowid, 
    NULL, NULL, NULL, NULL, 
    NULL, NULL, NULL, NULL, NULL, 
    NULL, ]

cdef public int sqlite3_extension_init(
      sqlite3 *db, 
      char **pzErrMsg, 
      const sqlite3_api_routines *pApi
    ):
    global sqlite
    global firsttime
    cdef int rc = SQLITE_OK
    if firsttime:
        sqlite = pApi
        rc = sqlite.create_function(db, "numpydist", 2,
                       SQLITE_UTF8|SQLITE_DETERMINISTIC,
                       NULL, numpydist, NULL, NULL)
        if rc != SQLITE_OK:
            return rc
        rc = sqlite.create_function(db, "numpymult", 2,
                       SQLITE_UTF8|SQLITE_DETERMINISTIC,
                       NULL, numpymult, NULL, NULL)
        if rc != SQLITE_OK:
            return rc
        rc = sqlite.create_function(db, "numpyallgt", 2,
                       SQLITE_UTF8|SQLITE_DETERMINISTIC,
                       NULL, numpyallgt, NULL, NULL)
        if rc != SQLITE_OK:
            return rc
        rc = sqlite.create_function(db, "numpyanygt", 2,
                       SQLITE_UTF8|SQLITE_DETERMINISTIC,
                       NULL, numpyanygt, NULL, NULL)
        if rc != SQLITE_OK:
            return rc
        rc = sqlite.create_function(db, "numpyalllt", 2,
                       SQLITE_UTF8|SQLITE_DETERMINISTIC,
                       NULL, numpyalllt, NULL, NULL)
        if rc != SQLITE_OK:
            return rc
        rc = sqlite.create_function(db, "numpyanylt", 2,
                       SQLITE_UTF8|SQLITE_DETERMINISTIC,
                       NULL, numpyanylt, NULL, NULL)
        if rc != SQLITE_OK:
            return rc
        
        rc = sqlite.create_module(db, "balltree", &balltree_module, NULL)
        if rc != SQLITE_OK:
            return rc
        firsttime = False
    return rc
