# MiniTorch Module 3

<img src="https://minitorch.github.io/minitorch.svg" width="50%">

## Task 1: Parallelization

Diagnostics output from `project/parallel_check.py`:

````bash
MAP
 
================================================================================
 Parallel Accelerator Optimizing:  Function tensor_map.<locals>._map, /Users/str
aberry_yogurt_tart/Desktop/CS5781/mod3-ruitian1212/minitorch/fast_ops.py (164)  
================================================================================


Parallel loop listing for  Function tensor_map.<locals>._map, /Users/straberry_yogurt_tart/Desktop/CS5781/mod3-ruitian1212/minitorch/fast_ops.py (164) 
-----------------------------------------------------------------------------------------|loop #ID
    def _map(                                                                            | 
        out: Storage,                                                                    | 
        out_shape: Shape,                                                                | 
        out_strides: Strides,                                                            | 
        in_storage: Storage,                                                             | 
        in_shape: Shape,                                                                 | 
        in_strides: Strides,                                                             | 
    ) -> None:                                                                           | 
        out_size: int = len(out)                                                         | 
                                                                                         | 
        if np.array_equal(out_strides, in_strides) and np.array_equal(                   | 
            out_shape, in_shape                                                          | 
        ):                                                                               | 
            for i in prange(out_size):---------------------------------------------------| #2
                out[i] = fn(in_storage[i])                                               | 
        else:                                                                            | 
            for i in prange(out_size):---------------------------------------------------| #3
                out_index = np.zeros(MAX_DIMS, dtype=np.int32)---------------------------| #0
                in_index = np.zeros(MAX_DIMS, dtype=np.int32)----------------------------| #1
                to_index(i, out_shape, out_index)                                        | 
                broadcast_index(out_index, out_shape, in_shape, in_index)                | 
                mapped_data = fn(in_storage[index_to_position(in_index, in_strides)])    | 
                out[i] = mapped_data                                                     | 
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...
 
Fused loop summary:
+--0 has the following loops fused into it:
   +--1 (fused)
Following the attempted fusion of parallel for-loops there are 3 parallel for-
loop(s) (originating from loops labelled: #2, #3, #0).
--------------------------------------------------------------------------------
---------------------------- Optimising loop nests -----------------------------
Attempting loop nest rewrites (optimising for the largest parallel loops)...
 
+--3 is a parallel loop
   +--0 --> rewritten as a serial loop
--------------------------------------------------------------------------------
----------------------------- Before Optimisation ------------------------------
Parallel region 0:
+--3 (parallel)
   +--0 (parallel)
   +--1 (parallel)


--------------------------------------------------------------------------------
------------------------------ After Optimisation ------------------------------
Parallel region 0:
+--3 (parallel)
   +--0 (serial, fused with loop(s): 1)


 
Parallel region 0 (loop #3) had 1 loop(s) fused and 1 loop(s) serialized as part
 of the larger parallel loop (#3).
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------
 
---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
The memory allocation derived from the instruction at /Users/straberry_yogurt_ta
rt/Desktop/CS5781/mod3-ruitian1212/minitorch/fast_ops.py (181) is hoisted out of
 the parallel loop labelled #3 (it will be performed before the loop is executed
 and reused inside the loop):
   Allocation:: out_index = np.zeros(MAX_DIMS, dtype=np.int32)
    - numpy.empty() is used for the allocation.
The memory allocation derived from the instruction at /Users/straberry_yogurt_ta
rt/Desktop/CS5781/mod3-ruitian1212/minitorch/fast_ops.py (182) is hoisted out of
 the parallel loop labelled #3 (it will be performed before the loop is executed
 and reused inside the loop):
   Allocation:: in_index = np.zeros(MAX_DIMS, dtype=np.int32)
    - numpy.empty() is used for the allocation.
None
ZIP
 
================================================================================
 Parallel Accelerator Optimizing:  Function tensor_zip.<locals>._zip, /Users/str
aberry_yogurt_tart/Desktop/CS5781/mod3-ruitian1212/minitorch/fast_ops.py (214)  
================================================================================


Parallel loop listing for  Function tensor_zip.<locals>._zip, /Users/straberry_yogurt_tart/Desktop/CS5781/mod3-ruitian1212/minitorch/fast_ops.py (214) 
---------------------------------------------------------------------------|loop #ID
    def _zip(                                                              | 
        out: Storage,                                                      | 
        out_shape: Shape,                                                  | 
        out_strides: Strides,                                              | 
        a_storage: Storage,                                                | 
        a_shape: Shape,                                                    | 
        a_strides: Strides,                                                | 
        b_storage: Storage,                                                | 
        b_shape: Shape,                                                    | 
        b_strides: Strides,                                                | 
    ) -> None:                                                             | 
        out_size: int = len(out)                                           | 
                                                                           | 
        if (                                                               | 
            np.array_equal(out_strides, a_strides)                         | 
            and np.array_equal(out_strides, b_strides)                     | 
            and np.array_equal(out_shape, a_shape)                         | 
            and np.array_equal(out_shape, b_shape)                         | 
        ):                                                                 | 
            for i in prange(out_size):-------------------------------------| #7
                out[i] = fn(a_storage[i], b_storage[i])                    | 
        else:                                                              | 
            for i in prange(out_size):-------------------------------------| #8
                out_index: Index = np.zeros(MAX_DIMS, dtype=np.int32)------| #4
                a_index: Index = np.zeros(MAX_DIMS, dtype=np.int32)--------| #5
                b_index: Index = np.zeros(MAX_DIMS, dtype=np.int32)--------| #6
                to_index(i, out_shape, out_index)                          | 
                                                                           | 
                broadcast_index(out_index, out_shape, a_shape, a_index)    | 
                broadcast_index(out_index, out_shape, b_shape, b_index)    | 
                                                                           | 
                zipped_data = fn(                                          | 
                    a_storage[index_to_position(a_index, a_strides)],      | 
                    b_storage[index_to_position(b_index, b_strides)],      | 
                )                                                          | 
                                                                           | 
                out[i] = zipped_data                                       | 
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...
 
Fused loop summary:
+--4 has the following loops fused into it:
   +--5 (fused)
   +--6 (fused)
Following the attempted fusion of parallel for-loops there are 3 parallel for-
loop(s) (originating from loops labelled: #7, #8, #4).
--------------------------------------------------------------------------------
---------------------------- Optimising loop nests -----------------------------
Attempting loop nest rewrites (optimising for the largest parallel loops)...
 
+--8 is a parallel loop
   +--4 --> rewritten as a serial loop
--------------------------------------------------------------------------------
----------------------------- Before Optimisation ------------------------------
Parallel region 0:
+--8 (parallel)
   +--4 (parallel)
   +--5 (parallel)
   +--6 (parallel)


--------------------------------------------------------------------------------
------------------------------ After Optimisation ------------------------------
Parallel region 0:
+--8 (parallel)
   +--4 (serial, fused with loop(s): 5, 6)


 
Parallel region 0 (loop #8) had 2 loop(s) fused and 1 loop(s) serialized as part
 of the larger parallel loop (#8).
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------
 
---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
The memory allocation derived from the instruction at /Users/straberry_yogurt_ta
rt/Desktop/CS5781/mod3-ruitian1212/minitorch/fast_ops.py (237) is hoisted out of
 the parallel loop labelled #8 (it will be performed before the loop is executed
 and reused inside the loop):
   Allocation:: out_index: Index = np.zeros(MAX_DIMS, dtype=np.int32)
    - numpy.empty() is used for the allocation.
The memory allocation derived from the instruction at /Users/straberry_yogurt_ta
rt/Desktop/CS5781/mod3-ruitian1212/minitorch/fast_ops.py (238) is hoisted out of
 the parallel loop labelled #8 (it will be performed before the loop is executed
 and reused inside the loop):
   Allocation:: a_index: Index = np.zeros(MAX_DIMS, dtype=np.int32)
    - numpy.empty() is used for the allocation.
The memory allocation derived from the instruction at /Users/straberry_yogurt_ta
rt/Desktop/CS5781/mod3-ruitian1212/minitorch/fast_ops.py (239) is hoisted out of
 the parallel loop labelled #8 (it will be performed before the loop is executed
 and reused inside the loop):
   Allocation:: b_index: Index = np.zeros(MAX_DIMS, dtype=np.int32)
    - numpy.empty() is used for the allocation.
None
REDUCE
 
================================================================================
 Parallel Accelerator Optimizing:  Function tensor_reduce.<locals>._reduce, /Use
rs/straberry_yogurt_tart/Desktop/CS5781/mod3-ruitian1212/minitorch/fast_ops.py 
(276)  
================================================================================


Parallel loop listing for  Function tensor_reduce.<locals>._reduce, /Users/straberry_yogurt_tart/Desktop/CS5781/mod3-ruitian1212/minitorch/fast_ops.py (276) 
-----------------------------------------------------------------------|loop #ID
    def _reduce(                                                       | 
        out: Storage,                                                  | 
        out_shape: Shape,                                              | 
        out_strides: Strides,                                          | 
        a_storage: Storage,                                            | 
        a_shape: Shape,                                                | 
        a_strides: Strides,                                            | 
        reduce_dim: int,                                               | 
    ) -> None:                                                         | 
        out_size: int = len(out)                                       | 
        reduce_size: int = a_shape[reduce_dim]                         | 
                                                                       | 
        for i in prange(out_size):-------------------------------------| #10
            out_index: Index = np.zeros(MAX_DIMS, dtype=np.int32)------| #9
            to_index(i, out_shape, out_index)                          | 
            a_index = index_to_position(out_index, a_strides)          | 
            reduced_val = out[i]                                       | 
            for j in range(reduce_size):                               | 
                reduced_val = fn(                                      | 
                    reduced_val,                                       | 
                    a_storage[a_index + j * a_strides[reduce_dim]],    | 
                )                                                      | 
            out[i] = reduced_val                                       | 
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...
Following the attempted fusion of parallel for-loops there are 2 parallel for-
loop(s) (originating from loops labelled: #10, #9).
--------------------------------------------------------------------------------
---------------------------- Optimising loop nests -----------------------------
Attempting loop nest rewrites (optimising for the largest parallel loops)...
 
+--10 is a parallel loop
   +--9 --> rewritten as a serial loop
--------------------------------------------------------------------------------
----------------------------- Before Optimisation ------------------------------
Parallel region 0:
+--10 (parallel)
   +--9 (parallel)


--------------------------------------------------------------------------------
------------------------------ After Optimisation ------------------------------
Parallel region 0:
+--10 (parallel)
   +--9 (serial)


 
Parallel region 0 (loop #10) had 0 loop(s) fused and 1 loop(s) serialized as 
part of the larger parallel loop (#10).
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------
 
---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
The memory allocation derived from the instruction at /Users/straberry_yogurt_ta
rt/Desktop/CS5781/mod3-ruitian1212/minitorch/fast_ops.py (289) is hoisted out of
 the parallel loop labelled #10 (it will be performed before the loop is 
executed and reused inside the loop):
   Allocation:: out_index: Index = np.zeros(MAX_DIMS, dtype=np.int32)
    - numpy.empty() is used for the allocation.
None
MATRIX MULTIPLY
 
================================================================================
 Parallel Accelerator Optimizing:  Function _tensor_matrix_multiply, /Users/stra
berry_yogurt_tart/Desktop/CS5781/mod3-ruitian1212/minitorch/fast_ops.py (303)  
================================================================================


Parallel loop listing for  Function _tensor_matrix_multiply, /Users/straberry_yogurt_tart/Desktop/CS5781/mod3-ruitian1212/minitorch/fast_ops.py (303) 
---------------------------------------------------------------------------------------|loop #ID
def _tensor_matrix_multiply(                                                           | 
    out: Storage,                                                                      | 
    out_shape: Shape,                                                                  | 
    out_strides: Strides,                                                              | 
    a_storage: Storage,                                                                | 
    a_shape: Shape,                                                                    | 
    a_strides: Strides,                                                                | 
    b_storage: Storage,                                                                | 
    b_shape: Shape,                                                                    | 
    b_strides: Strides,                                                                | 
) -> None:                                                                             | 
    """NUMBA tensor matrix multiply function.                                          | 
                                                                                       | 
    Should work for any tensor shapes that broadcast as long as                        | 
                                                                                       | 
    ```                                                                                | 
    assert a_shape[-1] == b_shape[-2]                                                  | 
    ```                                                                                | 
                                                                                       | 
    Optimizations:                                                                     | 
                                                                                       | 
    * Outer loop in parallel                                                           | 
    * No index buffers or function calls                                               | 
    * Inner loop should have no global writes, 1 multiply.                             | 
                                                                                       | 
                                                                                       | 
    Args:                                                                              | 
    ----                                                                               | 
        out (Storage): storage for `out` tensor                                        | 
        out_shape (Shape): shape for `out` tensor                                      | 
        out_strides (Strides): strides for `out` tensor                                | 
        a_storage (Storage): storage for `a` tensor                                    | 
        a_shape (Shape): shape for `a` tensor                                          | 
        a_strides (Strides): strides for `a` tensor                                    | 
        b_storage (Storage): storage for `b` tensor                                    | 
        b_shape (Shape): shape for `b` tensor                                          | 
        b_strides (Strides): strides for `b` tensor                                    | 
                                                                                       | 
    Returns:                                                                           | 
    -------                                                                            | 
        None : Fills in `out`                                                          | 
                                                                                       | 
    """                                                                                | 
    a_batch_stride = a_strides[0] if a_shape[0] > 1 else 0                             | 
    b_batch_stride = b_strides[0] if b_shape[0] > 1 else 0                             | 
                                                                                       | 
    D = a_shape[-1]                                                                    | 
    A, B, C = out_shape[-3:]                                                           | 
    for a in prange(A):----------------------------------------------------------------| #13
        for b in prange(B):------------------------------------------------------------| #12
            for c in prange(C):--------------------------------------------------------| #11
                sum_result: float = 0.0                                                | 
                a_index: int = a * a_batch_stride + b * a_strides[-2]                  | 
                b_index: int = a * b_batch_stride + c * b_strides[-1]                  | 
                for _ in range(D):                                                     | 
                    sum_result += a_storage[a_index] * b_storage[b_index]              | 
                    a_index += a_strides[-1]                                           | 
                    b_index += b_strides[-2]                                           | 
                out_index = (                                                          | 
                    a * out_strides[-3] + b * out_strides[-2] + c * out_strides[-1]    | 
                )                                                                      | 
                out[out_index] = sum_result                                            | 
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...
Following the attempted fusion of parallel for-loops there are 2 parallel for-
loop(s) (originating from loops labelled: #13, #12).
--------------------------------------------------------------------------------
---------------------------- Optimising loop nests -----------------------------
Attempting loop nest rewrites (optimising for the largest parallel loops)...
 
+--13 is a parallel loop
   +--12 --> rewritten as a serial loop
      +--11 --> rewritten as a serial loop
--------------------------------------------------------------------------------
----------------------------- Before Optimisation ------------------------------
Parallel region 0:
+--13 (parallel)
   +--12 (parallel)
      +--11 (parallel)


--------------------------------------------------------------------------------
------------------------------ After Optimisation ------------------------------
Parallel region 0:
+--13 (parallel)
   +--12 (serial)
      +--11 (serial)


 
Parallel region 0 (loop #13) had 0 loop(s) fused and 2 loop(s) serialized as 
part of the larger parallel loop (#13).
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------
 
---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
No allocation hoisting found
None
````


## Task 4: CUDA Matrix Multiplication

Below is a table showing the comparison of matrix multiplication execution time between `FastOps` and `CudaOps`:

| Matrix Size (n x n) | FastOps Time (s) | CudaOps Time (s) |
|---------------------|------------------|------------------|
| 64                  | 0.00300          | 0.00567          |
| 128                 | 0.01393          | 0.01339          |
| 256                 | 0.08933          | 0.04640          |
| 512                 | 0.94337          | 0.19329          |
| 1024                | 8.11608          | 0.86809          |
