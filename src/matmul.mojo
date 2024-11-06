from memory import memset_zero
from random import rand
from algorithm import parallelize, vectorize, Static2DTileUnitFunc as Tile2DFunc
from sys import simdwidthof
import benchmark


alias type = DType.float32

# simdwidthof = number of float32 elements that fit into a single SIMD register
# using a 2x multiplier allows some SIMD operations to run in the same cycle
alias nelts = simdwidthof[DType.float32]() * 2

struct Matrix[rows: Int, cols: Int]:
    var data: UnsafePointer[Scalar[type]]

    # Initialize zeroeing all values
    fn __init__(inout self):
        self.data = UnsafePointer[Scalar[type]].alloc(rows * cols)
        memset_zero(self.data.address, rows * cols)

    # Initialize taking a pointer, don't set any elements
    fn __init__(inout self, data: UnsafePointer[Scalar[type]]):
        self.data = data

    # Initialize with random values
    @staticmethod
    fn rand() -> Self:
        var data = UnsafePointer[Scalar[type]].alloc(rows * cols)
        rand(data.address, rows * cols)
        return Self(data)

    fn __getitem__(self, y: Int, x: Int) -> Scalar[type]:
        return self.load[1](y, x)

    fn __setitem__(self, y: Int, x: Int, val: Scalar[type]):
        self.store[1](y, x, val)

    fn load[nelts: Int](self, y: Int, x: Int) -> SIMD[type, nelts]:
        return self.data.load[width=nelts](y * self.cols + x)

    fn store[nelts: Int](self, y: Int, x: Int, val: SIMD[type, nelts]):
        return self.data.store[width=nelts](y * self.cols + x, val)

# Perform 2D tiling on the iteration space defined by end_x and end_y.
fn tile[tiled_fn: Tile2DFunc, tile_x: Int, tile_y: Int](end_x: Int, end_y: Int):
    # Note: this assumes that ends are multiples of the tiles.
    for y in range(0, end_y, tile_y):
        for x in range(0, end_x, tile_x):
            tiled_fn[tile_x, tile_y](x, y)

# Use the above tile function to perform tiled matmul.
# Unroll the vectorized loop by a constant factor.
fn matmul_tiled_unrolled_parallelized(C: Matrix, A: Matrix, B: Matrix):
    @parameter
    fn calc_row(m: Int):
        @parameter
        fn calc_tile[tile_x: Int, tile_y: Int](x: Int, y: Int):
            for k in range(y, y + tile_y):
                @parameter
                fn dot[nelts: Int](n: Int):
                    C.store(m, n + x, C.load[nelts](m, n + x) + A[m, k] * B.load[nelts](k, n + x))

                # Vectorize by nelts and unroll by tile_x/nelts
                # Here unroll factor is 4
                alias unroll_factor = tile_x // nelts
                vectorize[dot, nelts, size=tile_x, unroll_factor=unroll_factor]()

        alias tile_size = 4
        tile[calc_tile, nelts * tile_size, tile_size](A.cols, C.cols)

    parallelize[calc_row](C.rows, C.rows)


alias M = 1024 * 100
alias N = 1024
alias K = 1024

@always_inline
fn bench[
    func: fn (Matrix, Matrix, Matrix) -> None]():
    var C = Matrix[M, N]()
    var A = Matrix[M, K].rand()
    var B = Matrix[K, N].rand()

    @always_inline
    @parameter
    fn test_fn():
        _ = func(C, A, B)

    var secs = benchmark.run[test_fn](max_runtime_secs=1).mean()

    A.data.free()
    B.data.free()
    C.data.free()

    var gflops = ((2 * M * N * K) / secs) / 1e9
    # var speedup: Float64 = gflops / base_gflops

    print(gflops, "GFLOP/s")  

fn main():
  bench[matmul_tiled_unrolled_parallelized]()