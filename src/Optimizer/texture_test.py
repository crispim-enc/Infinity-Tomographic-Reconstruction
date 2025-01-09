import time

import pycuda.driver as cuda
import pycuda.autoinit
import numpy
from pycuda.compiler import SourceModule
import math

tic = time.time()
realrow = numpy.array([1.0, 2.0, 3.0, 4.0, 5.0],
                      dtype=numpy.float32).reshape(1, 5)

free, total = cuda.mem_get_info()
n_el = 50000000
realrow = numpy.array(numpy.random.rand(n_el),
                   dtype=numpy.float32)

mod_copy_texture = SourceModule("""
texture<float, 1> tex;

__global__ void copy_texture_kernel(float * data) {
int ty = blockIdx.x * blockDim.x + threadIdx.x;
data[ty] = tex1Dfetch(tex, ty);
// data[ty] = 1;
}
""")

# cuda.cudaCreateTextureObject(&tex_ar[i], &resDesc, &texDesc, NULL);
copy_texture_func = mod_copy_texture.get_function("copy_texture_kernel")
texref = mod_copy_texture.get_texref("tex")

im_gpu = cuda.mem_alloc(
    realrow.size * realrow.dtype.itemsize)


cuda.memcpy_htod_async(im_gpu, realrow)
texref.set_address(im_gpu, realrow.nbytes)
texref.set_format(pycuda.driver.array_format.FLOAT, 1)

# cuda.matrix_to_texref(realrow, texref, order="C")

# texref.set_array(realrow)

texref.set_flags(cuda.TRSF_NORMALIZED_COORDINATES)
# texref.set_filter_mode(cuda.filter_mode.LINEAR)

gpu_output = numpy.zeros_like(realrow)
threadsperblock = (128, 1, 1)
blockspergrid_x = int(math.ceil(n_el/ threadsperblock[0]))
blockspergrid_y = int(math.ceil(1 / threadsperblock[1]))
blockspergrid_z = int(math.ceil(1 / threadsperblock[2]))
blockspergrid = (blockspergrid_x, blockspergrid_y, blockspergrid_z)
copy_texture_func(cuda.Out(gpu_output), block=threadsperblock, grid=blockspergrid, texrefs=[texref])
print(numpy.sum(gpu_output-realrow))
print(time.time()-tic)