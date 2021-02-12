import cv2
import numpy as np
import pyopencl as cl
import os
import time
os.environ['PYOPENCL_COMPILER_OUTPUT'] = '1'

def Difference(img1, img2):
        platforms = cl.get_platforms() # a platform corresponds to a driver (e.g. AMD)
        platform = platforms[0] # take first platform
        devices = platform.get_devices(cl.device_type.GPU) # get GPU devices of selected platform
        device = devices[0] # take first GPU
        context = cl.Context([device]) # put selected GPU into context object
        queue = cl.CommandQueue(context, device) # create command queue for selected GPU and context

        # (2) get shape of input image, allocate memory for output to which result can be copied to
        shape = img1.T.shape
        result = np.empty_like(img1)    
        
        # (2) create image buffers which hold images for OpenCL
        imgInBuf1 = cl.Image(context, cl.mem_flags.READ_ONLY, cl.ImageFormat(cl.channel_order.LUMINANCE, cl.channel_type.UNORM_INT8), shape=shape) # holds a gray-valued image of given shape
        imgInBuf2 = cl.Image(context, cl.mem_flags.READ_ONLY, cl.ImageFormat(cl.channel_order.LUMINANCE, cl.channel_type.UNORM_INT8), shape=shape)
        imgOutBuf = cl.Image(context, cl.mem_flags.WRITE_ONLY, cl.ImageFormat(cl.channel_order.LUMINANCE, cl.channel_type.UNORM_INT8), shape=shape) # placeholder for gray-valued image of given shape
        
        # (3) load and compile OpenCL program
        program = cl.Program(context, open('Difference.cl').read()).build()

        # (3) from OpenCL program, get kernel object and set arguments (input image, operation type, output image)
        kernel = cl.Kernel(program, 'Difference') # name of function according to kernel.py
        kernel.set_arg(0, imgInBuf1) # input image buffer
        kernel.set_arg(1, imgInBuf2) # operation type passed as an integer value (dilate=0, erode=1)
        kernel.set_arg(2, imgOutBuf) # output image buffer
        
        # (4) copy image to device, execute kernel, copy data back
        cl.enqueue_copy(queue, imgInBuf1, img1, origin=(0, 0), region=shape, is_blocking=False) # copy image from CPU to GPU
        cl.enqueue_copy(queue, imgInBuf2, img2, origin=(0, 0), region=shape, is_blocking=False)
        cl.enqueue_nd_range_kernel(queue, kernel, shape, None) # execute kernel, work is distributed across shape[0]*shape[1] work-items (one work-item per pixel of the image)
        cl.enqueue_copy(queue, result, imgOutBuf, origin=(0, 0), region=shape, is_blocking=True) # wait until finished copying resulting image back from GPU to CPU

        return result
