import cv2
import numpy as np
import pyopencl as cl
import os
import time
os.environ['PYOPENCL_COMPILER_OUTPUT'] = '1'

def Difference(img1, img2, threshold):
        img1 = np.array(img1).astype('uint8')
        img2 = np.array(img2).astype('uint8')
        platforms = cl.get_platforms()
        platform = platforms[0]
        devices = platform.get_devices(cl.device_type.GPU)
        device = devices[0]
        context = cl.Context([device])
        queue = cl.CommandQueue(context, device)

        shape = img1.T.shape
        result = np.empty_like(img1)    
        
        imgInBuf1 = cl.Image(context, cl.mem_flags.READ_ONLY, cl.ImageFormat(cl.channel_order.LUMINANCE, cl.channel_type.UNORM_INT8), shape=shape)
        imgInBuf2 = cl.Image(context, cl.mem_flags.READ_ONLY, cl.ImageFormat(cl.channel_order.LUMINANCE, cl.channel_type.UNORM_INT8), shape=shape)
        imgOutBuf = cl.Image(context, cl.mem_flags.WRITE_ONLY, cl.ImageFormat(cl.channel_order.LUMINANCE, cl.channel_type.UNORM_INT8), shape=shape)
        
        program = cl.Program(context, open('Difference.cl').read()).build()

        kernel = cl.Kernel(program, 'Difference')
        kernel.set_arg(0, imgInBuf1)
        kernel.set_arg(1, imgInBuf2)
        kernel.set_arg(2, imgOutBuf)
        kernel.set_arg(3, np.float32(threshold))

        cl.enqueue_copy(queue, imgInBuf1, img1, origin=(0, 0), region=shape, is_blocking=False)
        cl.enqueue_copy(queue, imgInBuf2, img2, origin=(0, 0), region=shape, is_blocking=False)
        cl.enqueue_nd_range_kernel(queue, kernel, shape, None)
        cl.enqueue_copy(queue, result, imgOutBuf, origin=(0, 0), region=shape, is_blocking=True)

        return result
