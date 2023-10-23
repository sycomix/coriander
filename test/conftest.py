import numpy as np
import pyopencl as cl
import pytest
import os


gpu_idx = int(os.environ.get('CL_GPUOFFSET', 0))

mf = cl.mem_flags


@pytest.fixture(scope='module')
def context():
    platforms = cl.get_platforms()
    i = 0
    ctx = None
    for platform in platforms:
        if os.environ.get('COCL_DEVICES_ALL', None) == '1':
            print('Warning!  Using COCL_DEVICES_ALL.  This is a maintainer-oriented option, and is likely to lead to errors')
            gpu_devices = platform.get_devices(device_type=cl.device_type.ALL)
        else:
            gpu_devices = platform.get_devices(device_type=cl.device_type.GPU)
        if gpu_idx < i + len(gpu_devices):
            ctx = cl.Context(devices=[gpu_devices[gpu_idx - i]])
            break
        i += len(gpu_devices)

    if ctx is None:
        raise Exception(f'unable to find gpu at index {gpu_idx}')
    print('context', ctx)
    # ctx = cl.create_some_context()
    return ctx


@pytest.fixture(scope='module')
def queue(context):
    return cl.CommandQueue(context)


@pytest.fixture(scope='module')
def q(queue):
    return queue


@pytest.fixture(scope='module')
def ctx(context):
    return context


@pytest.fixture
def int_data():
    np.random.seed(123)
    return np.random.randint(1024, size=(1024,), dtype=np.int32)


@pytest.fixture
def float_data():
    np.random.seed(124)
    return np.random.randn(1024).astype(np.float32)


@pytest.fixture
def int_data_gpu(int_data, ctx):
    return cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=int_data)


@pytest.fixture
def float_data_gpu(float_data, ctx):
    return cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=float_data)
