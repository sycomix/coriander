"""
Try to run a tf kernel
Check does/doesnt work
"""
import os
from os.path import join
import numpy as np
import pyopencl as cl
import subprocess
from test import test_common
from test.test_common import offset_type
import pytest


# CLANG_HOME = os.environ['CLANG_HOME']


def myrun(cmd_list):
    print(f"running [{' '.join(cmd_list)}]")
    with open('/tmp/testout.txt', 'w+') as f_out:
        res = subprocess.run(cmd_list, stdout=f_out, stderr=subprocess.STDOUT)
        f_out.seek(0)
        contents = f_out.read()
    print(contents)
    assert res.returncode == 0


@pytest.mark.skipif(os.environ.get('TRAVIS', None) == 'true', reason='fails on travis mac cpu, not looked into why yet')
def test_cwise_sqrt(context, q, float_data, float_data_gpu):
    with open('test/tf/samples/cwise_op_gpu_sqrt-device-noopt.ll', 'r') as f:
        ll_code = f.read()

    cl_sourcecode = test_common.ll_to_cl(
        ll_code,
        '_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIfLi1ELi1EiEELi16ENS_11MakePointerEEEKNS_18TensorCwiseUnaryOpINS0_14scalar_sqrt_opIfEEKNS4_INS5_IKfLi1ELi1EiEELi16ES7_EEEEEENS_9GpuDeviceEEEiEEvT_T0_',
        num_clmems=3)

    print('creating program...')
    prog_unbuilt = cl.Program(context, cl_sourcecode)
    print('building kernel...')
    prog = prog_unbuilt.build()

    N = 10

    # global struct Eigen__TensorEvaluator_nopointers* eval_nopointers, global float* eval_ptr0, long eval_ptr0_offset, global float* eval_ptr1, long eval_ptr1_offset, int size, local int *scratch

    # what we need:
    # struct Eigen__TensorEvaluator_nopointers   Note that none of the values we copy across are actually use, so we can just create a sufficiently large buffer...
    # global float *eval_ptr0  => this will receive the result.  just create a sufficiently large buffer
    # ptr0_offset => 0
    # eval_ptr1 => will contian the data we want to reduce
    # eval_ptr1_offset=> 0
    # size =>  eg 10, to reduce 10 values
    # scratch => set to workgroupsize * sizeof(float)

    eval_nopointers_gpu = cl.Buffer(context, cl.mem_flags.READ_WRITE, size=4096)

    eval_ptr0 = np.zeros(1024, dtype=np.float32)
    eval_ptr0_gpu = cl.Buffer(context, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf=eval_ptr0)
    eval_ptr0_offset = 0

    eval_ptr1 = np.random.uniform(0, 1, size=(1024,)).astype(np.float32) + 1.0
    eval_ptr1_gpu = cl.Buffer(context, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf=eval_ptr1)
    eval_ptr1_offset = 0

    size = N

    global_size = 256
    workgroup_size = 256
    scratch = workgroup_size * 4

    print('running kernel...')
    prog.__getattr__('_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIfLi1ELi1EiEELi16ENS_11MakePointerEEEKNS_18TensorCwiseUnaryOpINS0_14scalar_sqrt_opIfEEKNS4_INS5_IKfLi1ELi1EiEELi16ES7_EEEEEENS_9GpuDeviceEEEiEEvT_T0_')(
        q, (global_size,), (workgroup_size,),
        eval_nopointers_gpu, offset_type(0),
        eval_ptr0_gpu, offset_type(0),
        eval_ptr1_gpu, offset_type(0),
        offset_type(0),
        offset_type(eval_ptr0_offset),
        offset_type(eval_ptr1_offset),
        np.int32(size),
        cl.LocalMemory(scratch)
    )
    # check for errors
    q.finish()
    # copy eval_ptr0 back, and check the results...
    cl.enqueue_copy(q, eval_ptr0, eval_ptr0_gpu)
    q.finish()
    print('eval_ptr0[:N]', eval_ptr0[:N])

    expected = np.sqrt(eval_ptr1)
    print('expected[:10]', expected[:N])
    assert np.abs(expected[:N] - eval_ptr0[:N]).max() < 1e-4


@pytest.mark.skipif(os.environ.get('TRAVIS', None) == 'true', reason='fails on travis mac cpu, not looked into why yet')
def test_cwise_sqrt_singlebuffer(context, queue, float_data, float_data_gpu):
    with open('test/tf/samples/cwise_op_gpu_sqrt-device-noopt.ll', 'r') as f:
        ll_code = f.read()

    cl_sourcecode = test_common.ll_to_cl(
        ll_code,
        '_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIfLi1ELi1EiEELi16ENS_11MakePointerEEEKNS_18TensorCwiseUnaryOpINS0_14scalar_sqrt_opIfEEKNS4_INS5_IKfLi1ELi1EiEELi16ES7_EEEEEENS_9GpuDeviceEEEiEEvT_T0_',
        num_clmems=3)

    prog = cl.Program(context, cl_sourcecode).build()

    N = 10

    # global struct Eigen__TensorEvaluator_nopointers* eval_nopointers, global float* eval_ptr0, long eval_ptr0_offset, global float* eval_ptr1, long eval_ptr1_offset, int size, local int *scratch

    # what we need:
    # struct Eigen__TensorEvaluator_nopointers   Note that none of the values we copy across are actually use, so we can just create a sufficiently large buffer...
    # global float *eval_ptr0  => this will receive the result.  just create a sufficiently large buffer
    # ptr0_offset => 0
    # eval_ptr1 => will contian the data we want to reduce
    # eval_ptr1_offset=> 0
    # size =>  eg 10, to reduce 10 values
    # scratch => set to workgroupsize * sizeof(float)

    # by compariosn to the earlier test, we create a sigle buffer, containing both ptr0 and ptr1, and just use
    # offset into this

    src_host = np.random.uniform(0, 1, size=(N,)).astype(np.float32) + 1.0
    dst_host = np.zeros(N, dtype=np.float32)

    src_offset_bytes = 128
    dst_offset_bytes = 256

    huge_buf_gpu = cl.Buffer(context, cl.mem_flags.READ_WRITE, size=4096)
    # huge_buf_gpu_spare = cl.Buffer(context, cl.mem_flags.READ_WRITE, size=4096)

    eval_nopointers_gpu = cl.Buffer(context, cl.mem_flags.READ_WRITE, size=4096)

    eval_ptr0_gpu = huge_buf_gpu
    eval_ptr0_offset = dst_offset_bytes

    eval_ptr1_gpu = huge_buf_gpu
    eval_ptr1_offset = src_offset_bytes

    size = N

    test_common.enqueue_write_buffer_ext(
        cl,
        queue,
        huge_buf_gpu,
        src_host,
        device_offset=eval_ptr1_offset,
        size=N * 4,
    )
    global_size = 256
    workgroup_size = 256
    scratch = workgroup_size * 4

    prog.__getattr__('_ZN5Eigen8internal15EigenMetaKernelINS_15TensorEvaluatorIKNS_14TensorAssignOpINS_9TensorMapINS_6TensorIfLi1ELi1EiEELi16ENS_11MakePointerEEEKNS_18TensorCwiseUnaryOpINS0_14scalar_sqrt_opIfEEKNS4_INS5_IKfLi1ELi1EiEELi16ES7_EEEEEENS_9GpuDeviceEEEiEEvT_T0_')(
        queue, (global_size,), (workgroup_size,),
        eval_nopointers_gpu, offset_type(0),
        eval_ptr0_gpu, offset_type(0),
        eval_ptr1_gpu, offset_type(0),
        offset_type(0),
        offset_type(eval_ptr0_offset),
        offset_type(eval_ptr1_offset),
        np.int32(size),
        cl.LocalMemory(scratch)
    )
    # check for errors
    queue.finish()

    test_common.enqueue_read_buffer_ext(
        cl,
        queue,
        huge_buf_gpu,
        dst_host,
        device_offset=eval_ptr0_offset,
        size=N * 4,
    )
    # cl.enqueue_copy(queue, dst_host, huge_buf_gpu, device_offset=128, size=N * 4)
    queue.finish()
    print('dst_host[:N]', dst_host[:N])

    expected = np.sqrt(src_host)
    print('expected[:10]', expected[:N])
    assert np.abs(expected[:N] - dst_host[:N]).max() < 1e-4
