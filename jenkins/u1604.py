#/usr/bin/env python
# designed to be run with python 2.7, from the root of the repo

import os
import sys
import subprocess
import platform
from os import path
from os.path import join
import time
import argparse


current_dir = path.abspath(os.getcwd())

def cd(subdir):
    global current_dir
    current_dir = subdir if subdir.startswith('/') else join(current_dir, subdir)
    print(f'cd to [{current_dir}]')


def mkdir(subdir):
    global current_dir
    full_path = join(current_dir, subdir)
    if not path.isdir(full_path):
        os.makedirs(full_path)


def cd_repo_root():
    global current_dir
    current_dir = path.abspath(os.getcwd())  # since python never really changes its actual cwd


def run(cmdlist):
    global recursion_level
    print(' '.join(cmdlist))
    f_out = open('jenkins-out%s.txt', 'w', buffering=1)
    f_in = open('jenkins-out%s.txt', 'r', buffering=1)
    f_in.seek(0)
    p = subprocess.Popen(cmdlist, cwd=current_dir, stdout=f_out, stderr=subprocess.STDOUT, bufsize=1)
    res = ''
    def print_progress():
        line = f_in.readline()
        # if not is_py2():
        #     line = line.decode('utf-8')
        res_lines = ''
        while line != '':
            print(line[:-1])
            res_lines += line
            line = f_in.readline()
            # if not is_py2():
            #     line = line.decode('utf-8')
        return res_lines
        # print(lines)
    p.poll()
    while p.returncode is None:
        res += print_progress()
        time.sleep(1)
        p.poll()
    res += print_progress()
    print('p.returncode', p.returncode)
    assert p.returncode == 0
    return res


def run_until(cmdlist, until):
    """
    Runs until string until appears in output, then kills the process, without
    checking return code, and returns the output so far
    """
    global current_dir
    print(' '.join(cmdlist))
    f_out = open('jenkins-out%s.txt', 'w', buffering=1)
    f_in = open('jenkins-out%s.txt', 'r', buffering=1)
    f_in.seek(0)
    p = subprocess.Popen(cmdlist, cwd=current_dir, stdout=f_out, stderr=subprocess.STDOUT, bufsize=1)
    res = ''
    def print_progress():
        line = f_in.readline()
        # if not is_py2():
        #     line = line.decode('utf-8')
        res_lines = ''
        while line != '':
            print(line[:-1])
            res_lines += line
            line = f_in.readline()
            # if not is_py2():
            #     line = line.decode('utf-8')
        return res_lines
        # print(lines)
    p.poll()
    while p.returncode is None:
        res += print_progress()
        if until in res:
            p.terminate()
            return res
        time.sleep(1)
        p.poll()
    res += print_progress()
    print('p.returncode', p.returncode)
    assert p.returncode == 0
    return res


def maybe_rmtree(tree_dir):
    if path.isdir(tree_dir):
        if platform.uname()[0] == 'Windows':
            run(['rmdir', '/s', '/q', f'"{tree_dir}"'])
        else:
            run(['rm', '-Rf', tree_dir])


def wget(target_url):
    # should be generalized for Windows
    run(['wget', '--progress=dot:giga', target_url])


def gunzip(target):
    # should be generalized for Windows
    run(['gunzip', target])


def activate(activate_file):
    with open(activate_file) as f:
        contents = f.read()
    for line in contents.split('\n'):
        line = line.strip().replace('export ', '')
        if line == '':
            continue
        var = line.split('=')[0].strip()
        value = line.split('=')[1].strip().replace('$PATH', os.environ['PATH'])
        if value.startswith('"'):
            value = value[1:]
        if value.endswith('"'):
            value = value[:-1]
        os.environ[var] = value


def main(git_branch):
    # BASEDIR = os.getcwd()

    coriander_dir = join(os.environ['HOME'], 'coriander')

    maybe_rmtree(coriander_dir)

    run(['python2', 'install_distro.py', '--git-branch', git_branch])

    run(['hg', 'clone', 'https://bitbucket.org/hughperkins/eigen', '-b', 'tf-coriander'])
    # mkdir build
    EIGEN_HOME = join(os.getcwd(), 'eigen')
    cd('build')
    run(['cmake', '-DEIGEN_TESTS=ON', f'-DEIGEN_HOME={EIGEN_HOME}', '..'])
    run(['make', '-j', '16'])
    so_suffix = '.so'
    if platform.uname()[0] == 'Darwin':
        so_suffix = '.dylib'
    elif platform.uname()[0] == 'Windows':
        so_suffix = '.dll'
    artifacts_list = [
        f'lib{libname}{so_suffix}'
        for libname in ['cocl', 'clew', 'clblast', 'easycl']
    ]
    run(['zip', '../artifacts.zip'] + artifacts_list)
    run(['make', '-j', '16', 'gtest-tests'])
    run(['make', '-j', '16', 'endtoend-tests'])
    run(['make', '-j', '16', 'eigen-tests'])
    run(['make', 'run-gtest-tests'])
    run(['make', 'run-endtoend-tests'])
    run(['make', 'run-eigen-tests'])

    activate(join(coriander_dir, 'activate'))
    for plugin in ['coriander-clblast', 'coriander-dnn']:
        cd(join(coriander_dir, 'git', plugin, 'test'))
        mkdir('build')
        cd('build')
        run(['cmake', '..'])
        run(['cmake', '--build', '.'])
        run(['cmake', '--build', '.', '--target', 'tests'])
        run(['cmake', '--build', '.', '--target', 'run-tests'])

    cd_repo_root()
    run(['git', 'clone', '--recursive', 'https://github.com/hughperkins/cudnn-training', '-b', git_branch])
    cd('cudnn-training')
    mkdir('build')
    cd('build')
    run(['cmake', '..', '-DUSE_CUDA=OFF', '-DUSE_OPENCL=ON'])
    run(['cmake', '--build', '.'])

    wget('http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz')
    wget('http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz')
    wget('http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz')
    wget('http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz')
    gunzip('train-images-idx3-ubyte.gz')
    gunzip('train-labels-idx1-ubyte.gz')
    gunzip('t10k-images-idx3-ubyte.gz')
    gunzip('t10k-labels-idx1-ubyte.gz')

    run_until(['./lenet'], until='Training iter 2')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--git-branch', type=str, default='master', help='mostly affects plugins')
    args = parser.parse_args()
    main(**args.__dict__)
