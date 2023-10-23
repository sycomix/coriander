#!/usr/bin/env python
import subprocess
import argparse
import sys
import os
import platform
from os import path
from os.path import join


from cocl_env import CLANG_HOME, COCL_LIB, COCL_INCLUDE, COCL_INSTALL_PREFIX

# SCRIPT_DIR = path.dirname(path.realpath(__file__))
# CORIANDER_DIR = path.dirname(SCRIPT_DIR)
# CORIANDER_DIR = '/usr/local'
# print('CORIANDER_DIR', CORIANDER_DIR)


print('CLANG_HOME', CLANG_HOME)
print('COCL_INSTALL_PREFIX', COCL_INSTALL_PREFIX)
print('COCL_INCLUDE', COCL_INCLUDE)
print('COCL_LIB', COCL_LIB)


def check_output(cmdlist, cwd=None):
    res = subprocess.check_output(cmdlist, cwd=cwd)
    if int(platform.python_version_tuple()[0]) != 2:
        res = res.decode('utf-8')
    print(res)
    return res


def check_folder_writable(target):
    wrote_ok = False
    try:
        writecheck_path = join(target, 'writecheck.txt')
        with open(writecheck_path, 'w') as f:
            f.write('test')
        os.unlink(writecheck_path)
        wrote_ok = True
    except:
        pass
    if not wrote_ok:
        print(f'Please ensure the directory {target} is writable')
        sys.exit(-1)


def check_folder_exists(target):
    if not path.isdir(target):
        os.makedirs(target)


def run_install_checks():
    """
    We need non-sudo write access into `{CORIANDER_DIR}/include/coriander_plugins`, and 
    `{CORIANDER_DIR}/lib/coriander_plugins`
    """
    INCLUDES_DIR = join(COCL_INCLUDE, 'coriander_plugins')
    check_folder_exists(INCLUDES_DIR)
    check_folder_writable(INCLUDES_DIR)

    LIB_DIR = join(COCL_LIB, 'coriander_plugins')
    check_folder_exists(LIB_DIR)
    check_folder_writable(LIB_DIR)


def install(repo_url, git_branch):
    run_install_checks()

    # tmpdir = '/tmp/coriander_clone'
    git_dir = join(COCL_INSTALL_PREFIX, 'git')
    plugin_name = repo_url.split('/')[-1].split('.')[0]
    print(f'plugin_name [{plugin_name}]')
    plugin_git_dir = join(git_dir, plugin_name)
    if path.isdir(plugin_git_dir):
        print(check_output(['rm', '-Rf', plugin_git_dir]))
    os.makedirs(plugin_git_dir)
    print(check_output([
        'git', 'clone', '--recursive', repo_url, '-b', git_branch, plugin_name
    ], cwd=git_dir))
    repo_dir = plugin_git_dir
    # repo_dir = join(tmpdir, os.listdir(tmpdir)[0])
    print('repo_dir', repo_dir)
    build_dir = join(repo_dir, 'build')
    os.makedirs(build_dir)
    print(
        check_output(
            [
                'cmake',
                '..',
                f'-DCORIANDER_DIR={COCL_INSTALL_PREFIX}',
                f'-DCMAKE_INSTALL_PREFIX={COCL_INSTALL_PREFIX}',
                f'-DCLANG_HOME={CLANG_HOME}',
            ],
            cwd=build_dir,
        )
    )
    print(check_output([
        'make', '-j', '8'
    ], cwd=build_dir))
    print(check_output([
        'make', 'install'
    ], cwd=build_dir))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()

    parser_ = subparsers.add_parser('install')
    parser_.add_argument('--repo-url', type=str, required=True, help='eg: https://github.com/hughperkins/coriander-dnn')
    parser_.add_argument('--git-branch', type=str, default='master', help='for developers/maintainers')
    parser_.set_defaults(func=install)

    args = parser.parse_args()
    if 'func' not in args.__dict__:
        print('please choose a function, eg install')
    else:
        func = args.func
        args_dict = args.__dict__
        del args_dict['func']
        func(**args_dict)
