import os
import glob
import torch
from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension

import torch_npu
from torch_npu.utils.cpp_extension import NpuExtension

PYTORCH_NPU_INSTALL_PATH = os.path.dirname(os.path.abspath(torch_npu.__file__))
USE_NINJA = os.getenv('USE_NINJA') == '1'
BASE_DIR = os.path.dirname(os.path.realpath(__file__))

source_files = [os.path.join(BASE_DIR, name) for name in (
    ["SmlaVaOps.cpp"] if os.getenv("XLLM_OPS_TEST_OP") == "smla_va"
    else ["RegisterOps.cpp", "SmlaVaOps.cpp"]
)]

exts = []
ext = NpuExtension(
    name="custom_ops_lib",
    sources=source_files,
    extra_compile_args = (["-DXLLM_OPS_SMLA_TEST_MODULE"] if os.getenv("XLLM_OPS_TEST_OP") == "smla_va" else []) + [
        '-I' + os.path.join(PYTORCH_NPU_INSTALL_PATH, "include/third_party/acl/inc"),
    ],
)
exts.append(ext)

setup(
    name="custom_ops",
    version='1.0',
    keywords='custom_ops',
    ext_modules=exts,
    packages=find_packages(),
    cmdclass={"build_ext": BuildExtension.with_options(use_ninja=USE_NINJA)},
)
