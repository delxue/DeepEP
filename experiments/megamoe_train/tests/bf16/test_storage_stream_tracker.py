# Copyright 2026 Huawei Technologies Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Compile and execute storage-provenance contracts without initializing an NPU."""

import os
from pathlib import Path
import shlex
import subprocess

import torch
from torch.utils.cpp_extension import include_paths, library_paths


def test_storage_stream_tracker(tmp_path):
    root = Path(__file__).resolve().parents[2]
    binary = tmp_path / "storage_stream_tracker"
    libraries = library_paths(device_type="cpu")
    command = [
        *shlex.split(os.environ.get("CXX", "c++")),
        "-std=c++17",
        "-O2",
        "-UNDEBUG",
        f"-D_GLIBCXX_USE_CXX11_ABI={int(torch._C._GLIBCXX_USE_CXX11_ABI)}",
        *[f"-I{path}" for path in include_paths(device_type="cpu")],
        f"-I{root / 'csrc/bf16'}",
        str(Path(__file__).with_suffix(".cpp")),
        *[f"-L{path}" for path in libraries],
        *[f"-Wl,-rpath,{path}" for path in libraries],
        "-ltorch_cpu",
        "-lc10",
        "-o",
        str(binary),
    ]
    subprocess.run(command, check=True)
    subprocess.run([str(binary)], check=True)
