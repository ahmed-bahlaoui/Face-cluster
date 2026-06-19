import os
import site
from pathlib import Path

# Add every folder that contains NVIDIA DLLs.
# This catches paths like:
# .venv/Lib/site-packages/nvidia/cu13/bin/x86_64
# .venv/Lib/site-packages/nvidia/cudnn/bin
added = set()

for site_dir in site.getsitepackages():
    nvidia_dir = Path(site_dir) / "nvidia"

    if not nvidia_dir.exists():
        continue

    for dll in nvidia_dir.rglob("*.dll"):
        dll_dir = str(dll.parent)

        if dll_dir not in added:
            os.add_dll_directory(dll_dir)
            added.add(dll_dir)
            print("Added DLL dir:", dll_dir)

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper

print("ONNX Runtime:", ort.__version__)
print("Available providers:", ort.get_available_providers())

# Preload ORT CUDA/cuDNN/MSVC DLLs.
ort.preload_dlls(directory="")
ort.print_debug_info()

# Tiny ONNX model: y = x @ I
x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 4])
y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 4])

w = helper.make_tensor(
    name="w",
    data_type=TensorProto.FLOAT,
    dims=[4, 4],
    vals=np.eye(4, dtype=np.float32).flatten().tolist(),
)

node = helper.make_node("MatMul", ["x", "w"], ["y"])

# Correct argument order:
# make_graph(nodes, name, inputs, outputs, initializer)
graph = helper.make_graph(
    [node],
    "cuda_test_graph",
    [x],
    [y],
    [w],
)

model = helper.make_model(
    graph,
    opset_imports=[helper.make_opsetid("", 17)],
)

model.ir_version = 10
onnx.save(model, "cuda_test.onnx")

session = ort.InferenceSession(
    "cuda_test.onnx",
    providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
)

print("Session providers:", session.get_providers())

result = session.run(
    None,
    {"x": np.array([[1, 2, 3, 4]], dtype=np.float32)},
)

print("Result:", result[0])

if "CUDAExecutionProvider" in session.get_providers():
    print("CUDAExecutionProvider is working.")
else:
    print("CUDAExecutionProvider was not used.")