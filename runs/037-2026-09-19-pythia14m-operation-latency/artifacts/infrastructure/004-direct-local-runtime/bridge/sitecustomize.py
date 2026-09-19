"""Keep compiler paths identical while importing the byte-identical local venv."""
import torch.utils.cpp_extension as cpp

cpp._TORCH_PATH = '/workspace/run037/runtime/venv/lib/python3.12/site-packages/torch'
cpp.TORCH_LIB_PATH = cpp._TORCH_PATH + '/lib'
