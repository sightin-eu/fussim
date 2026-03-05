// Windows SDK (rpcndr.h) defines 'small' as 'char', which conflicts with
// PyTorch headers that use 'small' as a parameter name.
#ifdef small
#undef small
#endif

#include <torch/types.h>
#include <pybind11/pybind11.h>
#include "ssim.h"
#include "ssim_fp16.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("fusedssim", &fusedssim);
  m.def("fusedssim_backward", &fusedssim_backward);
  m.def("fusedssim_fp16", &fusedssim_fp16);
  m.def("fusedssim_fp16_train", &fusedssim_fp16_train);
  m.def("fusedssim_fp16_backward", &fusedssim_fp16_backward);
}
