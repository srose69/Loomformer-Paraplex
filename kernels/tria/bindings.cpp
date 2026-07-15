#include <torch/extension.h>
#include <vector>

std::vector<torch::Tensor> tria_init_forward_cuda(
    torch::Tensor r, torch::Tensor i, torch::Tensor o, double alpha, int64_t axis);
std::vector<torch::Tensor> tria_init_gate_forward_cuda(
    torch::Tensor r, torch::Tensor i, torch::Tensor o, torch::Tensor w,
    double alpha, int64_t axis);
std::vector<torch::Tensor> tria_init_seed_forward_cuda(
    torch::Tensor r, torch::Tensor i, torch::Tensor o, torch::Tensor seed,
    torch::Tensor valid, double alpha, int64_t axis);
std::vector<torch::Tensor> tria_init_seed_gate_forward_cuda(
    torch::Tensor r, torch::Tensor i, torch::Tensor o, torch::Tensor seed,
    torch::Tensor valid, torch::Tensor w, double alpha, int64_t axis);
std::vector<torch::Tensor> tria_init_backward_cuda(
    torch::Tensor grad_carry_1, torch::Tensor r, torch::Tensor i, torch::Tensor o,
    torch::Tensor scale, double alpha, int64_t axis);
std::vector<torch::Tensor> tria_init_gate_backward_cuda(
    torch::Tensor grad_carry_1, torch::Tensor grad_p_out,
    torch::Tensor r, torch::Tensor i, torch::Tensor o,
    torch::Tensor w, torch::Tensor scale, double alpha, int64_t axis);
std::vector<torch::Tensor> tria_init_seed_backward_cuda(
    torch::Tensor grad_carry, torch::Tensor r, torch::Tensor i, torch::Tensor o,
    torch::Tensor seed, torch::Tensor valid, double alpha, int64_t axis);
std::vector<torch::Tensor> tria_init_seed_gate_backward_cuda(
    torch::Tensor grad_carry, torch::Tensor grad_p,
    torch::Tensor r, torch::Tensor i, torch::Tensor o,
    torch::Tensor seed, torch::Tensor valid, torch::Tensor w, double alpha, int64_t axis);
std::vector<torch::Tensor> tria_step_forward_cuda(
    torch::Tensor r, torch::Tensor i, torch::Tensor o, torch::Tensor carry_prev,
    double alpha, int64_t axis);
std::vector<torch::Tensor> tria_step_gate_forward_cuda(
    torch::Tensor r, torch::Tensor i, torch::Tensor o, torch::Tensor carry_prev,
    torch::Tensor w, double alpha, int64_t axis);
std::vector<torch::Tensor> tria_step_backward_cuda(
    torch::Tensor grad_carry_new, torch::Tensor r, torch::Tensor i, torch::Tensor o,
    torch::Tensor carry_prev, torch::Tensor scale, double alpha, int64_t axis);
std::vector<torch::Tensor> tria_step_gate_backward_cuda(
    torch::Tensor grad_carry_new, torch::Tensor grad_p_out,
    torch::Tensor r, torch::Tensor i, torch::Tensor o,
    torch::Tensor carry_prev, torch::Tensor w, torch::Tensor scale,
    double alpha, int64_t axis);
std::vector<torch::Tensor> tria_step_reverse_backward_cuda(
    torch::Tensor grad_carry_new, torch::Tensor r, torch::Tensor i, torch::Tensor o,
    torch::Tensor current, double alpha, int64_t axis);
std::vector<torch::Tensor> tria_step_gate_reverse_backward_cuda(
    torch::Tensor grad_carry_new, torch::Tensor grad_p_out,
    torch::Tensor r, torch::Tensor i, torch::Tensor o,
    torch::Tensor current, torch::Tensor w, double alpha, int64_t axis);
torch::Tensor gate_slot_mix_forward_cuda(torch::Tensor carry, torch::Tensor w);
std::vector<torch::Tensor> gate_slot_mix_backward_cuda(
    torch::Tensor grad_p, torch::Tensor carry, torch::Tensor w);
std::vector<torch::Tensor> slot_attention_pool_forward_cuda(
    torch::Tensor carry, torch::Tensor score_w);
std::vector<torch::Tensor> slot_attention_pool_backward_cuda(
    torch::Tensor grad_pooled, torch::Tensor carry, torch::Tensor score_w, torch::Tensor lse);
std::vector<torch::Tensor> temporal_carry_forward_cuda(
    torch::Tensor depth_carry, torch::Tensor reset_mask);
std::vector<torch::Tensor> temporal_carry_endpoint_forward_cuda(
    torch::Tensor depth, torch::Tensor reset, torch::Tensor initial, torch::Tensor initial_valid);
torch::Tensor temporal_carry_backward_cuda(
    torch::Tensor grad_document_carry, torch::Tensor depth_carry,
    torch::Tensor document_carry, torch::Tensor scale, torch::Tensor reset_mask);
std::vector<torch::Tensor> temporal_carry_endpoint_backward_cuda(
    torch::Tensor grad_endpoint, torch::Tensor depth, torch::Tensor endpoint_fp32,
    torch::Tensor reset, torch::Tensor initial, torch::Tensor initial_valid);
std::vector<torch::Tensor> depth_replay_backward_cuda(
    torch::Tensor grad_carry, torch::Tensor r, torch::Tensor i, torch::Tensor o,
    torch::Tensor r_ptrs, torch::Tensor i_ptrs, torch::Tensor o_ptrs,
    torch::Tensor axes, torch::Tensor seed, torch::Tensor seed_valid,
    double alpha, int64_t axis, int64_t layer_index);
std::vector<torch::Tensor> depth_replay_gate_backward_cuda(
    torch::Tensor grad_carry, torch::Tensor grad_p, torch::Tensor r, torch::Tensor i, torch::Tensor o,
    torch::Tensor w, torch::Tensor r_ptrs, torch::Tensor i_ptrs, torch::Tensor o_ptrs,
    torch::Tensor axes, torch::Tensor seed, torch::Tensor seed_valid,
    double alpha, int64_t axis, int64_t layer_index);
std::vector<torch::Tensor> final_ca_sparse_forward_cuda(
    torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor allowed, double scale);
std::vector<torch::Tensor> final_ca_sparse_backward_cuda(
    torch::Tensor grad_out, torch::Tensor q, torch::Tensor k, torch::Tensor v,
    torch::Tensor allowed, torch::Tensor lse, double scale);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("tria_init_forward", &tria_init_forward_cuda);
    m.def("tria_init_gate_forward", &tria_init_gate_forward_cuda);
    m.def("tria_init_seed_forward", &tria_init_seed_forward_cuda);
    m.def("tria_init_seed_gate_forward", &tria_init_seed_gate_forward_cuda);
    m.def("tria_init_backward", &tria_init_backward_cuda);
    m.def("tria_init_gate_backward", &tria_init_gate_backward_cuda);
    m.def("tria_init_seed_backward", &tria_init_seed_backward_cuda);
    m.def("tria_init_seed_gate_backward", &tria_init_seed_gate_backward_cuda);
    m.def("tria_step_forward", &tria_step_forward_cuda);
    m.def("tria_step_gate_forward", &tria_step_gate_forward_cuda);
    m.def("tria_step_backward", &tria_step_backward_cuda);
    m.def("tria_step_gate_backward", &tria_step_gate_backward_cuda);
    m.def("tria_step_reverse_backward", &tria_step_reverse_backward_cuda);
    m.def("tria_step_gate_reverse_backward", &tria_step_gate_reverse_backward_cuda);
    m.def("gate_slot_mix_forward", &gate_slot_mix_forward_cuda);
    m.def("gate_slot_mix_backward", &gate_slot_mix_backward_cuda);
    m.def("slot_attention_pool_forward", &slot_attention_pool_forward_cuda);
    m.def("slot_attention_pool_backward", &slot_attention_pool_backward_cuda);
    m.def("temporal_carry_forward", &temporal_carry_forward_cuda);
    m.def("temporal_carry_backward", &temporal_carry_backward_cuda);
    m.def("temporal_carry_endpoint_forward", &temporal_carry_endpoint_forward_cuda);
    m.def("temporal_carry_endpoint_backward", &temporal_carry_endpoint_backward_cuda);
    m.def("depth_replay_backward", &depth_replay_backward_cuda);
    m.def("depth_replay_gate_backward", &depth_replay_gate_backward_cuda);
    m.def("final_ca_sparse_forward", &final_ca_sparse_forward_cuda);
    m.def("final_ca_sparse_backward", &final_ca_sparse_backward_cuda);
}
