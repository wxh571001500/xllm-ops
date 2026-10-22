#include <torch/library.h>
#include "pytorch_npu_helper.hpp"
#include "xllm_ops/attention/smla_va/smla_va_torch_adpt.h"

TORCH_LIBRARY_FRAGMENT(xllm_ops, ops) {
    ops.def(
        "smla_va_metadata(int num_heads_q, int num_heads_kv, int head_dim, *, "
        "Tensor? cu_seqlens_q=None, Tensor? cu_seqlens_ori_kv=None, Tensor? cu_seqlens_cmp_kv=None, "
        "Tensor? seqused_q=None, Tensor? seqused_ori_kv=None, Tensor? seqused_cmp_kv=None, "
        "Tensor? cmp_residual_kv=None, Tensor? ori_topk_length=None, Tensor? cmp_topk_length=None, "
        "int batch_size=0, int max_seqlen_q=0, int max_seqlen_ori_kv=0, int max_seqlen_cmp_kv=0, "
        "int ori_topk=0, int cmp_topk=0, int cmp_ratio=0, int ori_mask_mode=0, int cmp_mask_mode=0, "
        "int ori_win_left=-1, int ori_win_right=-1, str layout_q='BSND', str layout_kv='BSND', "
        "bool has_ori_kv=True, bool has_cmp_kv=True) -> Tensor"
    );
    ops.impl("smla_va_metadata", torch::kPrivateUse1,
             &xllm_ops::smla_va_metadata);

    ops.def(
        "smla_va(Tensor q, *, Tensor? ori_kv=None, Tensor? cmp_kv=None, "
        "Tensor? ori_sparse_indices=None, Tensor? cmp_sparse_indices=None, "
        "Tensor? ori_block_table=None, Tensor? cmp_block_table=None, "
        "Tensor? cu_seqlens_q=None, Tensor? cu_seqlens_ori_kv=None, Tensor? cu_seqlens_cmp_kv=None, "
        "Tensor? seqused_q=None, Tensor? seqused_ori_kv=None, Tensor? seqused_cmp_kv=None, "
        "Tensor? cmp_residual_kv=None, Tensor? ori_topk_length=None, Tensor? cmp_topk_length=None, "
        "Tensor? sinks=None, Tensor? metadata=None, float softmax_scale=1.0, int cmp_ratio=0, "
        "int ori_mask_mode=0, int cmp_mask_mode=0, int ori_win_left=-1, int ori_win_right=-1, "
        "str layout_q='BSND', str layout_kv='BSND', int topk_value_mode=1, "
        "bool return_softmax_lse=False) -> (Tensor, Tensor)"
    );
    ops.impl("smla_va", torch::kPrivateUse1,
             &xllm_ops::smla_va);
}
