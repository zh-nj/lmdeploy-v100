// Copyright (c) OpenMMLab. All rights reserved.

#include "src/turbomind/models/llama/gated_delta_net_weight.h"

namespace turbomind {

GatedDeltaNetWeight::GatedDeltaNetWeight(int                       hidden_size,
                                         const GatedDeltaNetParam& gdn,
                                         int                       tp_size,
                                         int                       tp_rank,
                                         DataType                  data_type)
{
    const int key_dim   = gdn.num_key_heads * gdn.key_head_dim;
    const int value_dim = gdn.num_value_heads * gdn.value_head_dim;

    // Merged in_proj_all: [hidden_size, (key_dim*2 + value_dim*2 + num_v_heads*2) / tp]
    // Layout: [Q, K, V, Z, b(beta_raw), a(alpha)]  (QKVZ + BA concatenated)
    const int all_out = (key_dim * 2 + value_dim * 2 + gdn.num_value_heads * 2) / tp_size;
    in_proj_all.emplace(hidden_size, all_out, data_type, false, data_type, 1);
    register_module("in_proj_all", in_proj_all, tp_rank);

    // out_proj: [value_dim / tp, hidden_size]
    out_proj.emplace(value_dim / tp_size, hidden_size, data_type, false, data_type, 1);
    register_module("out_proj", out_proj, tp_rank);

    // conv1d_weight: [conv_dim / tp, 1, kernel_size]
    const int conv_dim = (key_dim * 2 + value_dim) / tp_size;
    conv1d_weight = Tensor{{conv_dim, 1, gdn.conv_kernel_dim}, data_type, kDEVICE};
    register_parameter("conv1d." + std::to_string(tp_rank) + ".weight", conv1d_weight);

    // A_log: [num_v_heads / tp]
    A_log = Tensor{{gdn.num_value_heads / tp_size}, data_type, kDEVICE};
    register_parameter("A_log." + std::to_string(tp_rank), A_log);

    // dt_bias: [num_v_heads / tp]
    dt_bias = Tensor{{gdn.num_value_heads / tp_size}, data_type, kDEVICE};
    register_parameter("dt_bias." + std::to_string(tp_rank), dt_bias);

    // norm_weight: [head_v_dim] - NOT TP-sharded
    norm_weight = Tensor{{gdn.value_head_dim}, data_type, kDEVICE};
    register_parameter("norm.weight", norm_weight);
}

}  // namespace turbomind
