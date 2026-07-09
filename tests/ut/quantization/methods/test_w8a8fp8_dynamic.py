from unittest.mock import MagicMock, Mock, patch

import torch

from tests.ut.base import TestBase
from tests.ut.quantization.conftest_quantization import (
    create_mock_ascend_config,
    create_mock_vllm_config,
)
from vllm_ascend.ascend_forward_context import MoECommType
from vllm_ascend.quantization.methods.w8a8fp8_dynamic import (
    AscendW8A8FP8DynamicFusedMoEMethod,
    AscendW8A8FP8DynamicLinearMethod,
)


class TestAscendW8A8FP8DynamicLinearMethod(TestBase):
    def setUp(self):
        self.method = AscendW8A8FP8DynamicLinearMethod()

    def test_act_quant_type(self):
        self.assertEqual(self.method.act_quant_type, torch.float8_e4m3fn)

    def test_get_weight_various_sizes(self):
        sizes = [(64, 128), (256, 512), (1024, 2048)]
        for input_size, output_size in sizes:
            weight = self.method.get_weight(input_size, output_size, torch.bfloat16)
            self.assertEqual(weight["weight"].dtype, torch.float8_e4m3fn)
            self.assertEqual(weight["weight"].shape, (output_size, input_size))

    def test_get_perchannel_param_dtype_variations(self):
        dtypes = [torch.bfloat16, torch.float16]
        for dtype in dtypes:
            params = self.method.get_perchannel_param(128, dtype)
            self.assertEqual(params["weight_scale"].dtype, torch.float32)
            self.assertEqual(params["weight_offset"].dtype, dtype)
            self.assertEqual(params["weight_scale"].shape, (128, 1))
            self.assertEqual(params["weight_offset"].shape, (128, 1))

    def test_block_fp8_registers_scale_inv_as_group_param(self):
        method = AscendW8A8FP8DynamicLinearMethod({"weight_block_size": [128, 128]})

        self.assertTrue(method.has_block_scale_inv)
        self.assertEqual(method.get_perchannel_param(256, torch.bfloat16), {})
        params = method.get_pergroup_param(
            input_size=2048,
            output_size=768,
            params_dtype=torch.bfloat16,
        )

        self.assertEqual(params["weight_scale_inv"].shape, (6, 16))
        self.assertEqual(params["weight_scale_inv"].dtype, torch.bfloat16)
        self.assertEqual(params["_packed_dim"], 0)
        self.assertEqual(params["_packed_factor"], 128)

    def test_block_fp8_process_converts_scale_inv_to_dequant_scale(self):
        method = AscendW8A8FP8DynamicLinearMethod({"weight_block_size": [128, 128]})
        layer = torch.nn.Module()
        layer.weight = torch.nn.Parameter(torch.zeros(768, 2048, dtype=torch.float8_e4m3fn), requires_grad=False)
        layer.weight_scale_inv = torch.nn.Parameter(
            torch.full((6, 16), 2.0, dtype=torch.bfloat16),
            requires_grad=False,
        )

        method.process_weights_after_loading(layer)

        self.assertEqual(layer.weight.shape, (2048, 768))
        self.assertEqual(layer.weight_scale.shape, (6, 16))
        self.assertTrue(torch.equal(layer.weight_scale, torch.full((6, 16), 0.5, dtype=torch.float32)))


class TestAscendW8A8FP8FusedMoEMethod(TestBase):
    num_experts = 8
    hidden_size = 128
    intermediate_size = 128

    @patch("torch.distributed.get_rank")
    @patch("vllm_ascend.quantization.methods.w8a8_dynamic.get_mc2_group")
    @patch("vllm_ascend.quantization.methods.w8a8_dynamic.get_ascend_config")
    def setUp(self, mock_ascend, mock_mc2, mock_rank):
        with patch("vllm_ascend.quantization.methods.w8a8_dynamic.get_current_vllm_config") as mock_vllm:
            mock_vllm.return_value = create_mock_vllm_config()
            mock_ascend.return_value = create_mock_ascend_config()
            mock_mc2.return_value = MagicMock(
                device_group=Mock(
                    _get_backend=Mock(return_value=Mock(get_hccl_comm_name=Mock(return_value="test_comm")))
                )
            )
            mock_rank.return_value = 0
            self.quant_method = AscendW8A8FP8DynamicFusedMoEMethod()

    def test_quant_type_is_w8a8fp8(self):
        from vllm_ascend.quantization.quant_type import QuantType

        self.assertEqual(self.quant_method.quant_type, QuantType.W8A8FP8)

    def test_get_weight_dtype_is_float8_e4m3fn(self):
        param_dict = self.quant_method.get_weight(
            self.num_experts, self.intermediate_size, self.hidden_size, torch.bfloat16
        )
        self.assertEqual(param_dict["w13_weight"].dtype, torch.float8_e4m3fn)
        self.assertEqual(param_dict["w2_weight"].dtype, torch.float8_e4m3fn)
        self.assertEqual(
            param_dict["w13_weight"].shape, (self.num_experts, 2 * self.intermediate_size, self.hidden_size)
        )
        self.assertEqual(param_dict["w2_weight"].shape, (self.num_experts, self.hidden_size, self.intermediate_size))

    def test_get_weight_various_expert_counts(self):
        expert_counts = [4, 8, 16, 32]
        for num_experts in expert_counts:
            param_dict = self.quant_method.get_weight(
                num_experts, self.intermediate_size, self.hidden_size, torch.bfloat16
            )
            self.assertEqual(param_dict["w13_weight"].shape[0], num_experts)
            self.assertEqual(param_dict["w2_weight"].shape[0], num_experts)

    def test_block_fp8_registers_moe_scale_inv_params(self):
        method = AscendW8A8FP8DynamicFusedMoEMethod({"weight_block_size": [128, 128]})

        self.assertTrue(method.has_block_scale_inv)
        param_dict = method.get_dynamic_quant_param(
            num_experts=128,
            intermediate_size_per_partition=768,
            hidden_sizes=2048,
            params_dtype=torch.bfloat16,
        )

        self.assertEqual(param_dict["w13_weight_scale_inv"].shape, (128, 12, 16))
        self.assertEqual(param_dict["w2_weight_scale_inv"].shape, (128, 16, 6))
        self.assertEqual(param_dict["w13_weight_scale_inv"].dtype, torch.bfloat16)
        self.assertEqual(param_dict["w2_weight_scale_inv"].dtype, torch.bfloat16)

    def test_block_fp8_process_converts_moe_scale_inv_to_dequant_scale(self):
        method = AscendW8A8FP8DynamicFusedMoEMethod({"weight_block_size": [128, 128]})
        layer = torch.nn.Module()
        layer.w13_weight = torch.nn.Parameter(
            torch.zeros(2, 256, 128, dtype=torch.float8_e4m3fn),
            requires_grad=False,
        )
        layer.w2_weight = torch.nn.Parameter(
            torch.zeros(2, 128, 128, dtype=torch.float8_e4m3fn),
            requires_grad=False,
        )
        layer.w13_weight_scale_inv = torch.nn.Parameter(
            torch.full((2, 2, 1), 4.0, dtype=torch.bfloat16),
            requires_grad=False,
        )
        layer.w2_weight_scale_inv = torch.nn.Parameter(
            torch.full((2, 1, 1), 2.0, dtype=torch.bfloat16),
            requires_grad=False,
        )

        method.process_weights_after_loading(layer)

        self.assertEqual(layer.w13_weight.shape, (2, 128, 256))
        self.assertEqual(layer.w2_weight.shape, (2, 128, 128))
        self.assertEqual(layer.w13_weight_scale.shape, (2, 1, 2))
        self.assertEqual(layer.w2_weight_scale.shape, (2, 1, 1))
        self.assertTrue(torch.equal(layer.w13_weight_scale, torch.full((2, 1, 2), 0.25, dtype=torch.float32)))
        self.assertTrue(torch.equal(layer.w2_weight_scale, torch.full((2, 1, 1), 0.5, dtype=torch.float32)))

    @patch("vllm_ascend.quantization.methods.w8a8_dynamic._EXTRA_CTX")
    @patch("vllm_ascend.quantization.methods.w8a8_dynamic.select_experts")
    def test_apply_uses_explicit_dispatch_and_mlp_args(self, mock_select_experts, mock_extra_ctx):
        tokens = 4
        hidden_size = self.hidden_size
        layer = torch.nn.Module()
        layer.w13_weight = torch.randn(
            self.num_experts, 2 * self.intermediate_size, hidden_size, dtype=torch.bfloat16
        ).to(torch.float8_e4m3fn)
        layer.w2_weight = torch.randn(self.num_experts, hidden_size, self.intermediate_size, dtype=torch.bfloat16).to(
            torch.float8_e4m3fn
        )
        layer.w13_weight_scale_fp32 = torch.ones(self.num_experts, 2 * self.intermediate_size, dtype=torch.float32)
        layer.w2_weight_scale = torch.ones(self.num_experts, hidden_size, dtype=torch.float32)
        layer.swiglu_limit = 1000000

        x = torch.randn(tokens, hidden_size, dtype=torch.float32)
        router_logits = torch.randn(tokens, self.num_experts, dtype=torch.float32)
        topk_weights = torch.randn(tokens, 2, dtype=torch.float32)
        topk_ids = torch.randint(0, self.num_experts, (tokens, 2), dtype=torch.int64)
        mc2_mask = torch.tensor([1, 0, 1, 0], dtype=torch.bool)
        pertoken_scale = torch.randn(tokens, dtype=torch.float32)

        mock_select_experts.return_value = (topk_weights, topk_ids)
        mock_comm = Mock()
        mock_comm.fused_experts.return_value = torch.randn(tokens, hidden_size, dtype=torch.float32)
        mock_extra_ctx.moe_comm_method = mock_comm
        mock_extra_ctx.moe_comm_type = MoECommType.ALLGATHER
        self.quant_method.multistream_overlap_gate = False
        self.quant_method.in_dtype = torch.float32

        self.quant_method.apply(
            layer=layer,
            x=x,
            router_logits=router_logits,
            top_k=2,
            renormalize=True,
            num_experts=self.num_experts,
            activation="gelu",
            apply_router_weight_on_input=True,
            mc2_mask=mc2_mask,
            pertoken_scale=pertoken_scale,
        )

        fused_experts_input = mock_comm.fused_experts.call_args.kwargs["fused_experts_input"]
        self.assertEqual(fused_experts_input.activation, "gelu")
        self.assertTrue(fused_experts_input.routing.apply_router_weight_on_input)
        self.assertIs(fused_experts_input.routing.mc2_mask, mc2_mask)
        self.assertIs(fused_experts_input.routing.pertoken_scale, pertoken_scale)
        self.assertIs(fused_experts_input.topk_weights, topk_weights)
        self.assertIs(fused_experts_input.topk_ids, topk_ids)
