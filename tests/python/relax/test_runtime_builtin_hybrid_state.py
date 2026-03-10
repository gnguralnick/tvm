# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
"""Tests for HybridState — validates delegation to inner PagedAttentionKVCache and RNNState."""
from collections.abc import Sequence

import numpy as np
import pytest

import tvm
import tvm.testing
from tvm import tir
from tvm.relax.frontend.nn.llm.kv_cache import (
    AttnKind,
    RopeMode,
    _attention_decode_cpu,
    _attention_prefill_cpu,
    _attention_prefill_ragged_cpu,
    _compact_kv_copy_cpu,
    _copy_single_page_cpu,
    _kv_cache_debug_get_kv,
    _kv_cache_transpose_append,
    _merge_state_inplace_cpu,
    llama_rope_with_position_map,
    tree_attn_cpu,
    tree_attn_with_paged_kv_cache_cpu,
)
from tvm.runtime import ShapeTuple
from tvm.s_tir import dlight as dl
from tvm.script import tir as T

# pylint: disable=invalid-name

# ---------- shared constants ----------
device = tvm.cpu()
reserved_nseq = 4
max_history = 4

# KV cache parameters (kept small for test speed)
kv_num_layers = 1
num_qo_heads = 4
num_kv_heads = 2
head_dim = 64
page_size = 16
maximum_total_seq_length = 256
prefill_chunk_size = 128
rope_scale = 1.0
rope_theta = 1e4
rope_scaling = {}
kv_dtype = "float32"

# RNN state parameters
rnn_num_layers = 1
rnn_states = [((8, 8), "float32")]
np_rnn_init = np.zeros((8, 8), dtype="float32")
np_rnn_two = np.full((8, 8), 2.0, dtype="float32")

# ---------- built function handles (populated by setup) ----------
f_clear = None
f_add_sequence = None
f_remove_sequence = None
f_begin_forward = None
f_end_forward = None
f_is_empty = None
f_hybrid_create = None
f_hybrid_rnn_get = None
f_hybrid_rnn_set = None
f_rnn_debug_get = None

# KV cache TIR functions
_kv_tir_funcs = {}

# RNN TIR functions
f_tir_gets = []
f_tir_sets = []


# ---------- RNN TIR kernel builders (from test_runtime_builtin_rnn_state.py) ----------
def rnn_state_get(shape: Sequence[int], dtype: str):
    # fmt: off
    @T.prim_func
    def _rnn_state_get(
        var_storage: T.handle,
        var_seq_slot_ids: T.handle,
        var_history_slot_ids: T.handle,
        var_output: T.handle,
    ):
        batch_size = T.int32(is_size_var=True)
        storage = T.match_buffer(var_storage, (reserved_nseq, max_history, *shape), dtype)
        seq_slot_ids = T.match_buffer(var_seq_slot_ids, (batch_size,), "int32")
        history_slot_ids = T.match_buffer(var_history_slot_ids, (batch_size,), "int32")
        output = T.match_buffer(var_output, (batch_size, *shape), dtype)
        for i in range(batch_size):
            for s in T.grid(*shape):
                with T.sblock("copy"):
                    vi, *vs = T.axis.remap("S" * (len(shape) + 1), [i, *s])
                    seq_id: T.int32 = seq_slot_ids[vi]
                    history_id: T.int32 = history_slot_ids[vi]
                    T.buffer_store(
                        output, T.BufferLoad(storage, [seq_id, history_id, *vs]), [vi, *vs]
                    )
    # fmt: on
    return _rnn_state_get


def rnn_state_set(shape: Sequence[int | tir.Var], dtype: str):
    # fmt: off
    @T.prim_func
    def _rnn_state_set(
        var_storage: T.handle,
        var_seq_slot_ids: T.handle,
        var_history_slot_ids: T.handle,
        var_data: T.handle,
    ):
        batch_size = T.int32(is_size_var=True)
        storage = T.match_buffer(var_storage, (reserved_nseq, max_history, *shape), dtype)
        seq_slot_ids = T.match_buffer(var_seq_slot_ids, (batch_size,), "int32")
        history_slot_ids = T.match_buffer(var_history_slot_ids, (batch_size,), "int32")
        data = T.match_buffer(var_data, (batch_size, *shape), dtype)
        for i in range(batch_size):
            for s in T.grid(*shape):
                with T.sblock("copy"):
                    vi, *vs = T.axis.remap("S" * (len(shape) + 1), [i, *s])
                    seq_id: T.int32 = seq_slot_ids[vi]
                    history_id: T.int32 = (history_slot_ids[vi] + 1) % T.cast(
                        max_history, "int32"
                    )
                    T.buffer_store(
                        storage, T.BufferLoad(data, [vi, *vs]), [seq_id, history_id, *vs]
                    )
    # fmt: on
    return _rnn_state_set


# ---------- setup ----------
def _build_tir(tir_func, target):
    mod = tvm.IRModule({"main": tir_func})
    with target:
        mod = dl.ApplyDefaultSchedule(dl.gpu.Fallback())(mod)
    f = tvm.tir.build(mod["main"], target=target)
    return f.main


def setup_module():
    global f_clear, f_add_sequence, f_remove_sequence
    global f_begin_forward, f_end_forward, f_is_empty
    global f_hybrid_create, f_hybrid_rnn_get, f_hybrid_rnn_set, f_rnn_debug_get
    global _kv_tir_funcs, f_tir_gets, f_tir_sets

    f_clear = tvm.get_global_func("vm.builtin.kv_state_clear")
    f_add_sequence = tvm.get_global_func("vm.builtin.kv_state_add_sequence")
    f_remove_sequence = tvm.get_global_func("vm.builtin.kv_state_remove_sequence")
    f_begin_forward = tvm.get_global_func("vm.builtin.kv_state_begin_forward")
    f_end_forward = tvm.get_global_func("vm.builtin.kv_state_end_forward")
    f_is_empty = tvm.get_global_func("vm.builtin.attention_kv_cache_empty")
    f_hybrid_create = tvm.get_global_func("vm.builtin.hybrid_state_create")
    f_hybrid_rnn_get = tvm.get_global_func("vm.builtin.hybrid_state_rnn_get")
    f_hybrid_rnn_set = tvm.get_global_func("vm.builtin.hybrid_state_rnn_set")
    f_rnn_debug_get = tvm.get_global_func("vm.builtin.rnn_state_debug_get")

    target = tvm.target.Target.from_device(device)

    # Build KV cache TIR kernels (mirrors test_runtime_builtin_paged_attention_kv_cache_cpu.py)
    kv_tir_sources = [
        _kv_cache_transpose_append(num_kv_heads, head_dim, kv_dtype),
        _kv_cache_debug_get_kv(kv_num_layers, num_kv_heads, head_dim, kv_dtype),
        _attention_prefill_cpu(num_kv_heads, num_qo_heads, head_dim, kv_dtype, False, rope_scaling),
        _attention_decode_cpu(num_kv_heads, num_qo_heads, head_dim, kv_dtype, False, rope_scaling),
        _attention_prefill_cpu(num_kv_heads, num_qo_heads, head_dim, kv_dtype, True, rope_scaling),
        _attention_decode_cpu(num_kv_heads, num_qo_heads, head_dim, kv_dtype, True, rope_scaling),
        _attention_prefill_ragged_cpu(
            num_kv_heads, num_qo_heads, head_dim, head_dim, kv_dtype, rope_scaling
        ),
        tree_attn_cpu(num_kv_heads, num_qo_heads, head_dim, kv_dtype, rope_scaling),
        tree_attn_with_paged_kv_cache_cpu(
            num_kv_heads, num_qo_heads, head_dim, kv_dtype, rope_scaling
        ),
        _merge_state_inplace_cpu(kv_dtype),
        llama_rope_with_position_map(
            rope_theta, rope_scale, head_dim, num_qo_heads, num_kv_heads, kv_dtype, rope_scaling
        ),
        _copy_single_page_cpu(num_kv_heads, page_size, head_dim, kv_dtype),
        _compact_kv_copy_cpu(num_kv_heads, head_dim, kv_dtype),
    ]
    built = [_build_tir(f, target) for f in kv_tir_sources]
    _kv_tir_funcs = {
        "transpose_append": built[0],
        "copy_cache": built[1],
        "attn_prefill": built[2],
        "attn_decode": built[3],
        "attn_prefill_sw": built[4],
        "attn_decode_sw": built[5],
        "attn_prefill_ragged": built[6],
        "tree_attn": built[7],
        "tree_attn_paged": built[8],
        "merge_state": built[9],
        "split_rotary": built[10],
        "copy_single_page": built[11],
        "compact_copy": built[12],
    }

    # Build RNN TIR kernels
    _gets, _sets = [], []
    for shape, dtype in rnn_states:
        _gets.append(_build_tir(rnn_state_get(shape, dtype), target))
        _sets.append(_build_tir(rnn_state_set(shape, dtype), target))
    f_tir_gets = _gets
    f_tir_sets = _sets


def _create_kv_cache():
    fcreate = tvm.get_global_func("vm.builtin.paged_attention_kv_cache_create")
    t = _kv_tir_funcs
    return fcreate(
        ShapeTuple([
            reserved_nseq,
            maximum_total_seq_length,
            prefill_chunk_size,
            page_size,
            0,  # no sliding window
        ]),
        ShapeTuple([0, kv_num_layers]),
        num_qo_heads,
        num_kv_heads,
        head_dim,
        head_dim,
        ShapeTuple([int(AttnKind.MHA)] * kv_num_layers),
        False,  # enable_kv_transfer
        int(RopeMode.NONE),
        rope_scale,
        rope_theta,
        None,  # rope_ext_factors
        tvm.runtime.empty((), kv_dtype, device=device),
        t["transpose_append"],
        None,  # f_transpose_append_mla
        ["tir", t["attn_prefill_ragged"]],
        ["tir", t["attn_prefill"]],
        ["tir", t["attn_decode"]],
        ["tir", t["attn_prefill_sw"]],
        ["tir", t["attn_decode_sw"]],
        ["tir", t["tree_attn_paged"]],
        ["tir", t["tree_attn"]],
        [],  # f_mla_prefill
        [t["merge_state"]],
        t["split_rotary"],
        t["copy_single_page"],
        t["copy_cache"],
        t["compact_copy"],
    )


def _create_rnn_state():
    fcreate = tvm.get_global_func("vm.builtin.rnn_state_create")
    init_values = [tvm.runtime.tensor(np_rnn_init, device=device)]
    return fcreate(rnn_num_layers, reserved_nseq, max_history, f_tir_gets, f_tir_sets, init_values)


@pytest.fixture
def hybrid_state():
    kv_cache = _create_kv_cache()
    rnn = _create_rnn_state()
    return f_hybrid_create(kv_cache, rnn)


@pytest.fixture
def rnn_state_standalone():
    """A standalone RNNState for verifying inner state via debug_get."""
    return _create_rnn_state()


# ---------- tests ----------


def test_hybrid_state_rnn_get_set(hybrid_state):
    """Verify RNN get/set through the hybrid wrapper, which also validates
    that add_sequence/begin_forward/end_forward propagate to the inner RNN state."""
    state = hybrid_state
    f_clear(state)
    f_add_sequence(state, 0)

    # Write via hybrid_state_rnn_set
    f_begin_forward(state, ShapeTuple([0]), ShapeTuple([1]))
    f_hybrid_rnn_set(
        state, 0, 0, tvm.runtime.tensor(np_rnn_two.reshape(1, 8, 8), device=device)
    )
    f_end_forward(state)

    # Read back via hybrid_state_rnn_get
    f_begin_forward(state, ShapeTuple([0]), ShapeTuple([1]))
    out = tvm.runtime.tensor(np.empty((1, 8, 8), "float32"), device=device)
    f_hybrid_rnn_get(state, 0, 0, out)
    f_end_forward(state)

    tvm.testing.assert_allclose(out.numpy(), np_rnn_two.reshape(1, 8, 8))


def test_hybrid_state_clear_propagates(hybrid_state):
    """Verify that Clear on hybrid state resets the inner RNN state."""
    state = hybrid_state
    f_clear(state)
    f_add_sequence(state, 0)

    # Set a non-default value
    f_begin_forward(state, ShapeTuple([0]), ShapeTuple([1]))
    f_hybrid_rnn_set(
        state, 0, 0, tvm.runtime.tensor(np_rnn_two.reshape(1, 8, 8), device=device)
    )
    f_end_forward(state)

    # Clear the hybrid state — should reset RNN state
    f_clear(state)
    f_add_sequence(state, 0)

    # Read back — should be the init value (zeros)
    f_begin_forward(state, ShapeTuple([0]), ShapeTuple([1]))
    out = tvm.runtime.tensor(np.empty((1, 8, 8), "float32"), device=device)
    f_hybrid_rnn_get(state, 0, 0, out)
    f_end_forward(state)

    tvm.testing.assert_allclose(out.numpy(), np.zeros((1, 8, 8), "float32"))


def test_hybrid_state_kv_cache_empty(hybrid_state):
    """Verify that AttentionKVCache.Empty() delegates to the inner KV cache."""
    state = hybrid_state
    f_clear(state)
    assert f_is_empty(state), "Fresh hybrid state should report empty via inner KV cache"


def test_hybrid_state_sequence_ops_propagate(hybrid_state):
    """Verify add/remove sequence propagates to both sides by checking
    that the KV cache and RNN state both accept subsequent operations."""
    state = hybrid_state
    f_clear(state)

    # Add sequences to both inner objects via hybrid
    f_add_sequence(state, 0)
    f_add_sequence(state, 1)

    # Remove one — should propagate to both
    f_remove_sequence(state, 1)

    # Operations on seq 0 should still work for both KV and RNN sides
    f_begin_forward(state, ShapeTuple([0]), ShapeTuple([1]))
    f_hybrid_rnn_set(
        state, 0, 0, tvm.runtime.tensor(np_rnn_two.reshape(1, 8, 8), device=device)
    )
    f_end_forward(state)

    # Verify RNN side still works
    f_begin_forward(state, ShapeTuple([0]), ShapeTuple([1]))
    out = tvm.runtime.tensor(np.empty((1, 8, 8), "float32"), device=device)
    f_hybrid_rnn_get(state, 0, 0, out)
    f_end_forward(state)
    tvm.testing.assert_allclose(out.numpy(), np_rnn_two.reshape(1, 8, 8))

    # Verify KV side still works (not empty after adding a sequence)
    assert not f_is_empty(state)


if __name__ == "__main__":
    setup_module()
    hs = f_hybrid_create(_create_kv_cache(), _create_rnn_state())
    test_hybrid_state_rnn_get_set(hs)
    hs = f_hybrid_create(_create_kv_cache(), _create_rnn_state())
    test_hybrid_state_clear_propagates(hs)
    hs = f_hybrid_create(_create_kv_cache(), _create_rnn_state())
    test_hybrid_state_kv_cache_empty(hs)
    hs = f_hybrid_create(_create_kv_cache(), _create_rnn_state())
    test_hybrid_state_sequence_ops_propagate(hs)
    print("All tests passed!")
