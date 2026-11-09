"""Compare independent eager/FlashAttention executions on a real CUDA backend."""
import copy

import pytest
import torch
from transformers.cache_utils import DynamicCache
from transformers.utils import is_flash_attn_2_available

from model_src.config_custom import GPTNeoXConfig
from model_src import modeling_custom as implementation

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or not is_flash_attn_2_available(),
    reason='These tests require a CUDA GPU and the real flash-attn package.',
)


def models(dtype=torch.bfloat16, dropout=0.0):
    torch.manual_seed(42)
    cfg = GPTNeoXConfig(vocab_size=256, hidden_size=128, num_hidden_layers=2,
                       num_attention_heads=2, intermediate_size=256,
                       max_position_embeddings=2048, hidden_dropout=0.0,
                       attention_dropout=dropout, use_cache=False)
    cfg._attn_implementation = 'eager'
    eager = implementation.GPTNeoXForCausalLM(cfg).cuda().to(dtype).eval()
    fa_cfg = copy.deepcopy(cfg)
    fa_cfg._attn_implementation = 'flash_attention_2'
    flash = implementation.GPTNeoXForCausalLM(fa_cfg).cuda().to(dtype).eval()
    flash.load_state_dict(eager.state_dict())
    assert isinstance(eager.gpt_neox.layers[0].attention, implementation.GPTNeoXAttention)
    assert isinstance(flash.gpt_neox.layers[0].attention, implementation.GPTNeoXFlashAttention2)
    return eager, flash


def close(actual, expected, dtype, gradient=False):
    assert torch.isfinite(actual).all() and torch.isfinite(expected).all()
    a, b = actual.float(), expected.float()
    # Check both absolute and relative errors; low precision is not bitwise exact.
    rtol = 0.04 if dtype == torch.bfloat16 else 0.006
    atol = 0.004 if dtype == torch.bfloat16 else 0.0008
    if gradient:
        rtol *= 2
    torch.testing.assert_close(a, b, atol=atol, rtol=rtol)
    relative_l2 = (a-b).norm() / b.norm().clamp_min(1e-10)
    assert relative_l2 < rtol, f'relative L2 error {relative_l2.item():.5f}'


@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16])
@pytest.mark.parametrize('batch,length', [(1, 1), (2, 17), (2, 512), (1, 2048)])
@pytest.mark.parametrize('padding', ['none', 'left', 'right'])
def test_logits_loss_and_backward(dtype, batch, length, padding):
    eager, flash = models(dtype)
    ids = torch.randint(1, 256, (batch, length), device='cuda')
    mask = torch.ones_like(ids)
    if padding != 'none' and length > 1:
        for row in range(batch):
            count = min(row+2, length-1)
            mask[row, :count] = 0 if padding == 'left' else 1
            if padding == 'right':
                mask[row, -count:] = 0
    labels = ids.clone().masked_fill(mask == 0, -100)
    if length > 1:
        labels[:, 1:].masked_fill_(mask[:, :-1] == 0, -100)
    # One-token causal LM losses have no targets; still exercise outputs/gradients.
    outputs = []
    for model in (eager, flash):
        out = model(input_ids=ids, attention_mask=mask, labels=labels if length > 1 else None)
        outputs.append(out)
    close(outputs[1].logits[mask.bool()], outputs[0].logits[mask.bool()], dtype)
    if length > 1:
        close(outputs[1].loss, outputs[0].loss, dtype)
    upstream = torch.randn_like(outputs[0].logits, dtype=torch.float32)
    upstream *= mask.unsqueeze(-1)
    for out in outputs:
        (out.logits.float()*upstream).sum().backward()
    for (name, ep), (_, fp) in zip(eager.named_parameters(), flash.named_parameters()):
        if ep.grad is None:
            assert fp.grad is None, name
        else:
            assert fp.grad is not None, name
            close(fp.grad, ep.grad, dtype, gradient=True)


@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16])
def test_future_tokens_cannot_change_prefix(dtype):
    _, flash = models(dtype)
    ids = torch.randint(1, 256, (2, 19), device='cuda')
    changed = ids.clone()
    changed[:, 11:] = (changed[:, 11:] + 37) % 255 + 1
    with torch.no_grad():
        before = flash(ids).logits
        after = flash(changed).logits
    torch.testing.assert_close(before[:, :11], after[:, :11], atol=0, rtol=0)
    assert not torch.equal(before[:, 11:], after[:, 11:])


@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16])
@pytest.mark.parametrize('new_tokens', [1, 3])
@pytest.mark.parametrize('left_padding', [False, True])
def test_cached_decoding_matches_eager_and_full_sequence(dtype, new_tokens, left_padding):
    eager, flash = models(dtype)
    ids = torch.randint(1, 256, (2, 7 + new_tokens), device='cuda')
    mask = torch.ones_like(ids)
    if left_padding:
        mask[0, :2] = 0
    with torch.no_grad():
        full = eager(ids, attention_mask=mask).logits[:, 7:]
        cached = []
        for model in (eager, flash):
            cache = DynamicCache()
            model(ids[:, :7], attention_mask=mask[:, :7], past_key_values=cache, use_cache=True)
            out = model(ids[:, 7:], attention_mask=mask, past_key_values=cache, use_cache=True)
            assert cache.get_seq_length() == ids.shape[1]
            cached.append(out.logits)
    close(cached[0], full, dtype)
    close(cached[1], full, dtype)


def test_dropout_switch_and_attention_return(monkeypatch):
    _, flash = models(dropout=0.2)
    calls = []
    real_forward = implementation._flash_attention_forward
    def record(*args, **kwargs):
        calls.append(kwargs['dropout'])
        return real_forward(*args, **kwargs)
    monkeypatch.setattr(implementation, '_flash_attention_forward', record)
    ids = torch.randint(1, 256, (1, 17), device='cuda')
    flash.train()
    flash(ids)
    assert calls == [0.2, 0.2]
    calls.clear()
    flash.eval()
    with torch.no_grad():
        out = flash(ids, output_attentions=True)
    assert calls == [0.0, 0.0]
    assert out.attentions == (None, None)
    with pytest.raises(ValueError, match='head_mask'):
        flash(ids, head_mask=torch.ones(2, device='cuda'))
