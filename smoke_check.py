"""CPU functional check using generated event-disjoint tiles only."""
import io
import json
import sys
from pathlib import Path

import numpy as np
import torch

from reliable_change.cli import main as cli_main, read, tensor_batch
from reliable_change.evaluate_modes import evaluate
from reliable_change.method import Model, event_metrics
from reliable_change.prepare_bright import EVENTS, prepare


def main():
    torch.set_num_threads(1)
    torch.manual_seed(13)
    root = Path('runs/smoke')
    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(13)
    raw = dict(x=rng.uniform(0, 1, (10, 7, 32, 32)).astype(np.float32),
               y=(rng.random((10, 1, 32, 32)) > .7).astype(np.float32),
               available=np.ones((10, 2), np.float32), event=np.repeat(EVENTS, 2),
               group=np.arange(10), quality=np.ones((10, 2, 32, 32), np.float32))
    np.savez_compressed(root/'raw.npz', **raw)
    state = prepare(root/'raw.npz', root/'prepared', 0)
    shifted = dict(raw, x=raw['x'].copy())
    shifted['x'][:4] *= 1000  # Test/validation cannot affect fitted statistics.
    np.savez_compressed(root/'shifted.npz', **shifted)
    state2 = prepare(root/'shifted.npz', root/'shifted', 0)
    assert state == state2
    train = read(root/'prepared/train.npz')
    batch = tensor_batch(train, np.array([0, 1]), 'cpu')
    c = dict(widths=[4, 8, 12, 16], alignment_bound=4, alignment_temperature=.1, variant='full')
    model = Model(c).train()
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    opt = torch.optim.AdamW(model.parameters(), lr=.001)
    prediction = model(batch)
    loss = model.objective(batch, prediction)
    assert torch.isfinite(loss)
    opt.zero_grad()
    loss.backward()
    for prefix in ['optical', 'sar', 'encoder', 'quality', 'difference', 'decoder', 'head', 'edge', 'reference_projection', 'post_projection']:
        assert any(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0
                   for n, p in model.named_parameters() if n.startswith(prefix)), prefix
    opt.step()
    for prefix in ['encoder', 'quality', 'head', 'edge']:
        assert any(not torch.equal(before[n], p) for n, p in model.named_parameters() if n.startswith(prefix)), prefix
    model.eval()
    with torch.no_grad():
        expected = model(batch)['probability']
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    buffer.seek(0)
    restored = Model(c).eval()
    restored.load_state_dict(torch.load(buffer, weights_only=True))
    with torch.no_grad():
        result = restored(batch)
        torch.testing.assert_close(expected, result['probability'], rtol=1e-6, atol=1e-6)
        assert 'masked_probability' not in result
        for availability in [[0, 1], [1, 0], [1, 1]]:
            changed = dict(batch, available=torch.tensor([availability]*2, dtype=torch.float32))
            assert torch.isfinite(restored(changed)['probability']).all()
    for variant in ['baseline', 'no_alignment', 'no_quality', 'no_mask_consistency', 'no_boundary', 'no_change']:
        control = Model(dict(c, variant=variant))
        value = control.objective(batch, control(batch))
        assert torch.isfinite(value)
        value.backward()
    checkpoint = dict(config=c, state=model.state_dict())
    torch.save(checkpoint, root/'model.pt')
    acquisition = evaluate(read(root/'prepared/test.npz'), checkpoint, 1)
    (root/'acquisition.json').write_text(json.dumps(acquisition, indent=2))
    probability = np.array([[[[1., 1.], [0., 0.]]], [[[1., 0.], [0., 0.]]]])
    truth = np.array([[[[1., 0.], [1., 0.]]], [[[1., 0.], [0., 0.]]]])
    values = event_metrics(probability, truth, np.array(['a', 'b']))
    assert np.isclose(values['event_macro_f1'], .75)
    assert np.isclose(values['event_macro_iou'], 2/3)
    (root/'config.json').write_text(json.dumps(c), encoding='utf-8')
    for command in [
        ['train', '--input', str(root/'prepared'), '--output', str(root/'cli'), '--config', str(root/'config.json'), '--epochs', '1', '--batch-size', '2'],
        ['evaluate', '--input', str(root/'prepared/test.npz'), '--output', str(root/'cli/evaluation'), '--model', str(root/'cli/model.pt')],
        ['plot', '--input', str(root/'cli'), '--output', str(root/'figures')],
    ]:
        sys.argv = ['reliable-change']+command
        cli_main()
    print(f'PASS loss={float(loss.detach()):.6f}: train-only preprocessing, full graph backward/update, reload, three modes, six ablations, hand event-macro metrics')


if __name__ == '__main__':
    main()
