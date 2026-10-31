"""Evaluate one trained checkpoint in all three valid acquisition modes."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .cli import read, tensor_batch
from .method import event_metrics
from .baselines import build_model
from .protocol import tiled_predict


def evaluate(data, state, batch_size=8):
    network = build_model(state['config']).eval()
    network.load_state_dict(state['state'])
    if not (data['available'] == 1).all():
        raise ValueError('Acquisition intervention needs complete measured triplets')
    result = {}
    for name, availability in [('full', [1, 1]), ('missing_optical', [0, 1]), ('missing_sar', [1, 0])]:
        intervention = dict(data, available=np.tile(np.array(availability, np.float32), (len(data['x']), 1)))
        values = []
        with torch.no_grad():
            for begin in range(0, len(data['x']), batch_size):
                batch = tensor_batch(intervention, np.arange(begin, min(begin+batch_size, len(data['x']))), 'cpu')
                values.append(tiled_predict(network,batch)['probability'].numpy())
        result[name] = event_metrics(np.concatenate(values), data['y'], data['event'])
    result['worst_mode_f1'] = min(value['event_macro_f1'] for value in result.values())
    for name in ['missing_optical', 'missing_sar']:
        result[f'acquisition_drop_{name}'] = result['full']['event_macro_f1']-result[name]['event_macro_f1']
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', required=True)
    p.add_argument('--model', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--batch-size', type=int, default=8)
    a = p.parse_args()
    state = torch.load(a.model, map_location='cpu', weights_only=True)
    values = evaluate(read(a.input), state, a.batch_size)
    out = Path(a.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(values, indent=2, allow_nan=False), encoding='utf-8')


if __name__ == '__main__':
    main()
