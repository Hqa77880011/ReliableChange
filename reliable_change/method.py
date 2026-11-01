"""Availability-masked multiscale change detector and training objectives."""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .layers import boundary, image_demo


def block(inputs, outputs, stride=1):
    return nn.Sequential(nn.Conv2d(inputs, outputs, 3, stride, 1),
                         nn.GroupNorm(1, outputs), nn.GELU(),
                         nn.Conv2d(outputs, outputs, 3, padding=1),
                         nn.GroupNorm(1, outputs), nn.GELU())


def lovasz(probability, target):
    """Binary foreground Lovasz extension on a batch of probabilities."""
    errors = (target-probability).abs().flatten()
    order = errors.argsort(descending=True)
    truth = target.flatten()[order]
    total = truth.sum()
    intersection = total-truth.cumsum(0)
    union = total+(1-truth).cumsum(0)
    gradient = 1-intersection/union.clamp_min(1)
    gradient = torch.cat([gradient[:1], gradient[1:]-gradient[:-1]])
    return errors[order] @ gradient


class Model(nn.Module):
    """Separate optical/SAR stems, shared encoder and confidence-aware U decoder."""
    def __init__(self, c):
        super().__init__()
        self.c = c
        widths = c.get('widths', [64, 128, 256, 512])
        self.optical = block(3, widths[0], 2)
        self.sar = block(1, widths[0], 2)
        self.encoder = nn.ModuleList([block(widths[max(i-1, 0)], w, 2)
                                      for i, w in enumerate(widths)])
        self.difference = nn.ModuleList([block(4*w, w) for w in widths])
        self.quality = nn.ModuleList([nn.ModuleList([nn.Conv2d(4*w, 1, 1) for _ in range(2)]) for w in widths])
        self.confidence = nn.ModuleList([nn.Conv2d(w+2, w, 1) for w in widths])
        self.decoder = nn.ModuleList([block(widths[i+1]+widths[i], widths[i]) for i in range(3)])
        self.reference_projection = nn.Conv2d(widths[-1], widths[-1], 1)
        self.post_projection = nn.Conv2d(widths[-1], widths[-1], 1)
        self.head = nn.Conv2d(widths[0], 1, 1)
        self.edge = nn.Conv2d(widths[0], 1, 1)

    def align(self, reference, post, level):
        if self.c['variant'] in {'baseline', 'no_alignment'}:
            return post
        radius = self.c.get('alignment_bound', 4) // (4*2**level)
        if not radius:
            return post
        k = 2*radius+1
        padded = F.pad(post, (radius,)*4, mode='reflect')
        patches = F.unfold(padded, k).reshape(post.shape[0], post.shape[1], k*k, *post.shape[-2:])
        scores = (F.normalize(reference, dim=1, eps=1e-6)[:, :, None]
                  * F.normalize(patches, dim=1, eps=1e-6)).sum(1)
        return ((scores/self.c.get('alignment_temperature', .1)).softmax(1)[:, None]*patches).sum(2)

    def encode(self, x):
        reference = self.optical(x[:, :3])
        posts = [self.optical(x[:, 3:6]), self.sar(x[:, 6:7])]
        features = []
        for stage in self.encoder:
            reference = stage(reference)
            posts = [stage(post) for post in posts]
            features.append((reference, posts))
        return features

    def fused(self, features, available, shape):
        fused, qualities, post_embeddings = [], [], []
        valid = available[:, :, None, None, None].float()
        for level, (reference, posts) in enumerate(features):
            descriptors, quality, aligned = [], [], []
            for q, post in enumerate(posts):
                post = self.align(reference, post, level)
                descriptor = torch.cat([reference, post, post-reference, abs(post-reference)], 1)
                descriptors.append(self.difference[level](descriptor))
                quality.append(self.quality[level][q](descriptor).sigmoid())
                aligned.append(post)
            quality = torch.stack(quality, 1)
            if self.c['variant'] in {'baseline', 'no_quality'}:
                quality = torch.ones_like(quality)
            weights = valid*(quality+1e-6)
            weights = weights/weights.sum(1, keepdim=True).clamp_min(1e-6)
            z = (weights*quality*torch.stack(descriptors, 1)).sum(1)
            post_embeddings.append((weights*quality*torch.stack(aligned, 1)).sum(1))
            qualities.append(quality.squeeze(2))
            fused.append(self.confidence[level](torch.cat([z, quality.squeeze(2)*available[:, :, None, None]], 1)))
        z = fused[-1]
        for level in [2, 1, 0]:
            z = self.decoder[level](torch.cat([F.interpolate(z, fused[level].shape[-2:], mode='bilinear', align_corners=False), fused[level]], 1))
        resize = lambda a: F.interpolate(a, shape, mode='bilinear', align_corners=False)
        reference = self.reference_projection(features[-1][0])
        post = self.post_projection(post_embeddings[-1])
        return dict(probability=resize(self.head(z)).sigmoid(), boundary=resize(self.edge(z)).sigmoid(),
                    quality=resize(qualities[0]), quality_scales=qualities,
                    feature=resize((post-reference).abs().sum(1, keepdim=True)), availability=available)

    def forward(self, b):
        x = b['x'].float()
        available = b['available'].float()
        if x.ndim != 4 or x.shape[1] != 7 or min(x.shape[-2:]) < 32:
            raise ValueError('Expected x[N,7,H,W] with H,W >= 32')
        if not ((available == 0) | (available == 1)).all() or not (available.sum(1) > 0).all():
            raise ValueError('At least one binary post-event availability flag is required')
        features = self.encode(x)
        sampled = available.clone()
        full = available.sum(1) == 2
        if self.training and self.c['variant'] != 'baseline':
            choice = torch.rand(len(x), device=x.device)
            sampled[full & (choice >= .5) & (choice < .75), 0] = 0
            sampled[full & (choice >= .75), 1] = 0
        prediction = self.fused(features, sampled, x.shape[-2:])
        if self.training and full.any() and self.c['variant'] not in {'baseline', 'no_mask_consistency'}:
            masked = available.clone()
            removed = torch.randint(2, (len(x),), device=x.device)
            masked[full, removed[full]] = 0
            prediction['teacher'] = self.fused(features, available, x.shape[-2:])['probability'].detach()
            prediction['masked_probability'] = self.fused(features, masked, x.shape[-2:])['probability']
            prediction['full_observation'] = full
        return prediction

    def objective(self, b, p):
        y = b['y'].float()
        probability = p['probability'].clamp(1e-6, 1-1e-6)
        weight = torch.where(y > .5, self.c.get('positive_weight', 1.), 1.)
        loss = F.binary_cross_entropy(probability, y, weight=weight)+lovasz(probability, y)
        if self.c['variant'] == 'baseline':
            return loss
        if self.c['variant'] != 'no_boundary':
            loss = loss+.2*F.binary_cross_entropy(p['boundary'], boundary(y, 1))
        if self.c['variant'] != 'no_change':
            loss = loss+.3*((1-y)*p['feature']+y*F.relu(1-p['feature'])).mean()
        if 'masked_probability' in p:
            teacher = p['teacher'].clamp(1e-6, 1-1e-6)
            student = p['masked_probability'].clamp(1e-6, 1-1e-6)
            kl = teacher*(teacher/student).log()+(1-teacher)*((1-teacher)/(1-student)).log()
            loss = loss+.2*kl[p['full_observation']].mean()
        if 'quality' in b and self.c['variant'] != 'no_quality':
            quality_loss = loss*0
            for quality in p['quality_scales']:
                target = F.interpolate(b['quality'].float(), quality.shape[-2:], mode='area')
                error = (quality-target).square()*p['availability'][:, :, None, None]
                quality_loss = quality_loss+error.mean((2, 3)).sum(1).mean()
            loss = loss+.1*quality_loss
        return loss


def demo(seed, n, c):
    d = image_demo(seed, n, 7)
    d['available'] = np.ones((n, 2), np.float32)
    d['quality'] = np.ones((n, 2, 32, 32), np.float32)
    return d


def event_metrics(probability, truth, event):
    """Pool pixels within each event, then macro-average independent events."""
    from .metrics import measure as base_measure
    values = {}
    for key in np.unique(event):
        chosen = event == key
        value = base_measure({'probability': probability[chosen]}, {'y': truth[chosen]})
        values[str(key)] = {name: value[name] for name in ['precision', 'recall', 'dice', 'iou']}
        values[str(key)]['f1'] = values[str(key)].pop('dice')
        values[str(key)]['boundary_f1'] = value.get('boundary_f1', 1.)
    return dict(events=values, **{f'event_macro_{key}': float(np.mean([r[key] for r in values.values()])) for key in next(iter(values.values()))})


def measure(p, d, c):
    if 'event' not in d:
        return {'f1': float(2*((p['probability'] >= .5) & (d['y'] >= .5)).sum()/max(1, (p['probability'] >= .5).sum()+(d['y'] >= .5).sum()))}
    return event_metrics(p['probability'], d['y'], d['event'])
