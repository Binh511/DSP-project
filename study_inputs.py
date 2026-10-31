"""Versioned classifier input encoding used by new experiments and evaluation."""
import math
import re

MODES = ('text', 'section', 'section_position', 'neighbors', 'full')
_BULLET = re.compile(r'^\s*(?:[-*●•]|\d+[.)])\s')
_MODAL = re.compile(r'\b(will|must|should|shall|need to|needs to)\b', re.I)
_IMPER = re.compile(r'^\W*(?:\d+[.)]\s*)?(implement|add|create|build|develop|design|fix|write|update|integrate|test|deploy|refactor|provide|release|document)\b', re.I)
_TIME = re.compile(r'\b(week|day|month|phase|milestone|sprint)\b|\bQ[1-4]\b', re.I)


def context_for(item, config):
    mode = config.get('context_mode', 'full')
    if mode not in MODES:
        raise ValueError(f'Unsupported context_mode: {mode}')
    section, prev, nxt, text = (str(item.get(k, '') or '') for k in ('section', 'prev_text', 'next_text', 'text'))
    pos = float(item.get('pos', 0))
    if not math.isfinite(pos) or not 0 <= pos <= 1:
        raise ValueError('pos must be finite in [0, 1]')
    if mode == 'text':
        return None
    fields = [f'Section: {section.strip()[-120:]}']
    if mode in ('section_position', 'full'):
        fields.append(f'Pos: {int(pos * 10)}')
    if mode == 'full':
        cue = ''.join(str(int(bool(p.search(text)))) for p in (_MODAL, _IMPER, _TIME))
        fields += [f'Bullet: {int(bool(_BULLET.match(text)))}', f'Cue: {cue}']
    if mode in ('neighbors', 'full'):
        fields += [f'Previous: {prev.strip()[-150:]}', f'Next: {nxt.strip()[:150]}']
    return ' | '.join(fields)


def encode_items(tokenizer, items, config, **kwargs):
    texts = [str(i['text'])[:config.get('text_max_chars', 600)] for i in items]
    contexts = [context_for(i, config) for i in items]
    options = dict(truncation='longest_first', max_length=config.get('max_len', 256), **kwargs)
    return tokenizer(texts, **options) if config.get('context_mode') == 'text' else tokenizer(texts, contexts, **options)
