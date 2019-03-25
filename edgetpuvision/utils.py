import collections
import os
import re
import time

LABEL_PATTERN = re.compile(r'\s*(\d+)(.+)')

def load_labels(path):
    with open(path, 'r', encoding='utf-8') as f:
       lines = (LABEL_PATTERN.match(line).groups() for line in f.readlines())
       return {int(num): text.strip() for num, text in lines}


def input_image_size(engine):
    _, h, w, _ = engine.get_input_tensor_shape()
    return w, h

def same_input_image_sizes(engines):
    return len({input_image_size(engine) for engine in engines}) == 1

def avg_fps_counter(window_size):
    window = collections.deque(maxlen=window_size)
    prev = time.monotonic()
    yield 0.0  # First fps value.

    while True:
        curr = time.monotonic()
        window.append(curr - prev)
        prev = curr
        yield len(window) / sum(window)

def make_engines(models, engine_class):
    engines, titles = [], {}
    for model in models.split(','):
        if '@' in model:
            model_path, title = model.split('@')
        else:
            model_path, title = model, os.path.basename(os.path.normpath(model))
        engine = engine_class(model_path)
        engines.append(engine)
        titles[engine] = title
    return engines, titles
