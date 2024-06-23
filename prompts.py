from importlib import resources
import os
import functools
import random
import inflect

IE = inflect.engine()
ASSETS_PATH = resources.files("assets")
# ASSETS_PATH = "assets"


@functools.cache
def _load_lines(path):
    """
    Load lines from a file. First tries to load from `path` directly, and if that doesn't exist, searches the
    `ddpo_pytorch/assets` directory for a file named `path`.
    """
    if not os.path.exists(path):
        newpath = ASSETS_PATH.joinpath(path)
    if not os.path.exists(newpath):
        raise FileNotFoundError(f"Could not find {path} or assets/{path}")
    path = newpath
    with open(path, "r") as f:
        return [line.strip() for line in f.readlines()]

def hps_v2_all():
    return from_file("hps_v2_all.txt")

def hps_custom(nouns_file, activities_file):
    return from_file("hps_custom.txt")

def hps_debug():
    return from_file("hps_debug.txt")

def hps_single(nouns_file, activities_file):
    return from_file("hps_single.txt")

def kinetics_4rand():
    return from_file("kinetics_4rand.txt")

def kinetics_50rand():
    return from_file("kinetics_50rand.txt")

def simple_animals():
    return from_file("simple_animals.txt")

def eval_simple_animals():
    return from_file("eval_simple_animals.txt")

def eval_hps_v2_all():
    return from_file("hps_v2_all_eval.txt")


def from_file(path, low=None, high=None, **kwargs):
    prompts = _load_lines(path)[low:high]
    return random.choice(prompts), {}

def from_str(_str, **kwargs):
    return _str, {}

def nouns_activities(nouns_file, activities_file, **kwargs):
    nouns = _load_lines(nouns_file)
    activities = _load_lines(activities_file)
    return f"{IE.a(random.choice(nouns))} {random.choice(activities)}", {}