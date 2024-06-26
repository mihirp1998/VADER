from importlib_resources import files
import os
import functools
import random
import inflect

IE = inflect.engine()
ASSETS_PATH = files("assets")
# ASSETS_PATH = "assets"


@functools.lru_cache(maxsize=None)
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

def chatgpt_custom(nouns_file, activities_file):
    return from_file("chatgpt_custom.txt")

def chatgpt_custom_instruments(nouns_file, activities_file):
    return from_file("chatgpt_custom_instruments.txt")

def chatgpt_custom_human(nouns_file, activities_file):
    return from_file("chatgpt_custom_human.txt")

def chatgpt_custom_human_activity(nouns_file, activities_file):
    return from_file("chatgpt_custom_human_activity.txt")

def chatgpt_custom_animal(nouns_file, activities_file):
    return from_file("chatgpt_custom_animal.txt")

def chatgpt_custom_animal_sport(nouns_file, activities_file):
    return from_file("chatgpt_custom_animal_sport.txt")

def chatgpt_custom_animal_sportV2(nouns_file, activities_file):
    return from_file("chatgpt_custom_animal_sportV2.txt")

def chatgpt_custom_animal_clothes(nouns_file, activities_file):
    return from_file("chatgpt_custom_animal_clothes.txt")

def chatgpt_custom_animal_clothesV2(nouns_file, activities_file):
    return from_file("chatgpt_custom_animal_clothesV2.txt")

def chatgpt_custom_animal_clothesV3(nouns_file, activities_file):
    return from_file("chatgpt_custom_animal_clothesV3.txt")

def chatgpt_custom_animal_technology(nouns_file, activities_file):
    return from_file("chatgpt_custom_animal_technology.txt")

def chatgpt_custom_animal_housework(nouns_file, activities_file):
    return from_file("chatgpt_custom_animal_housework.txt")

def chatgpt_custom_animal_action(nouns_file, activities_file):
    return from_file("chatgpt_custom_animal_action.txt")

def chatgpt_custom_outdoor(nouns_file, activities_file):
    return from_file("chatgpt_custom_outdoor.txt")

def chatgpt_custom_rainy(nouns_file, activities_file):
    return from_file("chatgpt_custom_rainy.txt")

def chatgpt_custom_snowy(nouns_file, activities_file):
    return from_file("chatgpt_custom_snowy.txt")

def chatgpt_custom_dog(nouns_file, activities_file):
    return from_file("chatgpt_custom_dog.txt")

def chatgpt_custom_banana(nouns_file, activities_file):
    return from_file("chatgpt_custom_banana.txt")

def chatgpt_custom_forest(nouns_file, activities_file):
    return from_file("chatgpt_custom_forest.txt")

def chatgpt_custom_forest_vivid(nouns_file, activities_file):
    return from_file("chatgpt_custom_forest_vivid.txt")

def chatgpt_custom_cruel_animal(nouns_file, activities_file):
    return from_file("chatgpt_custom_cruel_animal.txt")

def chatgpt_custom_cruel_animal2(nouns_file, activities_file):
    return from_file("chatgpt_custom_cruel_animal2.txt")

def chatgpt_custom_bottle_glass(nouns_file, activities_file):
    return from_file("chatgpt_custom_bottle_glass.txt")

def chatgpt_custom_book_cup(nouns_file, activities_file):
    return from_file("chatgpt_custom_book_cup.txt")

def chatgpt_custom_book_cup_character(nouns_file, activities_file):
    return from_file("chatgpt_custom_book_cup_character.txt")

def chatgpt_custom_animal_cute(nouns_file, activities_file):
    return from_file("chatgpt_custom_animal_cute.txt")

def chatgpt_custom_animal_cute_select(nouns_file, activities_file):
    return from_file("chatgpt_custom_animal_cute_select.txt")

def chatgpt_custom_compression(nouns_file, activities_file):
    return from_file("chatgpt_custom_compression.txt")

def chatgpt_custom_compression_animals(nouns_file, activities_file):
    return from_file("chatgpt_custom_compression_animals.txt")

def chatgpt_custom_actpred(nouns_file, activities_file):
    return from_file("chatgpt_custom_actpred.txt")

def chatgpt_custom_actpred2(nouns_file, activities_file):
    return from_file("chatgpt_custom_actpred2.txt")

def chatgpt_custom_instruments_unseen(nouns_file, activities_file):
    return from_file("chatgpt_custom_instruments_unseen.txt")

def from_file(path, low=None, high=None, **kwargs):
    prompts = _load_lines(path)[low:high]
    return random.choice(prompts), {}

def from_str(_str, **kwargs):
    return _str, {}

def nouns_activities(nouns_file, activities_file, **kwargs):
    nouns = _load_lines(nouns_file)
    activities = _load_lines(activities_file)
    return f"{IE.a(random.choice(nouns))} {random.choice(activities)}", {}