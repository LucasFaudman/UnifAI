from typing import Optional, MutableSequence, Collection

def combine_dicts(*dicts: Optional[dict]) -> dict:
    combined = {}
    for d in dicts:
        if d:
            combined.update(d)
    return combined

def clean_locals(_locals: dict, exclude: Collection={"self", "cls"}, kwargs_key="kwargs") -> dict:
    return {k: v for k, v in _locals.items() if k != kwargs_key and k not in exclude}

def update_kwargs_with_locals(_kwargs: dict, _locals: dict, exclude: Collection={"self", "cls"}, kwargs_key="kwargs") -> dict:
    _kwargs.update(clean_locals(_locals, exclude, kwargs_key))
    return _kwargs

def recursive_clear(d: dict) -> dict:
    """Recursively clear all nested dicts until all values are empty."""
    for value in d.values():
        if isinstance(value, dict):
            recursive_clear(value)
        elif isinstance(value, MutableSequence):
            value.clear()
    d.clear()
    return d
    
def recursive_pop(
        d: dict, 
        keep_keys: Optional[Collection] = None, 
        remove_keys: Optional[Collection] = None, 
        replace_keys: Optional[dict] = None, 
        remove_values: Optional[Collection] = [None, "", {}, [], set(), tuple()], 
        replace_values: Optional[dict] = None
    ):
    """
    Recursively remove keys from a dict.

    Args:
        d (dict): The dict to remove keys from.
        keep_keys (Optional[Collection], optional): Keys to keep. If keep_keys is set, only keep_keys are kept. Defaults to None.
        remove_keys (Optional[Collection], optional): Keys to remove.  If remove_keys is set, remove_keys are removed. Defaults to None.
        replace_keys (Optional[dict], optional): Keys to replace. If replace_keys is set and a key is in replace_keys, the key is replaced with the value in replace_keys. Defaults to None.
        remove_values (Optional[Collection], optional): Values to remove. If remove_values is set, all keys whose value is in remove_values are removed. Defaults to [None, "", {}, [], set(), tuple()].
        replace_values (Optional[dict], optional): Values to replace. If replace_values is set and a value is in replace_values, the value is replaced with the value in replace_values. Defaults to None.
    """
    if isinstance(d, dict):
            if remove_keys:
                for key in remove_keys:
                    d.pop(key, None)
            if keep_keys:
                for key in set(d.keys()) - set(keep_keys):
                    d.pop(key, None)

            for key, value in list(d.items()):
                value = recursive_pop(value, keep_keys, remove_keys, replace_keys, remove_values, replace_values)
                
                if replace_values and value in replace_values:
                    d[key] = replace_values.get(value)
                    value = d[key]

                if remove_values and value in remove_values:
                    d.pop(key, None)
                    continue
                
                if replace_keys and key in replace_keys:
                    new_key = replace_keys[key]
                    d[new_key] = d.pop(key)

    elif isinstance(d, Collection):
        for i, item in enumerate(d):
            d[i] = recursive_pop(item, keep_keys, remove_keys, replace_keys, remove_values, replace_values)

    return d
  