from typing import Optional, MutableMapping, MutableSequence, Mapping, Collection

def combine_dicts(*dicts: Optional[dict]) -> dict:
    combined = {}
    for d in dicts:
        if d:
            combined.update(d)
    return combined

def update_kwargs_with_locals(_kwargs: dict, _locals: dict, kwargs_key="kwargs", exclude=("self", "cls")):
    _exclude = exclude + (kwargs_key,)
    _kwargs.update({k: v for k, v in _locals.items() if k not in _exclude})
    return _kwargs

def recursive_pop(
        d: MutableMapping, 
        keep_keys: Optional[Collection] = None, 
        remove_keys: Optional[Collection] = None, 
        replace_keys: Optional[Mapping] = None, 
        remove_values: Optional[Collection] = [None, "", {}, [], set(), tuple()], 
        replace_values: Optional[Mapping] = None
    ):
    """
    Recursively remove keys from a dict.

    Args:
        d (MutableMapping): The dict to remove keys from.
        keep_keys (Optional[Collection], optional): Keys to keep. If keep_keys is set, only keep_keys are kept. Defaults to None.
        remove_keys (Optional[Collection], optional): Keys to remove.  If remove_keys is set, remove_keys are removed. Defaults to None.
        replace_keys (Optional[MutableMapping], optional): Keys to replace. If replace_keys is set and a key is in replace_keys, the key is replaced with the value in replace_keys. Defaults to None.
        remove_values (Optional[Collection], optional): Values to remove. If remove_values is set, all keys whose value is in remove_values are removed. Defaults to [None, "", {}, [], set(), tuple()].
        replace_values (Optional[MutableMapping], optional): Values to replace. If replace_values is set and a value is in replace_values, the value is replaced with the value in replace_values. Defaults to None.
    """
    if isinstance(d, MutableMapping):
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

    elif isinstance(d, MutableSequence):
        for i, item in enumerate(d):
            d[i] = recursive_pop(item, keep_keys, remove_keys, replace_keys, remove_values, replace_values)

    return d
  