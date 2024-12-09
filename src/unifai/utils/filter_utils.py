from typing import Any

def check_filter(filter: dict|str, value: Any) -> bool:
    if isinstance(filter, str):
        return value == filter
    filter_operator, filter_value = next(iter(filter.items()))
    if filter_operator == "$eq":
        return value == filter_value
    if filter_operator == "$ne":
        return value != filter_value
    if filter_operator == "$gt":
        return value > filter_value
    if filter_operator == "$gte":
        return value >= filter_value
    if filter_operator == "$lt":
        return value < filter_value
    if filter_operator == "$lte":
        return value <= filter_value
    if filter_operator == "$in":
        return value in filter_value
    if filter_operator == "$nin":
        return value not in filter_value
    if filter_operator == "$exists":
        return bool(value) == filter_value
    if filter_operator == "$contains":
        return filter_value in value
    if filter_operator == "$not_contains":
        return filter_value not in value
    raise ValueError(f"Invalid filter {filter}")

def check_metadata_filters(where: dict, metadata: dict) -> bool:
    for key, filter in where.items():
        if key == "$and":
            for sub_filter in filter:
                if not check_metadata_filters(sub_filter, metadata):
                    return False
            continue
        
        if key == "$or":
            _any = False
            for sub_filter in filter:
                if check_metadata_filters(sub_filter, metadata):
                    _any = True
                    break
            if not _any:                    
                return False
            continue
        
        value = metadata.get(key)
        if not check_filter(filter, value):
            return False
        
    return True