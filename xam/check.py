def is_an_int(val):
    return isinstance(val, int)


def is_a_positive_int(val, strict=False):
    if not is_an_int(val):
        return False
    return val > 0 if strict else val >= 0


def is_a_float(val):
    return isinstance(val, float)


def is_a_ratio(val, strict=False):
    if not is_a_float(val):
        return False
    return 0 < val < 1 if strict else 0 <= val <= 1


def is_a_bool(val):
    return isinstance(val, bool)
