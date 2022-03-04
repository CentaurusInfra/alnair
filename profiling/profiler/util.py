def dict_changes(old, new):
    diff ="no changes"
    for k in new:
        if k not in old:
            diff = "new key {} value {}".format(k, new[k])
            return diff
        else:
            if new[k] != old[k]:
                diff = "key {} value change from {} to {}".format(k, old[k], new[k])
                return diff
    return diff

