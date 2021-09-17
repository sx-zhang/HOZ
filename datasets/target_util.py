from datasets.constants import AI2THOR_TARGET_CLASSES


def get_object_list(object_list_str):
    return AI2THOR_TARGET_CLASSES


def get_object_index(target_objects, all_objects):
    idxs = []
    for o in target_objects:
        idxs.append(all_objects.index(o))
    return idxs
