def name_to_num(name):
    return ["kitchen", "living_room", "bedroom", "bathroom"].index(name)


def num_to_name(num):
    return ["kitchen", "", "living_room", "bedroom", "bathroom"][int(num / 100)]
