"""convert pretrained models"""

def key_transformation(old_key):
    # print(old_key)
    if '.' in old_key:
        a, b = old_key.split('.', 1)
        if a == "cnn":
            return f"body.{b}"
    return old_key
    # body.0.weight", "cnn.1.weight

rename_state_dict_keys(e['NoRoIPoolModel'].path/'models'/'bestmodel.pth', key_transformation)
