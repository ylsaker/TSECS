import json
def out_put_metadata(meta_data, fn):
    json.dump(meta_data, open(fn, 'w'))
