import ast
import os
import shutil

def key_exists(dict, key):
    try:
        if dict[key] is not None:
            return True
    except KeyError as e:
        return False


def transfer_parameter(dict, key, not_found):
    if key_exists(dict, key):
        param = dict[key]
    else:
        param = not_found
    return param


def copy_folders(src, dst, beginning_of_name):
    l = get_folders_starting_with(src, beginning_of_name)
    for f in l:
        shutil.copytree(src + "/" + f, dst + "/" + f)


def copy_folder_and_duplicate(src, dst, beginning_of_name, duplicates=1):
    l = get_folders_starting_with(src, beginning_of_name)[0]
    for i in range(duplicates):
        shutil.copytree(src + "/" + l, dst + "/" + "Task" + str(i+1))


def copy_file_and_duplicate(src, dst, beginning_of_name, duplicates=1):
    lst = get_files_starting_with(src, beginning_of_name)
    for l in lst:
        for i in range(duplicates):
            shutil.copy(src + "/" + l, dst + "/" + l.replace(beginning_of_name, "Task" + str(i+1)))


def get_folders_starting_with(src, beginning_of_name):
    res = []
    for f in os.listdir(src + "/"):
        if f.startswith(beginning_of_name) and os.path.isdir(src + "/" + f):
            res.append(f)
    return res


def get_files_starting_with(src, beginning_of_name):
    res = []
    for f in os.listdir(src + "/"):
        if f.startswith(beginning_of_name) and os.path.isfile(src + "/" + f):
            res.append(f)
    return res


def strdict_to_numdict(d):
    res_dict = {}
    for k in d:
        res_dict[k] = converter(d[k])
    return res_dict


def converter(i):
    try:
        return ast.literal_eval(i)
    except ValueError:
        return i


def fill_default_paramers_for_net(net):
    return net