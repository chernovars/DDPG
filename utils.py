import ast
import os
import shutil

from matplotlib import pyplot as plt




def copy_folders(src, dst, starts_with):
    l = get_folders_starting_with(src, starts_with)
    for f in l:
        shutil.copytree(src + "/" + f, dst + "/" + f)


def copy_folder_and_duplicate(src, dst, starts_with, names_list):
    l = get_folders_starting_with(src, starts_with)[0]
    for name in names_list:
        shutil.copytree(src + "/" + l, dst + "/" + "Task" + name)

def copy_file_and_duplicate(src, dst, starts_with, names_list):
    lst = get_files_starting_with(src, starts_with)
    for l in lst:
        for name in names_list:
            shutil.copy(src + "/" + l, dst + "/" + l.replace(starts_with, "Task" + name))

def get_folders_starting_with(src, beginning_of_name):
    res = []
    for f in os.listdir(src + "/"):
        if f.startswith(beginning_of_name) and os.path.isdir(src + "/" + f):
            res.append(f)
    return res


def get_files_starting_with(src, beginning_of_name, ends_with=None):
    res = []
    for f in os.listdir(src + "/"):
        if f.startswith(beginning_of_name) and os.path.isfile(src + "/" + f):
            if ends_with:
                if f.endswith(ends_with):
                    res.append(f)
            else:
                res.append(f)
    return res

def converter(i):
    try:
        return ast.literal_eval(i)
    except ValueError:
        return i

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
    return converter(param)


def strdict_to_numdict(d):
    res_dict = {}
    for k in d:
        res_dict[k] = converter(d[k])
    return res_dict





def fill_default_paramers_for_net(net):
    return net


def generatePlot(*y, x=None, scatter=None, title="", labels=None, legend=None, save_folder=None, show_picture=True,
                 color='b', x_start=0, POI=None):
    '''



    :param y:
    :param x:
    :param scatter:
    :param title:
    :param labels:
    :param legend:
    :param save_folder:
    :param show_picture:
    :param color:
    :param x_start:
    :param POI:
    [ x_list y_list labels_list ]
    :return:
    '''
    colors = ['b', 'y', 'r', 'c', 'm', 'g', 'k']
    plt.figure()
    plot_args = []

    legths_y = [len(i) for i in y]
    if min(legths_y) != max(legths_y):
        print("Lenghts of y-lists should be the same")
        raise AssertionError

    if x is None and len(y) > 0:
        x = list(range(x_start, x_start + len(y[0])))

    if legend is None or len(legend) != len(y):
        legend = ["y" + str(y.index(i)) for i in y]

    ax = plt.gca()
    if len(y) > 1:
        for i in range(0, len(y)):
            ax.plot(x, y[i], colors[i % (len(colors))], label=legend[i])
    elif len(y) == 1:
        plot_args += [x, y[0], color]
        ax.plot(plot_args)

    if labels is None:
        ax.xlabel('x label')
        ax.ylabel('y label')
    else:
        plt.xlabel(labels[0])
        plt.ylabel(labels[1])

    if title:
        plt.title(title)

    if scatter is not None:
        ax.scatter(scatter[0], scatter[1], c='r', zorder=100)


    if POI is not None:
        for X, Y, Z in zip(*POI):
            # Annotate the points 5 _points_ above and to the left of the vertex
            ax.annotate('{}'.format(Z), xy=(X + x_start, Y), xytext=(15, 5), rotation=90, ha='right',
                        textcoords='offset points')

    plt.legend()
    if save_folder:
        plt.savefig(save_folder)

    if show_picture:
        plt.show(block=False)