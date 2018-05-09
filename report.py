import argparse
import os

import xml.etree.ElementTree as ET

import automation
import main
import numpy as np
import utils
import matplotlib.pyplot as plt
import csv

def processPicture(full_path, scenario):
    files = utils.get_files_starting_with(full_path, "Task")
    files_filtered = list(filter(lambda s: not (s.endswith("test") or s.endswith("png")), files))
    for f in files_filtered:
        dc = main.DataCollector("","")
        temp_path = full_path + "/" + f
        print(temp_path)
        #test = dc.load_test_list(temp_path + "_test")

        test_points_x_start = 0

        info = dc.load_rewards_list(temp_path)
        last_points_to_show = len(info)
        #test_points = len(test[0])

        '''test[0] = np.array(test[0])[-test_points:]
        test[1] = np.array(test[1])[-test_points:]
        test_filtered = list(filter(lambda x: x[0] >= test_points_x_start, zip(test[0], test[1])))
        test_filtered = [np.array(t) for t in zip(*test_filtered)]
'''
        ema = dc.listToEMA(info)[-last_points_to_show:]
        info = info[-last_points_to_show:]

        ema = ema[test_points_x_start:]
        info = info[test_points_x_start:]

        title = scenario + " " + f
        labels = ["episodes", "rewards"]
        legend = ["Reward on noise", "EMA"]

        POI = [[len(ema)-1], [ema[-1]], [int(ema[-1])]]
        #utils.generatePlot(info, ema, x_start=test_points_x_start, scatter=test_filtered, title=title, labels=labels, legend=legend, save_folder=temp_path + ".png", POI=POI)
        utils.generatePlot(info, ema, x_start=test_points_x_start, title=title, labels=labels, legend=legend, save_folder=temp_path + ".png", POI=POI)

def get_decay_list(tasks):
    res = []
    for i, t in enumerate(tasks):
        executions_num = utils.transfer_parameter(t[0][0].attrib, "decay", False)
        res.append(str(executions_num))
    return res

def generateReport(full_path, from_file=False):
    env_name = "Walker2d-v1" # TODO: replace with input variable

    scenario = os.path.join(full_path, utils.get_files_starting_with(full_path, "scenario", ends_with="xml")[0])
    tree = ET.parse(scenario)

    task_names = automation.__get_tasks_names(tree.getroot(), lists_in_list=True)
    task_names = [["Task" + ordinal + env_name for ordinal in list] for list in task_names]

    ema_list_of_tasks = []

    if from_file:
        decay_list = []
        with open(os.path.join(full_path, 'report.csv')) as file:
            reader = csv.reader(file)
            #l = list(map(list, filter(lambda s: bool(s) , reader)))
            l = list(map(list, reader))
        for line in l:
            ema_task = []
            counter = 0
            try:
                if "[" in line[0]:
                    decay = ""
                    while not "]" in line[counter]:
                        decay += line[counter] + ", "
                        counter += 1
                    decay += line[counter]
                    counter += 1

                    decay_list.append(decay)
                else:
                    decay_list.append(line[0])

                for rep in line[counter+1:]:
                    if rep:
                        ema_task.append(float(rep))
            except IndexError:
                i = 0
                continue
            ema_list_of_tasks.append(ema_task)


    else:
        decay_list = get_decay_list(tree.getroot())
        with open(os.path.join(full_path, 'report.csv'), 'w') as f:
            for decay, task in zip(decay_list, task_names):
                line_to_write=decay + ","
                ema_task = []
                for rep in task:

                    dc = main.DataCollector("", "")
                    temp_path = full_path + "/" + rep
                    info = dc.load_rewards_list(temp_path)
                    ema = dc.listToEMA(info)
                    if ema is None:
                        print(decay, rep)
                        continue
                    else:
                        ema = ema[-1]
                    ema_task.append(ema)
                    line_to_write += (str(round(ema)) + ",")
                ema_list_of_tasks.append(ema_task)
                f.write(line_to_write + "\n")

    title = os.path.basename(os.path.normpath(full_path))
    boxplot(ema_list_of_tasks, title, labels=decay_list, save_path=full_path)

def boxplot(rows, title, save_path=None, labels=None, text=False):
    plt.figure()
    bp_dict = plt.boxplot(rows, labels=labels, showmeans=True)
    plt.xticks(rotation=-90)
    xoff = 0
    yoff = 0

    for line in bp_dict['means']:
        x, y = line.get_xydata()[0]
        if text:
            plt.text(x + xoff, y + yoff, '%3.f' % y,
                    horizontalalignment='center',  # centered
                    verticalalignment='top')

    plt.title(title)
    print(bp_dict["means"])
    print(bp_dict.keys())
    plt.tight_layout()
    if save_path:
        plt.savefig(os.path.join(save_path, title.replace(" ", "_") + ".png"), dpi=300)
    plt.show()



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--scenario", type=str,
                        help="scenario to report on")
    parser.add_argument("-v", "--video", action="store_true",
                        help="shoot video for scenario")
    parser.add_argument("-t", "--test", action="store_true", help="evaluate average reward")
    parser.add_argument("-n", type=int,
                        help="number of videos to shoot")
    parser.add_argument("-s", action="store_true", help="save")
    parser.add_argument("-p", "--picture", action="store_true", help="generate plot from rewards list")
    parser.add_argument("-r", "--report", action="store_true", help="generate report")
    parser.add_argument("-b", "--box", action="store_true", help="generate boxplot from report.csv")

    args = parser.parse_args()
    path = "./experiments/"
    temp_path = ""

    if args.scenario: # If path to scenario experiments folder was supplied
        scenario = args.scenario
        if os.path.exists(path + args.scenario):
            temp_path = path + args.scenario
        else:
            print("Scenario is not found. Aborting...")
            exit(1)
    else:
        lst = os.listdir(path)
        start = "scenario"
        scenario = max(filter(lambda k: k.startswith(start), lst)) # Find the latest scenario
        temp_path = path + scenario


    if args.video:
        automation.demo(temp_path, type="video")
    elif args.test:
        automation.demo(temp_path, type="test")
    elif args.picture:
        processPicture(temp_path, scenario)
    elif args.report:
        generateReport(temp_path)
    elif args.box:
        generateReport(temp_path, from_file=True)
        #automation.demo(temp_path, type="video")
        #automation.demo(temp_path, type="test")
        #processPicture(temp_path, scenario)

