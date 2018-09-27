import argparse
import os
import pycronos.agent.DDPG.automation
import datetime
import xml.etree.ElementTree as ET
import pycronos.agent.DDPG.main
import shutil
import sys
import time
import numpy as np


def processPicture(path):
    files = automation.get_files_starting_with(path, "Task")
    files_filtered = list(filter(lambda s: not s.endswith("test"), files))
    for f in files_filtered:
        dc = main.DataCollector("","")
        print(path + "/" + f)
        test = dc.load_test_list(path + "/" + f + "_test")

        test_points_x_start = 0

        info = dc.load_rewards_list(path + "/" + f)
        last_points_to_show = len(info)
        test_points = len(test[0])

        test[0] = np.array(test[0])[-test_points:] + test_points_x_start
        test[1] = np.array(test[1])[-test_points:]

        ema = dc.listToEMA(info)[-last_points_to_show:]
        info = info[-last_points_to_show:]

        title = args.scenario + " " + f
        labels = ["episodes", "rewards"]
        legend = ["Reward on noise", "EMA"]
        main.generatePlot(info, ema, x_start=test_points_x_start, scatter=test, title=title, labels=labels, legend=legend, save_folder=temp_path)

def generateReport():
    print("Report Generated")



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

    args = parser.parse_args()
    path = "./experiments/"
    temp_path = ""

    if args.scenario: # If path to scenario experiments folder was supplied
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
        processPicture(temp_path)
    elif args.report:
        automation.demo(temp_path, type="video")
        automation.demo(temp_path, type="test")
        processPicture(temp_path)

