import argparse
import os
import automation
import datetime
import xml.etree.ElementTree as ET
import main
import shutil
import sys
import time







if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--scenario", type=str,
                        help="scenario to report on")
    parser.add_argument("-v", "--video", type=str,
                        help="execute scenario using weights from old scenario c")
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
        files = automation.get_files_starting_with(temp_path, "Task")
        for f in files:
            dc = main.DataCollector("")
            print(temp_path + "/" + f)
            info = dc.load_rewards_list(temp_path + "/" + f)
            ema = dc.listToEMA(info)
            title = args.scenario + " " + f
            labels = ["episodes", "rewards"]
            legend = ["Reward on noise", "EMA"]
            main.generatePlot(info, ema, title=title, labels=labels, legend=legend, save_folder=temp_path)



    elif args.report:
        pass