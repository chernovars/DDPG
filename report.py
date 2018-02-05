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
    parser.add_argument("-n", type=int,
                        help="number of videos to shoot")
    parser.add_argument("-s", action="store_true", help="save")

    args = parser.parse_args()
    path = "./experiments/"
    if args.scenario:
        if os.path.exists(path + args.scenario):
            automation.demo(path + args.scenario)
        else:
            print("Scenario is not found. Aborting...")
            exit(1)
    else:
        lst = os.listdir(path)
        start = "scenario"
        scenario = max(filter(lambda k: k.startswith(start), lst))
        automation.demo(path + scenario)

