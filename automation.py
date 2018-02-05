import gc
import argparse
import os
import datetime
import xml.etree.ElementTree as ET
import main
import shutil
import sys
import time
gc.enable()

def scenario(_scenario, cont=""):
    cur_time = '{0:%Y-%m-%d_%H-%M-%S}'.format(datetime.datetime.now()) # before scenario becomes scenario.xml
    save_folder = "./experiments/" + _scenario + "_" + cur_time
    os.makedirs(save_folder, exist_ok=True)


    _scenario = _scenario + ".xml"
    if not os.path.isfile("./" + _scenario):
        print("Create and fill file xml scenario file.")
    else:
        tree = ET.parse(_scenario)
        root = tree.getroot()
        i = 0
        with open(save_folder + '/status.txt', 'w') as the_file:
            the_file.write('Executing scenario\n')
        if cont:
            copy_folders("./experiments/" + cont, save_folder, "Task")
        for t in root:
            i = i+1
            t_folder = save_folder + "/Task" + str(i)
            os.makedirs(t_folder, exist_ok=True)
            shutil.copy("./" + _scenario, save_folder)
            start_time = time.time()
            task(t, t_folder)
            time_took = (time.time() - start_time) / 60
            with open(save_folder + '/status.txt', 'a') as the_file:
                the_file.write(str(i)+ " " + str(time_took) + ' m\n')
        with open(save_folder + '/status.txt', 'a') as the_file:
            the_file.write('Finished\n')

def task(_task, save_folder):
    rl_world = main.World(RENDER_STEP=False, RENDER_delay=0, TRAIN=True, NOISE=True)
    if _task.attrib["type"] == "simulation":
        os.makedirs(save_folder + "/saved_actor_networks", exist_ok=True)
        os.makedirs(save_folder + "/saved_critic_networks", exist_ok=True)

        rl_world.ENV_NAME = _task[0].attrib["name"]
        rl_world.TEST = 10
        rl_world.TEST_ON_EPISODE = 200

        el_actor = _task[0][0]
        el_critic = _task[0][1]
        el_end_criteria = _task[0][2]
        rl_world.ACTOR_SETTINGS = el_actor.attrib
        rl_world.ACTOR_SETTINGS["layers"] = []
        for child in el_actor:
            rl_world.ACTOR_SETTINGS["layers"].append(int(child.text))
        rl_world.CRITIC_SETTINGS = el_critic.attrib
        rl_world.CRITIC_SETTINGS["layers"] = []
        for child in el_critic:
            rl_world.CRITIC_SETTINGS["layers"].append(int(child.text))

        end_criteria = el_end_criteria.attrib["criteria"]
        if end_criteria == "episodes":
            rl_world.EPISODES = int(el_end_criteria.text)
        elif end_criteria == "time": # TODO: Implement
            rl_world.TIME_LIMIT = int(el_end_criteria.text) * 60  # minutes * seconds
        elif end_criteria == "solved":
            rl_world.EPISODES = 10000000000000
            rl_world.UNTIL_SOLVED = True
            rl_world.AVG_REWARD = int(el_end_criteria[0].text)
            rl_world.OVER_LAST = int(el_end_criteria[1].text)
        try:
            rl_world.main(save_folder)
        except Exception as e:
            print(e)
            with open(save_folder + '/exception.txt', 'a') as the_file:
                the_file.write('Exception '+str(e)+'\n')

def copy_folders(src, dst, beginning_of_name):
    for f in os.listdir(src + "/"):
        if f.startswith(beginning_of_name) and os.path.isdir(src + "/" + f):
            shutil.copytree(src + "/" + f, dst + "/" + f)
#def read_file_to_list:


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("scenario", type=str,
                        help="scenario to execute")
    parser.add_argument("-c", "--cont", type=str,
                        help="execute scenario using weights from old scenario c")
    args = parser.parse_args()


    if args.cont:
        if os.path.exists("./experiments/" + args.cont):
            scenario(args.scenario, cont=args.cont)
        else:
            print("Source directory does not exist. Aborting...")
            exit(1)

    else:
        scenario(args.scenario)
        pass
