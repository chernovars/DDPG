import gc
import argparse
import os
import datetime
import xml.etree.ElementTree as ET
import pycronos.agent.DDPG.main
import shutil
import time


# gc.enable()

def scenario(_scenario, old_scenario_folder="", copy_task=None):
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
        if old_scenario_folder:
            load_folder = "./experiments/" + old_scenario_folder
            if copy_task is None:
                copy_folders(load_folder, save_folder, "Task")
                reward_files = get_files_starting_with(load_folder, "Task")
                for f in reward_files:
                    src = load_folder + "/" + f
                    print(src)
                    print(save_folder)
                    shutil.copy(load_folder + "/" + f, save_folder)
            else:
                copy_folder_and_duplicate(load_folder, save_folder, "Task" + str(copy_task), len(list(root)))
                copy_file_and_duplicate(load_folder, save_folder, "Task" + str(copy_task), len(list(root)))

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

def demo(old_scenario_folder, type=""):
    assert old_scenario_folder is not ""

    lst = os.listdir(old_scenario_folder)
    start = "scenario"
    scenario = max(filter(lambda k: k.startswith(start), lst)) # Find scenario file to get the settings

    tree = ET.parse(scenario)
    root = tree.getroot()
    i = 0
    for t in root:
        i = i+1
        t_folder = old_scenario_folder + "/Task" + str(i)
        if type == "video":
            os.makedirs(old_scenario_folder + "/Videos", exist_ok=True)

        task(t, t_folder, demo=True, demo_type=type)

def task(_task, save_folder, demo=False, demo_type=None):
    if demo:
        if demo_type == "video":
            rl_world = pycronos.agent.DDPG.main.World(RENDER_STEP=True, RENDER_delay=0.0002, TRAIN=False, NOISE=False)
        elif demo_type == "test":
            rl_world = pycronos.agent.DDPG.main.World(RENDER_STEP=False, RENDER_delay=0, TRAIN=False, NOISE=False)
    else:
        rl_world = pycronos.agent.DDPG.main.World(RENDER_STEP=False, RENDER_delay=0, TRAIN=True, NOISE=True)

    if _task.attrib["type"] == "simulation":
        os.makedirs(save_folder + "/saved_actor_networks", exist_ok=True)
        os.makedirs(save_folder + "/saved_critic_networks", exist_ok=True)

        rl_world.ENV_NAME = _task[0].attrib["name"]
        rl_world.TEST = 10
        rl_world.TEST_ON_EPISODE = 100

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

        if demo:
            if demo_type == "video":
                rl_world.RECORD_VIDEO = True
                rl_world.EPISODES = 3
                rl_world.UNTIL_SOLVED = False
                rl_world.TEST_ON_EPISODE = 10 #so that we never test (because we have only 3 episodes)
            elif demo_type == "test":
                rl_world.EPISODES = 1
                rl_world.UNTIL_SOLVED = False
                rl_world.TEST_ON_EPISODE = 1
                rl_world.TEST = 100
            else:
                print("Wrong demo type. Exiting...")
                exit(1)
        try:
            rl_world.main(save_folder, data_save=demo)
        except Exception as e:
            print(e)
            with open(save_folder + '/exception.txt', 'a') as the_file:
                the_file.write('Exception '+str(e)+'\n')

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

#def read_file_to_list:


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("scenario", type=str,
                        help="scenario to execute")
    parser.add_argument("-c", "--cont", type=str,
                        help="execute scenario by copying tasks results from old scenario c")

    parser.add_argument("-n", "--task", type=int,
                        help="execute scenario by copying old task for each new task")

    args = parser.parse_args()


    if args.cont:
        if os.path.exists("./experiments/" + args.cont):
            scenario(args.scenario, old_scenario_folder=args.cont, copy_task=args.task)
        else:
            print("Source directory does not exist. Aborting...")
            exit(1)

    else:
        scenario(args.scenario)
        pass
