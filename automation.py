import gc
import argparse
import os
import datetime
import xml.etree.ElementTree as ET
import main
import shutil
import time
import traceback
import report

from utils import *


gc.enable()

SCENARIOS_FOLDER = "./scenarios/"
EXPERIMENTS_FOLDER = "./experiments/"

def __get_tasks_names(tasks, lists_in_list=False):
    res = []

    for i, t in enumerate(tasks):
        executions_num = transfer_parameter(t.attrib, "executions", 1)
        if executions_num > 1:
            if lists_in_list:
                rep_list = []
                for j in range(executions_num):
                    rep_list.append(str(i + 1) + "_" + str(j + 1))
                res.append(rep_list)
            else:
                for j in range(executions_num):
                    res.append(str(i+1) + "_" + str(j+1))
        else:
            if lists_in_list:
                res.append([str(i + 1)])
            else:
                res.append(str(i + 1))
    return res

def scenario(scenario, old_scenario_folder="", copy_task=None, plot=False):
    cur_time = '{0:%Y-%m-%d_%H-%M-%S}'.format(datetime.datetime.now()) # before scenario becomes scenario.xml
    save_folder = EXPERIMENTS_FOLDER + scenario + "_" + cur_time
    os.makedirs(save_folder, exist_ok=True)
    ask_for_description(save_folder, cur_time)
    scenario = SCENARIOS_FOLDER + scenario + ".xml"
    if not os.path.isfile(scenario):
        print("Create and fill file xml scenario file.")
    else:
        tree = ET.parse(scenario)
        tasks = tree.getroot()
        save_nets = transfer_parameter(tasks.attrib, "save_nets", True)
        tasks_names = __get_tasks_names(tasks)
        with open(save_folder + '/status.txt', 'w') as the_file:
            the_file.write('Executing scenario\n')
        if old_scenario_folder:
            load_folder = EXPERIMENTS_FOLDER + old_scenario_folder
            if copy_task is None:
                copy_folders(load_folder, save_folder, starts_with="Task")
                reward_files = get_files_starting_with(load_folder, "Task")
                for f in reward_files:
                    shutil.copy(load_folder + "/" + f, save_folder)
            else:
                copy_folder_and_duplicate(load_folder, save_folder, "Task" + copy_task, tasks_names)
                copy_file_and_duplicate(load_folder, save_folder, "Task" + copy_task, tasks_names)

        counter = 0
        for i, t in enumerate(tasks):
            executions = transfer_parameter(t.attrib, "executions", 1)
            for rep in range(executions):
                ordinal = tasks_names[counter]
                counter += 1
                t_folder = save_folder + "/Task" + ordinal
                os.makedirs(t_folder, exist_ok=True)
                shutil.copy(scenario, save_folder)
                start_time = time.time()
                task(t, t_folder, save_when_training=save_nets)
                time_took = (time.time() - start_time) / 60
                with open(save_folder + '/status.txt', 'a') as the_file:
                    the_file.write(str(i)+ " " + str(time_took) + ' m\n')
        if plot:
            report.processPicture(save_folder, scenario + "_" + cur_time)
        with open(save_folder + '/status.txt', 'a') as the_file:
            the_file.write('Finished\n')

def demo(old_scenario_folder, type=""):
    assert old_scenario_folder is not ""

    lst = os.listdir(old_scenario_folder)
    start = "scenario"
    scenario = max(filter(lambda k: k.startswith(start) and not k.endswith("png"), lst)) # Find scenario file to get the settings

    tree = ET.parse(os.path.join(SCENARIOS_FOLDER, scenario))
    tasks = tree.getroot()

    counter = 0
    tasks_names = __get_tasks_names(tasks)
    for i, t in enumerate(tasks):
        executions = transfer_parameter(t.attrib, "executions", 1)
        for rep in range(executions):
            ordinal = tasks_names[counter]
            counter += 1
            t_folder = old_scenario_folder + "/Task" + ordinal
            if type == "video":
                os.makedirs(old_scenario_folder + "/Videos", exist_ok=True)
            if os.path.isdir(t_folder):
                task(t, t_folder, demo=True, demo_type=type)
            else:
                print("{0} not exists".format(t_folder))


def task(_task, save_folder, demo=False, demo_type=None, save_when_training=True):
    if demo:
        if demo_type == "video":
            rl_world = main.World(RENDER_STEP=True, RENDER_delay=0.0002, TRAIN=False, NOISE=False)
        elif demo_type == "test":
            rl_world = main.World(RENDER_STEP=False, RENDER_delay=0, TRAIN=False, NOISE=False)

        else:
            print("Demo type undefined")
            exit(1)
    else:
        noise = transfer_parameter(_task[0].attrib, "noise", not_found=True)
        render_step = transfer_parameter(_task[0].attrib, "render", not_found=False)
        rl_world = main.World(RENDER_STEP=render_step, RENDER_delay=0, TRAIN=True, NOISE=noise)
        if noise:
            noise_switching_period = transfer_parameter(_task[0].attrib, "noise_switch", not_found=False)
            if noise_switching_period is not False:
                rl_world.NOISE_PERIOD = noise_switching_period

    if _task.attrib["type"] == "simulation":
        os.makedirs(save_folder + "/saved_actor_networks", exist_ok=True)
        os.makedirs(save_folder + "/saved_critic_networks", exist_ok=True)

        rl_world.ENV_NAME = _task[0].attrib["name"]
        rl_world.OBSERVATIONS = transfer_parameter(_task[0].attrib, "observations", not_found="state")
        rl_world.TEST_NUM = 10
        rl_world.TEST_ON_EPISODE = 100


        rl_world.SAVE = save_when_training
        el_actor = _task[0][0]
        el_critic = _task[0][1]
        el_end_criteria = _task[0][2]


        rl_world.ACTOR_SETTINGS = fill_default_paramers_for_net(strdict_to_numdict(el_actor.attrib))
        rl_world.CRITIC_SETTINGS = fill_default_paramers_for_net(strdict_to_numdict(el_critic.attrib))

        #This block is an example of bad code, i know. I put info from layers xml blocks as another attribute in network settings.
        rl_world.ACTOR_SETTINGS["layers"] = []
        rl_world.ACTOR_SETTINGS["layers_settings"] = []
        for child in el_actor:
            rl_world.ACTOR_SETTINGS["layers"].append(int(child.text))
            rl_world.ACTOR_SETTINGS["layers_settings"].append(strdict_to_numdict(child.attrib))
        rl_world.CRITIC_SETTINGS["layers"] = []
        rl_world.CRITIC_SETTINGS["layers_settings"] = []
        for child in el_critic:
            rl_world.CRITIC_SETTINGS["layers"].append(int(child.text))
            rl_world.CRITIC_SETTINGS["layers_settings"].append(strdict_to_numdict(child.attrib))

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
                rl_world.EPISODES = 2
                rl_world.UNTIL_SOLVED = False
                rl_world.TEST_ON_EPISODE = 1
                rl_world.TEST_NUM = 24
                rl_world.TEST = True
                rl_world.TEST_SAVE = True
            else:
                print("Wrong demo type. Exiting...")
                exit(1)
        try:
            rl_world.main(save_folder, change_saved=(not demo))
        except EnvironmentError:
            print("\n\n MUJOCO UNSTABLE, exitting task \n\n")
            rl_world.finish(change_saved=False)
        except Exception as e:
            print(e)
            traceback.print_exc()
            with open(save_folder + '/exception.txt', 'a') as the_file:
                the_file.write('Exception '+str(e)+'\n')
            exit(1)


def ask_for_description(save_folder, time):
    print("Please write a description for the experiment or press enter otherwise.")
    desc = input()
    if desc:
        msg = time + "\n" + desc
        with open(save_folder + "/description.txt", "w") as myfile:
            myfile.write(msg)
        with open(EXPERIMENTS_FOLDER + "journal.txt", "a") as myfile:
            myfile.write("\n" + msg + "\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("scenario", type=str,
                        help="scenario to execute")
    parser.add_argument("-c", "--cont", type=str,
                        help="execute scenario by copying tasks results from old scenario c")

    parser.add_argument("-n", "--task", type=str,
                        help="execute scenario by copying old task for each new task")

    parser.add_argument("-p", "--picture", action="store_true", help="generate plot after scenario executed")

    args = parser.parse_args()


    if args.cont:
        if os.path.exists(EXPERIMENTS_FOLDER + args.cont):
            scenario(args.scenario, old_scenario_folder=args.cont, copy_task=args.task, plot=args.picture)
        else:
            print("Source directory does not exist. Aborting...")
            exit(1)

    else:
        scenario(args.scenario, plot=args.picture)
        pass
