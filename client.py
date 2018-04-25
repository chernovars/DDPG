import xml.etree.ElementTree as ET
import paramiko as pssh
import os
import time
import shlex
import subprocess
from ssh_wrapper import *
import argparse
from automation import EXPERIMENTS_FOLDER

CREDENTIALS_FOLDER = "./credentials/"
CLIENT_PATH = "/media/ars/lin_part2/Git2/DDPG/"
CLIENT_EXP_PATH = os.path.join(CLIENT_PATH, EXPERIMENTS_FOLDER)

def main(_host, _port, _login, _password, server_folder):
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--download", action="store_true",
                        help="download experiments which you don't have yet")
    parser.add_argument("-s", "--stats", action="store_true",
                        help="option for -d to downoald experiments stats only, not networks")
    parser.add_argument("-e", "--experiment", type=str,
                        help="specify experiment name")
    parser.add_argument("-R", "--remove", action="store_true",
                        help="shoot video for scenario")
    args = parser.parse_args()

    ssh = None
    try:
        ssh = pssh.SSHClient()
        ssh.set_missing_host_key_policy(pssh.AutoAddPolicy())
        ssh.connect(_host, port=_port, username=_login, password=_password)
        ssh_wrap = StdPipeWrapper(ssh)

        if args.download:
            list_of_folders_serv = _get_list_of_folders_serv(ssh_wrap, os.path.join(server_folder, EXPERIMENTS_FOLDER))
            if args.stats:
                exp_full_path = os.path.join(server_folder, EXPERIMENTS_FOLDER, args.experiment)
                stats_names = ssh_wrap.exec("ls -p " + exp_full_path + " | grep -v /").split("\n")
                download_stats(_host, str(_port), _login, server_folder, args.experiment, stats_names)
            else:
                folders_client = _get_list_of_folders_client(CLIENT_EXP_PATH)
                diff = list(set(list_of_folders_serv) - set(folders_client))
                if args.remove:
                    print(diff," to be downloaded")
                    download_experiments(ssh_wrap, _host, str(_port), _login, server_folder,
                                         delete_originals=True, experiments=diff)
                else:
                    download_experiments(ssh_wrap, _host, str(_port), _login, server_folder, experiments=diff)
    finally:
        if ssh:
            ssh.close()


def _get_list_of_folders_serv(ssh_wrap, path):
    return ssh_wrap.exec("ls " + path).split("\n")[:-1]

def _get_list_of_folders_client(path):
    filenames = os.listdir(path)
    result = []
    for filename in filenames:
        if os.path.isdir(os.path.join(path, filename)):
            result.append(filename)
    return result

def download_experiments(ssh_wrap, _host, _port, _login, server_folder, experiments=None, delete_originals=False):
    print(ssh_wrap.exec("ls", "-l"))

    if delete_originals:
        do_remove = "--remove-files "
    else:
        do_remove = ""

    if experiments is None:
        print(ssh_wrap.exec("tar cvf " + do_remove + "DDPG.tar -C " + server_folder + " ./experiments"))
    else:
        if len(experiments) == 0:
            print("Nothing new to download. Exiting...")
            return

        for i, f in enumerate(experiments):
            if i == 0:
                print(ssh_wrap.exec("tar cvf " + do_remove + "DDPG.tar -C " + server_folder + " ./experiments/" + f))
            else:
                print(ssh_wrap.exec("tar uvf " + do_remove + "DDPG.tar -C " + server_folder + " ./experiments/" + f))

    print(ssh_wrap.exec("mkdir " + server_folder + "experiments")) #in case if deleted
    scp_path = _login + "@" + str(_host) + ":~/DDPG.tar"
    shell("scp -P " + _port + " " + scp_path + " " + CLIENT_PATH)
    shell("tar xvf "+ CLIENT_PATH +" DDPG.tar -C " + CLIENT_PATH)
    shell("ls -l " + CLIENT_EXP_PATH)


def download_stats(_host, _port, _login, server_folder, experiment_name, stat_names):
    shell("mkdir -p " + os.path.join(CLIENT_EXP_PATH, experiment_name))
    print("Files to download: ", len(stat_names))
    for i, s in enumerate(stat_names):
        print(i)
        scp_path = _login + "@" + str(_host) + ":" + server_folder + "experiments/" + experiment_name + "/" + s
        shell("scp -P " + _port + " " + scp_path + " /media/ars/lin_part2/Git2/DDPG/experiments/" + \
              experiment_name + '/' + s, sleep=0.1)

def shell(command, sleep=2.0):
    subprocess.run([command], shell=True)
    time.sleep(sleep)

if __name__ == '__main__':
    credentials_file = "credentials.xml"
    if not os.path.isfile(CREDENTIALS_FOLDER + credentials_file):
        print("Create and fill file credentials.xml")
    else:
        tree = ET.parse(CREDENTIALS_FOLDER + credentials_file)
        root = tree.getroot()
        credentials = {elem.tag:elem.text for elem in root}
        main(credentials['host'], credentials['port'], credentials['login'], credentials['password'], credentials['path'])




