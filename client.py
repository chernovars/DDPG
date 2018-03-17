import xml.etree.ElementTree as ET
import paramiko as pssh
import os
import time
import shlex
import subprocess
from ssh_wrapper import *
import argparse



def main(_host, _port, _login, _password):
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--download", action="store_true",
                        help="download experiments which you don't have yet")
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
            a = _get_list_of_folders_serv(ssh_wrap, "~/DDPG_working/DDPG/experiments")
            b = _get_list_of_folders_client("/media/ars/lin_part2/Git2/DDPG/experiments")
            diff = list(set(a) - set(b))
            if args.remove:
                print(diff," to be downloaded")
                download_experiments(ssh_wrap, _host, delete_originals=True, experiments=diff)
            else:
                download_experiments(ssh_wrap, _host, experiments=diff)
    finally:
        if ssh:
            ssh.close()

#    print("scp", "-P 2345", "arseniy@" + str(_host) + ":~/DDPG.tar.gz", "./Git/DDPG")

def _get_list_of_folders_serv(ssh_wrap, path):
    return ssh_wrap.exec("ls " + path).split("\n")[:-1]

def _get_list_of_folders_client(path):
    filenames = os.listdir(path)
    result = []
    for filename in filenames:
        if os.path.isdir(os.path.join(path, filename)):
            result.append(filename)
    return result

def download_experiments(ssh_wrap, _host, experiments=None, delete_originals=False):
    path = "~/DDPG_working/DDPG/"

    print(ssh_wrap.exec("ls", "-l"))

    if delete_originals:
        do_remove = "--remove-files "
    else:
        do_remove = ""

    if experiments is None:
        print(ssh_wrap.exec("tar cvf " + do_remove + "DDPG.tar -C ./DDPG_working/DDPG ./experiments"))
    else:
        if len(experiments) == 0:
            print("Nothing new to download. Exiting...")
            return

        for i, f in enumerate(experiments):
            if i == 0:
                print(ssh_wrap.exec("tar cvf " + do_remove + "DDPG.tar -C ./DDPG_working/DDPG ./experiments/" + f))
            else:
                print(ssh_wrap.exec("tar uvf " + do_remove + "DDPG.tar -C ./DDPG_working/DDPG ./experiments/" + f))

    print(ssh_wrap.exec("mkdir ./DDPG_working/DDPG/experiments")) #in case if deleted

    scp_path = "arseniy@" + str(_host) + ":~/DDPG.tar"

    #scp -P 2345 arseniy@" + str(_host) + ":~/DDPG.tar /media/ars/lin_part2/Git2/DDPG/
    shell("scp -P 2345 " + scp_path + " /media/ars/lin_part2/Git2/DDPG/")

    shell("tar xvf /media/ars/lin_part2/Git2/DDPG/DDPG.tar -C /media/ars/lin_part2/Git2/DDPG/")

    shell("ls -l /media/ars/lin_part2/Git2/DDPG/experiments")

def shell(command):
    subprocess.run([command], shell=True)
    time.sleep(2)

if __name__ == '__main__':
    if not os.path.isfile("./credentials.xml"):
        print("Create and fill file credentials.xml")
    else:
        tree = ET.parse('credentials.xml')
        root = tree.getroot()
        credentials = {elem.tag:elem.text for elem in root}
        main(credentials['host'], credentials['port'], credentials['login'], credentials['password'])




