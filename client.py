import xml.etree.ElementTree as ET
import paramiko as pssh
import os
import time
import shlex
import main
import subprocess
from ssh_wrapper import *

def main(_host, _port, _login, _password):
    ssh = None
    try:
        ssh = pssh.SSHClient()
        ssh.set_missing_host_key_policy(pssh.AutoAddPolicy())
        ssh.connect(_host, port=_port, username=_login, password=_password)
        ssh_wrap = StdPipeWrapper(ssh)

        path = "~/DDPG_working/DDPG/"

        print(ssh_wrap.exec("ls", "-l"))


        print(ssh_wrap.exec("tar czvf --remove-files DDPG.tar.gz -C ./DDPG_working/DDPG ./experiments")) # Compress (with overwriting the archive)

        print(ssh_wrap.exec("mkdir ./DDPG_working/DDPG/experiments"))

        scp_path = "arseniy@" + str(_host) + ":~/DDPG.tar.gz"

        shell("scp -P 2345 " + scp_path + " ~/Git/DDPG") # Donwload from the client side

        shell("tar xvzf ~/Git/DDPG/DDPG.tar.gz -C ~/Git/DDPG/") # Uncompress

        shell("ls -l ~/Git/DDPG/experiments")


    finally:
        if ssh:
            ssh.close()

    print("scp", "-P 2345", "arseniy@" + str(_host) + ":~/DDPG.tar.gz", "./Git/DDPG")

def exec_scenario(scenario, ssh_wrapper):
    pass

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

