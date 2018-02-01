import xml.etree.ElementTree as ET
import paramiko as pssh
import os
import main
from ssh_wrapper import *

def main(_host, _port, _login, _password):
    ssh = None
    try:
        ssh = pssh.SSHClient()
        ssh.set_missing_host_key_policy(pssh.AutoAddPolicy())
        ssh.connect(_host, port=_port, username=_login, password=_password)
        ssh_wrap = StdPipeWrapper(ssh)

        path = "~/DDPG_working/DDPG/"

        #print(ssh_wrap.exec("ls", "-l"))
        main.exec_scenario("scenario1")

    finally:
        if ssh:
            ssh.close()



def exec_scenario(scenario, ssh_wrapper):
    pass


if __name__ == '__main__':
    if not os.path.isfile("./credentials.xml"):
        print("Create and fill file credentials.xml")
    else:
        tree = ET.parse('credentials.xml')
        root = tree.getroot()
        credentials = {elem.tag:elem.text for elem in root}
        main(credentials['host'], credentials['port'], credentials['login'], credentials['password'])

