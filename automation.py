import xml.etree.ElementTree as ET
import paramiko as pssh
import os

def main(_host, _port, _login, _password):

    ssh = pssh.SSHClient()
    ssh.set_missing_host_key_policy(pssh.AutoAddPolicy())
    ssh.connect(_host, port=_port, username=_login, password=_password)

    ssh_stdin, ssh_stdout, ssh_stderr = ssh.exec_command("ls")
    output = ssh_stdout.read()
    print(output.decode("utf-8"))


    #print(root.)


if __name__ == '__main__':
    if not os.path.isfile("./credentials.xml"):
        print("Create and fill file credentials.xml")
    else:
        tree = ET.parse('credentials.xml')
        root = tree.getroot()
        credentials = {elem.tag:elem.text for elem in root}
        main(credentials['host'], credentials['port'], credentials['login'], credentials['password'])

