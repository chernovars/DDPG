class StdPipeWrapper:
    def __init__(self, ssh):


        self.ssh = ssh
        self.ssh_stderr = None
        self.ssh_stdin = None
        self.ssh_stdout = None

        self.errors = []
        self.ins = []
        self.outs = []

    def exec(self, command, *args):

        comm_with_args = "" + command
        for arg in args:
            comm_with_args += (" " + arg)

        self.ssh_stdin, self.ssh_stdout, self.ssh_stderr = self.ssh.exec_command(comm_with_args)
        self.errors.append(self.ssh_stderr)
        self.ins.append(self.ssh_stdin)
        self.outs.append(self.ssh_stdout)
        return self.read_out()

    def read_out(self):
        output = self.ssh_stdout.read()
        return output.decode("utf-8")