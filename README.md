# DDPG

Fork of DDPG from songrotek, with added functionality for remote execution on server + result analysis

[http://arxiv.org/abs/1509.02971](http://arxiv.org/abs/1509.02971)


Requirements:
Mujoco 1.31
OpenAI Gym 0.5.7


### Interface:

##### automation.py

Schedule different experiments to be simulated, specifying settings in xml file.

required arguments:
  xml-scenario name to execute: "scenario2", located in the project's folder

optional arguments:
  -c, --cont      continue executing scenario by copying tasks results from old scenario c
  -n, --task      execute scenario by copying old task for each new task

##### client.py 

Download results of experiments from remote server 

optional arguments:
  -h, --help      show this help message and exit
  -d, --download  download experiments which you don't have yet
  -R, --remove    shoot video for scenario


##### report.py









### References:

1 [https://github.com/songrotek/DDPG](https://github.com/songrotek/DDPG)

2 [https://github.com/rllab/rllab](https://github.com/rllab/rllab)

3 [https://github.com/MOCR/DDPG](https://github.com/MOCR/DDPG)

4 [https://github.com/SimonRamstedt/ddpg](https://github.com/SimonRamstedt/ddpg)




