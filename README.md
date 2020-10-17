# peg-in-hole-gym

gym env for multi-task simulation with Panda robotic arm engined by pybullet

## Task List

- peg-in-hole

  ![peg-in-hole](./img/peg-in-hole.gif)

- random-fly

  ![random-fly](./img/random-fly.gif)

## Installation

```bash
$ git clone git@github.com:guodashun/peg-in-hole-gym.git
$ cd peg-in-hole-gym
$ pip install -e .
```

## Usage

##### Initial the environment

```python
import gym
import peg_in_hole_gym

env = gym.make('peg-in-hole-v0')
env.render()
env.reset()
```

##### Perform a task(e.g. random fly)

```python
env.reset_fly()
while True:
	time.sleep(0.01)
	p.stepSimulation()
```

## Reference

- [PyBullet Quickstart Guide](https://docs.google.com/document/d/10sXEhzFRSnvFcl3XxNGhnD4N2SedqwdAvK3dsihxVUA/edit?usp=sharing)

