### Check video waypoint_vid.mp4

# EcoDrive Simulator

**EcoDrive Simulator** is a web-based smart mobility platform that leverages **Reinforcement Learning (RL)** to optimize vehicle trajectories and promote sustainable driving.  
It enables users to compare **AI-generated racing lines** with real driver telemetry, simulate **eco-friendly vs. aggressive** driving styles, and visualize energy efficiency across multiple race tracks.

---

## Project Overview

EcoDrive Simulator combines artificial intelligence, data visualization, and smart mobility concepts into a single platform.  
It demonstrates how **machine learning**, particularly **Proximal Policy Optimization (PPO)**, can be applied to improve energy efficiency and driving performance in both **autonomous** and **human-driven** scenarios.

---

## Objectives

1. **Conduct a systematic review** of reinforcement learning techniques, mobility datasets, and driver telemetry to define simulator requirements.  
2. **Design and implement** a web-based simulator with RL-powered racing line optimization and interactive comparative dashboards.  
3. **Pilot test** the system with users to evaluate trajectory accuracy, energy efficiency, and usability.

---

TODO list:
- [x] clip lidar after adding noise, put lidar noise in config
- [x] remove unnecessary use of dictionary. model st is 50% faster, mb is 150% faster.
- [x] toggle renderer
- [ ] verify collision
- [x] scan toggle
- [x] separate time step and integrater time step
- [x] add frenet
- [x] use correct loop count frenet_based, added max loop num
- [ ] also add winding_angle
- [x] simplify std_state and observation
- [x] add option "state" in reset
- [x] control_buffer_size, lidar fov, lidar nums in config
- [ ] double check dynamics result
- [x] implemented new rendering with pyqtgraph.opengl
- [x] added 3d mesh renderering for proof of concept
- [ ] merge lidar scan fix

## Quickstart
I recommend installing the simulation inside a virtualenv. You can install the environment by running:

```bash
virtualenv gym_env
source gym_env/bin/activate
cd f1tenth_gym
pip install -e .
```

Then you can run a quick waypoint follow example by:
```bash
cd examples
python3 waypoint_follow.py
```
