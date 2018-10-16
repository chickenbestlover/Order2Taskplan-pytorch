# Order2Taskplan-pytorch

This is a PyTorch implementation of the Episodic memory module described in our IROS 2018 paper "Adaptive Task Planner for Performing Home Service Tasks in Cooperation with a Human".
Please note that this repository contains only the implementation of the episodic memory module(the deep learning part) in the paper.

![concept](./fig/order2taskplan_concept.pdf)


## Requirements
* Ubuntu 16.04+ 
* Python 3.6+
* Pytorch 0.3.1+
* Numpy
* Matplotlib

## Installing Order2Taskplan-pytorch

Order2Taskplan requires Python 3.6 or higher. It also requires installing PyTorch 0.3.1+ (warnings occur on 0.4.0). Its other dependencies are listed in requirements.txt. CUDA is strongly recommended for speed, but not necessary.

Run the following commands to clone the repository and install Order2Taskplan:
```
git clone https://github.com/chickenbestlover/Order2Taskplan-pytorch.git
cd Order2Taskplan-pytorch; pip install -r requirements.txt
```


## Citation

Please cite the IROS 2018 paper if you use Order2Taskplan in your work:

```
@inproceedings{lee2018taskplan,
  title={Adaptive Task Planner for Performing Home Service Tasks in Cooperation with a Human},
  author={Lee, Seung-Jae and Park, Jin-Man and Kim, Deok-Hwa and Kim, Jong-Hwan},
  booktitle={Intelligent Robots and Systems (IROS), 2018 IEEE/RSJ International Conference on},
  year={2018}
  organization={IEEE}
}
```
