# IVMR suite - accepted by KDD 2025 DB Track
This repository implements IVMR suite which is the datasets and codes of the paper "IVMR suite: An Industrial-scale Virtual Machine Rescheduling Dataset and Benchmark for Elastic Cloud Service" (https://doi.org/10.1145/3711896.3737381).


## Dataset: IVMR-D
The dataset IVMR-D is in the directory IVMR_Suite/data
Before you use the dataset, you need unzip the file to the directory ./IVMR_Suite/data
```
$ cd ./IVMR_Suite/data
$ unzip ecs_data.zip
```


## Algorithms: IVMR-B
The algorithms are in the directories:
- Optimal Assignment Based Static Algorithms
    * Integer Programming Solver (IP): ./IVMR_Suite/ecs_opt
    * Genetic Algorithm (GA): ./IVMR_Suite/ecs_metaheuristic/ga
    * Ant Colony Optimization (ACO): ./IVMR_Suite/ecs_metaheuristic/aco
- Migration Planning Based Dynamic Algorithms
    * Heuristic Greedy Algorithm (HGA): ./IVMR_Suite/ecs_greedy
    * Imitation Learning (IL): ./IVMR_Suite/ecs_imitation
    * Reinforcement Learning (RL): ./IVMR_Suite/ecs_rl (implemented based on the repository https://github.com/bytedance/DRL-based-VM-Rescheduling)

Before you run the algorithms, you need pip install the packages in the requirements.txt


## Citation
If you find this repository useful, please cite our paper:
https://doi.org/10.1145/3711896.3737381