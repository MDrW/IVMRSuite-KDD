## Migration Planning Based Dynamic Algorithms - Machine Learning-based Category

### Run Imitation Learning (IL)
!!! Before you run imitation learning algorithm, you need run ecs_greedy and ecs_opt first to generate expert data.

- To run imitation learning
```
$ python process_expert_data.py
$ bash run_imitation.sh
```
You can generate dataset for IL by running command "python process_expert_data.py"

You can train and evaluate IL by running command "bash run_imitation.sh"

The parameters can be set for IL in the file "train_imitation_model.py" and "eval_imitation_model.py"
