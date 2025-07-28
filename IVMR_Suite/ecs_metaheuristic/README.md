## Optimal Assignment Based Static Algorithms - Meta Heuristic Algorithms

### Run Genetic Algorithm (GA)
- To run genetic algorithm
```
$ cd ./ga
$ python genetic_algorithm.py
$ python placement2moveseq_ga.py
$ parse_logs_ga.py
```
You can get the optimal assignment for each case by running command "python genetic_algorithm.py"

You can get the move sequence using the optimal assignment by running command "python placement2moveseq_ga.py"

You can parse the logs to get the results of GA by running command "parse_logs_ga.py"

The parameters can be set for GA in the file "genetic_algorithm.py"

### Run Ant Colony Optimization (ACO)
- To run ant colony optimization
```
$ cd ./aco
$ python aco_algo.py
$ python placement2moveseq_aco.py
$ python parse_logs_aco.py
```
You can get the optimal assignment for each case by running command "python aco_algo.py"

You can get the move sequence using the optimal assignment by running command "python placement2moveseq_aco.py"

You can parse the logs to get the results of GA by running command "python parse_logs_aco.py"

The parameters can be set for ACO in the file "aco_algo.py"
