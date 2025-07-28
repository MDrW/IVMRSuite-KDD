## Optimal Assignment Based Static Algorithms - Optimization Category

### Run Integer Programming Solver (IP)
- To run integer programming solver
```
$ python run_cplex_model_and_solve.py
$ python run_cplex_move_seq.py
$ python opt_log_parse.py
$ python resort_and_reindex_moveseq.py
```
You can get the optimal assignment for each case by running command "python run_cplex_model_and_solve.py"

You can get the move sequence using the optimal assignment by running command "python run_cplex_move_seq.py"

You can parse the logs to get the results of IP by running command "python opt_log_parse.py"

The parameters can be set for IP in the file "run_cplex_model_and_solve.py" and "run_cplex_move_seq.py"
