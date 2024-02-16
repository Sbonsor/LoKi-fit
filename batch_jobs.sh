#!/bin/sh

mpiexec -n 10 './parallel_fisher_calculation.py'

echo 'In between runs now!'

mpiexec -n 10 './run_11.py'

echo 'In between runs now!'

mpiexec -n 10 './run_12.py'


