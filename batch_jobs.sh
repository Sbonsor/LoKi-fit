#!/bin/sh

#mpiexec -n 10 './run_1.py'

#echo 'In between runs now!'

#mpiexec -n 10 './run_2.py'

#echo 'In betwen runs now!'

mpiexec -n 10 './run_3.py'

echo 'In between runs now!'

mpiexec -n 10 './run_4.py'

echo 'In between runs now!'

mpiexec -n 10 './run_3a.py'

echo 'In between runs now!'

mpiexec -n 10 './run_4a.py'

echo 'In between runs now!'

mpiexec -n 10 './run_5.py'
