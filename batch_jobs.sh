#!/bin/sh

mpiexec -n 10 './run_9.py'

echo 'In between runs now!'

mpiexec -n 10 './run_10.py'


