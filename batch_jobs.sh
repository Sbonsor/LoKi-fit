#!/bin/sh

mpiexec -n 10 './parallel_max_likelihood.py'

echo 'In between runs now!'

mpiexec -n 10 './parallel__single_derivatives.py'




