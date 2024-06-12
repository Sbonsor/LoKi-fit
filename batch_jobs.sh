#!/bin/sh

mpiexec -n 10 './MCMC_auto_proposal_no_BH.py'

echo 'In between runs now!'

mpiexec -n 10 './run_14.py'




