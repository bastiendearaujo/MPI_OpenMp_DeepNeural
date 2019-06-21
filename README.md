# MPI_OpenMp_DeepNeural

Recurrent Neural Network developed with MPI and OpenMp.
Use an NSL-KDD database. This algorithm detect the differents attacks presents in this database. You can see database in "examplesFiles" directory. 
The files that he used to learn and test is define at the begin of the file "rnn.c" in sources directory. 

Usage : 
$ make
$ mpiexec -np NumberOfProcess ./rnnProgramm

Or for debug and see all information : 
$ make alinf
$ mpiexec -np NumberOfProcess ./rnnProgramm