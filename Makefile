all: main clean

CC=mpicc
SRC_DIR=sources
INCLUDE=-I./headers
OPTION=-Wall -fopenmp

OPTIM=-O3

# Executable
main:
	$(CC) $(OPTION) $(INCLUDE) ./sources/*.c -o rnnProgramm -lm $(OPTIM)

allinf:
	$(CC) $(OPTION) $(INCLUDE) -D ALLINF ./sources/*.c -o rnnProgramm -lm $(OPTIM)

clean:
	rm -f *.o *~

del:
	rm rnnProgramm