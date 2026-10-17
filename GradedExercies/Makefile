CC = gcc
MPICC = mpicc
CFLAGS ?= -O3
LDLIBS = -lm

TARGETS = serial_abc.out mpi_abc_split_a.out mpi_abc_split_ab.out
HEADERS = Libraries/arrays.h Libraries/linearalgebra.h Libraries/mt19937ar.h

.PHONY: all clean

all: $(TARGETS)

serial_abc.out: serial_abc.c $(HEADERS)
	$(CC) $(CPPFLAGS) $(CFLAGS) $(LDFLAGS) -o $@ $< $(LDLIBS)

mpi_abc_split_a.out: mpi_abc_split_a.c $(HEADERS)
	$(MPICC) $(CPPFLAGS) $(CFLAGS) $(LDFLAGS) -o $@ $< $(LDLIBS)

mpi_abc_split_ab.out: mpi_abc_split_ab.c $(HEADERS)
	$(MPICC) $(CPPFLAGS) $(CFLAGS) $(LDFLAGS) -o $@ $< $(LDLIBS)

clean:
	$(RM) $(TARGETS)
