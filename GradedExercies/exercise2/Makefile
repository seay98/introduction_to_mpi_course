CC = gcc
MPICC = mpicc
CFLAGS ?= -O3
LDLIBS = -lm

TARGETS = mpi_2d_heat.out serial_2d_heat.out

.PHONY: all clean

all: $(TARGETS)

mpi_2d_heat.out: mpi_2d_heat.c Libraries/arrays.h Libraries/mt19937ar.h
	$(MPICC) $(CPPFLAGS) $(CFLAGS) $(LDFLAGS) -o $@ $< $(LDLIBS)

serial_2d_heat.out: serial_2d_heat.c
	$(CC) $(CPPFLAGS) $(CFLAGS) $(LDFLAGS) -o $@ $< $(LDLIBS)

clean:
	$(RM) $(TARGETS)
