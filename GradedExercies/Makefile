MPICC = mpicc
CFLAGS ?= -O3
LDLIBS = -lm

TARGETS = mpi_abc_split_ab.out
HEADERS = Libraries/arrays.h Libraries/linearalgebra.h Libraries/mt19937ar.h

.PHONY: all clean

all: $(TARGETS)

mpi_abc_split_ab.out: mpi_abc_split_ab.c $(HEADERS)
	$(MPICC) $(CPPFLAGS) $(CFLAGS) $(LDFLAGS) -o $@ $< $(LDLIBS)

clean:
	$(RM) $(TARGETS)
