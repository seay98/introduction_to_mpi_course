# mpi_abc_split_ab

`mpi_abc_split_ab.c` uses MPI to distribute matrices A and B by rows, compute `y = A × (B × c)`, and report the Euclidean norm of the result and execution times.

## Build

The environment should provide `mpicc` and `mpirun`, and `make` for the compilation, run:

```bash
make
```

This creates the executable `mpi_abc_split_ab.out`.

## Run

Run a small example with 4 MPI processes:

```bash
mpirun -np 4 ./mpi_abc_split_ab.out 1000 100 200 10 20 30
```

Arguments must appear in this order:

```text
N M K seed_a seed_b seed_c
```

- `N M K`: A has dimensions N × M, B has dimensions M × K, and vector c has length K. All three values must be positive integers.
- `seed_a seed_b seed_c`: Random seeds used to generate A, B, and c, respectively.
- `-np 4`: Starts 4 MPI processes. Adjust this value as needed.

Omit all program arguments to use the defaults `128000 500 1000 10 20 30`:

```bash
mpirun -np 4 ./mpi_abc_split_ab.out
```

The output includes the Euclidean norm of the result vector, the average initialization, calculation, and reduction times per process, and the maximum total elapsed time across all processes. All times are in seconds.

## Clean

```bash
make clean
```

This removes the generated `mpi_abc_split_ab.out` executable.

## Script for Aion

`aion_batch.sh` is the script for submitting the task to Aion.

## Aion result

`slurm_mpi_abc_16050113.log` is the output of the calcultions from Aion.
