# Computacão Paralela
Projeto 1 MPI/OpenMP.

Como rodar o sequencial
g++ -o kmeans_seq kmeans.cpp -std=c++11 -O3
time ./kmeans_seq < input_kmeans.txt

Como rodar o OPENMP
g++ -o kmeans_omp kmeans_omp.cpp -std=c++11 -O3 -fopenmp
export OMP_NUM_THREADS=<numero de threads>
time ./kmeans_omp < input_kmeans.txt

Como rodar o OPENMP + MPI
mpic++ -fopenmp -O3 kmeans_mpi.cpp -o kmeans_mpi
export OMP_NUM_THREADS=<numero de threads>
time mpirun -np <numero de processos> ./kmeans_mpi input_kmeans.txt
