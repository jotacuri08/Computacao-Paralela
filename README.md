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


Algoritmo de agrupamento que divide 𝑁 pontos em K clusters.

A cada iteração:

  Atribuição: cada ponto é associado ao centróide mais próximo.

  Atualização: cada centróide vira a média dos pontos do seu cluster.
  Repete até convergir ou atingir o máx. de iterações.

Versões implementadas

Sequencial (kmeans.cpp)
Um único processo/CPU executa as fases de atribuição e atualização.

OpenMP (kmeans_omp.cpp)
Paraleliza a atribuição com múltiplos threads (compartilhando memória).
OMP_NUM_THREADS define o número de threads.

MPI + OpenMP (kmeans_mpi.cpp)
MPI divide os pontos entre processos; OpenMP paraleliza dentro de cada processo.
Ao final de cada iteração, agregamos somas/contagens via MPI_Allreduce para atualizar centróides globais.
