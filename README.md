
# Paralelização do Algoritmo K-Means

Este repositório contém implementações sequencial e paralela (usando OpenMP e MPI+OpenMP) do algoritmo de clusterização K-Means. O projeto foi desenvolvido com o objetivo de analisar e comparar o ganho de performance (speedup) obtido com diferentes estratégias de paralelismo em relação à versão puramente sequencial.

## Sobre a Aplicação: O Algoritmo K-Means

K-Means é um popular algoritmo de aprendizado de máquina não supervisionado usado para **clusterização**. Seu objetivo é particionar um conjunto de $N$ pontos de dados em $K$ clusters distintos, onde cada ponto pertence ao cluster com o centroide (ponto médio) mais próximo.

O algoritmo funciona de forma iterativa em duas etapas principais:

1.  **Etapa de Atribuição (Assignment Step):** Cada ponto de dado é associado ao centroide mais próximo, geralmente com base na distância euclidiana.
    
2.  **Etapa de Atualização (Update Step):** O centroide de cada cluster é recalculado como a média de todos os pontos de dados que foram atribuídos a ele na etapa anterior.
    

Esses dois passos são repetidos até que os centroides não mudem significativamente (convergência) ou um número máximo de iterações seja atingido.

## Objetivo da Paralelização

O algoritmo K-Means, especialmente a **Etapa de Atribuição**, pode ser computacionalmente muito intensivo, principalmente com grandes conjuntos de dados (muitos pontos, $N$) ou alta dimensionalidade. Para cada ponto, é preciso calcular sua distância para _todos_ os $K$ centroides.

A intenção de paralelizar este código é **reduzir o tempo de execução** e medir o ganho de performance.

A Etapa de Atribuição é um gargalo de performance e também um candidato perfeito para **paralelismo de dados**. O cálculo da distância e a atribuição de um ponto ao seu centroide mais próximo são operações independentes para cada ponto. Isso significa que podemos dividir o conjunto de dados e processar múltiplos pontos simultaneamente.

### Estratégias Utilizadas

-   **OpenMP (Memória Compartilhada):** Esta versão usa threads para paralelizar o loop principal da Etapa de Atribuição dentro de um único processo (executando em uma máquina com múltiplos núcleos). As threads dividem o trabalho, onde cada uma processa um subconjunto dos pontos de dados, aproveitando a memória compartilhada.
    
-   **MPI + OpenMP (Híbrido):** Esta abordagem combina paralelismo de memória distribuída (MPI) e memória compartilhada (OpenMP).
    
    -   **MPI:** É usado para distribuir o conjunto de dados entre múltiplos processos, que podem estar em nós/máquinas diferentes em um cluster. Cada processo MPI é responsável por uma parte do conjunto de dados.
        
    -   **OpenMP:** Dentro de cada processo MPI, o OpenMP é usado para paralelizar o trabalho entre os núcleos daquela máquina específica, assim como na versão OpenMP pura.
        
    -   Esta abordagem híbrida é ideal para clusters de computadores modernos, onde cada nó possui múltiplos núcleos. A Etapa de Atualização requer comunicação entre os processos (via MPI, como `MPI_Allreduce`) para calcular os novos centroides globais.
        

## Como Compilar e Executar

### 1. Versão Sequencial

```
# Compilar
g++ -o kmeans_seq kmeans.cpp -std=c++11 -O3

# Executar
# O arquivo input_kmeans.txt é passado via redirecionamento de entrada
time ./kmeans_seq < input_kmeans.txt

```

### 2. Versão OpenMP

```
# Compilar
g++ -o kmeans_omp kmeans_omp.cpp -std=c++11 -O3 -fopenmp

# Definir o número de threads (ex: 8)
export OMP_NUM_THREADS=8

# Executar
time ./kmeans_omp < input_kmeans.txt

```

### 3. Versão MPI + OpenMP

```
# Compilar
mpic++ -fopenmp -O3 kmeans_mpi.cpp -o kmeans_mpi

# Definir o número de threads por processo (ex: 4)
export OMP_NUM_THREADS=4

# Executar (ex: com 2 processos MPI)
# Note que o arquivo de input é passado como argumento de linha de comando
time mpirun -np 2 ./kmeans_mpi input_kmeans.txt

```

