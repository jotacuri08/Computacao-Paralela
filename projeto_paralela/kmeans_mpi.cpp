/*
kmeans_mpi_omp.cpp

Hybrid MPI+OpenMP K-Means implementation based on original sequential version
Original code: https://github.com/marcoscastro/kmeans
Reference: http://mnemstudio.org/clustering-k-means-example-1.htm
*/

#include <bits/stdc++.h> // Includes most standard headers
#include <mpi.h>
#include <omp.h>
#include <fstream>       // <-- ADICIONADO para std::ifstream

using namespace std;
using vec = vector<double>; // <-- ADICIONADO
using mat = vector<vec>;   // <-- ADICIONADO

// Function to compute squared Euclidean distance
inline double dist2(const vec &a, const vec &b) {
    double s = 0.0;
    size_t n = min(a.size(), b.size());
    for (size_t i = 0; i < n; ++i) {
        double d = a[i] - b[i];
        s += d * d;
    }
    return s;
}

struct KMeansResult {
    mat centroids;
    vector<int> labels; // only filled on rank 0 after gather
    int iterations;
};

// Declaração da função (protótipo) para que o main() possa encontrá-la
void displayResults(const KMeansResult &res, const mat &data, int total_points, int total_values, int K);


KMeansResult kmeans_mpi_omp(const mat &data_full_root, int K, int max_iter, mt19937 &rng) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Root (rank 0) knows N_total and D; broadcast them to everyone.
    int N_total = 0;
    int D = 0;
    if (rank == 0) {
        N_total = (int)data_full_root.size();
        if (N_total > 0) D = (int)data_full_root[0].size();
    }
    MPI_Bcast(&N_total, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&D, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (N_total == 0 || D == 0) throw runtime_error("Empty dataset or unknown dimensionality.");

    // Compute counts and displacements (in number of samples)
    vector<int> counts(size, 0), displs(size, 0);
    int base = N_total / size;
    int rem = N_total % size;
    for (int i = 0; i < size; ++i) {
        counts[i] = base + (i < rem ? 1 : 0);
        displs[i] = (i == 0 ? 0 : displs[i - 1] + counts[i - 1]);
    }
    int local_N = counts[rank];

    // Prepare flattened send buffer on root
    vector<double> sendbuf;
    if (rank == 0) {
        sendbuf.reserve((size_t)N_total * D);
        for (const auto &row : data_full_root) {
            sendbuf.insert(sendbuf.end(), row.begin(), row.end());
        }
    }

    // Prepare recv buffer for local data (flattened)
    vector<double> recvbuf((size_t)local_N * D);
    vector<int> counts_d(size), displs_d(size);
    for (int i = 0; i < size; ++i) {
        counts_d[i] = counts[i] * D;
        displs_d[i] = displs[i] * D;
    }

    MPI_Scatterv(rank == 0 ? sendbuf.data() : nullptr, counts_d.data(), displs_d.data(), MPI_DOUBLE,
                 recvbuf.data(), local_N * D, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Build local_data matrix
    mat local_data(local_N, vec(D));
    for (int i = 0; i < local_N; ++i)
        for (int j = 0; j < D; ++j)
            local_data[i][j] = recvbuf[(size_t)i * D + j];

    // Initialize centroids on root by random sampling unique indices
    mat centroids(K, vec(D, 0.0));
    if (rank == 0) {
        unordered_set<int> chosen;
        uniform_int_distribution<int> uid(0, N_total - 1);
        for (int k = 0; k < K; ++k) {
            int idx;
            do { idx = uid(rng); } while (!chosen.insert(idx).second);
            centroids[k] = data_full_root[idx];
        }
    }

    // Broadcast initial centroids to all ranks (flattened)
    vector<double> cent_flat(K * D);
    if (rank == 0) {
        for (int k = 0; k < K; ++k)
            for (int d = 0; d < D; ++d)
                cent_flat[k * D + d] = centroids[k][d];
    }
    MPI_Bcast(cent_flat.data(), K * D, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    if (rank != 0) {
        for (int k = 0; k < K; ++k) {
            for (int d = 0; d < D; ++d) centroids[k][d] = cent_flat[k * D + d];
        }
    }

    vector<int> local_labels(local_N, -1);
    int iter;
    const double tol = 1e-6; // convergence tolerance

    for (iter = 0; iter < max_iter; ++iter) {
        // Local buffers for sums and counts
        vector<double> local_sum_flat((size_t)K * D, 0.0);
        vector<int> local_count(K, 0);

        int nthreads = omp_get_max_threads();
        vector<vector<double>> thread_sum_flat(nthreads, vector<double>((size_t)K * D, 0.0));
        vector<vector<int>> thread_count(nthreads, vector<int>(K, 0));

        // Parallel assign step: each thread writes to its private buffer
        #pragma omp parallel
        {
            int tid = omp_get_thread_num();
            #pragma omp for schedule(static)
            for (int i = 0; i < local_N; ++i) {
                double best = numeric_limits<double>::infinity();
                int bi = -1;
                for (int k = 0; k < K; ++k) {
                    // Compute distance between local_data[i] and centroids[k]
                    double s = 0.0;
                    for (int d = 0; d < D; ++d) {
                        double diff = local_data[i][d] - centroids[k][d];
                        s += diff * diff;
                    }
                    if (s < best) { best = s; bi = k; }
                }
                local_labels[i] = bi;
                // Accumulate to thread-local buffers
                for (int d = 0; d < D; ++d)
                    thread_sum_flat[tid][(size_t)bi * D + d] += local_data[i][d];
                thread_count[tid][bi]++;
            }
        }

        // Reduce thread buffers into local_sum_flat and local_count
        for (int t = 0; t < nthreads; ++t) {
            for (int k = 0; k < K; ++k) {
                local_count[k] += thread_count[t][k];
                for (int d = 0; d < D; ++d)
                    local_sum_flat[(size_t)k * D + d] += thread_sum_flat[t][(size_t)k * D + d];
            }
        }

        // MPI Allreduce for sums (flattened) and counts
        vector<double> global_sum_flat((size_t)K * D, 0.0);
        vector<int> global_count(K, 0);
        MPI_Allreduce(local_sum_flat.data(), global_sum_flat.data(), K * D, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(local_count.data(), global_count.data(), K, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

        // Update centroids and check convergence
        bool converged_local = true;
        for (int k = 0; k < K; ++k) {
            if (global_count[k] == 0) continue; // leave centroid unchanged
            vec updated(D);
            for (int d = 0; d < D; ++d)
                updated[d] = global_sum_flat[(size_t)k * D + d] / global_count[k];
            if (dist2(updated, centroids[k]) > tol) converged_local = false;
            centroids[k].swap(updated);
        }

        int local_conv_int = converged_local ? 1 : 0;
        int all_conv = 0;
        MPI_Allreduce(&local_conv_int, &all_conv, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
        if (all_conv) break;
    }

    // Gather labels to root
    vector<int> global_labels;
    if (rank == 0) global_labels.resize(N_total);
    MPI_Gatherv(local_labels.data(), local_N, MPI_INT,
                rank == 0 ? global_labels.data() : nullptr,
                counts.data(), displs.data(), MPI_INT, 0, MPI_COMM_WORLD);

    return {centroids, global_labels, iter + 1};
}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc < 2) {
        if (rank == 0) {
            cerr << "Erro: Forneça o nome do arquivo como argumento." << endl;
            cerr << "Uso: mpirun -np <N> " << argv[0] << " <arquivo_dados>" << endl;
        }
        MPI_Finalize();
        return 1;
    }

    string filename = argv[1];
    int total_points, total_values, K, max_iterations, has_name;
    mat data; // only filled on rank 0

    if (rank == 0) {
        ifstream infile(filename); // Abre o arquivo
        if (!infile) {
            cerr << "Erro: Nao foi possivel abrir o arquivo " << filename << endl;
            // Sinaliza erro para outros ranks definindo total_points como 0
            total_points = 0;
        } else {
            infile >> total_points >> total_values >> K >> max_iterations >> has_name;

            data.resize(total_points, vec(total_values));
            string point_name;

            for (int i = 0; i < total_points; i++) {
                for (int j = 0; j < total_values; j++) {
                    double value;
                    infile >> value;
                    data[i][j] = value;
                }
                if (has_name) {
                    infile >> point_name;
                    // We ignore point names for now in this simplified version
                }
            }
        }
    }

    // Broadcast dos parâmetros (inclusive o 'total_points' 0 em caso de erro)
    MPI_Bcast(&total_points, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Se total_points for 0, todos os ranks saem
    if (total_points == 0) {
        MPI_Finalize();
        return 1;
    }

    MPI_Bcast(&total_values, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&K, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&max_iterations, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&has_name, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Make RNG; different seed per rank
    unsigned seed = chrono::high_resolution_clock::now().time_since_epoch().count();
    mt19937 rng(seed + rank);

    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();

    KMeansResult res = kmeans_mpi_omp(data, K, max_iterations, rng);

    MPI_Barrier(MPI_COMM_WORLD);
    double t1 = MPI_Wtime();

    if (rank == 0) {
        cerr << "Converged in " << res.iterations << " iterations.\n";
        cerr << "Elapsed time: " << (t1 - t0) << " s\n";

        // Display results in the original format
        displayResults(res, data, total_points, total_values, K);
    }

    MPI_Finalize();
    return 0;
}

// Function to display results in original format
void displayResults(const KMeansResult &res, const mat &data, int total_points, int total_values, int K) {
    // Reorganize points by cluster
    vector<vector<int>> cluster_points(K);

    for (int i = 0; i < total_points; i++) {
        int cluster_id = res.labels[i];
        if (cluster_id >= 0 && cluster_id < K) {
            cluster_points[cluster_id].push_back(i);
        }
    }

    // Display clusters
    for (int k = 0; k < K; k++) {
        cout << "Cluster " << k + 1 << endl;

        for (int point_idx : cluster_points[k]) {
            cout << "Point " << point_idx + 1 << ": ";
            for (int d = 0; d < total_values; d++) {
                cout << data[point_idx][d] << " ";
            }
            cout << endl;
        }

        cout << "Cluster values: ";
        for (int d = 0; d < total_values; d++) {
            cout << res.centroids[k][d] << " ";
        }
        cout << "\n\n";
    }
}