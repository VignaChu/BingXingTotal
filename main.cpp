#include "PCFG.h"
#include <chrono>
#include <fstream>
#include "md5_neon2.h"
#include <iomanip>
#include <mpi.h>
using namespace std;
using namespace chrono;

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);
    int world_rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    PriorityQueue q;
    double time_train = 0; // 声明在外部，便于最后输出
    if (world_rank == 0) {
        auto start_train = system_clock::now();
        q.m.train("./input/Rockyou-singleLined-full.txt");
        q.m.order();
        auto end_train = system_clock::now();
        auto duration_train = duration_cast<microseconds>(end_train - start_train);
        time_train = double(duration_train.count()) * microseconds::period::num / microseconds::period::den;
        q.init();
        // 不在这里输出
    }

    // 假设模型和priority都能被所有进程访问（如用文件共享或每个进程都训练一次模型）
    int total_pt = q.priority.size();
    MPI_Bcast(&total_pt, 1, MPI_INT, 0, MPI_COMM_WORLD);

    int chunk = (total_pt + world_size - 1) / world_size;
    int start = world_rank * chunk;
    int end = min(start + chunk, total_pt);

    int curr_num = 0;
    int history = 0;
    double time_hash = 0;
    double time_guess = 0;
    auto start_time = system_clock::now();

    for (int i = start; i < end && i < total_pt; ++i) {
        q.PopNext();
        q.total_guesses = q.guesses.size();
        if (q.total_guesses - curr_num >= 100000) {
            cout << "[Rank " << world_rank << "] Guesses generated: " << history + q.total_guesses << endl;
            curr_num = q.total_guesses;
        }
        if (curr_num > 1000000) {
            auto start_hash = system_clock::now();
            bit32 state[4];
            for (size_t j = 0; j < q.guesses.size(); j += 4) {
                string pw1 = (j + 0 < q.guesses.size()) ? q.guesses[j + 0] : "";
                string pw2 = (j + 1 < q.guesses.size()) ? q.guesses[j + 1] : "";
                string pw3 = (j + 2 < q.guesses.size()) ? q.guesses[j + 2] : "";
                string pw4 = (j + 3 < q.guesses.size()) ? q.guesses[j + 3] : "";
                MD5Hash(pw1, pw2, pw3, pw4, state);
            }
            auto end_hash = system_clock::now();
            auto duration = duration_cast<microseconds>(end_hash - start_hash);
            time_hash += double(duration.count()) * microseconds::period::num / microseconds::period::den;
            history += curr_num;
            curr_num = 0;
            q.guesses.clear();
        }
    }

    // 汇总所有进程的猜测数
    int total_history = 0;
    MPI_Reduce(&history, &total_history, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    if (world_rank == 0) {
        auto end_time = system_clock::now();
        auto duration = duration_cast<microseconds>(end_time - start_time);
        time_guess = double(duration.count()) * microseconds::period::num / microseconds::period::den;
        cout << "Guess time:" << time_guess - time_hash << "seconds" << endl;
        cout << "Hash time:" << time_hash << "seconds" << endl;
        cout << "Train time:" << time_train << "seconds" << endl;
    }

    MPI_Finalize();
    return 0;
}