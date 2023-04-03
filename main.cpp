#include"BoltzmannSimulation.h"
#include<iostream>
#include <omp.h>
#include <Kokkos_Core.hpp>


// argv = "W H GIRD_SIZE NUMBER_OF_ITERATIONS NUM_THREADS"
int main(int argc, char** argv)
{
    Kokkos::initialize(
        Kokkos::InitializationSettings()
        .set_disable_warnings(false)
        .set_num_threads(1));
    int W = 11000;
    int H = 11000;
    int NUMBER_OF_ITERATIONS = 3;



    if (argc != 1 && argc != 4)
    {
        Kokkos::finalize();
        return -1;
    }

    if (argc > 1)
    {
        W = atoi(argv[1]);
        H = atoi(argv[2]);
        NUMBER_OF_ITERATIONS = atoi(argv[3]);
    }



    BoltzmannSumulation bs(W, H);

    double start = omp_get_wtime();
    for (int i = 0; i < NUMBER_OF_ITERATIONS; i++)
    {
        bs.update();
        //std::cout << i << "\n";
    }
    double end = omp_get_wtime();

    //freopen("awdef.txt", "w", stdout);
    std::cout << end - start << " ";


    Kokkos::finalize();

    return 0;
}