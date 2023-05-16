#include"BoltzmannSimulation.h"
#include<iostream>
#include <fstream>
#include <iomanip>
#include <Kokkos_Core.hpp>


// argv = "W H GIRD_SIZE NUMBER_OF_ITERATIONS NUM_THREADS"
int main(int argc, char** argv)
{
    Kokkos::initialize(argc, argv);
    {
        int W = 110;
        int H = 110;
        int NUMBER_OF_ITERATIONS = 3;



        if (argc != 1 && argc != 4 && argc != 5)
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
        Kokkos::Timer timer;

        timer.reset();
        double start = timer.seconds();
        bs.update(NUMBER_OF_ITERATIONS);
        Kokkos::fence();  // synchronize


        if (argc == 5)
        {
            freopen(argv[4], "w", stdout);

            std::ios::sync_with_stdio(0); std::cin.tie(0); std::cout.tie(0);
            std::cout << std::fixed << std::showpoint;
            std::cout << std::setprecision(8);

            auto table = bs.get_table();
            for (int l = 0; l < 9; ++l)
                for (int i = 0; i < W * H; ++i)
                    std::cout << table(i, l) << "\n";

        }
        else
        {
            std::cout << timer.seconds() - start << std::endl;
        }
    }
    Kokkos::finalize();

    return 0;
}