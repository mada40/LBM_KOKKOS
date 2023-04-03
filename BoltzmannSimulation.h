#ifndef __TBoltzmannSumulation_H__
#define __TBoltzmannSumulation_H__

#include <iostream>
#include <string>
#include <cmath>
#include <Kokkos_Core.hpp>
#include "omp.h"

const int NUM_THREADS = 8;

const int NL = 9;


Kokkos::View<int*> invert_k;
Kokkos::View<int*> dxs_k;
Kokkos::View<int*> dys_k;
Kokkos::View<double*> weights_k;

class BoltzmannSumulation
{
private:
	const int Ny;
	const int Nx;
	const double TAU = 0.58;
	Kokkos::View<float*> new_table_k;
	Kokkos::View<float*> cur_table_k;
	Kokkos::View<bool*> isBondary_k;

	//812
	//703
	//654


	BoltzmannSumulation& operator = (const BoltzmannSumulation& other) { throw std::logic_error("MNE LEN' DELAT' ETOT OPERATOR"); }


	void init_sumulation()
	{

		Kokkos::parallel_for("init_sum", NL * Ny * Nx, KOKKOS_LAMBDA(int i)
		{
			cur_table_k(i) = 0.6;
			if (i / (Nx * Ny) == invert_k(4))
				cur_table_k(i) = 1.2;
		});

		Kokkos::parallel_for("init_sum2", Ny * Nx, KOKKOS_LAMBDA(int coor)
		{
			int y = coor / Nx;
			int x = coor % Nx;
			int R = Ny / 10;

			int cylX = x - Nx / 2;
			int cylY = y - Ny / 2;

			if (abs(cylX) + abs(cylY) <= 2 * R)
			{
				isBondary_k(coor) = 1;
			}
			else
			{
				isBondary_k(coor) = 0;
			}
		});

	}


	void update_stream()
	{

		Kokkos::parallel_for("stream", NL * Ny * Nx, KOKKOS_LAMBDA(int i)
		{
			int l = i / (Ny * Nx);
			int coor = i % (Ny * Nx);
			int y = coor / Nx;
			int x = coor % Nx;
			int new_y = (y + dys_k(l) + Ny) % Ny;
			int new_x = (x + dxs_k(l) + Nx) % Nx;
			new_table_k(l * (Ny * Nx) + new_y * Nx + new_x) = cur_table_k(i);
		});

	}
	void update_collide()
	{
		Kokkos::parallel_for("collide", Ny * Nx, KOKKOS_LAMBDA(int coor)
		{
			float ux = 0.0;
			float uy = 0.0;
			float rho = 0.0;
			for (int l = 0; l < NL; l++)
			{
				int i = l * (Ny * Nx) + coor;
				rho += new_table_k(i);
				if (isBondary_k(coor))
				{
					new_table_k(i) = new_table_k(invert_k(l) * Ny * Nx + coor);
				}
				else
				{
					ux += dxs_k(l) * new_table_k(i);
					uy += dys_k(l) * new_table_k(i);
				}
			}
			ux /= rho;
			uy /= rho;


			float sqU = ux * ux + uy * uy;
			for (int l = 0; l < NL; l++)
			{
				int i = l * (Ny * Nx) + coor;
				float tmp = dxs_k(l) * ux + dys_k(l) * uy;
				double Feq = rho * weights_k(l) * (1.0 + 3.0 * tmp + 4.5 * tmp * tmp - 1.5 * sqU);
				new_table_k(i) = new_table_k(i) - (new_table_k(i) - Feq) / TAU;
			}

		});



	}

public:

	int get_W() { return Nx; }
	int get_H() { return Ny; }
	Kokkos::View<float*> get_table() { return new_table_k; }

	BoltzmannSumulation(int _Nx, int _Ny) : Ny(_Ny), Nx(_Nx)
	{
		size_t SZ = Ny * Nx;
		new_table_k = Kokkos::View<float*>("new_table", NL*SZ);
		cur_table_k = Kokkos::View<float*>("cur_table", NL*SZ);
		isBondary_k = Kokkos::View<bool*>("isBondary", SZ);
		invert_k = Kokkos::View<int*>("invert", 9);
		dxs_k = Kokkos::View<int*>("dxs", 9);
		dys_k = Kokkos::View<int*>("dys", 9);
		weights_k = Kokkos::View<double*>("w", 9);


		invert_k(0) = 0;
		invert_k(1) = 5;
		invert_k(2) = 6;
		invert_k(3) = 7;
		invert_k(4) = 8;
		invert_k(5) = 1;
		invert_k(6) = 2;
		invert_k(7) = 3;
		invert_k(8) = 4;

		//const int dys[] = { 0, 1, 1, 0,-1,-1,-1, 0, 1 };
		//const int dxs[] = { 0, 0, 1, 1, 1, 0,-1,-1,-1 };
		dys_k(0) = 0;
		dys_k(1) = 1;
		dys_k(2) = 1;
		dys_k(3) = 0;
		dys_k(4) = -1;
		dys_k(5) = -1;
		dys_k(6) = -1;
		dys_k(7) = 0;
		dys_k(8) = 1;

		dxs_k(0) = 0;
		dxs_k(1) = 0;
		dxs_k(2) = 1;
		dxs_k(3) = 1;
		dxs_k(4) = 1;
		dxs_k(5) = 0;
		dxs_k(6) = -1;
		dxs_k(7) = -1;
		dxs_k(8) = -1;


		//{ , , , ,  };
		weights_k(0) = 4.0 / 9;
		weights_k(1) = 1.0 / 9;
		weights_k(2) = 1.0 / 36;
		weights_k(3) = 1.0 / 9;
		weights_k(4) = 1.0 / 36;
		weights_k(5) = 1.0 / 9;
		weights_k(6) = 1.0 / 36;
		weights_k(7) = 1.0 / 9;
		weights_k(8) = 1.0 / 36;


		init_sumulation();
	}

	BoltzmannSumulation(const BoltzmannSumulation& bs) : Ny(bs.Ny), Nx(bs.Nx), TAU(bs.TAU)
	{
		new_table_k = bs.new_table_k;
		cur_table_k = bs.cur_table_k;
		isBondary_k = bs.isBondary_k;
	}



	void update()
	{
		update_stream();
		update_collide();
		std::swap(cur_table_k, new_table_k);
	}



};
#endif#pragma once