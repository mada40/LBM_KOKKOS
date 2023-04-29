#ifndef __TBoltzmannSumulation_H__
#define __TBoltzmannSumulation_H__

#include <iostream>
#include <string>
#include <cmath>
#include <Kokkos_Core.hpp>

#define CREATE_DXS const int dxs_k[] = { 0, 0, 1, 1, 1, 0,-1,-1,-1 };
#define CREATE_DYS const int dys_k[] = { 0, 1, 1, 0,-1,-1,-1, 0, 1 };
#define CREATE_W const double weights_k[] = { 4.0 / 9, 1.0 / 9, 1.0 / 36, 1.0 / 9, 1.0 / 36, 1.0 / 9, 1.0 / 36, 1.0 / 9, 1.0 / 36 };
#define CREATE_INV const int invert_k[] = { 0, 5,6,7,8,1,2,3,4 };


class BoltzmannSumulation
{
public:

	Kokkos::DefaultExecutionSpace defExecSpace;

	const int Ny;
	const int Nx;
	const int NL = 9;
	const double TAU = 0.58;
	Kokkos::View<float*, Kokkos::SharedSpace> new_table_k;
	Kokkos::View<float*, Kokkos::SharedSpace> cur_table_k;
	Kokkos::View<bool*, Kokkos::SharedSpace> isBondary_k;


	//812
	//703
	//654

	BoltzmannSumulation& operator = (const BoltzmannSumulation& other) = default;
	BoltzmannSumulation(const BoltzmannSumulation& bs) = default;

	void init_sumulation()
	{

		Kokkos::parallel_for("qwe", NL * Ny * Nx, KOKKOS_CLASS_LAMBDA(int i)
		{
			cur_table_k(i) = 0.6;
			if (i / (Nx * Ny) == 8)
				cur_table_k(i) = 1.2;
		});

		Kokkos::parallel_for("edwdewdw", Ny * Nx, KOKKOS_CLASS_LAMBDA(int coor)
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
		Kokkos::fence();

	}


	void update_stream()
	{

		Kokkos::parallel_for("aefef", NL * Ny * Nx, KOKKOS_CLASS_LAMBDA(int i)
		{
			CREATE_DXS;
			CREATE_DYS;

			int l = i / (Ny * Nx);
			int coor = i % (Ny * Nx);
			int y = coor / Nx;
			int x = coor % Nx;
			int new_y = (y + dys_k[l] + Ny) % Ny;
			int new_x = (x + dxs_k[l] + Nx) % Nx;
			new_table_k(l* (Ny* Nx) + new_y * Nx + new_x) = cur_table_k(i);
		});


	}
	void update_collide()
	{
		Kokkos::parallel_for("a100", Ny * Nx, KOKKOS_CLASS_LAMBDA(int coor)
		{
			CREATE_INV;
			CREATE_DXS;
			CREATE_DYS;
			CREATE_W;

			float ux = 0.0;
			float uy = 0.0;
			float rho = 0.0;
			for (int l = 0; l < NL; l++)
			{
				int i = l * (Ny * Nx) + coor;
				rho += new_table_k(i);
				if (isBondary_k(coor))
				{
					new_table_k(i) = new_table_k(invert_k[l] * Ny * Nx + coor);
				}
				else
				{
					ux += dxs_k[l] * new_table_k(i);
					uy += dys_k[l] * new_table_k(i);
				}
			}
			ux /= rho;
			uy /= rho;


			float sqU = ux * ux + uy * uy;
			for (int l = 0; l < NL; l++)
			{
				int i = l * (Ny * Nx) + coor;
				float tmp = dxs_k[l] * ux + dys_k[l] * uy;
				double Feq = rho * weights_k[l] * (1.0 + 3.0 * tmp + 4.5 * tmp * tmp - 1.5 * sqU);
				new_table_k(i) = new_table_k(i) - (new_table_k(i) - Feq) / TAU;
			}

		});


	}

public:

	int get_W() { return Nx; }
	int get_H() { return Ny; }
	Kokkos::View<float*, Kokkos::SharedSpace> get_table() { return cur_table_k; }

	BoltzmannSumulation(int _Nx, int _Ny) : Ny(_Ny), Nx(_Nx)
	{
		size_t SZ = Ny * Nx;

		Kokkos::resize(new_table_k, NL * SZ);
		Kokkos::resize(cur_table_k, NL * SZ);
		Kokkos::resize(isBondary_k, SZ);


		init_sumulation();
	}


	void update(int count)
	{
		for (size_t i = 0; i < count; ++i)
		{
			update_stream();
			update_collide();
			Kokkos::fence();
			std::swap(cur_table_k, new_table_k);
		}
		//Kokkos::fence();
	}




};
#endif
