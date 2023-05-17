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


#define myAccess(array, coor, l) array(coor, l)

#define FEQ(index) {									      \
	float tmp = dxs_k[index] * ux + dys_k[index] * uy;				      \
	double Feq = rho * weights_k[index] * (1.0 + 3.0 * tmp + 4.5 * tmp * tmp - 1.5 * sqU);\
	myAccess(new_table_k,coor,index) = myAccess(new_table_k,coor,index) - (myAccess(new_table_k,coor,index) - Feq) / TAU;}


#define SUM_RHO(index) {rho += myAccess(new_table_k,coor,index);}
#define UP_NT(index) {myAccess(new_table_k,coor,index) = myAccess(new_table_k,coor,invert_k[index]);}
#define SUM_DU(index) {ux += dxs_k[index] * myAccess(new_table_k,coor,index); uy += dys_k[index] * myAccess(new_table_k,coor,index);}


class BoltzmannSumulation
{
public:

	Kokkos::DefaultExecutionSpace defExecSpace;

	const int Ny;
	const int Nx;
	const int NL = 9;
	const double TAU = 0.58;
	Kokkos::View<float**, Kokkos::SharedSpace> new_table_k;
	Kokkos::View<float**, Kokkos::SharedSpace> cur_table_k;
	Kokkos::View<bool*, Kokkos::SharedSpace> isBondary_k;


	//812
	//703
	//654

	BoltzmannSumulation& operator = (const BoltzmannSumulation& other) = default;
	BoltzmannSumulation(const BoltzmannSumulation& bs) = default;

	void init_sumulation()
	{
		Kokkos::parallel_for("edwdewdw", Ny * Nx, KOKKOS_CLASS_LAMBDA(int coor)
		{
			myAccess(cur_table_k, coor, 0) = 0.6;
			myAccess(cur_table_k, coor, 1) = 0.6;
			myAccess(cur_table_k, coor, 2) = 0.6;
			myAccess(cur_table_k, coor, 3) = 0.6;
			myAccess(cur_table_k, coor, 4) = 0.6;
			myAccess(cur_table_k, coor, 5) = 0.6;
			myAccess(cur_table_k, coor, 6) = 0.6;
			myAccess(cur_table_k, coor, 7) = 0.6;
			myAccess(cur_table_k, coor, 8) = 1.2;

			int y = coor / Nx;
			int x = coor % Nx;
			int R = Ny / 10;

			int cylX = x - Nx / 2;
			int cylY = y - Ny / 2;
			isBondary_k(coor) = (abs(cylX) + abs(cylY) <= 2 * R);

		});
		Kokkos::fence();
	}


	void update_stream()
	{

		Kokkos::parallel_for("aefef", Ny * Nx, KOKKOS_CLASS_LAMBDA(int coor)
		{
			CREATE_DXS;
			CREATE_DYS;

			int y = coor / Nx;
			int x = coor % Nx;

			for (int l = 0; l < 9; ++l)
			{
				int new_y = (y + dys_k[l] + Ny) % Ny;
				int new_x = (x + dxs_k[l] + Nx) % Nx;
				myAccess(new_table_k, new_y * Nx + new_x, l) = myAccess(cur_table_k, coor, l);
			}
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
			for (int l = 0; l < 9; ++l)
			{
				SUM_RHO(l)
			}

			if (isBondary_k(coor))
			{
				for (int l = 0; l < 9; ++l)
				{
					UP_NT(l)
				}
			}
			else
			{
				for (int l = 0; l < 9; ++l)
				{
					SUM_DU(l)
				}

			}



			ux /= rho;
			uy /= rho;
			float sqU = ux * ux + uy * uy;
			for (int l = 0; l < 9; ++l)
			{
				FEQ(l)
			}

		});


	}

public:

	int get_W() { return Nx; }
	int get_H() { return Ny; }
	Kokkos::View<float**, Kokkos::SharedSpace> get_table() { return cur_table_k; }

	BoltzmannSumulation(int _Nx, int _Ny) : Ny(_Ny), Nx(_Nx)
	{
		Kokkos::resize(new_table_k, Ny * Nx, NL);
		Kokkos::resize(cur_table_k, Ny * Nx, NL);
		Kokkos::resize(isBondary_k, Ny * Nx);


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
