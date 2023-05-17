#ifndef __TBoltzmannSumulation_H__
#define __TBoltzmannSumulation_H__

//////(9*(y*Nx + x) + l)

#include <iostream>
#include <string>
#include <cmath>
#include <Kokkos_Core.hpp>

#define CREATE_DXS const int dxs_k[] = { 0, 0, 1, 1, 1, 0,-1,-1,-1 };
#define CREATE_DYS const int dys_k[] = { 0, 1, 1, 0,-1,-1,-1, 0, 1 };
#define CREATE_W const double weights_k[] = { 4.0 / 9, 1.0 / 9, 1.0 / 36, 1.0 / 9, 1.0 / 36, 1.0 / 9, 1.0 / 36, 1.0 / 9, 1.0 / 36 };
#define CREATE_INV const int invert_k[] = { 0, 5,6,7,8,1,2,3,4 };


#define myAccess(array, coor, l) array(9*coor+l)

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


			int new_y = (y + dys_k[0] + Ny) % Ny;
			int new_x = (x + dxs_k[0] + Nx) % Nx;
			myAccess(new_table_k, new_y* Nx + new_x, 0) = myAccess(cur_table_k, coor, 0);

			new_y = (y + dys_k[1] + Ny) % Ny;
			new_x = (x + dxs_k[1] + Nx) % Nx;
			myAccess(new_table_k, new_y* Nx + new_x, 1) = myAccess(cur_table_k, coor, 1);

			new_y = (y + dys_k[2] + Ny) % Ny;
			new_x = (x + dxs_k[2] + Nx) % Nx;
			myAccess(new_table_k, new_y* Nx + new_x, 2) = myAccess(cur_table_k, coor, 2);

			new_y = (y + dys_k[3] + Ny) % Ny;
			new_x = (x + dxs_k[3] + Nx) % Nx;
			myAccess(new_table_k, new_y* Nx + new_x, 3) = myAccess(cur_table_k, coor, 3);

			new_y = (y + dys_k[4] + Ny) % Ny;
			new_x = (x + dxs_k[4] + Nx) % Nx;
			myAccess(new_table_k, new_y* Nx + new_x, 4) = myAccess(cur_table_k, coor, 4);

			new_y = (y + dys_k[5] + Ny) % Ny;
			new_x = (x + dxs_k[5] + Nx) % Nx;
			myAccess(new_table_k, new_y* Nx + new_x, 5) = myAccess(cur_table_k, coor, 5);

			new_y = (y + dys_k[6] + Ny) % Ny;
			new_x = (x + dxs_k[6] + Nx) % Nx;
			myAccess(new_table_k, new_y* Nx + new_x, 6) = myAccess(cur_table_k, coor, 6);


			new_y = (y + dys_k[7] + Ny) % Ny;
			new_x = (x + dxs_k[7] + Nx) % Nx;
			myAccess(new_table_k, new_y* Nx + new_x, 7) = myAccess(cur_table_k, coor, 7);


			new_y = (y + dys_k[8] + Ny) % Ny;
			new_x = (x + dxs_k[8] + Nx) % Nx;
			myAccess(new_table_k, new_y* Nx + new_x, 8) = myAccess(cur_table_k, coor, 8);
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

			SUM_RHO(0)
			SUM_RHO(1)
			SUM_RHO(2)
			SUM_RHO(3)
			SUM_RHO(4)
			SUM_RHO(5)
			SUM_RHO(6)
			SUM_RHO(7)
			SUM_RHO(8)

			if (isBondary_k(coor))
			{
				UP_NT(0)
				UP_NT(1)
				UP_NT(2)
				UP_NT(3)
				UP_NT(4)
				UP_NT(5)
				UP_NT(6)
				UP_NT(7)
				UP_NT(8)
			}
			else
			{
				SUM_DU(0)
				SUM_DU(1)
				SUM_DU(2)
				SUM_DU(3)
				SUM_DU(4)
				SUM_DU(5)
				SUM_DU(6)
				SUM_DU(7)
				SUM_DU(8)
			}



			ux /= rho;
			uy /= rho;
			float sqU = ux * ux + uy * uy;

			FEQ(0)
			FEQ(1)
			FEQ(2)
			FEQ(3)
			FEQ(4)
			FEQ(5)
			FEQ(6)
			FEQ(7)
			FEQ(8)
		});


	}

public:

	int get_W() { return Nx; }
	int get_H() { return Ny; }
	//Kokkos::View<float**, Kokkos::SharedSpace> get_table() { return cur_table_k; }

	BoltzmannSumulation(int _Nx, int _Ny) : Ny(_Ny), Nx(_Nx)
	{
		Kokkos::resize(new_table_k, Ny * Nx* NL);
		Kokkos::resize(cur_table_k, Ny * Nx* NL);
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
