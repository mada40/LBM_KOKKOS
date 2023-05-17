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


#define myAccess(array, y,x, l) array(y, x, l)

#define FEQ(index) {									      \
	float tmp = dxs_k[index] * ux + dys_k[index] * uy;				      \
	double Feq = rho * weights_k[index] * (1.0 + 3.0 * tmp + 4.5 * tmp * tmp - 1.5 * sqU);\
	myAccess(new_table_k,y,x,index) = myAccess(new_table_k,y,x,index) - (myAccess(new_table_k,y,x,index) - Feq) / TAU;}

#define UP_STREAM_NT(index) {							\
	int new_y = (y + dys_k[index] + Ny) % Ny;				\
	int new_x = (x + dxs_k[index] + Nx) % Nx;				\
	myAccess(new_table_k,new_y, new_x,index) = myAccess(cur_table_k,y,x,index);}



#define SUM_RHO(index) {rho += myAccess(new_table_k,y,x,index);}
#define UP_NT(index) {myAccess(new_table_k,y,x,index) = myAccess(new_table_k,y,x,invert_k[index]);}
#define SUM_DU(index) {ux += dxs_k[index] * myAccess(new_table_k,y,x,index); uy += dys_k[index] * myAccess(new_table_k,y,x,index);}


class BoltzmannSumulation
{
public:

	Kokkos::DefaultExecutionSpace defExecSpace;

	const int Ny;
	const int Nx;
	const int NL = 9;
	const double TAU = 0.58;
	Kokkos::View<float***, Kokkos::SharedSpace> new_table_k;
	Kokkos::View<float***, Kokkos::SharedSpace> cur_table_k;
	Kokkos::View<bool**, Kokkos::SharedSpace> isBondary_k;


	//812
	//703
	//654

	BoltzmannSumulation& operator = (const BoltzmannSumulation& other) = default;
	BoltzmannSumulation(const BoltzmannSumulation& bs) = default;

	void init_sumulation()
	{
		Kokkos::parallel_for("edwdewdw", Ny, KOKKOS_CLASS_LAMBDA(int y)
		{
			for (int x = 0; x < Nx; x++)
			{
				myAccess(cur_table_k, y, x, 0) = 0.6;
				myAccess(cur_table_k, y, x, 1) = 0.6;
				myAccess(cur_table_k, y, x, 2) = 0.6;
				myAccess(cur_table_k, y, x, 3) = 0.6;
				myAccess(cur_table_k, y, x, 4) = 0.6;
				myAccess(cur_table_k, y, x, 5) = 0.6;
				myAccess(cur_table_k, y, x, 6) = 0.6;
				myAccess(cur_table_k, y, x, 7) = 0.6;
				myAccess(cur_table_k, y, x, 8) = 1.2;

				//int y = coor / Nx;
				//int x = coor % Nx;
				int R = Ny / 10;

				int cylX = x - Nx / 2;
				int cylY = y - Ny / 2;
				isBondary_k(y, x) = (abs(cylX) + abs(cylY) <= 2 * R);
			}

		});
		Kokkos::fence();
	}


	void update_stream()
	{

		Kokkos::parallel_for("aefef", Ny, KOKKOS_CLASS_LAMBDA(int y)
		{
			CREATE_DXS;
			CREATE_DYS;


			for (int x = 0; x < Nx; x++)
			{
				UP_STREAM_NT(0)
				UP_STREAM_NT(1)
				UP_STREAM_NT(2)
				UP_STREAM_NT(3)
				UP_STREAM_NT(4)
				UP_STREAM_NT(5)
				UP_STREAM_NT(6)
				UP_STREAM_NT(7)
				UP_STREAM_NT(8)
			}
		});


	}
	void update_collide()
	{
		Kokkos::parallel_for("a100", Ny, KOKKOS_CLASS_LAMBDA(int y)
		{
			CREATE_INV;
			CREATE_DXS;
			CREATE_DYS;
			CREATE_W;


			for (int x = 0; x < Nx; x++)
			{
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

				if (isBondary_k(y, x))
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
			}
		});


	}

public:

	int get_W() { return Nx; }
	int get_H() { return Ny; }
	//Kokkos::View<float**, Kokkos::SharedSpace> get_table(){return cur_table_k;}

	BoltzmannSumulation(int _Nx, int _Ny) : Ny(_Ny), Nx(_Nx)
	{
		Kokkos::resize(new_table_k, Ny, Nx, NL);
		Kokkos::resize(cur_table_k, Ny, Nx, NL);
		Kokkos::resize(isBondary_k, Ny, Nx);


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
	}

};
#endif
