/*
Tested on LANL Darwin with $salloc -N 2 -p scaling 
Broadwell CPUs only, with infiniband

[gchen@cn365 benchmark]$ module list
Currently Loaded Modulefiles:
  1) gcc/9.3.0                 2) openmpi/3.1.6-gcc_9.3.0   3) cmake/3.12.4              4) git/2.17.1

[gchen@cn365 build]$ mpirun -n 2 -npernode 1 core/performance_test/benchmark/TestMigration
#message lenght(B), throughput(MB/s)-POD, Cabana-AOSOA
0 0.000000 0.000000
8 8.830810 2.614360
16 17.566939 4.888903
32 33.504226 10.307015
64 62.027676 18.832625
128 112.393137 38.442561
256 168.479593 67.131918
512 303.513044 123.976650
1024 525.261120 205.854071
2048 938.191438 321.368087
4096 1426.430744 447.210929
8192 2107.546071 543.655575
10000 2460.586242 584.973117
100000 7548.404926 762.756590
1000000 11423.070896 835.884683
4000000 12031.619527 987.722785
[gchen@cn365 build]$ date
Mon Nov 16 17:28:54 MST 2020
 */

#include <Cabana_Core.hpp>

#include <Kokkos_Core.hpp>

#include <algorithm>
#include <iostream>
#include <vector>

#include <mpi.h>
#include <sys/time.h>

#define NTST 16 //number of tests varying the message length
#define REPS 1001 //within each test, repeat REPS times
//#define print_string_8b

double
secs()
{

        struct timeval ru;
        gettimeofday(&ru, (struct timezone *)0);
        return(ru.tv_sec + ((double)ru.tv_usec)/1000000);
}

template <class DataDevice, class CommDevice>
void testMPI_Cabana(MPI_Comm comm)
{
    double t,q; //for timing 
    int src,dst;
    MPI_Status status;

    int comm_rank = -1;
    MPI_Comm_rank( comm, &comm_rank );
    int comm_size = -1;
    MPI_Comm_size( comm, &comm_size );

    //std::cout << "Rank " << comm_rank << " of " << comm_size << ": Hello world from Cabana!\n";
    if(comm_size>2) {
      std::cout << "-np != 2. exit.\n";
      exit(1);
    }
    /*
      Declare the AoSoA parameters.
    */
    using DataTypes = Cabana::MemberTypes<char>;
    const int VectorLength = 1;
    using MemorySpace = Kokkos::HostSpace;
    using ExecutionSpace = Kokkos::Serial;
    using DeviceType = Kokkos::Device<ExecutionSpace, MemorySpace>;

    //message length for each test
    int len_arry[NTST] = {0, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 10000, 100000, 1000000, 4000000};

    if(comm_rank==0) printf("#message lenght(B), throughput(MB/s)-POD, Cabana-AOSOA\n");    

    for(int it=0; it<NTST; it++){
	int num_tuple = len_arry[it];
	// Create POD
	char *vpod = (char *) malloc(num_tuple);
	
	if(comm_rank==0){
	    for(int i=0;i<num_tuple;++i)  vpod[i] = 'a';
	    src=dst=1;
	    double avrg = 0;
	    for(int i=0;i<3;i++){ /* warmup */
	       MPI_Send(vpod,num_tuple,MPI_BYTE,dst,0,comm);
	       MPI_Recv(vpod,num_tuple,MPI_BYTE,src,0,comm,&status);
	   }

	   for(int i=0;i<REPS;i++){
	       t = secs();
	       MPI_Send(vpod,num_tuple,MPI_BYTE,dst,0,comm);
	       MPI_Recv(vpod,num_tuple,MPI_BYTE,src,0,comm,&status);
	       q=secs();
	       t = q-t;
	       avrg += t;      	
	   }
	   //output 
           avrg = avrg/REPS;
	   printf ("%d\t %f \t",
		    num_tuple, 2.e-6*num_tuple/avrg);	   
	}else{
	    for(int i=0;i<num_tuple;++i)  vpod[i] = 'b';
	    src=dst=0;
	    for(int i=0;i<3;i++){ /* warmup */
	       MPI_Recv(vpod,num_tuple,MPI_BYTE,src,0,comm,&status);
	       MPI_Send(vpod,num_tuple,MPI_BYTE,dst,0,comm);
	   }

	    for(int i=0;i<REPS;i++){
	       MPI_Recv(vpod,num_tuple,MPI_BYTE,src,0,comm,&status);
	       MPI_Send(vpod,num_tuple,MPI_BYTE,dst,0,comm);
	    }
	    
	}

	free(vpod);

	MPI_Barrier(comm);
        // Create aosoa
        Cabana::AoSoA<DataTypes, DeviceType, VectorLength> aosoa( "A", num_tuple );
	auto vtmp = Cabana::slice<0>( aosoa );

	Kokkos::View<int *, DeviceType> export_ranks( "export_ranks", num_tuple );
	int previous_rank = ( comm_rank == 0 ) ? 1 : 0;
	int next_rank = ( comm_rank == 1 ) ? 0 : 1;
	for (int i=0;i<num_tuple;++i)  export_ranks( i ) = next_rank;
	std::vector<int> neighbors = {previous_rank, comm_rank, next_rank};
	std::sort( neighbors.begin(), neighbors.end() );
	auto unique_end = std::unique( neighbors.begin(), neighbors.end() );
	neighbors.resize( std::distance( neighbors.begin(), unique_end ) );
	Cabana::Distributor<DeviceType> distributor( comm, export_ranks,
						     neighbors );

	if(comm_rank==0){
	    for(int i=0;i<num_tuple;++i)  vtmp(i) = 'a';
	    src=dst=1;
	    double avrg = 0;

	   for(int i=0;i<REPS;i++){
	       t = secs();
	       Cabana::migrate( distributor, aosoa );
	       q=secs();
	       t = q-t;
	       avrg += t;      	
	   }
	   //output 
           avrg = avrg/REPS;
	   printf ("%f\n", 2.e-6*num_tuple/avrg);	   

#ifdef  print_string_8b
	   if(it==1){
	       std::cout << "AFTER migration" << std::endl
                  << "(Rank " << comm_rank << ") ";
	       for ( std::size_t i = 0; i < num_tuple; ++i )
		   std::cout << vtmp( i ) << " ";
	       std::cout << std::endl;
	   }
#endif
	}else{
	    for(int i=0;i<num_tuple;++i)  vtmp(i) = 'b';
	    src=dst=0;
	    for(int i=0;i<REPS;i++){
	       Cabana::migrate( distributor, aosoa );
	    }

#ifdef  print_string_8b
	   if(it==1){
	       std::cout << "AFTER migration" << std::endl
                  << "(Rank " << comm_rank << ") ";
	       for ( std::size_t i = 0; i < num_tuple; ++i )
		   std::cout << vtmp( i ) << " ";
	       std::cout << std::endl;
	   }
#endif
	    
	}

    } //NIST
}

//---------------------------------------------------------------------------//
// Main.
//---------------------------------------------------------------------------//
int main( int argc, char* argv[] )
{
    MPI_Init( &argc, &argv );

    Kokkos::ScopeGuard scope_guard( argc, argv );

    //using OpenMPDevice = Kokkos::Device<Kokkos::OpenMP, Kokkos::HostSpace>;
    using OpenMPDevice = Kokkos::Device<Kokkos::Serial, Kokkos::HostSpace>;


    testMPI_Cabana<OpenMPDevice, OpenMPDevice>(MPI_COMM_WORLD);

    MPI_Finalize();

    return 0;
}
