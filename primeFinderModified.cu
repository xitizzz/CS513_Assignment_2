/*Eratosthenes' method - CUDA

Theoretical time complexity is sqrt(n)* log(log n)
Pass in input to command line to specify n
Average running time is up to 10 milliseconds
*/

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <time.h>

#define THREAD 128

using namespace std;

__global__
void markPrimes(unsigned int step, unsigned int k, unsigned int *d_primes, unsigned int n) {
	unsigned int index = blockIdx.x*blockDim.x + threadIdx.x;
	//unsigned int begin = ((2 * index + step) / step) * step;
	unsigned int begin = index * step;
	if (begin == 0){
		begin = step*step;
	}
	unsigned int end = (index + 1) * k;
	for (unsigned int i = begin; i <= end; i += step) //begin + step
	{
		if (i<n) 
			d_primes[i] = 0;
	}
}

int main(int argc, char *argv[]) {
	unsigned int n, N, k, blocks;
	unsigned int *primes, *d_primes;

	n = atoi(argv[1]); //pow(2, 12) + 5;;   //find all primes < n, n itself will not be considedred 
	N = ceill((long double)sqrt(n)); //Number of processors
	k = ceill((long double)n / (long double)N); //Size of the partition 
	blocks = ceill((long double)N / (long double)THREAD); //Number of blocks

	primes = (unsigned int*)malloc(n * sizeof(int));
	for (int i = 0; i < n; i++)
	{
		primes[i] = 1;
	}

	cudaMalloc(&d_primes, n * sizeof(int));
	cudaMemcpy(d_primes, primes, n * sizeof(int), cudaMemcpyHostToDevice);
	clock_t begin = clock();
	for (int i = 2; i <= N; i++)
	{
		if (primes[i] != 1)
			continue;
		markPrimes << <blocks, THREAD >> > (i, k, d_primes, n);
		//each thread will have work sqrt(n)/i
		//cudaMemcpy(primes, d_primes, n*sizeof(int), cudaMemcpyDeviceToHost);
		//Due to this memory transfer the running time will be HORRIBLE
	}
	clock_t end = clock();
	cudaMemcpy(primes, d_primes, n * sizeof(int), cudaMemcpyDeviceToHost);
	primes[0] = 0;
	primes[1] = 0;
	unsigned int count = 0;
	for (int i = 0; i < n; i++)
	{
		if (primes[i] == 1)
		{
			cout << i << "\t";
			count++;
		}
	}
	cout << "\nNumber of primes less than " << n << ": " << count << endl;
	double time_spent = (double)(end - begin) / CLOCKS_PER_SEC * 1000;
	cout << "The running time is " << time_spent << " milliseconds." << endl;
}