/*Eratosthenes' method CUDA
	
Theroritical time complexity is sqrt(n)* log(log n)
Howver actual runningtime is not be bad, it is HORRIBLE
You will see why in the program

Works on any number 
4 Appears as prime for n<=12
Not checked rigorously for correctness 

*/

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>

#define THREAD 128

using namespace std;

__global__
void markPrimes(unsigned int step, unsigned int k, unsigned int *d_primes, unsigned int n) {
	unsigned int index = blockIdx.x*blockDim.x + threadIdx.x;
	unsigned int begin = ((2 * index + step)/step) * step ;
	unsigned int end = (index + 1) * k;
	for (unsigned int i = begin + step; i <= end; i += step) {
		if(i<n) d_primes[i] = 0;
	}
}

int main() {
	unsigned int n, N, k, blocks;
	unsigned int *primes, *d_primes;

	n = pow(2, 12) + 5;   //find all primes <n, n it self will not be considedred 
	N = ceill((long double)sqrt(n)); //Number of processors
	k = ceill((long double)n / (long double)N); //Size of the partition 
	blocks = ceill((long double)N / (long double)THREAD); //Number of blocks

	primes = (unsigned int*)malloc(n * sizeof(int));
	for (int i = 0; i < n; i++) {
		primes[i] = 1;
	}

	cudaMalloc(&d_primes, n * sizeof(int));
	cudaMemcpy(d_primes, primes, n * sizeof(int), cudaMemcpyHostToDevice);

	for (int i = 2; i <= N; i++) {  
		if (primes[i] != 1) continue;
		markPrimes << <blocks, THREAD >> > (i, k, d_primes,n);  //ecah thraed will have work sqrt(n)/i
		cudaMemcpy(primes, d_primes, n*sizeof(int), cudaMemcpyDeviceToHost); //Due to this memory transfer the running time will be HORRIBLE
	}

	//cudaMemcpy(primes, d_primes, n * sizeof(int), cudaMemcpyDeviceToHost);
	primes[0] = 0;
	primes[1] = 0;
	unsigned int count = 0;
	for (int i = 0; i < n; i++) {
		if (primes[i] == 1) cout << i << "\t";
	}
	cout << count << endl;
}
