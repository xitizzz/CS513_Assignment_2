#include <stdio.h> //for printf
#include <stdint.h> //for uint32_t

/* for now, using uint32_t everywhere, we may overflow. If so, 
   must attempt to use floats for performance reasons. uint64_t is unusably 
   slow. We must use multiplication instead of repeated addition to limit
   error accumulation, but floats should be workable. */
__global__
void multiples(uint32_t step, uint32_t *prime_array, uint32_t n){
    /*this function should compute all multiples of step, and mark 
    indices of prime_array at those points. This function should use
    threadID to divide the problem set */

    uint32_t cpus = blockDim.x * gridDim.x; //total number of threads
    //compute absolute 1 dimensional thead ID
    uint32_t id = blockIdx.x * blockDim.x + threadIdx.x;

    uint32_t start = id * (n/cpus); //starts at 0, increments by n/cpus
    uint32_t start_mult = start/step;
    uint32_t end = (id + 1) * (n/cpus) - 1; //ensure range has no overlap
    uint32_t end_mult = end/step;
    //TODO check for end>n

    for(int i = start_mult; i<=end_mult; i++){
        prime_array[step * i] = 1;
    }




}



int main(){
    uint32_t n = 1<<10; //find all primes upto and including this number
    uint32_t *prime_array = (uint32_t *)calloc(n , sizeof(uint32_t)); //allocate and zero
    uint32_t *d_array;
    cudaMalloc(&d_array, n * sizeof(uint32_t));
    cudaMemcpy(d_array, prime_array, n * sizeof(uint32_t), cudaMemcpyHostToDevice);

    for(int loop = 2; loop <= sqrt(n); loop++){ //TODO careful of sqrt here, check for primes only
        multiples<<<1,1>>>(loop,d_array,n);
    }

    cudaMemcpy(prime_array, d_array, n * sizeof(uint32_t), cudaMemcpyDeviceToHost);








    for(int i=0; i<n; i++){
        if(prime_array[i] == 0){
            printf("%d ",i);
        }
    }


    free(prime_array); //clean up
    return 0;
}
