#include <stdio.h> //for printf
#include <stdint.h> //for uint32_t

/* for now, using uint32_t everywhere, we may overflow. If so, 
   must attempt to use floats for performance reasons. uint64_t is unusably 
   slow. We must use multiplication instead of repeated addition to limit
   error accumulation, but floats should be workable. */
__global__
void multiples(uint32_t step, uint32_t *array, uint32_t n, uint32_t *next_prime){
    /*this function should compute all multiples of step, and mark 
    indices of prime_array at those points. This function should use
    threadID to divide the problem set */
    uint32_t cpus, id, start, start_mult, end, end_mult;

    cpus = blockDim.x * gridDim.x; //total number of threads

    //ensure extra loop runs if n/cpus has remainder. Can be tuned to ensure
    //at least sqrt(n) thread executions by changing condition
    for( id = blockIdx.x * blockDim.x + threadIdx.x;
            id * (n/cpus) < n;
            id += cpus)
    {

        start = id * (n/cpus); //starts at 0, increments by n/cpus
        if (start < (step*step)){ //ensure we start at n^2, fixes 2, saves work
            start = step*step;
        }
        start_mult = start/step;

        end = (id + 1) * (n/cpus) - 1; //ensure range has no overlap
        if(end > n){ //avoid overflow
            end = n;
        }
        if(start >= end){
            return; //make sure extra threads exit instead of working
        }
        end_mult = end/step;

        printf("thread id %d, num cpus %d, start %d, end %d\n"
               , id, cpus, start, end); //debug

        for(int i = start_mult; i<=end_mult; i++){
            array[step * i] = 1;
        }
        __syncthreads(); //barrier until all threads done
        int j = step+1;
        for(; j<n; j++){
            if(array[j]==0){
                *next_prime = j;//write host memory
                break;
            }
        }
        if(j == n){
            *next_prime = 0;
        }


    }
}

int main(){
    uint32_t n = 1<<10; //find all primes upto and including this number
    uint32_t *prime_array = (uint32_t *)calloc(n , sizeof(uint32_t)); //allocate and zero
    uint32_t *next_prime = (uint32_t *)malloc(sizeof(uint32_t)); //allocate index for signalling
    cudaMallocHost(&next_prime, sizeof(uint32_t));
    *next_prime = 2; //initialize

    uint32_t *d_array;
    cudaMalloc(&d_array, n * sizeof(uint32_t));
    cudaMemcpy(d_array, prime_array, n * sizeof(uint32_t), cudaMemcpyHostToDevice);


    for(int loop = 2; loop <= sqrt(n); loop = *next_prime){ //TODO careful of sqrt here, check for primes only
//    while(*next_prime !=0){
        printf("starting loop for multiples of %d\n", loop);
        multiples<<<1,1>>>(loop,d_array,n,next_prime);
        cudaDeviceSynchronize();
        printf("next prime is %d\n", *next_prime);
    }

    cudaMemcpy(prime_array, d_array, n * sizeof(uint32_t), cudaMemcpyDeviceToHost);

    FILE *output_file = fopen("output.txt", "w+");
    
    for(int i=1; i<n; i++){
        if(prime_array[i] == 0){
            fprintf(output_file, "%d ",i);
        }
    }
    fprintf(output_file, "\n");

    fclose(output_file);
    free(prime_array); //clean up
    return 0;
}
