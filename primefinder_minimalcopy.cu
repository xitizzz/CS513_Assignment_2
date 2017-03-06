#include <stdio.h> //for printf
#include <stdint.h> //for uint32_t
#include <time.h> //for clock_gettime, high precision timer

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
    id = blockIdx.x * blockDim.x + threadIdx.x;
    //ensure extra loop runs if n/cpus has remainder. Can be tuned to ensure
    //at least sqrt(n) thread executions by changing condition
    for( id = blockIdx.x * blockDim.x + threadIdx.x;
            (id * (n/cpus)) < n;
            id += cpus)
    {
        start = id * (n/cpus); //starts at 0, increments by n/cpus
        if(start <= n){
        
            if (start < (step*step)){ //ensure we start at n^2, fixes 2, saves work
                start = step*step;
            }
            start_mult = start/step;

            end = (id + 1) * (n/cpus) - 1; //ensure range has no overlap
            if(end > n){ //avoid overflow
                end = n;
            }
            if(start < end){
            
                end_mult = end/step;

                //printf("thread id %d, num cpus %d, start %d, end %d\n"
                //       , id, cpus, start, end); //debug

                for(int i = start_mult; i<=end_mult; i++){
                    array[step * i] = 1;
                }
            }
        }
    }
    __syncthreads(); //barrier until all threads done
    if(threadIdx.x == 0){ //only valid within a block
        uint32_t j = step+1;
        for(; j<n; j++){
            if(array[j]==0){
                //atomicMin(next_prime, j);
                *next_prime = j;//write host memory
                break;
            }
        }
        if(j == n){
            //printf("found no further primes");
            *next_prime = n;
        }
    }
}

timespec time_diff(timespec start, timespec end)
{
    timespec temp;
    if ((end.tv_nsec-start.tv_nsec)<0) {
        temp.tv_sec = end.tv_sec-start.tv_sec-1;
        temp.tv_nsec = 1000000000+end.tv_nsec-start.tv_nsec;
    } else {
        temp.tv_sec = end.tv_sec-start.tv_sec;
        temp.tv_nsec = end.tv_nsec-start.tv_nsec;
    }
    return temp;
}

int main(int arc, char *argv[]){
    //for(int pow = 1; pow < 30; pow++){
    //    uint32_t n = 2<<pow;
    //    uint32_t n = 1<<28; //find all primes upto and including this number
    //    uint32_t n = 1<<12; //find all primes upto and including this number
        uint32_t n = strtol(argv[1], NULL, 10); //take n from input, conver to long
        //int threads = 512;
        int threads = 1;
        if(512*512 > n){
            threads = sqrt(n);
        }else{
            threads = 512;
        }
        uint32_t *prime_array = (uint32_t *)calloc(n , sizeof(uint32_t)); //allocate and zero
        uint32_t *next_prime; //= (uint32_t *)malloc(sizeof(uint32_t)); //allocate index for signalling
        cudaMallocHost(&next_prime, sizeof(uint32_t));
        *next_prime = 2; //initialize

        uint32_t *d_array;
        cudaMalloc(&d_array, n * sizeof(uint32_t));
        cudaMemcpy(d_array, prime_array, n * sizeof(uint32_t), cudaMemcpyHostToDevice);


        struct timespec start, end, difference;
        clock_gettime(CLOCK_MONOTONIC, &start);
        for(int loop = 2; loop <= sqrt(n); loop = *next_prime){ //TODO careful of sqrt here, check for primes only
            //printf("starting loop for multiples of %d\n", loop);
            multiples<<<1,threads>>>(loop,d_array,n,next_prime);
            cudaDeviceSynchronize();
            //printf("next prime is %d\n", *next_prime);
        }
        clock_gettime(CLOCK_MONOTONIC, &end);
        difference = time_diff(start,end);
        printf("%d %ld\n", difference.tv_sec, difference.tv_nsec);

        cudaMemcpy(prime_array, d_array, n * sizeof(uint32_t), cudaMemcpyDeviceToHost);
        char filename[16];
        //sprintf(filename,"output_pow_%d_thd_%d.txt", pow, threads);
        sprintf(filename,"output_n_%d_thd_%d.txt", n, threads);
        FILE *output_file = fopen(filename, "w+");
        
        for(int i=1; i<n; i++){
            if(prime_array[i] == 0){
                fprintf(output_file, "%d ",i);
            }
        }
        fprintf(output_file, "\n");

        fclose(output_file);
        cudaFree(d_array); //clean up
        cudaFreeHost(next_prime); //clean up
        free(prime_array); //clean up
        cudaDeviceReset();
    //}
    return 0;
}
