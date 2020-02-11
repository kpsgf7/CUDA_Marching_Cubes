// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <string>
#include <math.h>
#include <time.h>

// includes, kernels
#include <cuda_runtime.h>
#include "kpsgf7_marching_cubes_kernel.cu"

// includes, project
#include <helper_functions.h>  // helper for shared that are common to CUDA SDK samples
#include <helper_cuda.h>       // helper for checking cuda initialization and error checking
#include "kpsgf7_marching_cubes_helpers.cu"




int main(int argc, char **argv){

	if (argc != 6){
		printf("ERROR: IMPROPER PROGRAM USAGE\nCorrect usage: ./kpsgf7_marching_cubes <threshold value> <input directory> <number of input slices> <input image step> <output file>\nTerminating Program...\n");
		return -1;
	}
	cudaDeviceProp deviceProp;
    deviceProp.major = 0;
    deviceProp.minor = 0;
    int dev = 0;

    // This will pick the best possible CUDA capable device
    dev = findCudaDevice(argc, (const char **)argv);

    checkCudaErrors(cudaGetDeviceProperties(&deviceProp, dev));

    // Statistics about the GPU device
    printf("> GPU device has %d Multi-Processors, SM %d.%d compute capabilities\n\n",
           deviceProp.multiProcessorCount, deviceProp.major, deviceProp.minor);

    int version = (deviceProp.major * 0x10 + deviceProp.minor);

    if (version < 0x20)
    {
        printf(" requires a minimum CUDA compute 2.0 capability\n");
        exit(EXIT_SUCCESS);
    }


    // Load one image to determine the width and height of the passed images
    unsigned char *h_img0c = NULL;

    unsigned int w, h;
    std::string directory(argv[2]);

    std::string filepath = directory + "/0.pgm";

    const char *fname = filepath.c_str();

    if(!sdkLoadPGM(fname, &h_img0c, &w, &h)){
    	fprintf(stderr, "Failed to load <%s>\n", fname);
    	return -1;
    }

    printf("Loaded <%s> as image 0\n", fname);

    // using that width and height, allocate host memory
    unsigned int input_slice_n = atoi(argv[3]);
    unsigned int numData = w*h;
    unsigned int memSize = sizeof(unsigned int) * numData * input_slice_n;

    unsigned int *host_images = (unsigned int*)malloc(memSize);
   
    if(host_images == NULL){
    	printf("Failed to allocate. Exiting.\n");
    	return -1;
    }
 
    // cast and copy in the first image
    for (int i=0; i<numData; i++){
    	host_images[i] = (unsigned int)h_img0c[i];
    }

    int step = atoi(argv[4]);

    // load the remainder of the images into the host memory
    for (int img_idx = 1; img_idx<input_slice_n; img_idx++){

        filepath = directory + "/" + std::to_string(img_idx*step)  + ".pgm";

        fname = filepath.c_str();

        if(!sdkLoadPGM(fname, &h_img0c, &w, &h)){
            fprintf(stderr, "Failed to load <%s>\n", fname);
            return -1;
        }   

        for (int i=0; i<numData; i++){
        host_images[i+img_idx * numData] = (unsigned int)h_img0c[i];
        }
    }


    printf("Allocated Host Memory\n");

    //set up timing events
    cudaEvent_t start, stop;
    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&stop));

    checkCudaErrors(cudaEventRecord(start,NULL));

    // allocate device memory 
    unsigned int *device_output, *device_images;

    checkCudaErrors(cudaMalloc((void **) &device_output, memSize));
    checkCudaErrors(cudaMalloc((void **) &device_images, memSize));

    //copy host mem to device memory
    checkCudaErrors(cudaMemcpy(device_images,  host_images, memSize, cudaMemcpyHostToDevice));

    unsigned int thresh = atoi(argv[1]);
    std::cout << "Launching thresholding kernel\n";

    dim3 grid_size_thresh(32,32,input_slice_n);
    dim3 block_size_thresh(16,16,1);
    thresholding_filter_kernel<<<grid_size_thresh,block_size_thresh>>>(device_images, device_output ,thresh);
    cudaDeviceSynchronize();
    std::cout << cudaGetErrorString(cudaPeekAtLastError());

    //uncomment to see intermediate images

    // checkCudaErrors(cudaMemcpy(host_images,  device_output, memSize, cudaMemcpyDeviceToHost));
    // int out=0;
    // for(out=0; out<input_slice_n; out++){

    //     unsigned char *hold = (unsigned char*)malloc(sizeof(unsigned char) * numData);

    //     for (int i=0; i<numData; i++){
    //         hold[i] = (unsigned char)host_images[i + out * numData];
    //     }

    //     // save output
    //     std::string fname_out = "output_scan/" + std::to_string(out) + ".pgm";
    //     printf("Saving file at %s\n", fname_out.c_str());
    //     sdkSavePGM(fname_out.c_str(), hold, w, h);
    // }


    // clean up after the thresholding and set up for marching cubes
    cudaFree(device_images);
    //free(host_images);
    free(h_img0c);
    int *host_lookup_one = get_lookup_one();
    int *host_lookup_two = get_lookup_two();



    std::cout << "\nAllocating Cubes memory\n";
    int *device_lookup_one;
    int *device_lookup_two;

    float *device_triangles;
    int max_triangle_points = (5 * 3 *3 * numData * input_slice_n);
    float *host_triangles = (float *)malloc(sizeof(float) *max_triangle_points);
    if (host_triangles == NULL){
    	std::cout << "Allocating failed\nTerminating\n";
    	return -1;
    }

    std::cout << max_triangle_points * sizeof(float) << " bytes allocated for triangles\n";

    checkCudaErrors(cudaMalloc((void **) &device_triangles, sizeof(float) * max_triangle_points));
    checkCudaErrors(cudaMalloc((void **) &device_lookup_one, sizeof(int) * 256));
    checkCudaErrors(cudaMalloc((void **) &device_lookup_two, sizeof(int) * 256 * 16));

    //checkCudaErrors(cudaMemcpy(device_lookup_one,  host_lookup_one, sizeof(int)*256, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(device_lookup_two,  host_lookup_two, sizeof(int) * 256 * 16, cudaMemcpyHostToDevice));
    std::cout << "Launching cubes kernel\n";
   

    dim3 grid_size(64,64,input_slice_n/4); //512,512
    dim3 block_size(8,8,8);
    marching_cubes_filter<<<grid_size, block_size>>>(device_output, device_lookup_one, device_lookup_two, device_triangles,w,h,input_slice_n,step);
    cudaDeviceSynchronize();

    std::cout << cudaGetErrorString(cudaPeekAtLastError());

    //Copy result from device to host for verification
    checkCudaErrors(cudaMemcpy(host_triangles, device_triangles, sizeof(float)*max_triangle_points, cudaMemcpyDeviceToHost));

    checkCudaErrors(cudaEventRecord(stop,NULL));
    checkCudaErrors(cudaEventSynchronize(stop));

    float msecTotal = 0.0f;
    checkCudaErrors(cudaEventElapsedTime(&msecTotal, start, stop));

    std::cout << "\nGPU processing time : " << msecTotal << " (ms)\n";

   
    // // write a ply file
    write_ply(host_triangles,max_triangle_points, argv[5]);
   
    cudaFree(device_triangles);
    cudaFree(device_output);
    

    std::cout << "Comparing GPU time to CPU time\n";
    clock_t cpu_start = clock();
    compute_cpu_marching_cubes(host_images, thresh, w, h, input_slice_n, host_lookup_one, host_lookup_two, host_triangles);
    clock_t cpu_end = clock();
    std::cout << "CPU processing time: " << (((double)(cpu_end - cpu_start)/ CLOCKS_PER_SEC)*1000) << " ms\n";

    free(host_images);
    free(host_triangles);


    return 0;

}

