	
 __global__ void thresholding_filter_kernel(unsigned int *input, unsigned int *output, unsigned int thresh){

 	const int blockid = blockIdx.x + blockIdx.y *gridDim.x + gridDim.x * gridDim.y *blockIdx.z;
 	const int out_idx = blockid * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;

 	// const int global_x = blockIdx.x * blockDim.x + threadIdx.x;
 	// const int global_y = blockIdx.y * blockDim.y + threadIdx.y;
 	// const int global_z = blockIdx.z * blockDim.z + threadIdx.z;

 	// const int out_idx = global_x + 512 * (global_y + 24 * global_z);

 	output[out_idx] = 0;
 	if (input[out_idx] > thresh){
 		output[out_idx] = 255;//input[out_idx];
 	}

 	return;
}


__global__ void marching_cubes_filter(unsigned int *input,  int *lookup_one, int *lookup_two, float *triangles, unsigned int data_width, unsigned int data_height, unsigned int data_depth, unsigned int step){


 	const int global_x = blockIdx.x * blockDim.x + threadIdx.x;
 	const int global_y = blockIdx.y * blockDim.y + threadIdx.y;
 	const int global_z = blockIdx.z * blockDim.z + threadIdx.z;

 	const int master_vertex = global_z * data_width * data_height + global_y * data_width + global_x;

 	if(global_x + 1 < data_width-1 && global_y + 1 < data_height-1  && global_z + 1 < data_depth-1){

	 	for(unsigned int tm=0; tm<(5*3*3); tm++){
				triangles[master_vertex* (5*3*3) + tm] = 0.0;
			}


	 	// double check that these refer to the right vertices
	 	int cube[8][4]{
	 		{global_x, global_y, global_z,1},
	 		{global_x, global_y+1, global_z,2},
	 		{global_x+1, global_y+1, global_z,4},
	 		{global_x+1, global_y, global_z,8},
	 		{global_x, global_y, global_z+1,16},
	 		{global_x, global_y+1, global_z+1,32},
	 		{global_x+1, global_y+1, global_z+1,64},
	 		{global_x+1, global_y, global_z+1,128}};

 	

 		int case_lookup_idx = 0;
 		for(unsigned int ci=0; ci<8; ci++){
 			const int x = cube[ci][0];
 			const int y = cube[ci][1];
 			const int z = cube[ci][2];

 	
 			const int vertex = z * data_width * data_height + y * data_width + x;

 			if (input[vertex] ==255){
 				case_lookup_idx |= cube[ci][3];
 			}
 			
 		}

 		int edge_actual[12][6] = {

 			{cube[0][0],cube[0][1],cube[0][2],cube[1][0],cube[1][1],cube[1][2]}, 
 			{cube[1][0],cube[1][1],cube[1][2],cube[2][0],cube[2][1],cube[2][2]},
 			{cube[2][0],cube[2][1],cube[2][2],cube[3][0],cube[3][1],cube[3][2]},
 			{cube[3][0],cube[3][1],cube[3][2],cube[0][0],cube[0][1],cube[0][2]},

 			{cube[4][0],cube[4][1],cube[4][2],cube[5][0],cube[5][1],cube[5][2]},
 			{cube[5][0],cube[5][1],cube[5][2],cube[6][0],cube[6][1],cube[6][2]},
 			{cube[6][0],cube[6][1],cube[6][2],cube[7][0],cube[7][1],cube[7][2]},
 			{cube[7][0],cube[7][1],cube[7][2],cube[4][0],cube[4][1],cube[4][2]},

 			{cube[4][0],cube[4][1],cube[4][2],cube[0][0],cube[0][1],cube[0][2]},
 			{cube[5][0],cube[5][1],cube[5][2],cube[1][0],cube[1][1],cube[1][2]},
 			{cube[6][0],cube[6][1],cube[6][2],cube[2][0],cube[2][1],cube[2][2]},
 			{cube[7][0],cube[7][1],cube[7][2],cube[3][0],cube[3][1],cube[3][2]}

 		};


		if(case_lookup_idx != 255 && case_lookup_idx != 0){
			int current =0;
			int edge_counter = 0;

			for(int w=0; w<16; w++){
						current = lookup_two[case_lookup_idx * 16 + w];
						// current now gives an edge index so we need to add the point to the triangle list
						
						if(current != -1){
							int point1_x = edge_actual[current][0];
							int point1_y = edge_actual[current][1];
							int point1_z = edge_actual[current][2];

							int point2_x = edge_actual[current][3];
							int point2_y = edge_actual[current][4];
							int point2_z = edge_actual[current][5];


							triangles[master_vertex * (5*3*3) +(edge_counter*3) + 0] = (((float)point1_x + (float)point2_x)/2.0); 
							triangles[master_vertex * (5*3*3) +(edge_counter*3) + 1] = (((float)point1_y + (float)point2_y)/2.0); 
							triangles[master_vertex * (5*3*3) +(edge_counter*3) + 2] = (((float)point1_z + (float)point2_z)/2.0) * step;// could do better interpolation here
							edge_counter++;
						}
					
                
        	}
            // printf("\n");
 		}

 	}
 			

	return;
}