#include <stdio.h>

__global__ void multiply_them(float *dest, float *a, float *b,int rows,int cols , int channels)
{   
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;

    if(row<rows && col <cols){
        
        for (int i =0 ; i < channels;i++){
            dest[channels* (row*cols + col) + i] =   2.0* a[channels* (row*cols + col) +i ]* b[channels* (row*cols + col)  +i] ;
        
        }
    }
}


__global__ void multiply_them_2d(float *dest, float *a, float *b,int rows,int cols)
{   
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (row < rows && col < cols)
        dest[row*cols + col] =   15.0;
        

}




__device__ void testing(){
    printf("OK");
}


__device__ float compute_distance(float *a, 
                                  float *aa,
                                  float *b, 
                                  float *bb,
                                  int rows,
                                  int cols,
                                  int channels,
                                  int patch_size,
                                  int ax,
                                  int ay, 
                                  int bx,
                                  int by){
    int num_points = 0;
    float pixel_sum = 0;
    float temp_distance = 0; 
    int curr_pix_ax = 0;
    int curr_pix_ay = 0;
    
    int curr_pix_bx = 0;
    int curr_pix_by = 0;
    
    
    for(int y = -patch_size/2 ; y <= patch_size/2 ; y++ ){
        for(int x = -patch_size/2 ; x <= patch_size/2 ; x++){
            
            curr_pix_ax = ax + x; 
            curr_pix_ay = ay + y; 
            
            curr_pix_bx = bx + x; 
            curr_pix_by = by + y; 
            
            if ( curr_pix_ax > 0 && curr_pix_ax < cols && curr_pix_ay > 0 && curr_pix_ay < rows 
               &&
                 curr_pix_bx > 0 && curr_pix_bx < cols && curr_pix_by > 0 && curr_pix_by < rows ){
                
                for(int ch = 0 ; ch < channels ; ch++){
                    
                    temp_distance =  a[channels*(curr_pix_ay*cols + curr_pix_ax ) +ch] 
                                  - bb[channels*(curr_pix_by*cols + curr_pix_bx ) +ch] ;
                    pixel_sum += temp_distance * temp_distance;
                    
                    
                    temp_distance = aa[channels*(curr_pix_ay*cols + curr_pix_ax ) +ch] 
                                  -  b[channels*(curr_pix_by*cols + curr_pix_bx ) +ch] ;
                    pixel_sum += temp_distance * temp_distance;
                }
                num_points ++;
            }
        }
    }
    return pixel_sum / num_points;
       
}



__global__ void patch_match(float *a, 
                            float *aa,
                            float *b, 
                            float *bb,
                            int *nnf,
                            float *nnd,
                            int rows, 
                            int cols , 
                            int channels, 
                            int patch_size, 
                            int iters,
                            int jump_size)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
//     compute_distance(a,aa,b,bb,rows,cols,channels,patch_size,col,row,col,row)
    
    
    nnd[row*cols + col] = compute_distance(a,aa,b,bb,rows,cols,channels,patch_size,col,row,col,row) ; 
    
    if (row < rows && col < cols){
        nnf[2*(row*cols + col) ] = 84.0;
        nnf[2*(row*cols + col) +1] = 84.0;
    }

    for(int i = 0 ; i < iters; i++){
        for(int jump = jump_size ; jump >0 ; jump /=2){
        
        }
        
    }
    
    
}