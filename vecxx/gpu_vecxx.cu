#include <iostream>
#include <vector>

#define NUM 512

class Double
{
public:
    __device__
    float operator()(float val)
    {
        return 2*val;
    }
};

template<typename F>
__global__
void gpu_kernel(float * buf, F func)
{
    int idx = threadIdx.x;

    buf[idx] = func(buf[idx]);
}

void gpu_run(void)
{
    std::cout << std::endl << "gpu" << std::endl;
    std::vector<float> h_buf(NUM, 1);

    float * d_buf;
    
    cudaMalloc(&d_buf, NUM*sizeof(float));

    cudaMemcpy(d_buf, h_buf.data(), NUM*sizeof(float), cudaMemcpyHostToDevice);
    
    gpu_kernel<<<1, NUM>>>(d_buf, Double());

    cudaMemcpy(h_buf.data(), d_buf, NUM*sizeof(float), cudaMemcpyDeviceToHost);
   
    std::cout << "  0:" << h_buf[0] << std::endl;
    std::cout << "511:" << h_buf[511] << std::endl;

    cudaFree(d_buf);
}

int main()
{
    gpu_run();
}
