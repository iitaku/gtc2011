#include <iostream>
#include <vector>

#define NUM 512

class Double
{
public:
    __host__ __device__
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

template<typename F>
void cpu_kernel(float * buf, F func)
{
    for (int idx = 0; idx<NUM; ++idx)
    {
        buf[idx] = func(buf[idx]);
    }
}

void gpu_run(void)
{
    std::cout << "gpu" << std::endl;
    std::vector<float> h_buf(NUM, 1);

    float * d_buf;
    
    cudaMalloc(&d_buf, NUM*sizeof(float));

    cudaMemcpy(d_buf, h_buf.data(), NUM*sizeof(float), cudaMemcpyHostToDevice);
    
    gpu_kernel<<<1, NUM>>>(d_buf, Double());

    cudaMemcpy(h_buf.data(), d_buf, NUM*sizeof(float), cudaMemcpyDeviceToHost);
   
    std::cout << 0 << ":" << h_buf[0] << std::endl << std::endl;

    cudaFree(d_buf);
}

void cpu_run(void)
{
    std::cout << "cpu" << std::endl;
    std::vector<float> h_buf(NUM, 1);
    
    cpu_kernel(h_buf.data(), Double());
   
    std::cout << 0 << ":" << h_buf[0] << std::endl << std::endl;
}

int main()
{
    cpu_run();
    gpu_run();
}
