#ifndef DISPLAY_HPP
#define DISPLAY_HPP

#include <iostream>
#include <iomanip>
#include <sstream>
#include <cassert>

#ifdef __APPLE__
#include <GLUT/glut.h>
#else
#include <GL/glut.h>
#endif

#include "performance.hpp"
#include "element_collection.hpp"
#include "ray_trace.hpp"

#define CUDA_ERROR_CHECK() {                  \
	cudaError_t err = cudaGetLastError();     \
	if (cudaSuccess != err) {                 \
        std::cerr << __FILE__ << ":"          \
                  << __LINE__ << ":"          \
                  << cudaGetErrorString(err); \
    }                                         \
	assert(cudaSuccess == err);               \
}

namespace gtc
{
    template<typename F>
    class Display
    {
        private:
            
            static void display_callback(void)
            {
                Performance perf("all");
                
                F::compute();
                F::display();
                               
                perf.stop();
                               
                double fps = 1e3f / perf.mean_ms();
                std::stringstream ss;
                ss << "Real Time Raytracing : "
                   << std::setw(5) << std::left << std::setprecision(4) 
                   << fps << " fps";
#ifdef USE_CUDA
                Performance kernel_perf("kernel", false);
                Performance transfer_perf("transfer", false);

                ss << " kernel : " << kernel_perf.mean_ms() << "ms ";
                ss << " transfer : " << transfer_perf.mean_ms() << "ms ";
#endif
                glutSetWindowTitle(ss.str().c_str());

                frame_count++;
            }

            static void keyboard_callback(unsigned char key , int x , int y)
            {
                F::keyboard(key, x, y);
            }

        public:
            Display(int& argc, char* argv[], int width, int height)
            {
                glutInit(&argc, argv);
                glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA);
                
                glutInitWindowSize(width, height);
                
                glutCreateWindow("Real Time Raytracing");
                glutDisplayFunc(display_callback);
                glutKeyboardFunc(keyboard_callback);

                F::init(width, height);
            }

            void loop()
            {
                glutMainLoop();
            }

            void finish()
            {
                F::finish();
            }

    };

#ifdef USE_CUDA
    __global__
    static void new_kernel(Scene ** d_scene_p, int width, int height)
    {
        (*d_scene_p) = new Scene(width, height);
    }

    __global__
    static void delete_kernel(Scene ** d_scene_p)
    {
        delete (*d_scene_p);
    }

    __global__
    static void render_kernel(Scene ** d_scene_p, RGBA8U * d_image, int width, int height)
    {
        int x = blockDim.x * blockIdx.x + threadIdx.x;
        int y = blockDim.y * blockIdx.y + threadIdx.y;

        d_image[y*width+x] = (*d_scene_p)->render(x, y);
    }

    __global__
    static void displace_view_kernel(Scene ** d_scene_p, float v1, float v2, float v3)
    {
        (*d_scene_p)->displace_view(Vector(v1, v2, v3));
    }

    __global__
    static void displace_light_kernel(Scene ** d_scene_p, float v1, float v2, float v3)
    {
        (*d_scene_p)->displace_light(Vector(v1, v2, v3));
    }

#endif

    struct DrawImage
    {
        static int width_;
        static int height_;
        static int counter_;
        static GLuint texture_;
        static RGBA8U * image_;
        static Scene * scene_;

#ifdef USE_CUDA
        static RGBA8U * d_image_;
        static Scene ** d_scene_p_;
#endif

        static void init(int width, int height)
        {
            width_ = width;
            height_ = height;

            scene_ = new Scene(width_, height_);
            
            image_ = new RGBA8U[width_*height_];
            memset(image_, 0, width_*height_*sizeof(RGBA8U));

#ifdef USE_CUDA
            cudaMalloc(&d_scene_p_, sizeof(Scene*));
            new_kernel<<<1, 1>>>(d_scene_p_, width_, height_);

            cudaMalloc(&d_image_, width_*height_*sizeof(RGBA8U));
            cudaMemset(d_image_, 0, width_*height_*sizeof(RGBA8U));
#endif 
            
            glEnable(GL_TEXTURE_2D);
 
            glGenTextures(1, &texture_);

            glBindTexture(GL_TEXTURE_2D, texture_);
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, 
                         width_, height_, 0, GL_RGBA,
                         GL_UNSIGNED_BYTE, image_);
            
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        }

        static void finish(void)
        {
            delete [] image_;
            delete scene_;

#ifdef USE_CUDA
            cudaFree(d_image_);

            delete_kernel<<<1, 1>>>(d_scene_p_);
            cudaFree(d_scene_p_);
#endif
        }

        static void compute(void)
        {

#ifdef USE_CUDA
            dim3 grid_size(width_/16, height_/16);
            dim3 block_size(16, 16);
            
            Performance perf_kernel("kernel");
            
            render_kernel<<<grid_size, block_size>>>(d_scene_p_, d_image_, width_, height_);

            perf_kernel.stop();

            Performance perf_transfer("transfer");
            
            cudaMemcpy(image_, d_image_, width_*height_*sizeof(RGBA8U), cudaMemcpyDeviceToHost);
            
            perf_transfer.stop();
#else
            for (int y=0; y<height_; ++y)
            //for (int y=256; y<height_; ++y)
            {
                for (int x=0; x<width_; ++x)
                //for (int x=256; x<width_; ++x)
                {
                    image_[y*width_+x] = scene_->render(x, y);
                }
            }
#endif
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, 
                         width_, height_, 0, GL_RGBA,
                         GL_UNSIGNED_BYTE, image_);
            
            counter_++;

        }

        static void display(void)
        {
            glClear(GL_COLOR_BUFFER_BIT);

            glBegin(GL_POLYGON);
            glTexCoord2f(0, 0); glVertex2f(-0.95 , -0.95);
            glTexCoord2f(1, 0); glVertex2f(0.95 , -0.95);
            glTexCoord2f(1, 1); glVertex2f(0.95 , 0.95);
            glTexCoord2f(0, 1); glVertex2f(-0.95 , 0.95);
            glEnd();

            glutSwapBuffers();
            glutPostRedisplay();
        }

        static void keyboard(unsigned char key, int x, int y)
        {
            Vector displace;
            
            switch(key)
            {
                case 'h':
                    displace = Vector(-0.1, 0.0, 0.0);
                    break;
                case 'j':
                    displace = Vector(0.0, 0.0, -0.1);
                    break;
                case 'k':
                    displace = Vector(0.0, 0.0, +0.1);
                    break;
                case 'l':
                    displace = Vector(+0.1, 0.0, 0.0);
                    break;
                case 'L':
                    std::cout << x << ":" << y << std::endl;
                    break;
                case 'q':
                    finish();
                    exit(0);
                    break;

                default:
                    break;
            }
#ifdef USE_CUDA
            //displace_view_kernel<<<1, 1>>>(d_scene_p_, displace.V1(), displace.V2(), displace.V3());
            displace_light_kernel<<<1, 1>>>(d_scene_p_, displace.V1(), displace.V2(), displace.V3());
#else
            //scene_->displace_view(displace);
            scene_->displace_light(displace);
#endif
        }
    };

    int DrawImage::width_ = 0;
    int DrawImage::height_ = 0;
    int DrawImage::counter_ = 0;
    GLuint DrawImage::texture_ = 0;
    RGBA8U * DrawImage::image_ = NULL;
    Scene * DrawImage::scene_ = NULL;

#ifdef USE_CUDA
    RGBA8U * DrawImage::d_image_ = NULL;
    Scene ** DrawImage::d_scene_p_ = NULL;
#endif

} /* namespace gtc */

#endif /* DISPLAY_HPP */
