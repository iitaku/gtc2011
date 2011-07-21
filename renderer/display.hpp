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
                  << cudaGetErrorString(err)  \
                  << std::endl;               \
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
                F::compute();
                F::display();
                                              
                std::stringstream ss;
                ss << "Real Time Raytracing";

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

    struct DrawImage
    {
        static int width_;
        static int height_;
        static int counter_;
        static GLuint texture_;
        static RGBA8U * image_;
        static Scene * scene_;

        static void init(int width, int height)
        {
            width_ = width;
            height_ = height;

            scene_ = new Scene(width_, height_);
            
            image_ = new RGBA8U[width_*height_];
            memset(image_, 0, width_*height_*sizeof(RGBA8U));
            
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
        }

        static void compute(void)
        {
            
            for (int y=0; y<height_; ++y)
            {
                for (int x=0; x<width_; ++x)
                {
                    image_[y*width_+x] = scene_->render(x, y);
                }
            }

            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, 
                         width_, height_, 0, GL_RGBA,
                         GL_UNSIGNED_BYTE, image_);
            
            counter_++;

        }

        static void display(void)
        {
            glClear(GL_COLOR_BUFFER_BIT);

            glBegin(GL_POLYGON);
            glTexCoord2f(0, 1); glVertex2f(-0.95 , -0.95);
            glTexCoord2f(1, 1); glVertex2f(0.95 , -0.95);
            glTexCoord2f(1, 0); glVertex2f(0.95 , 0.95);
            glTexCoord2f(0, 0); glVertex2f(-0.95 , 0.95);
            glEnd();

            glutSwapBuffers();
            glutPostRedisplay();
        }

        static void keyboard(unsigned char key, int x, int y)
        {
            Vector displacement;
            
            switch(key)
            {
                case 'h':
                    displacement = Vector(-0.2, 0.0, 0.0);
                    break;
                case 'j':
                    displacement = Vector(0.0, 0.0, -0.2);
                    break;
                case 'k':
                    displacement = Vector(0.0, 0.0, +0.2);
                    break;
                case 'l':
                    displacement = Vector(+0.2, 0.0, 0.0);
                    break;
                case 'u':
                    displacement = Vector(0.0, -0.2, 0.0);
                    break;
                case 'i':
                    displacement = Vector(0.0, +0.2, 0.0);
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

            scene_->displace_primitive(displacement);
        }
    };

    int DrawImage::width_ = 0;
    int DrawImage::height_ = 0;
    int DrawImage::counter_ = 0;
    GLuint DrawImage::texture_ = 0;
    RGBA8U * DrawImage::image_ = NULL;
    Scene * DrawImage::scene_ = NULL;

} /* namespace gtc */

#endif /* DISPLAY_HPP */
