project(renderer)
cmake_minimum_required(VERSION 2.8)

find_package(CUDA)

if(APPLE)
find_library(GL_LIBRARIES opengl HINTS /System/Library)
find_library(GLUT_LIBRARIES glut)
elseif(WIN32)
set(GL_LIBRARIES "")
set(GLUT_LIBRARIES freeglut)
else()
set(GL_LIBRARIES "")
set(GLUT_LIBRARIES glut)
endif()

add_definitions(-g)
add_executable(cpu_renderer main.cpp)
target_link_libraries(cpu_renderer ${GL_LIBRARIES} ${GLUT_LIBRARIES})

