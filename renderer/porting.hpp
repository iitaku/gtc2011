#ifndef PORTING_HPP
#define PORTING_HPP

#ifdef USE_CUDA
#define FUNC_DECL __host__ __device__
#else
#define FUNC_DECL
#endif

#endif /* PORTING_HPP */
