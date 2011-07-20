#ifndef PERF_TRACER_HPP
#define PERF_TRACER_HPP

#include <vector>
#include <map>
#include <functional>
#include <numeric>

#if WIN32
#include <windows.h>
#else
#include <sys/time.h>
#endif

#define RANGE 10

namespace gtc
{

static std::map<const char*, int> counters;
static std::map<const char*, std::vector<double> > ms_results;
static std::map<const char*, std::vector<double> > us_results;

class Performance
{
private:
    const char * tag_;
    bool is_measured;

#ifdef WIN32
    LARGE_INTEGER start_;
    LARGE_INTEGER stop_;
#else
    struct timeval start_;
    struct timeval stop_;
#endif

public:
    Performance(const char * tag = NULL, bool auto_start = true)
        : tag_(tag), is_measured(false)
    {
        if (counters.end() == counters.find(tag_))
        {
            counters.insert(std::make_pair(tag_, 0));
            ms_results.insert(std::make_pair(tag_, std::vector<double>(RANGE)));
            us_results.insert(std::make_pair(tag_, std::vector<double>(RANGE)));
        }

        if (auto_start)
        {
            start();
        }
    }

    ~Performance(void)
    {
        if (is_measured)
        {
            int counter = counters[tag_];

            ms_results[tag_][counter] = ms();
            us_results[tag_][counter] = us();

            counters[tag_] = (counter+1) % RANGE;
        }
    }

    void start(void)
    {
#ifdef WIN32
        QueryPerformanceCounter(&start_);
#else
        gettimeofday(&start_, NULL);
#endif
    }

    void stop(void)
    {
#if USE_CUDA
        cudaDeviceSynchronize();
#endif

#ifdef WIN32
        QueryPerformanceCounter(&stop_);
#else
        gettimeofday(&stop_, NULL);
#endif
        is_measured = true;
    }

    double ms(void)
    {
#ifdef WIN32   
        LARGE_INTEGER freq;
        QueryPerformanceFrequency(&freq);
        return (static_cast<double>(stop_.QuadPart-start_.QuadPart)*1e3f)/static_cast<double>(freq.QuadPart);
#else
        return static_cast<double>((stop_.tv_sec-start_.tv_sec)*1e3f + (stop_.tv_usec-start_.tv_usec)/1e3f);
#endif
    }

    double us(void)
    {
#ifdef WIN32   
        LARGE_INTEGER freq;
        QueryPerformanceFrequency(&freq);
        return (static_cast<double>(stop_.QuadPart-start_.QuadPart)*1e6f)/static_cast<double>(freq.QuadPart);
#else
        return static_cast<double>((stop_.tv_sec-start_.tv_sec)*1e6f + (stop_.tv_usec-start_.tv_usec));
#endif
    }

    double mean_ms(void)
    {
        return std::accumulate(&ms_results[tag_][0], &ms_results[tag_][RANGE-1], 0, std::plus<double>()) / static_cast<double>(RANGE);
    }

    double mean_us(void)
    {
        return std::accumulate(&us_results[tag_][0], &us_results[tag_][RANGE-1], 0, std::plus<double>()) / static_cast<double>(RANGE);
    }
};

} /* namespace gtc */

#endif /* PERF_TRACER_HPP */
