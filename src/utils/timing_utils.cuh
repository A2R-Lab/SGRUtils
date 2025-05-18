#ifndef SGRUTILS_TIMING_UTILS_CUH
#define SGRUTILS_TIMING_UTILS_CUH

/*
    Utility functions related to timing.
*/
#include <vector>
#include <algorithm>
#include <numeric>
#include <string>
#include <iostream>
#include <fstream>
#include <cmath>

namespace sgrutils {

void computeStats(double *_median, double *_avg, double *_stdev, double *_min, double *_max, int size, std::vector<double> v){
	// sort gives us the median, max and min
    std::sort(v.begin(),v.end());
	*_median = size % 2 ? v[size / 2] : (v[size / 2 - 1] + v[size / 2]) / 2.0;	
	*_max = v.back();	
	*_min = v.front();
	// sum gives use the average
	double sum = std::accumulate(v.begin(), v.end(), 0.0);
	*_avg = sum / (double)size;	
	// and then the std dev	
	*_stdev = 0.0;
	for(std::vector<double>::iterator it = v.begin(); it != v.end(); ++it){
		*_stdev += pow(*it-*_avg,2.0);
	}
	*_stdev /= (double) size;	
	*_stdev = pow(*_stdev,0.5);
}

void printTimingStats(const std::vector<double>& arr, const char *type, std::ofstream& outfile){
    // std::cout << "Vector elements for " << type << " : \n";
    // for (double d : arr) {
    //     std::cout << d << " ";
    // }
	double _median, _avg, _stdev, _min, _max;
    std::cout << std::endl;
    computeStats(&_median, &_avg, &_stdev, &_min, &_max, arr.size(), arr);
	// report results
	printf("%s: Median[%f] Average[%f] StdDev[%f] max[%f] min[%f]\n",type,_median,_avg,_stdev,_max,_min);

	// if output file provided, write results to csv as well
	outfile << type << "," << _median << "," << _avg << "," << _stdev << "," << _max << "," << _min << std::endl;
}

void printAllTimingStats(const std::vector<double>& tTime,
                         const std::vector<double>& initTime,
                         const std::vector<double>& fsimTime,
                         const std::vector<double>& fsweepTime,
                         const std::vector<double>& linesearchTime,
                         const std::vector<double>& bpTime,
                         const std::vector<double>& nisTime,
						 const std::vector<double>& alOuterLoopTime,
						 const std::vector<double>& costFSTime) {
    printf("Printing timing information: \n");
	std::cout << "Length of tTime: " << tTime.size() << std::endl;
    std::cout << "Length of initTime: " << initTime.size() << std::endl;
    std::cout << "Length of fsimTime: " << fsimTime.size() << std::endl;
    std::cout << "Length of fsweepTime: " << fsweepTime.size() << std::endl;
    std::cout << "Length of linesearchTime: " << linesearchTime.size() << std::endl;
    std::cout << "Length of bpTime: " << bpTime.size() << std::endl;
    std::cout << "Length of nisTime: " << nisTime.size() << std::endl;
	std::cout << "Length of alOuterLoopTime: " << alOuterLoopTime.size() << std::endl;
	std::cout << "Length of costFSTime: " << costFSTime.size() << std::endl;
	std::ofstream outfile("output_12ch_46TS_100Tests_v2.csv");
	printTimingStats(tTime,"Total", outfile);
	printTimingStats(initTime,"Initialization", outfile);
	printTimingStats(fsimTime,"Forward Simulation", outfile);
	printTimingStats(costFSTime,"FS Cost Function", outfile);
	printTimingStats(fsweepTime,"Forward Sweep", outfile);
	printTimingStats(linesearchTime,"Line Search", outfile);
	printTimingStats(bpTime,"Backwards Pass", outfile);
	if (alOuterLoopTime.size() > 0) {
		printTimingStats(alOuterLoopTime,"AL Outer Loop", outfile);
	}
	if (nisTime.size() > 0) {
		printTimingStats(nisTime,"Next Iteration Setup", outfile);
	}
	else {
		printf("WARN: Next iteration setup timing array empty \n");
	}
}

} // namespace sgrutils

#endif // SGRUTILS_TIMING_UTILS_CUH