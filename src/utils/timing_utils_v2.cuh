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
#include <iomanip>  // For std::setprecision and std::fixed

namespace sgrutils {

class ColumnarCSVWriter {
private:
    std::vector<std::vector<double>> dataColumns;
    std::vector<std::string> columnHeaders;
    
public:
    // Add a column of data with a header
    void addColumn(const std::vector<double>& data, const std::string& header) {
        dataColumns.push_back(data);
        columnHeaders.push_back(header);
    }
    
    // Write all columns to file in CSV format
    void writeToFile(std::ofstream& outfile) {
        // Set high precision and fixed-point notation for full decimal precision
        outfile << std::fixed << std::setprecision(12);
        
        // Write column headers
        for (size_t i = 0; i < columnHeaders.size(); ++i) {
            outfile << columnHeaders[i];
            if (i != columnHeaders.size() - 1) outfile << ",";
        }
        outfile << std::endl;
        
        // Find the maximum size among all columns
        size_t maxSize = 0;
        for (const auto& column : dataColumns) {
            maxSize = std::max(maxSize, column.size());
        }
        
        // Write data row by row
        for (size_t i = 0; i < maxSize; ++i) {
            for (size_t j = 0; j < dataColumns.size(); ++j) {
                if (i < dataColumns[j].size()) {
                    outfile << dataColumns[j][i];
                }
                if (j != dataColumns.size() - 1) outfile << ",";
            }
            outfile << std::endl;
        }
    }
    
    // Clear all data and headers
    void clear() {
        dataColumns.clear();
        columnHeaders.clear();
    }
};

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

void printTimingStats(const std::vector<double>& arr, const char *type, std::ofstream& outfile, int test_iter = 0) {
    // std::cout << "Vector elements for " << type << " : \n";
    // for (double d : arr) {
    //     std::cout << d << " ";
    // }
	double _median, _avg, _stdev, _min, _max;
    // 
    if (TEST_ITERS == 1){
		computeStats(&_median, &_avg, &_stdev, &_min, &_max, arr.size(), arr);
	} else if (TEST_ITERS > 1) {
		
		if (arr.size() > 1) {
			// Remove first array element if array size is greater than 1
			std::vector<double> arr_new(arr.begin() + 1, arr.end());
			computeStats(&_median, &_avg, &_stdev, &_min, &_max, arr_new.size(), arr_new); 
		} else {
			computeStats(&_median, &_avg, &_stdev, &_min, &_max, arr.size(), arr);
		}
	}
	
	// report results
	if (test_iter == TEST_ITERS - 1) {
		std::cout << std::endl;

		printf("%s: Median[%f] Average[%f] StdDev[%f] max[%f] min[%f]\n",type,_median,_avg,_stdev,_max,_min);
	}
	
	// if output file provided, write results to csv as well
	outfile << type << "," << _median << "," << _avg << "," << _stdev << "," << _max << "," << _min << std::endl;
}

void saveMetricsData(const std::vector<double>& arr, const char *type, std::ofstream& outfile){
    // 
	outfile << type << ",";
	for (size_t i = 0; i < arr.size(); ++i) {
		outfile << arr[i];
		if (i != arr.size() - 1) outfile << ",";
	}
	outfile << std::endl;
}

void printAllTimingStats(const std::vector<double>& tTime,
                         const std::vector<double>& initTime,
                         const std::vector<double>& fsimTime,
                         const std::vector<double>& fsweepTime,
                         const std::vector<double>& linesearchTime,
                         const std::vector<double>& bpTime,
                         const std::vector<double>& nisTime,
						 const std::vector<double>& alOuterLoopTime,
						 const std::vector<double>& costFSTime,
						 const std::string& filename = "nch_nTS_nit_vn",
						 int test_iter = 0) {
    
	// Print timing information
	if (test_iter == TEST_ITERS - 1) {
		printf("\n=========================================================================\n");
		printf("Printing timing information: \n");
		printf("Length of tTime: %d\n", tTime.size());
		printf("Length of initTime: %d\n", initTime.size());
		printf("Length of fsimTime: %d\n", fsimTime.size());
		printf("Length of fsweepTime: %d\n", fsweepTime.size());
		printf("Length of linesearchTime: %d\n", linesearchTime.size());
		printf("Length of bpTime: %d\n", bpTime.size());
		printf("Length of nisTime: %d\n", nisTime.size());
		printf("Length of alOuterLoopTime: %d\n", alOuterLoopTime.size());
		printf("Length of costFSTime: %d\n", costFSTime.size());
	}

	// Timing Statistics
	std::ofstream outfile("z_output_csv/timing_stats_" + filename + ".csv");
	printTimingStats(tTime,"Total", outfile, test_iter);
	printTimingStats(initTime,"Initialization", outfile, test_iter);
	printTimingStats(fsimTime,"Forward Simulation", outfile, test_iter);
	printTimingStats(costFSTime,"FS Cost Function", outfile, test_iter);
	printTimingStats(fsweepTime,"Forward Sweep", outfile, test_iter);
	printTimingStats(linesearchTime,"Line Search", outfile, test_iter);
	printTimingStats(bpTime,"Backwards Pass", outfile, test_iter);
	if (alOuterLoopTime.size() > 0) {
		printTimingStats(alOuterLoopTime,"AL Outer Loop", outfile, test_iter);
	}
	if (nisTime.size() > 0) {
		printTimingStats(nisTime,"Next Iteration Setup", outfile, test_iter);
	}
	else {
		printf("WARN: Next iteration setup timing array empty \n");
	}
	outfile.close();

	// Timing Data - Save in columnar format using new function
	std::ofstream outfile2("z_output_csv/timing_data_" + filename + ".csv");
	// Use the new columnar function
	ColumnarCSVWriter csvWriter;
	csvWriter.addColumn(tTime, "Total");           // Add column 1
	csvWriter.addColumn(initTime, "Initialization"); // Add column 2
	csvWriter.addColumn(fsimTime, "Forward_Simulation"); // Add column 3
	csvWriter.addColumn(costFSTime, "FS_Cost_Function"); // Add column 4
	csvWriter.addColumn(fsweepTime, "Forward_Sweep"); // Add column 5
	csvWriter.addColumn(linesearchTime, "Line_Search"); // Add column 6
	csvWriter.addColumn(bpTime, "Backwards_Pass"); // Add column 7
	if (alOuterLoopTime.size() > 0) {
		csvWriter.addColumn(alOuterLoopTime, "AL_Outer_Loop"); // Add column 8
	}
	if (nisTime.size() > 0) {
		csvWriter.addColumn(nisTime, "Next_Iteration_Setup"); // Add column 9
	}
	csvWriter.writeToFile(outfile2); // Write everything at once
	outfile2.close();
}

template <typename T>
void printAllMetricsStats(const std::vector<T>& total_cost,
						 const std::vector<T>& norm_defects,
						 const std::vector<T>& state_error,
						 const std::vector<T>& line_search_index,
						 const std::vector<T>& rho_values,
						 const std::string& filename = "nch_nTS_nit_vn0",
						 int test_iter = 0) {
	// Convert input vectors to double vectors for compatibility if needed
	std::vector<double> total_cost_d, norm_defects_d, state_error_d, line_search_index_d, rho_values_d;

	if constexpr (std::is_same<T, double>::value) {
		// Already double, just copy
		total_cost_d = total_cost;
		norm_defects_d = norm_defects;
		state_error_d = state_error;
		line_search_index_d = line_search_index;
		rho_values_d = rho_values;
	} else if constexpr (std::is_same<T, float>::value) {
		// Convert float to double
		auto to_double_vec = [](const std::vector<T>& v) {
			return std::vector<double>(v.begin(), v.end());
		};
		total_cost_d = to_double_vec(total_cost);
		norm_defects_d = to_double_vec(norm_defects);
		state_error_d = to_double_vec(state_error);
		line_search_index_d = to_double_vec(line_search_index);
		rho_values_d = to_double_vec(rho_values);
	} else {
		static_assert(std::is_same<T, float>::value || std::is_same<T, double>::value, "T must be float or double");
	}

	// Metric Statistics
	std::ofstream outfile("z_output_csv/metrics_stats_" + filename + ".csv");
	if (test_iter == TEST_ITERS - 1) {
		printf("\n=========================================================================\n");
		printf("Printing metric information: \n");
	}
	printTimingStats(total_cost_d,"Total Cost", outfile, test_iter);
	printTimingStats(norm_defects_d,"Norm Defects", outfile, test_iter);
	printTimingStats(state_error_d,"State Error", outfile, test_iter);
	printTimingStats(line_search_index_d,"Line Search Index", outfile, test_iter);
	printTimingStats(rho_values_d,"Rho Values", outfile, test_iter);
	
	// Check if file is still good before closing
    if (outfile.is_open()) {
        outfile.close();
    }

	// Metric Data - Save in columnar format using new function
	std::ofstream outfile2("z_output_csv/metrics_data_" + filename + ".csv");
	// Use the new columnar function
	ColumnarCSVWriter csvWriter2;
	csvWriter2.addColumn(total_cost_d, "Total_Cost");           // Add column 1
	csvWriter2.addColumn(norm_defects_d, "Norm_Defects"); // Add column 2
	csvWriter2.addColumn(state_error_d, "State_Error"); // Add column 3
	csvWriter2.addColumn(line_search_index_d, "Line_Search_Index"); // Add column 4
	csvWriter2.addColumn(rho_values_d, "Rho_Values"); // Add column 5
	csvWriter2.writeToFile(outfile2); // Write everything at once
	outfile2.close();
}

} // namespace sgrutils

#endif // SGRUTILS_TIMING_UTILS_CUH