## CUDA-at-Image-processing

## GPU Accelerated Batch Image Processing using CUDA and OpenCV
# Project Overview
This project demonstrates GPU accelerated batch image processing using CUDA and OpenCV. The application processes a large number of images using GPU computation instead of CPU-only execution.

CUDA-enabled OpenCV functions are used to perform image operations such as:

Grayscale conversion
Gaussian blur
Edge detection
The program accepts command line arguments for input and output directories and processes multiple images in a single execution.

# Technologies Used
CUDA Toolkit
OpenCV with CUDA support
C++
NVIDIA GPU
# Project Structure
```
CUDA-GPU-Image-Processing/
│
├── input_images/
├── output_images/
├── main.cpp
├── Makefile
├── run.sh
├── README.md
└── execution_log.txt
```

# CUDA Image Processing Code
```

#include <opencv2/opencv.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudafilters.hpp>

#include <filesystem>
#include <iostream>

namespace fs = std::filesystem;

int main(int argc, char* argv[]) {
    if (argc != 3) {
        std::cout << "Usage: ./image_processor <input_folder> <output_folder>" << std::endl;
        return -1;
    }

    std::string input_folder = argv[1];
    std::string output_folder = argv[2];

    for (const auto& entry : fs::directory_iterator(input_folder)) {
        std::string image_path = entry.path().string();

        cv::Mat image = cv::imread(image_path);

        if (image.empty()) {
            std::cout << "Failed to load image: " << image_path << std::endl;
            continue;
        }

        cv::cuda::GpuMat gpu_input;
        gpu_input.upload(image);

        cv::cuda::GpuMat gpu_gray;
        cv::cuda::cvtColor(gpu_input, gpu_gray, cv::COLOR_BGR2GRAY);

        cv::Ptr<cv::cuda::Filter> gaussian_filter =
            cv::cuda::createGaussianFilter(
                gpu_gray.type(),
                gpu_gray.type(),
                cv::Size(5, 5),
                1.5);

        cv::cuda::GpuMat gpu_blur;
        gaussian_filter->apply(gpu_gray, gpu_blur);

        cv::cuda::GpuMat gpu_edges;
        cv::cuda::Canny(gpu_blur, gpu_edges, 50.0, 150.0);

        cv::Mat result;
        gpu_edges.download(result);

        std::string output_path = output_folder + "/processed_" +
                                  entry.path().filename().string();

        cv::imwrite(output_path, result);

        std::cout << "Processed: " << image_path << std::endl;
    }

    std::cout << "All images processed successfully." << std::endl;

    return 0;
}
```

# OUTPUT

<img width="1402" height="1122" alt="image" src="https://github.com/user-attachments/assets/95653bcb-b941-4ba8-b5c7-befd2de2b06b" />

## Result
The project successfully demonstrates GPU accelerated batch image processing using CUDA and OpenCV. The implementation shows how parallel GPU computation can efficiently process large-scale image datasets.
