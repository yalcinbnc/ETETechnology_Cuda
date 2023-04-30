///ETETechnology cuda assignment
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <vector>
#include <algorithm>
#include <ctime>
#include <thread>
#include <chrono>

using namespace std;

// Calculate the positions of the runners in parallel
__global__ void race(float *positions, float *speeds, int n) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n) {
        positions[i] += speeds[i];
    }
}

struct Runner {
    int id;
    float position;

    bool operator<(const Runner& other) const {
        return position > other.position;
    }
};

int main() {
    srand(time(0));
    const int n = 100;
    vector<Runner> runners(n);
    float speeds[n];
    float *d_positions, *d_speeds;

    cudaMalloc((void **)&d_positions, n * sizeof(float));
    cudaMalloc((void **)&d_speeds, n * sizeof(float));

    // Initialize runners and their speeds
    for (int i = 0; i < n; ++i) {
        runners[i].id = i + 1;  
        runners[i].position = 0.0f;
        speeds[i] = 1.0f + rand() % 5;
    }

    cudaMemcpy(d_positions, &runners[0].position, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_speeds, speeds, n * sizeof(float), cudaMemcpyHostToDevice);

    int blockSize = 256;
    int gridSize = (n + blockSize - 1) / blockSize;
    bool finished = false;

    // Run the race
    while (!finished) {
        race<<<gridSize, blockSize>>>(d_positions, d_speeds, n);
        cudaMemcpy(&runners[0].position, d_positions, n * sizeof(float), cudaMemcpyDeviceToHost);

        for (const auto &runner : runners) {
            if (runner.position >= 100.0f) {    
                finished = true;
                break;
            }
        }

        std::this_thread::sleep_for(std::chrono::seconds(1));
    }

    // Sort runners by their positions
    sort(runners.begin(), runners.end());

    // Display the results
    for (int i = 0; i < n; ++i) {
        cout << "Runner " << runners[i].id << " finished in position " << i + 1 << " with a distance of " << runners[i].position << " meters." << endl;
    }

    cudaFree(d_positions);
    cudaFree(d_speeds);

    return 0;
}
