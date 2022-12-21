#include <GLFW/glfw3.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cuda_gl_interop.h"
#include "surface_functions.h"

#include <iostream>

#include "Math.hpp"
#include "WorldTypes.hpp"

#define WIDTH 1280
#define HEIGHT 720

#define PI 3.1415926535

unsigned int screenTexture;
cudaGraphicsResource_t cudaImage; // Must be global

__global__ void launch_raytracer(cudaSurfaceObject_t screenSurfaceObj, dim3 cell) {
    int threadStartX = blockIdx.x * blockDim.x + threadIdx.x;
    int threadStartY = blockIdx.y * blockDim.y + threadIdx.y;

    constexpr int FOV = PI / 2;
    int screenZ = (WIDTH / 2) / tan(FOV / 2);
    
    // Loop through pixels in designated screen cell
    for (int y = 0; y < cell.y; y++) {
        for (int x = 0; x < cell.x; x++) {
            int screenX = threadStartX * cell.x + x;
            int screenY = threadStartY * cell.y + y;



            surf2Dwrite(make_uchar4(255 * screenX / WIDTH, 255 * screenY / HEIGHT, 0, 255), screenSurfaceObj, screenX * sizeof(uchar4), screenY);
        }
    }
}

void render() {
    cudaError error;
    error = cudaGraphicsMapResources(1, &cudaImage);

    // Map texture array to cuda
    cudaArray_t screenCudaArray;
    error = cudaGraphicsSubResourceGetMappedArray(&screenCudaArray, cudaImage, 0, 0);

    // Data describing array
    cudaResourceDesc screenArrayDesc;
    screenArrayDesc.resType = cudaResourceTypeArray;
    screenArrayDesc.res.array.array = screenCudaArray;

    // Create read-/writeable object for screen array
    cudaSurfaceObject_t screenSurfaceObj;
    error = cudaCreateSurfaceObject(&screenSurfaceObj, &screenArrayDesc);
    if (error != cudaSuccess)
        std::cout << "Failed to map screen array to cuda" << std::endl;

    // Calculate number of threads etc.
    //tex:Number of blocks (number of pixels / (threads/block * pixels/thread)):
    //$$\frac{1280*720}{256*9} = 400$$
    //$$400 = 20*20$$
    //(both 1280 and 720 are divisible by 20)
    
    // Hence, we have a 2D grid of 20 * 20 blocks consisting of 256 threads each (good number)
    // Each block handles a 64 * 36 pixel area
    // Each thread then handles a 1 * 9 pixel area
    dim3 grid(20, 20); // 20 * 20 blocks
    dim3 block(64, 4); // 64 * 4 threads
    dim3 cell(1, 9);

    launch_raytracer<<<grid, block>>>(screenSurfaceObj, cell);

    // Clean up cuda objects
    error = cudaDestroySurfaceObject(screenSurfaceObj);
    error = cudaGraphicsUnmapResources(1, &cudaImage);

    error = cudaStreamSynchronize(0); // Synchronize cuda stream 0 (the only one in use)
    if (error != cudaSuccess)
        std::cout << "Failed to clean up cuda objects and/or synchronize" << std::endl;

    // Draw screen texture
    glBindTexture(GL_TEXTURE_2D, screenTexture);
    glBegin(GL_QUADS);

    // Texture coordinates    Vertex coordinates on screen
    glTexCoord2f(0.0f, 0.0f); glVertex2f(-1.0f, -1.0f);
    glTexCoord2f(1.0f, 0.0f); glVertex2f(1.0f, -1.0f);
    glTexCoord2f(1.0f, 1.0f); glVertex2f(1.0f, 1.0f);
    glTexCoord2f(0.0f, 1.0f); glVertex2f(-1.0f, 1.0f);

    glEnd();
    glBindTexture(GL_TEXTURE_2D, 0); // Unbind texture
    glFinish();
}

int main() {
    GLFWwindow* window;

    // Initialize GLFW
    if (!glfwInit()) {
        std::cout << "Failed to initialize GLFW!" << std::endl;
        std::cin.get();
        return -1;
    }

    // Create a windowed mode window and its OpenGL context
    window = glfwCreateWindow(WIDTH, HEIGHT, "bwidman-raytracer", NULL, NULL);
    if (!window) {
        glfwTerminate();
        return -1;
    }

    // Make the window's context current
    glfwMakeContextCurrent(window);

    // Set up OpenGL screen texture

    cudaGLSetGLDevice(0); // Set target GPU

    glEnable(GL_TEXTURE_2D);
    glGenTextures(1, &screenTexture);

    // Texture settings (rendering canvas)
    glBindTexture(GL_TEXTURE_2D, screenTexture);

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, WIDTH, HEIGHT, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);

    glBindTexture(GL_TEXTURE_2D, 0);

    // Register texture and pixel buffer to cuda
    cudaError_t error = cudaGraphicsGLRegisterImage(&cudaImage, screenTexture, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsSurfaceLoadStore);
    if (error != cudaSuccess) {
        std::cout << "Failed to register screen texture to cuda" << std::endl;
    }

    double deltaTime = 0;
    int frameCount = 0;
    // Loop until the user closes the window
    while (!glfwWindowShouldClose(window)) {
        // Start timer
        double startTime = glfwGetTime();

        // Render here
        glClear(GL_COLOR_BUFFER_BIT);

        render();

        // Swap front and back buffers
        glfwSwapBuffers(window); // Locks FPS to monitor refresh rate

        // Poll for and process events
        glfwPollEvents();

        // Stop timer and print FPS if over a second has elapsed since last print
        deltaTime += glfwGetTime() - startTime;
        frameCount++;
        if (deltaTime > 1.0) {
            std::cout << "FPS: " << frameCount / deltaTime << std::endl;
            deltaTime = 0; frameCount = 0;
        }
    }

    glfwTerminate();

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaError cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    return 0;
}