// Simple GPU accelerated ray tracer
// Author: Benjamin Widman (benjaneb)
#define __CUDACC__
#include <GLFW/glfw3.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cuda_gl_interop.h"

#include <iostream>

#include "Math.cuh"
#include "WorldTypes.cuh"
#include "Controls.cuh"

#define WIDTH 1280
#define HEIGHT 720

unsigned int screenTexture;
cudaGraphicsResource_t cudaImage; // Must be global

scene allocateScene() {
    camera hCamera = { ZERO_VEC, { { 0, 0, 1 }, { 1, 0, 0 } }, 0, 0, PI / 2 };
    sphere hSpheres[] = {
        // Position, radius, color
        { { 2, 0, 8 }, 2, { 200, 50, 0 } },
        { { -2, 0, 6 }, 1, { 50, 0, 200 } },
    };
    int hSphereCount = sizeof(hSpheres) / sizeof(sphere);

    // Allocate spheres on GPU
    sphere* dSpheres;
    cudaMalloc(&dSpheres, sizeof(hSpheres));
    cudaMemcpy(dSpheres, &hSpheres, sizeof(hSpheres), cudaMemcpyHostToDevice);

    return { hCamera, dSpheres, hSphereCount };
}

// Check if camera ray intersects with sphere
__device__ bool sphereIntersect(ray cameraRay, sphere sphere, color* pixel, vec3d* intersection, float* closestHit) {
    // If you don't have the tex comments extension, good luck reading this
    //tex:
    // Sphere equation:
    // $$(x-p_1)^2 + (y-p_2)^2 + (z-p_3)^2 = r^2$$
    // Ray equation: $$\vec{r} = \vec{x} + t\vec{v}$$
    // Input ray into sphere equation:
    // $$(x_1+tv_1-p_1)^2 + (x_2+tv_2-p_2)^2 + (x_3+tv_3-p_3)^2 = r^2$$
    // $$(x_1^2 + x_1v_1t - x_1p_1 + x_1v_1t + v_1^2t^2 - v_1p_1t - p_1x_1 - p_1v_1t + p_1^2) + ... - r^2 = 0$$
    // $$(v_1^2 + v_2^2 + v_3^2)t^2 + (2x_1v_1 - 2p_1v_1 + 2x_2v_2 - 2p_2v_2 + 2x_3v_3 - 2p_3v_3)t + (x_1^2 - 2p_1x_1 + p_1^2 + x_2^2 - 2p_2x_2 + p_2^2 + x_3^2 - 2p_3x_3 + p_3^2 - r^2) = 0$$
    // $$(v_1^2 + v_2^2 + v_3^2)t^2 + 2(x_1v_1 - p_1v_1 + x_2v_2 - p_2v_2 + x_3v_3 - p_3v_3)t + ((x_1 - p_1)^2 + (x_2 - p_2)^2 + (x_3 - p_3)^2 - r^2) = 0$$
    // $$(\vec{v} \cdot \vec{v})t^2 + 2((\vec{x} - \vec{p}) \cdot \vec{v})t + ((\vec{x} - \vec{p}) \cdot (\vec{x} - \vec{p}) - r^2) = 0$$
    // Solve for t with quadratic formula:
    // $$t = \frac{-b\pm\sqrt{b^2 - 4ac}}{2a}$$
    // Where:
    // $$a = \vec{v} \cdot \vec{v}$$
    // $$b = 2((\vec{x} - \vec{p}) \cdot \vec{v})$$
    // $$c = (\vec{x} - \vec{p}) \cdot (\vec{x} - \vec{p}) - r^2$$
    vec3d p = sphere.position;
    vec3d x = cameraRay.origin;
    vec3d v = cameraRay.direction;

    float a = dotProduct(v, v);
    float b = 2 * dotProduct(x - p, v);
    float c = dotProduct(x - p, x - p) - sphere.radius * sphere.radius;

    float discriminant = b * b - 4 * a * c; // Discriminator of all

    // Negative root => no solutions => no intersection
    if (discriminant < 0) {
        return false;
    }

    // Only interested in negative solution to the root as it gives the
    // smallest value of t and is therefore the closest to the camera
    float t = (-b - sqrtf(discriminant)) / (2 * a);

    // Behind camera or further away than the so far closest hit
    if (t < 0 || t > *closestHit) {
        return false;
    }

    *closestHit = t;
    *pixel = sphere.color;
    *intersection = cameraRay.origin + t * cameraRay.direction;
    return true;
}

__device__ color raytrace(ray cameraRay, sphere spheres[], int sphereCount) {
    color pixel = ZERO_VEC;
    float closestHit = INFINITY; // Gets updated for every new closest hit
    
    // Check intersection with all spheres
    for (int i = 0; i < sphereCount; i++) {
        vec3d intersection;
        bool intersected = sphereIntersect(cameraRay, spheres[i], &pixel, &intersection, &closestHit);
    }

    return pixel;
}

__global__ void launch_raytracer(cudaSurfaceObject_t screenSurfaceObj, dim3 cell, scene scene, float screenZ, matrix3d rotLeft, matrix3d rotUp) {
    int pixelStartX = (blockIdx.x * blockDim.x + threadIdx.x) * cell.x;
    int pixelStartY = (blockIdx.y * blockDim.y + threadIdx.y) * cell.y;

    // Loop through pixels in designated screen cell
    for (int y = 0; y < cell.y; y++) {
        for (int x = 0; x < cell.x; x++) {
            float screenX = pixelStartX + x;
            float screenY = pixelStartY + y;

            vec3d pixelPosition = { screenX - WIDTH / 2, screenY - HEIGHT / 2, screenZ };
            pixelPosition = rotLeft * rotUp * pixelPosition; // Rotate to camera's facing direction

            ray cameraRay = { scene.camera.position, normalize(pixelPosition) };

            color pixel = raytrace(cameraRay, scene.spheres, scene.sphereCount);

            surf2Dwrite(make_uchar4(pixel.r, pixel.g, pixel.b, 255), screenSurfaceObj, screenX * sizeof(uchar4), screenY);
        }
    }
}

void render(scene scene) {
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

    // Screen coordinate calculations
    float screenZ = (WIDTH / 2) / tanf(scene.camera.FOV / 2);
    matrix3d rotLeft = rotationMatrix3DY(scene.camera.angle[0]);
    matrix3d rotUp = rotationMatrix3DX(scene.camera.angle[1]);

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
    dim3 cell(1, 9); // Dimensions of pixel cell handled by each thread

    launch_raytracer<<<grid, block>>>(screenSurfaceObj, cell, scene, screenZ, rotLeft, rotUp);

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

    // Allocate the scene on the GPU
    scene scene = allocateScene();

    double deltaTime = 0;
    int frameCount = 0;
    // Loop until the user closes the window
    while (!glfwWindowShouldClose(window)) {
        // Start timer
        double startTime = glfwGetTime();

        // Render here
        glClear(GL_COLOR_BUFFER_BIT);

        render(scene);

        // Swap front and back buffers
        glfwSwapBuffers(window);

        controls(window, scene.camera);

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

    // Clean up scene
    cudaFree(scene.spheres);

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaError cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    return 0;
}