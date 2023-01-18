// Simple GPU accelerated ray tracer
// Author: Benjamin Widman (benjaneb)

// If you don't have the "tex comments" extension,
// good luck reading the math derivations
#define __CUDACC__
#include <GLFW/glfw3.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cuda_gl_interop.h"
#include "curand_kernel.h"

#include <iostream>

#include "Math.cuh"
#include "WorldTypes.cuh"
#include "Intersection.cuh"
#include "Controls.cuh"

// Only works with 16:9 aspect ratios, such as:
// 640x360, 960x540, 1280x720, 1920x1080, 2560x1440
constexpr int windowWidth = 1280;
constexpr int windowHeight = 720;
constexpr bool fullscreen = false;
constexpr int maxBounces = 5;
constexpr int samplesPerPixel = 2;
constexpr color backgroundColor = { 0, 0, 0 };

unsigned int screenTexture;
cudaGraphicsResource_t cudaImage; // Must be global

#define CUDA_ARR_COPY(dArr, hArr)\
    cudaMalloc(&dArr, sizeof(hArr));\
    cudaMemcpy(dArr, &hArr, sizeof(hArr), cudaMemcpyHostToDevice);\

scene allocateScene() {
    camera camera = { { 0, 1, 0 }, { 0, 0 } , PI / 2 };

    // material = albedo, emittance, reflectivity
    sphere hSpheres[] = {
        // Position, radius, material
        { { -6, 3, -4 }, 1, { { 1, 0.6, 0.2 }, 20, 0 } }, // Orange light left
        { { 6, 3, -4 }, 1, { { 1, 0.2, 0.6 }, 20, 0 } }, // Purple light right
        { { -0.5, 0.2, -3 }, 0.2, { { 0.2, 0.8, 0.2 }, 5, 0 } }, // Green light center

        { { 0, 0.75, -4 }, 0.75, { { 1, 1, 1 }, 0, 1 } }, // Center white
        { { -4, 1, -6 }, 1, { { 0.2, 0, 0.8 }, 0, 1 } }, // Left purple
        { { 4, 2, -8 }, 2, { { 1, 0.1, 0 }, 0, 1 } }, // Right red
    };
    int sphereCount = sizeof(hSpheres) / sizeof(sphere);

    plane hPlanes[] = {
        // Origin,      directions,                     material
        { { 0, 0, 0 }, { { 0, 0, 1 }, { 1, 0, 0 } }, { { 0.5, 0.5, 0.5 }, 0, 0 } },
    };
    int planeCount = sizeof(hPlanes) / sizeof(plane);

    triangle hTriangles[] = {
        // Vertices,                                    material
        // Pyramid
        { { { -2, 0, -3 }, { -1, 0, -3 }, { -1.5, 1, -3.5 } }, { { 0.95, 0.9, 0.2 }, 0, 0 } }, // front
        { { { -1, 0, -4 }, { -2, 0, -4 }, { -1.5, 1, -3.5 } }, { { 0.95, 0.9, 0.2 }, 0, 0 } }, // back
        { { { -2, 0, -4 }, { -2, 0, -3 }, { -1.5, 1, -3.5 } }, { { 0.95, 0.9, 0.2 }, 0, 0 } }, // left
        { { { -1, 0, -3 }, { -1, 0, -4 }, { -1.5, 1, -3.5 } }, { { 0.95, 0.9, 0.2 }, 0, 0 } }, // right
    };
    int triangleCount = sizeof(hTriangles) / sizeof(triangle);

    // Allocate objects on GPU
    sphere* dSpheres;
    CUDA_ARR_COPY(dSpheres, hSpheres);

    plane* dPlanes;
    CUDA_ARR_COPY(dPlanes, hPlanes);

    triangle* dTriangles;
    CUDA_ARR_COPY(dTriangles, hTriangles);

    return { 
        camera, 
        dSpheres, sphereCount, 
        dPlanes, planeCount,
        dTriangles, triangleCount 
    };
}

__device__ float fresnel(vec3d incident, vec3d normal, float refractionIndex1, float refractionIndex2) {
    float c = dotProduct(incident, normal);
    float gRoot = (refractionIndex2 * refractionIndex2) / (refractionIndex1 * refractionIndex1) - 1 + c * c;

    if (gRoot < 0) // Total internal reflection
        return 1;
    float g = sqrtf(gRoot);

    return 0.5 * (g - c) * (g - c) / ((g + c) * (g + c)) *
        (1 + (c * (g + c) - 1) * (c * (g + c) - 1) / ((c * (g - c) + 1) * (c * (g - c) + 1)));
}

__device__ color diffuseBRDF(vec3d incident, vec3d normal, vec3d scatterDir) {
    vec3d halfDir = sign(dotProduct(incident, scatterDir)) * (incident + scatterDir);
}

__device__ vec3d reflect(vec3d direction, vec3d normal) {
    // direction and normal are both normalized so:
    //tex:$$proj_\vec{n}(\vec{d}) = (\vec{d} \cdot \vec{n})\vec{n}$$
    return direction - 2 * dotProduct(direction, normal) * normal;
}

__device__ vec3d genRandomDirection(curandStateXORWOW* randState, vec3d normal) {
    vec3d randomDirection = ZERO_VEC;
    do {
        randomDirection = {
            float(curand(randState)) / INT_MAX - 1.0f,
            float(curand(randState)) / INT_MAX - 1.0f,
            float(curand(randState)) / INT_MAX - 1.0f
        };
    } while (length(randomDirection) > 1);

    randomDirection = normalize(randomDirection);

    if (dotProduct(normal, randomDirection) < 0) {
        // Reflect to other hemisphere by subtracting twice it's projection on the normal
        randomDirection -= 2 * dotProduct(randomDirection, normal) * normal;
    }
    return randomDirection;
}

__device__ color tracePath(ray incidentRay, const scene& scene, curandStateXORWOW* randState, int bounces = 0) {
    color outgoingLight = backgroundColor;
    if (bounces > maxBounces) // Stop recursion
        return outgoingLight;

    // Gets updated for every new closest hit
    intersectionInfo closestHit = {};

    // We want to loop the number of times that is the largest array out of all the objects
    float largestArraySize = max(scene.sphereCount, max(scene.planeCount, scene.triangleCount));
    
    bool intersected = false;
    // Check intersection with all objects and shade accordingly
    for (int i = 0; i < largestArraySize; i++) {
        // Check index to avoid "index out of range"
        if (i < scene.sphereCount)
            intersected += sphereIntersection(incidentRay, scene.spheres[i], &closestHit);

        if (i < scene.planeCount)
            intersected += planeIntersection(incidentRay, scene.planes[i], &closestHit);

        if (i < scene.triangleCount)
            intersected += triangleIntersection(incidentRay, scene.triangles[i], &closestHit);
    }

    // If intersection was found with any of the objects shade the closest point
    if (intersected) {
        //ray reflectionRay = { closestHit.intersection, reflect(incidentRay.direction, closestHit.normal) };

        color emittedLight = closestHit.attributes.emittance * closestHit.attributes.albedo;

        vec3d randomDirection = genRandomDirection(randState, closestHit.normal);

        //color brdf = 2.0 * closestHit.attributes.albedo;
        color brdf = diffuseBRDF(-randomDirection, closestHit.normal, -incidentRay.direction);

        color incomingLight = tracePath({ closestHit.intersection, randomDirection }, scene, randState, bounces + 1);

        float cosAngle = dotProduct(randomDirection, closestHit.normal);

        //tex:The rendering equation
        //$$L_o(\omega_o) = L_e(\omega_o) + \int_\Omega f(\omega_i, \omega_o) L_i(\omega_i) (\omega_i \cdot n) d\omega_i$$
        outgoingLight = emittedLight + brdf * incomingLight * cosAngle;
    }

    return outgoingLight;
}

__global__ void launchRaytracer(cudaSurfaceObject_t screenSurfaceObj, dim3 cell, const scene scene, 
    float screenZ, matrix3d rotLeft, matrix3d rotUp, curandStateXORWOW* randStates, unsigned int accumulatedFrames, color* frameSum) {
    int pixelStartX = (blockIdx.x * blockDim.x + threadIdx.x) * cell.x;
    int pixelStartY = (blockIdx.y * blockDim.y + threadIdx.y) * cell.y;

    // Loop through pixels in designated screen cell
    for (int y = 0; y < cell.y; y++) {
        for (int x = 0; x < cell.x; x++) {
            int screenX = pixelStartX + x;
            int screenY = pixelStartY + y;
            int pixelIndex = screenY * windowWidth + screenX;
            curandStateXORWOW* randState = &randStates[pixelIndex];

            vec3d pixelPosition = { screenX - windowWidth / 2, screenY - windowHeight / 2, screenZ };
            pixelPosition = rotLeft * rotUp * pixelPosition; // Rotate to camera's facing direction

            ray cameraRay = { scene.camera.position, normalize(pixelPosition) };
            cameraRay.direction += 0.001 * (windowWidth / 1000) * genRandomDirection(randState, cameraRay.direction); // Jitter for anti-aliasing
            cameraRay.direction = normalize(cameraRay.direction);

            color pixel = ZERO_VEC;
            // Gather a number of samples and get the average color
            for (int i = 0; i < samplesPerPixel; i++) {
                pixel = tracePath(cameraRay, scene, randState);
            }
            pixel /= samplesPerPixel;

            if (accumulatedFrames == 1) // Reset accumulated frames
                frameSum[pixelIndex] = ZERO_VEC;

            frameSum[pixelIndex] += pixel;
            pixel = frameSum[pixelIndex] / float(accumulatedFrames);

            // Color correction
            pixel = acesToneMapping(pixel); // Also clamps value to 0-1
            pixel = gammaCorrection(pixel);

            pixel *= 255; // Scale to fit 0-255
            surf2Dwrite(make_uchar4(round(pixel.r), round(pixel.g), round(pixel.b), 255), screenSurfaceObj, screenX * sizeof(uchar4), screenY); // Draw pixel
        }
    }
}

void render(scene scene, dim3 grid, dim3 block, dim3 cell, curandStateXORWOW* randStates, int accumulatedFrames, color* frameSum) {
    cudaGraphicsMapResources(1, &cudaImage);

    // Map texture array to cuda
    cudaArray_t screenCudaArray;
    cudaGraphicsSubResourceGetMappedArray(&screenCudaArray, cudaImage, 0, 0);

    // Data describing array
    cudaResourceDesc screenArrayDesc;
    screenArrayDesc.resType = cudaResourceTypeArray;
    screenArrayDesc.res.array.array = screenCudaArray;

    // Create read-/writeable object for screen array
    cudaSurfaceObject_t screenSurfaceObj;
    cudaError error = cudaCreateSurfaceObject(&screenSurfaceObj, &screenArrayDesc);
    if (error != cudaSuccess)
        std::cout << "Failed to map screen array to cuda" << std::endl;

    // Screen coordinate calculations
    const float screenZ = -(windowWidth / 2) / tanf(scene.camera.FOV / 2);
    matrix3d rotLeft = rotationMatrix3DY(scene.camera.angle[0]);
    matrix3d rotUp = rotationMatrix3DX(scene.camera.angle[1]);
    

    // Call GPU kernel calculating the color of every pixel
    launchRaytracer<<<grid, block>>>(screenSurfaceObj, cell, scene, screenZ, rotLeft, rotUp, randStates, accumulatedFrames, frameSum);


    // Clean up cuda objects
    cudaDestroySurfaceObject(screenSurfaceObj);
    cudaGraphicsUnmapResources(1, &cudaImage);

    error = cudaStreamSynchronize(0); // Synchronize cuda stream 0 (the only one in use)
    if (error != cudaSuccess)
        std::cout << "Failed to synchronize cuda stream" << std::endl;

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

// Give a cuRAND state for every pixel on the screen
__global__ void initializeRand(curandStateXORWOW* randStates, dim3 cell) {
    int pixelStartX = (blockIdx.x * blockDim.x + threadIdx.x) * cell.x;
    int pixelStartY = (blockIdx.y * blockDim.y + threadIdx.y) * cell.y;

    for (int y = 0; y < cell.y; y++) {
        for (int x = 0; x < cell.x; x++) {
            int screenX = pixelStartX + x;
            int screenY = pixelStartY + y;
            curand_init(screenY * windowWidth + screenX, 0, 0, &randStates[screenY * windowWidth + screenX]);
        }
    }
}

cudaError_t setGLScreenTexture() {
    glEnable(GL_TEXTURE_2D);
    glGenTextures(1, &screenTexture);

    // Texture settings (rendering canvas)
    glBindTexture(GL_TEXTURE_2D, screenTexture);

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, windowWidth, windowHeight, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);

    glBindTexture(GL_TEXTURE_2D, 0);

    // Register texture and pixel buffer to cuda
    cudaError_t error = cudaGraphicsGLRegisterImage(&cudaImage, screenTexture, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsSurfaceLoadStore);
    return error;
}

int main() {
    GLFWwindow* window;

    // Initialize GLFW
    if (!glfwInit()) {
        std::cout << "Failed to initialize GLFW!" << std::endl;
        std::cin.get();
        return 1;
    }

    // Window settings
    glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);

    // Create a windowed mode window and its OpenGL context
    if (fullscreen)
        window = glfwCreateWindow(windowWidth, windowHeight, "bwidman-raytracer", glfwGetPrimaryMonitor(), NULL);
    else
        window = glfwCreateWindow(windowWidth, windowHeight, "bwidman-raytracer", NULL, NULL);
    if (!window) {
        glfwTerminate();
        return 1;
    }

    // Make the window's context current
    glfwMakeContextCurrent(window);

    std::cout << "OpenGL version: " << glGetString(GL_VERSION) << std::endl;

    // Set up OpenGL screen texture
    cudaError_t error = setGLScreenTexture();
    if (error != cudaSuccess) {
        std::cout << "Failed to register screen texture to CUDA" << std::endl;
        return 1;
    }

    // Calculate number of thread blocks (grid dimensions)
    //tex:$$Blocks = \frac{pixels}{threads/block * pixels/thread}$$
    // Example 1280x720:
    // $$\frac{1280*720}{256*9} = 400$$
    // Grid width: $\sqrt{400} = 20$
    // (both 1280 and 720 are divisible by 20)
    constexpr int threadsPerBlock = 256; // A good number
    constexpr int pixelsPerThread = 9;

    constexpr int blockAmount = windowWidth * windowHeight / (threadsPerBlock * pixelsPerThread);
    std::cout << "Using " << blockAmount << " CUDA thread blocks\n" << std::endl;

    const int gridWidth = ceil(sqrtf(blockAmount));

    // We have a 2D grid of gridWidth * gridWidth blocks consisting of 256 threads each
    // Each block handles a 64 * 36 pixel area
    // Each thread then handles a 1 * 9 pixel area
    dim3 grid(gridWidth, gridWidth);
    dim3 block(64, 4); // 64 * 4 threads (256)
    dim3 cell(1, 9); // Dimensions of pixel cell handled by each thread

    // Allocate the scene on the GPU
    scene scene = allocateScene();

    curandStateXORWOW* randStates;
    cudaMalloc(&randStates, windowHeight * windowWidth * sizeof(curandStateXORWOW));
    initializeRand<<<grid, block>>>(randStates, cell);

    color* frameSum;
    cudaMalloc(&frameSum, windowHeight * windowWidth * sizeof(color));

    int accumulatedFrames = 1;
    double deltaTime = 0;
    int frameCount = 0;
    // Loop until the user closes the window
    while (!glfwWindowShouldClose(window)) {
        double startTime = glfwGetTime(); // Start timer

        // Render here
        glClear(GL_COLOR_BUFFER_BIT);

        render(scene, grid, block, cell, randStates, accumulatedFrames, frameSum);

        glfwSwapBuffers(window); // Swap front and back buffers
        accumulatedFrames++;

        controls(window, scene.camera, glfwGetTime() - startTime, accumulatedFrames);

        glfwPollEvents(); // Poll for and process events

        // Stop timer and print FPS if over a second has elapsed since last print
        deltaTime += glfwGetTime() - startTime;
        frameCount++;
        if (deltaTime > 1.0) {
            if (frameCount > 1)
                std::cout << "FPS: " << int(frameCount / deltaTime) << " | Samples: " << accumulatedFrames * samplesPerPixel << std::endl;
            else 
                std::cout << "Rendering time: " << int(deltaTime) << "s\a" << std::endl;
            deltaTime = 0; frameCount = 0;
        }
    }

    glfwTerminate();

    // Clean up scene
    cudaFree(scene.spheres);
    cudaFree(scene.planes);
    cudaFree(scene.triangles);
    cudaFree(randStates);
    cudaFree(frameSum);

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaError cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    return 0;
}