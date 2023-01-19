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
constexpr int windowWidth = 1920;
constexpr int windowHeight = 1080;
constexpr bool fullscreen = false;
constexpr int maxBounces = 5;
constexpr int samplesPerPixel = 1;
constexpr color backgroundColor = { 0, 0, 0 };
constexpr float specularChance = 0.5f; // Chance of ray reflecting with specular BRDF

unsigned int screenTexture;
cudaGraphicsResource_t cudaImage; // Must be global

#define CUDA_ARR_COPY(dArr, hArr)\
    cudaMalloc(&dArr, sizeof(hArr));\
    cudaMemcpy(dArr, &hArr, sizeof(hArr), cudaMemcpyHostToDevice);\

scene allocateScene() {
    camera camera = { { 0, 1, 0 }, { 0, 0 } , PI / 2 };

    // material = albedo, emittance, roughness, refractiveIndex
    sphere hSpheres[] = {
        // Position, radius, material
        { { -6, 3, -4 }, 1, { { 1, 0.6, 0.2 }, 20 } }, // Orange light left
        { { 6, 3, -4 }, 1, { { 1, 0.2, 0.6 }, 20 } }, // Purple light right
        { { -0.5, 0.2, -3 }, 0.2, { { 0.2, 0.8, 0.2 }, 5 } }, // Green light center

        { { 0, 0.75, -4 }, 0.75, { { 1, 1, 1 }, 0, 0.001f, 10 } }, // Center white
        { { -4, 1, -6 }, 1, { { 0.2, 0, 0.8 }, 0, 1 } }, // Left purple
        { { 4, 2, -8 }, 2, { { 1, 0.1, 0 }, 0, 1 } }, // Right red
    };
    int sphereCount = sizeof(hSpheres) / sizeof(sphere);

    plane hPlanes[] = {
        // Origin      directions                     material
        { { 0, 0, 0 }, { { 0, 0, 1 }, { 1, 0, 0 } }, { { 0.5, 0.5, 0.5 } } },
    };
    int planeCount = sizeof(hPlanes) / sizeof(plane);

    triangle hTriangles[] = {
        // Vertices                                            material
        // Pyramid
        { { { -2, 0, -3 }, { -1, 0, -3 }, { -1.5, 1, -3.5 } }, { { 0.95, 0.9, 0.2 } } }, // front
        { { { -1, 0, -4 }, { -2, 0, -4 }, { -1.5, 1, -3.5 } }, { { 0.95, 0.9, 0.2 } } }, // back
        { { { -2, 0, -4 }, { -2, 0, -3 }, { -1.5, 1, -3.5 } }, { { 0.95, 0.9, 0.2 } } }, // left
        { { { -1, 0, -3 }, { -1, 0, -4 }, { -1.5, 1, -3.5 } }, { { 0.95, 0.9, 0.2 } } }, // right
    };
    int triangleCount = sizeof(hTriangles) / sizeof(triangle);

    int wallWidth = 10;
    //quad hQuads[] = {
    //    // Vertices,                                                                                                                                      material
    //    // Walls
    //    //{ { { -wallWidth, 0, -wallWidth }, { -wallWidth, wallWidth, -wallWidth }, { -wallWidth, wallWidth, 0 },          { -wallWidth, 0, 0 } },          { { 0.95, 0.9, 0.2 } } }, // left
    //    //{ { { wallWidth, 0, -wallWidth },  { wallWidth, wallWidth, -wallWidth },  { -wallWidth, wallWidth, -wallWidth }, { -wallWidth, 0, -wallWidth } }, { { 0, 0, 1 } } }, // back
    //    //{ { { wallWidth, 0, 0 },           { wallWidth, wallWidth, 0 },           { wallWidth, wallWidth, -wallWidth },  { wallWidth, 0, -wallWidth } },  { { 0.95, 0.9, 0.2 } } }, // right
    //    //{ { { -wallWidth, wallWidth, 0 },  { -wallWidth, wallWidth, -wallWidth }, { wallWidth, wallWidth, -wallWidth },  { wallWidth, wallWidth, 0 } },   { { 0.95, 0.9, 0.2 } } }, // top

    //    // Mirror
    //    { { { wallWidth, 0, -wallWidth },  { wallWidth, wallWidth, -wallWidth },  { -wallWidth, wallWidth, -wallWidth }, { -wallWidth, 0, -wallWidth } },                { { 1, 0.8, 0.2 }, 0, 0.005, 10 } }, // front
    //    { { { -wallWidth, 0, -wallWidth - 1 }, { -wallWidth, wallWidth, -wallWidth - 1 }, { wallWidth, wallWidth, -wallWidth - 1 },  { wallWidth, 0, -wallWidth - 1 } }, { { 1, 0.8, 0.2 }, 0, 0.005, 10 } }, // back
    //    { { { -wallWidth, 0, -wallWidth }, { -wallWidth, wallWidth, -wallWidth }, { -wallWidth, wallWidth, -wallWidth-1 },  { -wallWidth, 0, -wallWidth-1 } },           { { 1, 0.8, 0.2 }, 0, 0.005, 10 } }, // left
    //    { { { wallWidth, 0, -wallWidth - 1 }, { wallWidth, wallWidth, -wallWidth - 1 },  { wallWidth, wallWidth, -wallWidth }, { wallWidth, 0, -wallWidth } },           { { 1, 0.8, 0.2 }, 0, 0.005, 10 } }, // right
    //    { { { wallWidth, wallWidth, -wallWidth }, { wallWidth, wallWidth, -wallWidth - 1 },  { -wallWidth, wallWidth, -wallWidth - 1 }, { -wallWidth, wallWidth, -wallWidth } },           { { 1, 0.8, 0.2 }, 0, 0.005, 10 } }, // top
    //};
    //int quadCount = sizeof(hQuads) / sizeof(quad);
    int quadCount = 0;

    // Allocate objects on GPU
    sphere* dSpheres;
    CUDA_ARR_COPY(dSpheres, hSpheres);

    plane* dPlanes;
    CUDA_ARR_COPY(dPlanes, hPlanes);

    triangle* dTriangles;
    CUDA_ARR_COPY(dTriangles, hTriangles);

    quad* dQuads;
    //CUDA_ARR_COPY(dQuads, hQuads);

    return { 
        camera, 
        dSpheres, sphereCount, 
        dPlanes, planeCount,
        dTriangles, triangleCount,
        dQuads, quadCount
    };
}

// Geometry term
__device__ float shadowingMasking(vec3d direction, vec3d normal, vec3d microNormal, float roughness) {
    //tex:$$G_1(v,m) = \chi^+(\frac{v \cdot m}{v \cdot n}) \frac{2}{1 + \sqrt{1 + \alpha_g^2 \tan^2\theta_v}}$$
    //$$\tan^2\theta_v = \frac{\sin^2\theta_v}{\cos^2\theta_v} = \frac{1 - \cos^2\theta_v}{\cos^2\theta_v} = \frac{1 - (v \cdot n)^2}{(v \cdot n)^2} = \frac{1}{(v \cdot n)^2} - 1$$
    float vDotN = dot(direction, normal);
    float tanTheta = max(1.0f / (vDotN * vDotN) - 1.0f, 0.0f);

    return chi(dot(direction, microNormal) / vDotN) *
        2.0f / (1.0f + sqrtf(1.0f + roughness * roughness * tanTheta * tanTheta));
}

__device__ float fresnel(vec3d incident, vec3d normal, float refractionIndex1, float refractionIndex2) {
    float c = abs(dot(incident, normal));
    //tex:$$g = \sqrt{\frac{\eta_t^2}{\eta_i^2} - 1 + c^2}$$
    float gRoot = square(refractionIndex2) / square(refractionIndex1) - 1.0f + c * c;

    if (gRoot < 0) // Total internal reflection
        return 1;
    float g = sqrtf(gRoot);

    //tex:$$F(i,m) = \frac{1}{2} \frac{(g-c)^2}{(g+c)^2} (1 + \frac{(c(g+c)-1)^2}{(c(g-c)+1)^2})$$
    return 0.5f * square(g - c) / square(g + c) * (1.0f + square(c * (g + c) - 1.0f) / square(c * (g - c) + 1.0f));
}

__device__ float specularWeight(vec3d incident, vec3d scatterDir, vec3d normal, vec3d microNormal, float roughness) {
    //tex:$$weight(o) = \frac{|i \cdot m| G(i,o,m)}{|i \cdot n||m \cdot n|}$$
    float g = shadowingMasking(incident, normal, microNormal, roughness) * shadowingMasking(scatterDir, normal, microNormal, roughness);

    if (isnan(g))
        return 1;

    float denominator = abs(dot(incident, normal) * dot(microNormal, normal));
    if (denominator == 0.0f)
        denominator = nearZero;

    return abs(dot(incident, microNormal)) * g / denominator;
}

__device__ vec3d baseAroundNormalToRegular(vec3d microNormal, vec3d normal) {
    vec3d someDirection = { 1, 0, 0 };
    // Switch random direction if they happen to be parallel
    if (abs(dot(normal, someDirection)) < 1 - nearZero)
        someDirection = { 0, 1, 0 };

    vec3d tangent1 = cross(normal, someDirection);
    vec3d tangent2 = cross(normal, tangent1);

    // Use { normal, tangent1, tangent2 } as base
    matrix3d baseChangeMatrix = {
        {
            { tangent1.x, tangent2.x, normal.x },
            { tangent1.y, tangent2.y, normal.y },
            { tangent1.z, tangent2.z, normal.z }
        }
    };
    // Express vector in regular base
    return baseChangeMatrix * microNormal;
}

__device__ vec3d genMicrofacetNormal(float roughness, curandStateXORWOW* randState) {
    float epsilon1 = randRange(randState, 1);
    float epsilon2 = randRange(randState, 1);

    // Spherical coordinates
    float theta = atan(roughness * sqrtf(epsilon1) / sqrtf(1 - epsilon1));
    float phi = 2 * PI * epsilon2;

    // Convert to cartesian coordinates
    float sinTheta = sin(theta);
    float x = sinTheta * cos(phi);
    float y = sinTheta * sin(phi);
    float z = cos(theta);

    return { x, y, z };
}

__device__ vec3d reflect(vec3d direction, vec3d normal) {
    // direction and normal are both normalized so:
    //tex:$$proj_\vec{n}(\vec{d}) = (\vec{d} \cdot \vec{n})\vec{n}$$
    return direction - 2 * dot(direction, normal) * normal;
}

__device__ vec3d genRandomDirection(curandStateXORWOW* randState, vec3d normal) {
    vec3d randomDirection = ZERO_VEC;
    do {
        randomDirection = { randRange(randState, 2) - 1, randRange(randState, 2) - 1, randRange(randState, 2) - 1 };
    } while (length(randomDirection) > 1);

    randomDirection = normalize(randomDirection);

    if (dot(normal, randomDirection) < 0) {
        // Reflect to other hemisphere by subtracting twice it's projection on the normal
        randomDirection -= 2 * dot(randomDirection, normal) * normal;
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
    float largestArraySize = max(scene.sphereCount, max(scene.planeCount, max(scene.triangleCount, scene.quadCount)));
    
    bool intersected = false;
    // Check intersection with all objects and grab info about the closest intersection
    for (int i = 0; i < largestArraySize; i++) {
        // Check index to avoid "index out of range"
        if (i < scene.sphereCount)
            intersected += sphereIntersection(incidentRay, scene.spheres[i], &closestHit);

        if (i < scene.planeCount)
            intersected += planeIntersection(incidentRay, scene.planes[i], &closestHit);

        if (i < scene.triangleCount)
            intersected += triangleIntersection(incidentRay, scene.triangles[i], &closestHit);

        if (i < scene.quadCount)
            intersected += quadIntersection(incidentRay, scene.quads[i], &closestHit);
    }

    // If intersection was found with any of the objects shade the closest point
    if (intersected) {
        color emittedLight = closestHit.mat.emittance * closestHit.mat.albedo;

        vec3d scatterDirection{};
        color brdf{};

        float brdfChoice = randRange(randState, 1);

        if (brdfChoice < specularChance) {  // Dieletric BRDF
            // Microfacet normal for importance sampling
            vec3d microNormal = genMicrofacetNormal(closestHit.mat.roughness, randState);
            microNormal = baseAroundNormalToRegular(microNormal, closestHit.normal);

            scatterDirection = reflect(incidentRay.direction, microNormal);

            float fresnelTerm = fresnel(-incidentRay.direction, microNormal, 1.0f, closestHit.mat.refractiveIndex);
            float specularTerm = specularWeight(-incidentRay.direction, scatterDirection, closestHit.normal, microNormal, closestHit.mat.roughness);

            brdf = specularTerm * fresnelTerm / specularChance * vec3d{ 1, 1, 1 };
        }
        else { // Diffuse BRDF
            scatterDirection = genRandomDirection(randState, closestHit.normal);
            brdf = 2.0 / (1 - specularChance) * closestHit.mat.albedo;
        }

        color incomingLight = tracePath({ closestHit.intersection, scatterDirection }, scene, randState, bounces + 1);

        float cosAngle = dot(scatterDirection, closestHit.normal);

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
    cudaFree(scene.quads);
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