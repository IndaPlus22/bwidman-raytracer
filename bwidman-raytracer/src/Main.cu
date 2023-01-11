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
#include "Controls.cuh"

// Only works with 16:9 aspect ratios, such as:
// 1280x720, 1920x1080, 2560x1440
constexpr int windowWidth = 1920;
constexpr int windowHeight = 1080;
constexpr bool fullscreen = false;
constexpr int maxBounces = 3;
constexpr int samplesPerPixel = 1000;

unsigned int screenTexture;
cudaGraphicsResource_t cudaImage; // Must be global

#define CUDA_ARR_COPY(dArr, hArr)\
    cudaMalloc(&dArr, sizeof(hArr));\
    cudaMemcpy(dArr, &hArr, sizeof(hArr), cudaMemcpyHostToDevice);\

scene allocateScene() {
    camera camera = { { 0, 1, 0 }, { 0, 0 } , PI / 2 };

    sphere hSpheres[] = {
        // Position, radius, material
        { { -5, 6, -4 }, 1, { { 1, 0.6, 0.2 }, 15 } }, // Light left
        { { 5, 6, -4 }, 1, { { 1, 0.2, 0.6 }, 15 } }, // Light right

        { { 2, 2, -8 }, 2, { { 0.8, 0.2, 0 }, 0 } }, // Right
        { { -2, 1, -6 }, 1, { { 0.2, 0, 0.8 }, 0 } }, // Left
    };
    int sphereCount = sizeof(hSpheres) / sizeof(sphere);

    plane hPlanes[] = {
        // Origin,      directions,                     material
        { { 0, 0, 0 }, { { 0, 0, 1 }, { 1, 0, 0 } }, { { 0.5, 0.5, 0.5 }, 0 } },
    };
    int planeCount = sizeof(hPlanes) / sizeof(plane);

    triangle hTriangles[] = {
        // Vertices,                                    material
        { { { -2, 1, -8 }, { 2, 1, -8 }, { 0, 3, -8 } }, { { 0.8, 0.2, 0 }, 0 } },
    };
    int triangleCount = sizeof(hTriangles) / sizeof(triangle);

    // Allocate spheres on GPU
    sphere* dSpheres;
    CUDA_ARR_COPY(dSpheres, hSpheres);

    // Allocate planes on GPU
    plane* dPlanes;
    CUDA_ARR_COPY(dPlanes, hPlanes);

    // Allocate triangles on GPU
    triangle* dTriangles;
    CUDA_ARR_COPY(dTriangles, hTriangles);

    return { 
        camera, 
        dSpheres, sphereCount, 
        dPlanes, planeCount,
        dTriangles, triangleCount 
    };
}

__device__ vec3d reflect(vec3d direction, vec3d normal) {
    // direction and normal are both normalized so:
    //tex:$$proj_\vec{n}(\vec{d}) = (\vec{d} \cdot \vec{n})\vec{n}$$
    return direction - 2 * dotProduct(direction, normal) * normal;
}

// Check if ray intersects with plane
__device__ bool planeIntersect(ray ray, plane plane, vec3d* intersection, float* closestHit, vec3d* closestNormal, material* material) {
    //tex:$$P: ax + by + cz + d = 0$$
    // Input normal as (a,b,c) and plane origin as (x,y,z) into plane equation to solve for d.
    // $$d = -(ax + by + cz)$$
    // $$d = - \vec{n} \cdot \vec{x}$$
    vec3d normal = crossProduct(plane.directions[0], plane.directions[1]);

    // Check if ray is perpendicular to plane normal
    // In that case the ray is parallel to the plane and no intersection can occur
    if (dotProduct(normal, ray.direction) == 0) {
        return false;
    }

    float d = -dotProduct(normal, plane.origin);

    // To solve ray-plane intersection, input ray coordinates as (x,y,z)
    //tex:Where: $R = \vec{p} + t\vec{v}$
    // $$aR_x + bR_y + cR_z + d = 0$$
    // $$a(p_x + v_xt) + b(p_y + v_yt) + c(p_z + v_zt) + d = 0$$
    // $$ap_x + av_xt + bp_y + bv_yt + cp_z + cv_zt + d = 0$$
    // $$(av_x + bv_y + cv_z)t = - (ap_x + bp_y + cp_z + d)$$
    // $$t = - \frac{ap_x + bp_y + cp_z + d}{av_x + bv_y + cv_z}$$
    // $$t = - \frac{\vec{n} \cdot \vec{p} + d}{\vec{n} \cdot \vec{v}}$$
    float t = -(dotProduct(normal, ray.origin) + d) / dotProduct(normal, ray.direction);

    // Behind ray, inside ray origin or further away than the so far closest hit
    if (t <= 0.0001 || t > *closestHit) {
        return false;
    }

    *intersection = ray.origin + t * ray.direction;
    *closestHit = t;
    *closestNormal = normal;
    *material = plane.material;

    return true;
}

// Check if ray intersects with triangle
__device__ bool triangleIntersect(ray ray, triangle triangle, vec3d* intersection, float* closestHit, vec3d* normal, material* material) {
    vec3d directions[2] = { triangle.vertices[1] - triangle.vertices[0], triangle.vertices[2] - triangle.vertices[0] };
    plane trianglePlane = { triangle.vertices[0], { directions[0], directions[1] } };

    return false;

    bool intersectedPlane = planeIntersect(ray, trianglePlane, intersection, closestHit, normal, material);


    *material = triangle.material;
    return true;
}

// Check if ray intersects with sphere
__device__ bool sphereIntersect(ray ray, sphere sphere, vec3d* intersection, float* closestHit, vec3d* normal, material* material) {
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
    vec3d x = ray.origin;
    vec3d v = ray.direction;

    float a = dotProduct(v, v);
    float b = 2 * dotProduct(x - p, v);
    float c = dotProduct(x - p, x - p) - sphere.radius * sphere.radius;

    float discriminant = b * b - 4 * a * c; // Discriminator of all

    // Negative root => no solutions => no intersection
    if (discriminant < 0) {
        return false;
    }

    // Only interested in negative solution to the root as it gives the
    // smallest value of t and is therefore the closest to the ray origin
    float t = (-b - sqrtf(discriminant)) / (2 * a);

    // Behind ray, inside ray origin or further away than the so far closest hit
    if (t <= 0.0001 || t > *closestHit) {
        return false;
    }

    *closestHit = t;
    *intersection = ray.origin + t * ray.direction;
    *normal = normalize(*intersection - sphere.position);
    *material = sphere.material;

    return true;
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

__device__ color raytrace(ray incidentRay, const scene& scene, curandStateXORWOW* randState, int bounces = 0) {
    color outgoingLight = ZERO_VEC;
    if (bounces > maxBounces) // Stop recursion
        return outgoingLight;

    // Gets updated for every new closest hit
    float closestHit = INFINITY;

    // We want to loop the number of times that is the largest array out of all the objects
    float largestArraySize = max(scene.sphereCount, max(scene.planeCount, scene.triangleCount));
    
    vec3d intersection, normal;
    material material;
    bool intersected = false;
    // Check intersection with all objects and shade accordingly
    for (int i = 0; i < largestArraySize; i++) {
        // Check index to avoid "index out of range"
        if (i < scene.sphereCount)
            intersected += sphereIntersect(incidentRay, scene.spheres[i], &intersection, &closestHit, &normal, &material);

        if (i < scene.planeCount)
            intersected += planeIntersect(incidentRay, scene.planes[i], &intersection, &closestHit, &normal, &material);

        if (i < scene.triangleCount)
            intersected += triangleIntersect(incidentRay, scene.triangles[i], &intersection, &closestHit, &normal, &material);
    }

    // If intersection was found with any of the objects shade the closest point
    if (intersected) {
        // Reflect
        ray reflectionRay = { intersection, reflect(incidentRay.direction, normal) };

        color emittedLight = material.emmittance * material.albedo;
        color brdf = 2.0 * material.albedo;

        vec3d randomDirection = genRandomDirection(randState, normal);

        color incomingLight = raytrace({ intersection, randomDirection }, scene, randState, bounces + 1);

        float cosAngle = dotProduct(randomDirection, normal);

        //tex:$$L_o(\omega_o) = L_e(\omega_o) + \int_\Omega f(\omega_i, \omega_o) L_i(\omega_i) (\omega_i \cdot n) d\omega_i$$
        outgoingLight = emittedLight + brdf * incomingLight * cosAngle;
    }

    return outgoingLight;
}

__global__ void launch_raytracer(cudaSurfaceObject_t screenSurfaceObj, dim3 cell, const scene scene, float screenZ, matrix3d rotLeft, matrix3d rotUp, curandStateXORWOW* randStates) {
    int pixelStartX = (blockIdx.x * blockDim.x + threadIdx.x) * cell.x;
    int pixelStartY = (blockIdx.y * blockDim.y + threadIdx.y) * cell.y;

    // Loop through pixels in designated screen cell
    for (int y = 0; y < cell.y; y++) {
        for (int x = 0; x < cell.x; x++) {
            int screenX = pixelStartX + x;
            int screenY = pixelStartY + y;

            vec3d pixelPosition = { screenX - windowWidth / 2, screenY - windowHeight / 2, screenZ };
            pixelPosition = rotLeft * rotUp * pixelPosition; // Rotate to camera's facing direction

            ray cameraRay = { scene.camera.position, normalize(pixelPosition) };

            curandStateXORWOW* randState = &randStates[screenY * windowWidth + screenX];

            color pixel = ZERO_VEC;
            // Gather a number of samples and get the average color
            for (int i = 0; i < samplesPerPixel; i++) {
                pixel += raytrace(cameraRay, scene, randState);
            }
            pixel *= 1.0 / samplesPerPixel;

            // Scale to 0-255 and clamp to 255
            pixel.x = min(pixel.x * 255, 255.0);
            pixel.y = min(pixel.y * 255, 255.0);
            pixel.z = min(pixel.z * 255, 255.0);

            surf2Dwrite(make_uchar4(pixel.r, pixel.g, pixel.b, 255), screenSurfaceObj, screenX * sizeof(uchar4), screenY);
        }
    }
}

void render(scene scene, dim3 grid, dim3 block, dim3 cell, curandStateXORWOW* randStates) {
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
    launch_raytracer<<<grid, block>>>(screenSurfaceObj, cell, scene, screenZ, rotLeft, rotUp, randStates);


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

    double deltaTime = 0;
    int frameCount = 0;
    // Loop until the user closes the window
    while (!glfwWindowShouldClose(window)) {
        // Start timer
        double startTime = glfwGetTime();

        // Render here
        glClear(GL_COLOR_BUFFER_BIT);

        render(scene, grid, block, cell, randStates);

        // Swap front and back buffers
        glfwSwapBuffers(window);

        controls(window, scene.camera, glfwGetTime() - startTime);

        // Poll for and process events
        glfwPollEvents();

        // Stop timer and print FPS if over a second has elapsed since last print
        deltaTime += glfwGetTime() - startTime;
        frameCount++;
        if (deltaTime > 1.0) {
            std::cout << "FPS: " << int(frameCount / deltaTime) << std::endl;
            deltaTime = 0; frameCount = 0;
        }
    }

    glfwTerminate();

    // Clean up scene
    cudaFree(scene.spheres);
    cudaFree(scene.planes);
    cudaFree(scene.triangles);

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaError cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    return 0;
}