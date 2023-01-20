# bwidman-raytracer
This is a simple ""real time"" path tracing engine accelerated with the GPU via the CUDA API. There are camera controls and it will look ugly while it moves but it quickly accumulates frames after stopping.

NOTE: An Nvidia GPU is required to run this!

To compile:
1. Download Visual Studio 2022 with C++ desktop development
2. Download [CUDA Toolkit 12.0](https://developer.nvidia.com/cuda-downloads)
3. Open up `bwidman-raytracer.sln` in Visual Studio and run!

## Example renders (newest to oldest)

### Specular BRDF & importance sampling
![Specular BRDF](/Renders/07_specular_BRDF.png)

### Triangle intersection & tone mapping
![Edgy pyramid](/Renders/06_edgy_pyramid.png)

### Gamma correction
![Gamma correction](/Renders/05_gamma_correction.png)

### Fundamental path tracing
![Path tracing](/Renders/04_path_tracing.png)

### Specular reflections
![Reflections](/Renders/03_reflections.png)

### Lambert's cosine law
![Simple shading](/Renders/02_simple_shading.png)

### First render
![Red circle](/Renders/01_red_circle.png)
