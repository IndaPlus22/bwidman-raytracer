#pragma once
#include "Math.cuh"

struct ray {
	vec3d origin;
	vec3d direction;
};

struct camera {
	vec3d position;
	float angle[2];
	float FOV;
};

struct light {
	vec3d position;
	color color;
	float intensity;
};

struct sphere {
	vec3d position;
	float radius;
	color color;
};

struct plane {
	vec3d origin;
	vec3d directions[2];
	color color;
};

struct triangle {
	vec3d vertices[3];
	color color;
};

struct scene {
	camera camera;
	light* lights; // Pointer to device memory of lights
	int lightCount;
	sphere* spheres; // Pointer to device memory of spheres
	int sphereCount;
	plane* planes; // Pointer to device memory of planes
	int planeCount;
	triangle* triangles; // Pointer to device memory of triangles
	int triangleCount;
};