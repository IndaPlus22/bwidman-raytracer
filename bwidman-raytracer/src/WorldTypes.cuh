#pragma once
#include "Math.cuh"

struct camera {
	vec3d position;
	vec3d direction[2]; // Forward and left direction
	float angle[2];
	float FOV;
};

struct ray {
	vec3d origin;
	vec3d direction;
};

struct sphere {
	vec3d position;
	float radius;
	color color;
};

struct scene {
	camera camera;
	sphere* spheres; // Pointer to device memory of spheres
	int sphereCount;
};