#pragma once
#include "Math.cuh"

struct camera {
	vec3d position;
	float angle;
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

// Pointers to device memory of scene
struct scene {
	camera* camera;
	sphere* spheres;
	int* sphereCount;
};