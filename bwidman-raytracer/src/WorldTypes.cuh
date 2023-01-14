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

struct material {
	color albedo;
	float emittance;
	float reflectivity;
};

struct light {
	vec3d position;
	color color;
	float intensity;
};

struct sphere {
	vec3d position;
	float radius;
	material attributes;
};

struct plane {
	vec3d origin;
	vec3d directions[2];
	material attributes;
};

struct triangle {
	vec3d vertices[3];
	material attributes;
};

struct scene {
	camera camera;
	sphere* spheres; // Pointer to device memory of spheres
	int sphereCount;
	plane* planes; // Pointer to device memory of planes
	int planeCount;
	triangle* triangles; // Pointer to device memory of triangles
	int triangleCount;
};