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
	color albedo = ZERO_VEC;
	float emittance = 0;
	float roughness = 1;
	float refractiveIndex = 1.05;
};

struct sphere {
	vec3d position;
	float radius;
	material mat;
};

struct plane {
	vec3d origin;
	vec3d directions[2];
	material mat;
};

struct triangle {
	vec3d vertices[3];
	material mat;
};

struct quad {
	vec3d vertices[4];
	material mat;
};

struct scene {
	camera camera;
	sphere* spheres; // Pointer to device memory of spheres
	int sphereCount;
	plane* planes; // Pointer to device memory of planes
	int planeCount;
	triangle* triangles; // Pointer to device memory of triangles
	int triangleCount;
	quad* quads; // Pointer to device memory of quads
	int quadCount;
};