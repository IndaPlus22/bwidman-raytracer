#pragma once
#include "host_defines.h"

struct vec3d {
	float x, y, z;
};

__device__ __host__ vec3d operator + (vec3d a, vec3d b) {
	return { a.x + b.x, a.y + b.y, a.z + b.z };
}

__device__ __host__ vec3d operator - (vec3d a, vec3d b) {
	return { a.x - b.x, a.y - b.y, a.z - b.z };
}

__device__ __host__ vec3d operator * (vec3d a, vec3d b) {
	return { a.x * b.x, a.y * b.y, a.z * b.z };
}

__device__ __host__ vec3d operator * (float k, vec3d v) {
	return { k * v.x, k * v.y, k * v.z };
}

__device__ __host__ float dotProduct(vec3d a, vec3d b) {
	return a.x * b.x + a.y * b.y + a.z * b.z;
}

__device__ __host__ vec3d crossProduct(vec3d a, vec3d b) {
	float i = a.y * b.z - a.z * b.y;
	float j = -(a.x * b.z - a.z * b.x);
	float k = a.x * b.y - a.y * b.x;
	return { i, j, k };
}

__device__ __host__ float length(vec3d v) {
	return sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
}

__device__ __host__ float squaredLength(vec3d v) {
	return v.x * v.x + v.y * v.y + v.z * v.z;
}

__device__ __host__ vec3d normalized(vec3d v) {
	return (1.0f / length(v)) * v;
}