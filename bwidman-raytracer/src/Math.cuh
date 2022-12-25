#pragma once
#include "host_defines.h"

#define PI 3.1415926535f
#define ZERO_VEC { 0, 0, 0 }

//
// Vectors
//

struct vec2d {
	float x, y;
};

__device__ __host__ vec2d operator + (const vec2d& a, const vec2d& b) {
	return { a.x + b.x, a.y + b.y };
}

__device__ __host__ vec2d operator - (const vec2d& a, const vec2d& b) {
	return { a.x - b.x, a.y - b.y };
}

__device__ __host__ vec2d operator * (const vec2d& a, const vec2d& b) {
	return { a.x * b.x, a.y * b.y };
}

__device__ __host__ vec2d operator * (float k, const vec2d& v) {
	return { k * v.x, k * v.y };
}


struct vec3d {
	union { float x, r; };
	union { float y, g; };
	union { float z, b; };
};

typedef vec3d color;

__device__ __host__ vec3d operator + (const vec3d& a, const vec3d& b) {
	return { a.x + b.x, a.y + b.y, a.z + b.z };
}

__device__ __host__ void operator += (vec3d& a, const vec3d& b) {
	a = a + b;
}

__device__ __host__ vec3d operator - (const vec3d& a, const vec3d& b) {
	return { a.x - b.x, a.y - b.y, a.z - b.z };
}

__device__ __host__ void operator -= (vec3d& a, const vec3d& b) {
	a = a - b;
}

__device__ __host__ vec3d operator * (const vec3d& a, const vec3d& b) {
	return { a.x * b.x, a.y * b.y, a.z * b.z };
}

__device__ __host__ vec3d operator * (float k, const vec3d& v) {
	return { k * v.x, k * v.y, k * v.z };
}

__device__ __host__ float dotProduct(const vec3d& a, const vec3d& b) {
	return a.x * b.x + a.y * b.y + a.z * b.z;
}

__device__ __host__ vec3d crossProduct(const vec3d& a, const vec3d& b) {
	float i = a.y * b.z - a.z * b.y;
	float j = -(a.x * b.z - a.z * b.x);
	float k = a.x * b.y - a.y * b.x;
	return { i, j, k };
}

__device__ __host__ float length(const vec3d& v) {
	return sqrtf(v.x * v.x + v.y * v.y + v.z * v.z);
}

// Faster than length() because it does not calculate the root
__device__ __host__ float squaredLength(const vec3d& v) {
	return v.x * v.x + v.y * v.y + v.z * v.z;
}

__device__ __host__ vec3d normalize(const vec3d& v) {
	return (1.0f / length(v)) * v;
}

//
// Matrices
//

struct matrix2d {
	float cell[2][2];
};

__device__ __host__ vec2d operator * (const matrix2d& a, const vec2d& x) {
	return {
		a.cell[0][0] * x.x + a.cell[0][1] * x.y,
		a.cell[1][0] * x.x + a.cell[1][1] * x.y
	};
}

__device__ __host__ matrix2d rotationMatrix(float angle) {
	const float cosAngle = cosf(angle);
	const float sinAngle = sinf(angle);
	return {
		{
			{ cosAngle, -sinAngle },
			{ sinAngle, cosAngle }
		}
	};
}


struct matrix3d {
	float cell[3][3];
};

__device__ __host__ vec3d operator * (const matrix3d& a, const vec3d& x) {
	return {
		a.cell[0][0] * x.x + a.cell[0][1] * x.y + a.cell[0][2] * x.z,
		a.cell[1][0] * x.x + a.cell[1][1] * x.y + a.cell[1][2] * x.z,
		a.cell[2][0] * x.x + a.cell[2][1] * x.y + a.cell[2][2] * x.z,
	};
}