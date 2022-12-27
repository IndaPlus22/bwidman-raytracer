#pragma once
#include "host_defines.h"

#define PI 3.1415926535f
#define ZERO_VEC { 0, 0, 0 }

//
// Vectors
//

// 2D
struct vec2d {
	float x, y;

	__host__ __device__ vec2d() = default;
	__host__ __device__ vec2d(float x, float y)
		: x(x), y(y) {}
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


// 3D
struct vec3d {
	union { float x, r; };
	union { float y, g; };
	union { float z, b; };

	__host__ __device__ vec3d() = default;
	__host__ __device__ vec3d(float x, float y, float z)
		: x(x), y(y), z(z) {}
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

// 2D
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


// 3D
struct matrix3d {
	float cell[3][3];
};

__device__ __host__ vec3d row(const matrix3d& a, int row) {
	return { a.cell[row][0], a.cell[row][1], a.cell[row][2] };
}

__device__ __host__ vec3d col(const matrix3d& a, int col) {
	return { a.cell[0][col], a.cell[1][col], a.cell[2][col] };
}

__device__ __host__ vec3d operator * (const matrix3d& a, const vec3d& x) {
	return {
		dotProduct(row(a, 0), x),
		dotProduct(row(a, 1), x),
		dotProduct(row(a, 2), x)
	};
}

__device__ __host__ matrix3d operator * (const matrix3d& a, const matrix3d& b) {
	return {
		{
			{ dotProduct(row(a, 0), col(b, 0)), dotProduct(row(a, 0), col(b, 1)), dotProduct(row(a, 0), col(b, 2)) },
			{ dotProduct(row(a, 1), col(b, 0)), dotProduct(row(a, 1), col(b, 1)), dotProduct(row(a, 1), col(b, 2)) },
			{ dotProduct(row(a, 2), col(b, 0)), dotProduct(row(a, 2), col(b, 1)), dotProduct(row(a, 2), col(b, 2)) }
		}
	};
}

// Rotate around X axis
__device__ __host__ matrix3d rotationMatrix3DX(float angle) {
	const float cosAngle = cosf(angle);
	const float sinAngle = sinf(angle);
	return {
		{
			{ 1, 0,		   0 },
			{ 0, cosAngle, -sinAngle },
			{ 0, sinAngle, cosAngle }
		}
	};
}

// Rotate around Y axis
__device__ __host__ matrix3d rotationMatrix3DY(float angle) {
	const float cosAngle = cosf(angle);
	const float sinAngle = sinf(angle);
	return {
		{
			{ cosAngle,  0, sinAngle },
			{ 0,		 1, 0 },
			{ -sinAngle, 0, cosAngle }
		}
	};
}

// Rotate around Z axis
__device__ __host__ matrix3d rotationMatrix3DZ(float angle) {
	const float cosAngle = cosf(angle);
	const float sinAngle = sinf(angle);
	return {
		{
			{ cosAngle, -sinAngle, 0 },
			{ sinAngle, cosAngle,  0 },
			{ 0,		0,		   1 }
		}
	};
}