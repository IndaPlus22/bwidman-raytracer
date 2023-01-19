#pragma once
#include "host_defines.h"
#include <math.h>

#define PI 3.1415926535f
#define ZERO_VEC { 0, 0, 0 }

//
// Vectors
//

// 2D
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


// 3D
struct vec3d {
	union { float x, r; };
	union { float y, g; };
	union { float z, b; };
};

typedef vec3d color;

__device__ __host__ vec3d operator + (const vec3d& a, const vec3d& b) {
	return { a.x + b.x, a.y + b.y, a.z + b.z };
}

__device__ __host__ vec3d operator + (const vec3d& v, const float k) { // Not defined in math but useful
	return { v.x + k, v.y + k, v.z + k };
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

__device__ __host__ vec3d operator * (const float k, const vec3d& v) {
	return { k * v.x, k * v.y, k * v.z };
}

__device__ __host__ vec3d operator * (const vec3d& v, const float k) {
	return { k * v.x, k * v.y, k * v.z };
}

__device__ __host__ vec3d operator - (const vec3d& v) {
	return -1 * v;
}

__device__ __host__ void operator *= (vec3d& a, const vec3d& b) {
	a = a * b;
}

__device__ __host__ void operator *= (vec3d& v, const float k) {
	v = k * v;
}

__device__ __host__ vec3d operator / (const vec3d& a, const vec3d& b) { // Not defined in math but useful
	return { a.x / b.x, a.y / b.y, a.z / b.z };
}

__device__ __host__ vec3d operator / (const vec3d& v, const float k) {
	return (1.0f / k) * v;
}

__device__ __host__ void operator /= (vec3d& v, const float k) {
	v = v / k;
}

__device__ __host__ float dot(const vec3d& a, const vec3d& b) {
	return a.x * b.x + a.y * b.y + a.z * b.z;
}

__device__ __host__ vec3d cross(const vec3d& a, const vec3d& b) {
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
	float elements[2][2];

	__device__ __host__ float* operator[] (const int i) {
		return elements[i];
	}

	// For const objects
	__device__ __host__ float const* operator[] (const int i) const {
		return elements[i];
	}
};

__device__ __host__ vec2d operator * (const matrix2d& a, const vec2d& x) {
	return {
		a[0][0] * x.x + a[0][1] * x.y,
		a[1][0] * x.x + a[1][1] * x.y
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
	float elements[3][3];

	__device__ __host__ float* operator[] (const int i) {
		return elements[i];
	}

	// For const objects
	__device__ __host__ float const* operator[] (const int i) const {
		return elements[i];
	}

	// Access specific row or column as a vector
	__device__ __host__ vec3d row(const int row) const {
		return { elements[row][0], elements[row][1], elements[row][2] };
	}

	__device__ __host__ vec3d col(const int col) const {
		return { elements[0][col], elements[1][col], elements[2][col] };
	}
};

__device__ __host__ vec3d operator * (const matrix3d& a, const vec3d& x) {
	return {
		dot(a.row(0), x),
		dot(a.row(1), x),
		dot(a.row(2), x)
	};
}

__device__ __host__ matrix3d operator * (const matrix3d& a, const matrix3d& b) {
	return {
		{
			{ dot(a.row(0), b.col(0)), dot(a.row(0), b.col(1)), dot(a.row(0), b.col(2)) },
			{ dot(a.row(1), b.col(0)), dot(a.row(1), b.col(1)), dot(a.row(1), b.col(2)) },
			{ dot(a.row(2), b.col(0)), dot(a.row(2), b.col(1)), dot(a.row(2), b.col(2)) }
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


//
// Color
//

__device__ color clamp(color color, float k) {
	return { min(color.x, k), min(color.y, k), min(color.z, k) };
}

__device__ color gammaCorrection(color color) {
	return { sqrtf(color.x), sqrtf(color.y), sqrtf(color.z) };
}

__device__ color acesToneMapping(color color) {
	// Magical numbers
	color *= 0.6;
	constexpr float a = 2.51f;
	constexpr float b = 0.03f;
	constexpr float c = 2.43f;
	constexpr float d = 0.59f;
	constexpr float e = 0.14f;
	return clamp(color * (a * color + b) / (color * (c * color + d) + e), 1.0f);
}


//
// Miscellaneous
//

__device__ float sign(float x) {
	return (x < 0) ? -1 : 1;
}

__device__ float chi(float x) {
	return (x > 0) ? 1 : 0;
}

__device__ float randRange(curandStateXORWOW* randState, float max) {
	return float(curand(randState)) / INT_MAX * 0.5f * max;
}

__device__ __host__ constexpr float square(float x) {
	return x * x;
}

__device__ __host__ float clamp(float x, float lower, float upper) {
	return min(upper, max(x, lower));
}