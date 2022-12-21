#pragma once
#include "Math.hpp"

struct camera {
	vec3d position;
	float angle;
};

struct ray {
	vec3d origin;
	vec3d direction;
};

struct sphere {
	vec3d position;
	float radius;
	vec3d color;
};