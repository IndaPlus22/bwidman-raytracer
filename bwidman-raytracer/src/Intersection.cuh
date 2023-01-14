#pragma once
#include "WorldTypes.cuh"

struct intersectionInfo {
    vec3d intersection;
    float distance;
    vec3d normal;
    material attributes;
};

// Check if ray intersects with a certain object

__device__ bool sphereIntersection(ray ray, sphere sphere, intersectionInfo* closestHit) {
    //tex:
    // Sphere equation:
    // $$(x-p_1)^2 + (y-p_2)^2 + (z-p_3)^2 = r^2$$
    // Ray equation: $$\vec{r} = \vec{x} + t\vec{v}$$
    // Input ray into sphere equation:
    // $$(x_1+tv_1-p_1)^2 + (x_2+tv_2-p_2)^2 + (x_3+tv_3-p_3)^2 = r^2$$
    // $$(x_1^2 + x_1v_1t - x_1p_1 + x_1v_1t + v_1^2t^2 - v_1p_1t - p_1x_1 - p_1v_1t + p_1^2) + ... - r^2 = 0$$
    // $$(v_1^2 + v_2^2 + v_3^2)t^2 + (2x_1v_1 - 2p_1v_1 + 2x_2v_2 - 2p_2v_2 + 2x_3v_3 - 2p_3v_3)t + (x_1^2 - 2p_1x_1 + p_1^2 + x_2^2 - 2p_2x_2 + p_2^2 + x_3^2 - 2p_3x_3 + p_3^2 - r^2) = 0$$
    // $$(v_1^2 + v_2^2 + v_3^2)t^2 + 2(x_1v_1 - p_1v_1 + x_2v_2 - p_2v_2 + x_3v_3 - p_3v_3)t + ((x_1 - p_1)^2 + (x_2 - p_2)^2 + (x_3 - p_3)^2 - r^2) = 0$$
    // $$(\vec{v} \cdot \vec{v})t^2 + 2((\vec{x} - \vec{p}) \cdot \vec{v})t + ((\vec{x} - \vec{p}) \cdot (\vec{x} - \vec{p}) - r^2) = 0$$
    // Solve for t with quadratic formula:
    // $$t = \frac{-b\pm\sqrt{b^2 - 4ac}}{2a}$$
    // Where:
    // $$a = \vec{v} \cdot \vec{v}$$
    // $$b = 2((\vec{x} - \vec{p}) \cdot \vec{v})$$
    // $$c = (\vec{x} - \vec{p}) \cdot (\vec{x} - \vec{p}) - r^2$$
    vec3d p = sphere.position;
    vec3d x = ray.origin;
    vec3d v = ray.direction;

    float a = dotProduct(v, v);
    float b = 2 * dotProduct(x - p, v);
    float c = dotProduct(x - p, x - p) - sphere.radius * sphere.radius;

    float discriminant = b * b - 4 * a * c; // Discriminator of all

    // Negative root => no solutions => no intersection
    if (discriminant < 0) {
        return false;
    }

    // Only interested in negative solution to the root as it gives the
    // smallest value of t and is therefore the closest to the ray origin
    float t = (-b - sqrtf(discriminant)) / (2 * a);

    // Behind ray, inside ray origin or further away than the so far closest hit
    if (t <= 0.0001 || t > closestHit->distance) {
        return false;
    }

    closestHit->distance = t;
    closestHit->intersection = ray.origin + t * ray.direction;
    closestHit->normal = normalize(closestHit->intersection - sphere.position);
    closestHit->attributes = sphere.attributes;

    return true;
}

__device__ bool planeIntersection(ray ray, plane plane, intersectionInfo* closestHit) {
    //tex:$$P: ax + by + cz + d = 0$$
    // Input normal as (a,b,c) and plane origin as (x,y,z) into plane equation to solve for d.
    // $$d = -(ax + by + cz)$$
    // $$d = - \vec{n} \cdot \vec{x}$$
    vec3d normal = crossProduct(plane.directions[0], plane.directions[1]);

    // Check if ray is perpendicular to plane normal
    // In that case the ray is parallel to the plane and no intersection can occur
    if (dotProduct(normal, ray.direction) == 0) {
        return false;
    }

    float d = -dotProduct(normal, plane.origin);

    // To solve ray-plane intersection, input ray coordinates as (x,y,z)
    //tex:Where: $R = \vec{p} + t\vec{v}$
    // $$aR_x + bR_y + cR_z + d = 0$$
    // $$a(p_x + v_xt) + b(p_y + v_yt) + c(p_z + v_zt) + d = 0$$
    // $$ap_x + av_xt + bp_y + bv_yt + cp_z + cv_zt + d = 0$$
    // $$(av_x + bv_y + cv_z)t = - (ap_x + bp_y + cp_z + d)$$
    // $$t = - \frac{ap_x + bp_y + cp_z + d}{av_x + bv_y + cv_z}$$
    // $$t = - \frac{\vec{n} \cdot \vec{p} + d}{\vec{n} \cdot \vec{v}}$$
    float t = -(dotProduct(normal, ray.origin) + d) / dotProduct(normal, ray.direction);

    // Behind ray, inside ray origin or further away than the so far closest hit
    if (t <= 0.0001 || t > closestHit->distance) {
        return false;
    }

    closestHit->intersection = ray.origin + t * ray.direction;
    closestHit->distance = t;
    closestHit->normal = normal;
    closestHit->attributes = plane.attributes;

    return true;
}

__device__ bool triangleIntersection(ray ray, triangle triangle, intersectionInfo* closestHit) {
    vec3d edges[3] = { // All edges pointing in a roundabout
        triangle.vertices[1] - triangle.vertices[0], 
        triangle.vertices[2] - triangle.vertices[1], 
        triangle.vertices[0] - triangle.vertices[2] 
    };
    plane trianglePlane = { triangle.vertices[0], { edges[0], edges[1] }, triangle.attributes };

    // Check if ray intersects with the plane spanned out by the triangle
    intersectionInfo planeInfo = {};
    planeInfo.distance = INFINITY;
    bool intersectedPlane = planeIntersection(ray, trianglePlane, &planeInfo);

    if (!intersectedPlane || planeInfo.distance > closestHit->distance) {
        return false;
    }

    // Calculate normals of edges pointing towards the center
    vec3d innerNormal1 = crossProduct(planeInfo.normal, edges[0]);
    vec3d innerNormal2 = crossProduct(planeInfo.normal, edges[1]);
    vec3d innerNormal3 = crossProduct(planeInfo.normal, edges[2]);

    // Check if ray intersection is outside edges
    if (dotProduct(innerNormal1, planeInfo.intersection - triangle.vertices[0]) < 0 ||
        dotProduct(innerNormal2, planeInfo.intersection - triangle.vertices[1]) < 0 ||
        dotProduct(innerNormal3, planeInfo.intersection - triangle.vertices[2]) < 0) {
        return false;
    }

    *closestHit = planeInfo;
    return true;
}