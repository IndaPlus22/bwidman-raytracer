#pragma once
#include "WorldTypes.cuh"
#include <GLFW/glfw3.h>

void controls(GLFWwindow* window, camera& camera) {
	constexpr float moveSpeed = 0.1f;
	constexpr float rotSpeed = 0.02f;

	vec3d directionFront = rotationMatrix3DY(camera.angle[0]) * rotationMatrix3DX(camera.angle[1]) * vec3d(0, 0, -1);
	vec3d directionRight = rotationMatrix3DY(camera.angle[0]) * rotationMatrix3DX(camera.angle[1]) * vec3d(1, 0, 0);

	// Forward
	if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS) {
		camera.position += moveSpeed * directionFront;
	}

	// Left
	if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS) {
		camera.position -= moveSpeed * directionRight;
	}

	// Back
	if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS) {
		camera.position -= moveSpeed * directionFront;
	}

	// Right
	if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS) {
		camera.position += moveSpeed * directionRight;
	}

	// Up
	if (glfwGetKey(window, GLFW_KEY_SPACE) == GLFW_PRESS) {
		camera.position.y += moveSpeed;
	}

	// Down
	if (glfwGetKey(window, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS) {
		camera.position.y -= moveSpeed;
	}
	
	// Rotate left
	if (glfwGetKey(window, GLFW_KEY_LEFT) == GLFW_PRESS) {
		camera.angle[0] += rotSpeed;
	}

	// Rotate right
	if (glfwGetKey(window, GLFW_KEY_RIGHT) == GLFW_PRESS) {
		camera.angle[0] -= rotSpeed;
	}

	// Rotate up
	if (glfwGetKey(window, GLFW_KEY_UP) == GLFW_PRESS) {
		camera.angle[1] += rotSpeed;
	}

	// Rotate down
	if (glfwGetKey(window, GLFW_KEY_DOWN) == GLFW_PRESS) {
		camera.angle[1] -= rotSpeed;
	}

	// Close window
	if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS) {
		glfwSetWindowShouldClose(window, GLFW_TRUE);
	}
}