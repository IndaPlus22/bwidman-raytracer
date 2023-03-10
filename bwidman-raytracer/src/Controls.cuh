#pragma once
#include "WorldTypes.cuh"
#include <GLFW/glfw3.h>

void controls(GLFWwindow* window, camera& camera, float deltaTime, int& accumulatedFrames) {
	float moveSpeed = 5 * deltaTime;
	float rotSpeed = 2 * deltaTime;

	vec3d directionFront = rotationMatrix3DY(camera.angle[0]) * rotationMatrix3DX(camera.angle[1]) * vec3d{ 0, 0, -1 };
	vec3d directionRight = rotationMatrix3DY(camera.angle[0]) * rotationMatrix3DX(camera.angle[1]) * vec3d{ 1, 0, 0 };
	
	// Forward
	if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS) {
		camera.position += moveSpeed * directionFront;
		accumulatedFrames = 1;
	}

	// Left
	if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS) {
		camera.position -= moveSpeed * directionRight;
		accumulatedFrames = 1;
	}

	// Back
	if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS) {
		camera.position -= moveSpeed * directionFront;
		accumulatedFrames = 1;
	}

	// Right
	if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS) {
		camera.position += moveSpeed * directionRight;
		accumulatedFrames = 1;
	}

	// Up
	if (glfwGetKey(window, GLFW_KEY_SPACE) == GLFW_PRESS) {
		camera.position.y += moveSpeed;
		accumulatedFrames = 1;
	}

	// Down
	if (glfwGetKey(window, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS) {
		camera.position.y -= moveSpeed;
		accumulatedFrames = 1;
	}
	
	// Rotate left
	if (glfwGetKey(window, GLFW_KEY_LEFT) == GLFW_PRESS) {
		camera.angle[0] += rotSpeed;
		accumulatedFrames = 1;
	}

	// Rotate right
	if (glfwGetKey(window, GLFW_KEY_RIGHT) == GLFW_PRESS) {
		camera.angle[0] -= rotSpeed;
		accumulatedFrames = 1;
	}

	// Rotate up
	if (glfwGetKey(window, GLFW_KEY_UP) == GLFW_PRESS) {
		camera.angle[1] += rotSpeed;
		accumulatedFrames = 1;
	}

	// Rotate down
	if (glfwGetKey(window, GLFW_KEY_DOWN) == GLFW_PRESS) {
		camera.angle[1] -= rotSpeed;
		accumulatedFrames = 1;
	}

	// Close window
	if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS) {
		glfwSetWindowShouldClose(window, GLFW_TRUE);
	}
}