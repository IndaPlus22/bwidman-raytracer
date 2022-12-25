#pragma once
#include "WorldTypes.cuh"
#include <GLFW/glfw3.h>

void controls(GLFWwindow* window, camera& camera) {
	constexpr float moveSpeed = 0.1f;
	constexpr float rotSpeed = 0.02f;

	// Forward
	if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS) {
		camera.position += moveSpeed * camera.direction[0];
	}

	// Left
	if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS) {
		camera.position -= moveSpeed * camera.direction[1];
	}

	// Back
	if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS) {
		camera.position -= moveSpeed * camera.direction[0];
	}

	// Right
	if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS) {
		camera.position += moveSpeed * camera.direction[1];
	}

	// Up
	if (glfwGetKey(window, GLFW_KEY_SPACE) == GLFW_PRESS) {
		camera.position.y += moveSpeed;
	}

	// Down
	if (glfwGetKey(window, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS) {
		camera.position.y -= moveSpeed;
	}

	// Rotation around y axis
	
	// Rotate left
	if (glfwGetKey(window, GLFW_KEY_LEFT) == GLFW_PRESS) {
		matrix2d rotateLeft = rotationMatrix(rotSpeed);

		vec2d directionXZ = { camera.direction[0].x, camera.direction[0].z };
		directionXZ = rotateLeft * directionXZ;
		camera.direction[0] = { directionXZ.x, camera.direction[0].y, directionXZ.y };

		vec2d rightDirectionXZ = { camera.direction[1].x, camera.direction[1].z };
		rightDirectionXZ = rotateLeft * rightDirectionXZ;
		camera.direction[1] = { rightDirectionXZ.x, camera.direction[1].y, rightDirectionXZ.y };
	}

	// Rotate right
	if (glfwGetKey(window, GLFW_KEY_RIGHT) == GLFW_PRESS) {
		matrix2d rotateRight = rotationMatrix(-rotSpeed);

		vec2d directionXZ = { camera.direction[0].x, camera.direction[0].z };
		directionXZ = rotateRight * directionXZ;
		camera.direction[0] = { directionXZ.x, camera.direction[0].y, directionXZ.y };

		vec2d rightDirectionXZ = { camera.direction[1].x, camera.direction[1].z };
		rightDirectionXZ = rotateRight * rightDirectionXZ;
		camera.direction[1] = { rightDirectionXZ.x, camera.direction[1].y, rightDirectionXZ.y };
	}

	// Rotation around x axis (rotate )

	// Rotate up
	if (glfwGetKey(window, GLFW_KEY_UP) == GLFW_PRESS) {
		matrix2d rotateUp = rotationMatrix(rotSpeed);

		vec2d directionZY = { camera.direction[0].z, camera.direction[0].y };
		directionZY = rotateUp * directionZY;
		camera.direction[0] = { camera.direction[0].x, directionZY.y, directionZY.x };
	}

	// Rotate down
	if (glfwGetKey(window, GLFW_KEY_DOWN) == GLFW_PRESS) {
		matrix2d rotateDown = rotationMatrix(-rotSpeed);

		vec2d directionZY = { camera.direction[0].z, camera.direction[0].y };
		directionZY = rotateDown * directionZY;
		camera.direction[0] = { camera.direction[0].x, directionZY.y, directionZY.x };
	}
}