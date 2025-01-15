//
//  main.cpp
//  3D Object Drawing
//
//  Created by Nazirul Hasan on 4/9/23.
//  modified by Badiuzzaman on 3/11/24.
//

#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include "shader.h"
#include "basic_camera.h"
#include "camera.h"
#include <iostream>
#include <vector>
#include <cmath>
#include "pointLight.h"


#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif


#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

glm::mat4 perspectiveProjection(float fov, float aspect, float near, float far) {
    glm::mat4 matrix(0.0f); // Initialize all elements to 0

    float tanHalfFov = tanf(glm::radians(fov) / 2.0f);

    matrix[0][0] = 1.0f / (aspect * tanHalfFov);  // (1 / (aspect * tan(fov / 2)))
    matrix[1][1] = 1.0f / tanHalfFov;            // (1 / tan(fov / 2))
    matrix[2][2] = -(far + near) / (far - near);  // (-(far + near) / (far - near))
    matrix[2][3] = -1.0f;                         // -1
    matrix[3][2] = -(2.0f * far * near) / (far - near); 

    return matrix;
}

void orthogonalProjection(float left, float right, float bottom, float top, float near, float far, float matrix[16]) {
    // Compute the orthogonal projection matrix
    float r_l = right - left;
    float t_b = top - bottom;
    float n_f = far - near;

    // Set all elements of the matrix to 0
    for (int i = 0; i < 16; ++i)
        matrix[i] = 0.0f;

    // The orthogonal projection matrix (M_orth) structure:
    matrix[0] = 2.0f / r_l;  // (2 / (right - left))
    matrix[5] = 2.0f / t_b;  // (2 / (top - bottom))
    matrix[10] = 2.0f / n_f; // (2 / (near - far))
    matrix[12] = (right + left) / r_l;   // (right + left) / (right - left)
    matrix[13] = (top + bottom) / t_b;   // (top + bottom) / (top - bottom)
    matrix[14] = (far + near) / n_f;     // (far + near) / (near - far)
    matrix[15] = 1.0f;

    // Set other elements to 0 (already default, but to be explicit)
    matrix[1] = matrix[2] = matrix[3] = 0.0f;
    matrix[4] = matrix[6] = matrix[7] = 0.0f;
    matrix[8] = matrix[9] = matrix[11] = 0.0f;
}

glm::mat4 myRotate(const glm::mat4& baseMatrix, float angle, const glm::vec3& axis) {
    glm::vec3 normalizedAxis = glm::normalize(axis);

    float x = normalizedAxis.x;
    float y = normalizedAxis.y;
    float z = normalizedAxis.z;

    float cosTheta = cos(angle);
    float sinTheta = sin(angle);
    float oneMinusCosTheta = 1.0f - cosTheta;

    glm::mat4 rotationMatrix(1.0f);

    rotationMatrix[0][0] = cosTheta + x * x * oneMinusCosTheta;
    rotationMatrix[0][1] = x * y * oneMinusCosTheta - z * sinTheta;
    rotationMatrix[0][2] = x * z * oneMinusCosTheta + y * sinTheta;

    rotationMatrix[1][0] = y * x * oneMinusCosTheta + z * sinTheta;
    rotationMatrix[1][1] = cosTheta + y * y * oneMinusCosTheta;
    rotationMatrix[1][2] = y * z * oneMinusCosTheta - x * sinTheta;

    rotationMatrix[2][0] = z * x * oneMinusCosTheta - y * sinTheta;
    rotationMatrix[2][1] = z * y * oneMinusCosTheta + x * sinTheta;
    rotationMatrix[2][2] = cosTheta + z * z * oneMinusCosTheta;


    return baseMatrix * rotationMatrix;
}

using namespace std;


bool f = false;
void framebuffer_size_callback(GLFWwindow* window, int width, int height);
void mouse_callback(GLFWwindow* window, double xpos, double ypos);
void scroll_callback(GLFWwindow* window, double xoffset, double yoffset);
void processInput(GLFWwindow* window);
//void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods);
// draw object functions
void drawCylinder(unsigned int& VAO_C, Shader& lightingShader, glm::vec3 color, glm::mat4 model,std::vector<unsigned int>& indices);
void drawCube(Shader shaderProgram, unsigned int VAO, glm::mat4 parentTrans, float posX = 0.0, float posY = 0.0, float posz = 0.0, float scX = 1.0, float scY = 1.0, float scZ = 1.0, float R = 0.0, float G = 0.0, float B = 0.0);
void drawFan(Shader ourShader, glm::mat4 identityMatrix, unsigned int VAO, unsigned int VAO2, glm::mat4 Model_Center);

void drawTable(Shader ourShader, unsigned int VAO, glm::mat4 identityMatrix, float posX = 0.0, float posY = 0.0, float posz = 0.0, float scX = 1.0, float scY = 1.0, float scZ = 1.0, float rtX = 0.0f, float rtY = 0.0f, float rtZ = 0.0f);
void drawTVCAB(Shader ourShader, unsigned int VAO, glm::mat4 identityMatrix, float posX = 0.0, float posY = 0.0, float posz = 0.0, float scX = 1.0, float scY = 1.0, float scZ = 1.0, float rtX = 0.0f, float rtY = 0.0f, float rtZ = 0.0f);
void drawSofa(Shader ourShader, unsigned int VAO, glm::mat4 identityMatrix, float posX = 0.0, float posY = 0.0, float posz = 0.0, float scX = 1.0, float scY = 1.0, float scZ = 1.0, float rtX = 0.0f, float rtY = 0.0f, float rtZ = 0.0f);

void drawAlmirah(Shader ourShader, unsigned int VAO, glm::mat4 identityMatrix, float posX = 0.0, float posY = 0.0, float posz = 0.0, float scX = 1.0, float scY = 1.0, float scZ = 1.0, float rtX = 0.0f, float rtY = 0.0f, float rtZ = 0.0f);
void generateCylinder(float radius, float height, int segments, std::vector<float>& vertices, std::vector<unsigned int>& indices);

void generateCone(float radius, float height, int segments, std::vector<float>& vertices, std::vector<unsigned int>& indices);
void generateSphere(float radius, int sectorCount, int stackCount, std::vector<float>& vertices, std::vector<unsigned int>& indices);


//void drawWall(Shader ourShader, unsigned int VAO, glm::mat4 identityMatrix, float posX = 0.0, float posY = 0.0, float posz = 0.0, float scX = 1.0, float scY = 1.0, float scZ = 1.0, float rtX = 0.0f, float rtY = 0.0f, float rtZ = 0.0f);
// settings
const unsigned int SCR_WIDTH = 800;
const unsigned int SCR_HEIGHT = 600;

// modelling transform
float rotateAngle_X = 45.0;
float rotateAngle_Y = 45.0;
float rotateAngle_Z = 45.0;
float rotateAxis_X = 0.0;
float rotateAxis_Y = 0.0;
float rotateAxis_Z = 1.0;
float translate_X = 0.0;
float translate_Y = 0.0;
float translate_Z = 0.0;
float scale_X = 1.0;
float scale_Y = 1.0;
float scale_Z = 1.0;

// camera
float lastX = SCR_WIDTH / 2.0f;
float lastY = SCR_HEIGHT / 2.0f;
bool firstMouse = true;

float eyeX = 0.0, eyeY = -5.0, eyeZ = 3.0;
float lookAtX = 0.0, lookAtY = 0.0, lookAtZ = 0.0;
glm::vec3 V = glm::vec3(0.0f, 1.0f, 0.0f);
BasicCamera basic_camera(eyeX, eyeY, eyeZ, lookAtX, lookAtY, lookAtZ, V);

//Camera camera(glm::vec3(0.5, .5, .50));
// timing
float deltaTime = 0.0f;    // time between current frame and last frame
float lastFrame = 0.0f;


// lighting
// positions of the point lights
glm::vec3 pointLightPositions[] = {
    glm::vec3(1.30f,  1.5f,  1.0f),
    glm::vec3(-1.40f,  1.5f,  1.0f)

};
PointLight pointlight1(

    pointLightPositions[0].x, pointLightPositions[0].y, pointLightPositions[0].z,  // position
    0.05f, 0.05f, 0.05f,     // ambient
    0.8f, 0.8f, 0.8f,     // diffuse
    1.0f, 1.0f, 1.0f,        // specular
    1.0f,   //k_c
    0.09f,  //k_l
    0.032f, //k_q
    1       // light number
);
PointLight pointlight2(

    pointLightPositions[1].x, pointLightPositions[1].y, pointLightPositions[1].z,  // position
    0.05f, 0.05f, 0.05f,     // ambient
    0.8f, 0.8f, 0.8f,     // diffuse
    1.0f, 1.0f, 1.0f,        // specular
    1.0f,   //k_c
    0.09f,  //k_l
    0.032f, //k_q
    2       // light number
);

//directional light
bool directionLightOn = true;
bool directionalAmbient = true;
bool directionalDiffuse = true;
bool directionalSpecular = true;

//spot light
bool spotLightOn = true;

// light settings
bool pointLightOn = true;
bool ambientToggle = true;
bool diffuseToggle = true;
bool specularToggle = true;

//point light
bool point1 = true;
bool point2 = true;



int main()
{
    // glfw: initialize and configure
    // ------------------------------
    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

#ifdef _APPLE_
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
#endif

    // glfw window creation
    // --------------------
    GLFWwindow* window = glfwCreateWindow(SCR_WIDTH, SCR_HEIGHT, "CSE 4208: Computer Graphics Laboratory", NULL, NULL);
    if (window == NULL)
    {
        std::cout << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return -1;
    }
    glfwMakeContextCurrent(window);
    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);

    //glfwSetKeyCallback(window, key_callback);
    //glfwSetCursorPosCallback(window, mouse_callback);
    glfwSetScrollCallback(window, scroll_callback);

    // tell GLFW to capture our mouse
    // glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);

    // glad: load all OpenGL function pointers
    // ---------------------------------------
    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
    {
        std::cout << "Failed to initialize GLAD" << std::endl;
        return -1;
    }

    // configure global opengl state
    // -----------------------------
    glEnable(GL_DEPTH_TEST);

    // build and compile our shader zprogram
    // ------------------------------------
    Shader ourShader("vertexShader.vs", "fragmentShader.fs");
    Shader lightingShader("vertexShaderForGouraudShading.vs", "fragmentShaderForGouraudShading.fs");

    //Shader ourShader("vertexShader.vs", "fragmentShader.fs");
   // Shader constantShader("vertexShader.vs", "fragmentShaderV2.fs");

    // set up vertex data (and buffer(s)) and configure vertex attributes
    // ------------------------------------------------------------------
    float cube_vertices[] = {
        // positions          // normals
        // Front face
        0.0f, 0.0f, 0.0f,    0.0f, 0.0f, -1.0f,
        0.5f, 0.0f, 0.0f,    0.0f, 0.0f, -1.0f,
        0.5f, 0.5f, 0.0f,    0.0f, 0.0f, -1.0f,
        0.0f, 0.5f, 0.0f,    0.0f, 0.0f, -1.0f,

        // Right face
        0.5f, 0.0f, 0.0f,    1.0f, 0.0f, 0.0f,
        0.5f, 0.5f, 0.0f,    1.0f, 0.0f, 0.0f,
        0.5f, 0.0f, 0.5f,    1.0f, 0.0f, 0.0f,
        0.5f, 0.5f, 0.5f,    1.0f, 0.0f, 0.0f,

        // Back face
        0.0f, 0.0f, 0.5f,    0.0f, 0.0f, 1.0f,
        0.5f, 0.0f, 0.5f,    0.0f, 0.0f, 1.0f,
        0.5f, 0.5f, 0.5f,    0.0f, 0.0f, 1.0f,
        0.0f, 0.5f, 0.5f,    0.0f, 0.0f, 1.0f,

        // Left face
        0.0f, 0.0f, 0.5f,    -1.0f, 0.0f, 0.0f,
        0.0f, 0.5f, 0.5f,    -1.0f, 0.0f, 0.0f,
        0.0f, 0.5f, 0.0f,    -1.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 0.0f,    -1.0f, 0.0f, 0.0f,

        // Top face
        0.5f, 0.5f, 0.5f,    0.0f, 1.0f, 0.0f,
        0.5f, 0.5f, 0.0f,    0.0f, 1.0f, 0.0f,
        0.0f, 0.5f, 0.0f,    0.0f, 1.0f, 0.0f,
        0.0f, 0.5f, 0.5f,    0.0f, 1.0f, 0.0f,

        // Bottom face
        0.0f, 0.0f, 0.0f,    0.0f, -1.0f, 0.0f,
        0.5f, 0.0f, 0.0f,    0.0f, -1.0f, 0.0f,
        0.5f, 0.0f, 0.5f,    0.0f, -1.0f, 0.0f,
        0.0f, 0.0f, 0.5f,    0.0f,-1.0f,0.0f
    };

    float cylinder_vertix[] = {
        1,0.5,0,
        1,0,0,
        0.913,0.5,0.406,
        0.913,0,0.406,
        0.669,0.5,0.743,
        0.669,0,0.743,
        0.309,0.5,0.951,
        0.309,0,0.951,
        -0.104,0.5,0.994,
        -0.104,0,0.994,
        -0.5,0.5,0.866,
        -0.5,0,0.866,
        -0.809,0.5,0.588,
        -0.809,0,0.588,
        -0.978,0.5,0.208,
        -0.978,0,0.208,
        -0.978,0.5,-0.208,
        -0.978,0,-0.208,
        -0.809,0.5,-0.588,
        -0.809,0,-0.588,
        -0.5,0.5,-0.866,
        -0.5,0,-0.866,
        -0.104,0.5,-0.994,
        -0.104,0,-0.994,
        0.309,0.5,-0.951,
        0.309,0,-0.951,
        0.669,0.5,-0.743,
        0.669,0,-0.743,
        0.913,0.5,-0.406,
        0.913,0,-0.406,
    };

    unsigned int cylinder_indices[] = {
        // Top cap
        1,3,5,7,9,11,13,15,17,19,21,23,25,27,29,

        // Bottom cap
        0,28,26,24,22,20,18,16,14,12,10,8,6,4,2,

        //side bar
        //1,2,0,1,3,2,3,4,2,3,5,4,5,6,4,5,7,6,7,8,6,
         0, 2, 1,   2, 3, 1,
         2, 4, 3,   4, 5, 3,
         4, 6, 5,   6, 7, 5,
         6, 8, 7,   8, 9, 7,
         8, 10, 9,  10, 11, 9,
         10, 12, 11, 12, 13, 11,
         12, 14, 13, 14, 15, 13,
         14, 16, 15, 16, 17, 15,
         16, 18, 17, 18, 19, 17,
         18, 20, 19, 20, 21, 19,
         20, 22, 21, 22, 23, 21,
         22, 24, 23, 24, 25, 23,
         24, 26, 25, 26, 27, 25,
         26, 28, 27, 28, 29, 27,
         28, 0, 29,
         0,1,29
    };
    unsigned int cube_indices[] = {
        0, 3, 2,
        2, 1, 0,

        4, 5, 7,
        7, 6, 4,

        8, 9, 10,
        10, 11, 8,

        12, 13, 14,
        14, 15, 12,

        16, 17, 18,
        18, 19, 16,

        20, 21, 22,
        22, 23, 20
    };




    float cone_vertix[] = {
        // Apex vertex
        0.0f, 0.5f, 0.0f, // index 0

        // Circle vertices
         1, 0, 0,
         0.913, 0, 0.406,
         0.669, 0, 0.743,
         0.309, 0, 0.951,
        -0.104, 0, 0.994,
        -0.5, 0, 0.866,
        -0.809, 0, 0.588,
        -0.978, 0, 0.208,
        -0.978, 0, -0.208,
        -0.809, 0, -0.588,
        -0.5, 0, -0.866,
        -0.104, 0, -0.994,
         0.309, 0, -0.951,
         0.669, 0, -0.743,
         0.913, 0, -0.406,
         1, 0, 0 // Duplicate the first vertex for easy looping
    };

    unsigned int cone_indices[] = {
        // Sides
        0, 1, 2,
        0, 2, 3,
        0, 3, 4,
        0, 4, 5,
        0, 5, 6,
        0, 6, 7,
        0, 7, 8,
        0, 8, 9,
        0, 9, 10,
        0, 10, 11,
        0, 11, 12,
        0, 12, 13,
        0, 13, 14,
        0, 14, 15,

        // Base circle (triangles)
        1, 2, 3,
        3, 4, 5,
        5, 6, 7,
        7, 8, 9,
        9, 10, 11,
        11, 12, 13,
        13, 14, 15,
        15, 1, 2
    };

    // Generate cylinder vertices and indices
    std::vector<float> vertices;
    std::vector<unsigned int> indices;
    generateCylinder(0.25f, 0.5f, 36, vertices, indices);

    // Generate cone data
    std::vector<float> vertices_k;
    std::vector<unsigned int> indices_k;
    generateCone(1.0f, 2.0f, 36, vertices_k, indices_k);

    // Generate sphere data
    std::vector<float> vertices_s;
    std::vector<unsigned int> indices_s;
    generateSphere(1.0f, 36, 18, vertices_s, indices_s);

    // Create VAO_S, VBO_S, and EBO_S
    unsigned int VAO_S, VBO_S, EBO_S;
    glGenVertexArrays(1, &VAO_S);
    glGenBuffers(1, &VBO_S);
    glGenBuffers(1, &EBO_S);

    // Bind VAO
    glBindVertexArray(VAO_S);

    // Bind and set VBO
    glBindBuffer(GL_ARRAY_BUFFER, VBO_S);
    glBufferData(GL_ARRAY_BUFFER, vertices_s.size() * sizeof(float), vertices_s.data(), GL_STATIC_DRAW);

    // Bind and set EBO
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO_S);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices_s.size() * sizeof(unsigned int), indices_s.data(), GL_STATIC_DRAW);

    // Position attribute
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);

    // Normal attribute
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)(3 * sizeof(float)));
    glEnableVertexAttribArray(1);

    // Unbind VAO
    glBindVertexArray(0);


    // Create VAO, VBO, and EBO
    unsigned int VAO_K, VBO_K, EBO_K;
    glGenVertexArrays(1, &VAO_K);
    glGenBuffers(1, &VBO_K);
    glGenBuffers(1, &EBO_K);

    // Bind VAO
    glBindVertexArray(VAO_K);

    // Bind and set VBO
    glBindBuffer(GL_ARRAY_BUFFER, VBO_K);
    glBufferData(GL_ARRAY_BUFFER, vertices_k.size() * sizeof(float), vertices_k.data(), GL_STATIC_DRAW);

    // Bind and set EBO
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO_K);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices_k.size() * sizeof(unsigned int), indices_k.data(), GL_STATIC_DRAW);



    // position attribute
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);

    //normal attribute
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)12);
    glEnableVertexAttribArray(1);


    // Unbind VAO
    glBindVertexArray(0);



    unsigned int VBO_C, VAO_C, EBO_C;
    glGenVertexArrays(1, &VAO_C);
    glGenBuffers(1, &VBO_C);
    glGenBuffers(1, &EBO_C);

    glBindVertexArray(VAO_C);

    glBindBuffer(GL_ARRAY_BUFFER, VBO_C);
    glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(float), vertices.data(), GL_STATIC_DRAW);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO_C);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.size() * sizeof(unsigned int), indices.data(), GL_STATIC_DRAW);

    // position attribute
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);

    // normal attribute
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)12);
    glEnableVertexAttribArray(1);

    glBindVertexArray(0); // Unbind VAO









    unsigned int VBO, VAO, EBO;
    glGenVertexArrays(1, &VAO);
    glGenBuffers(1, &VBO);
    glGenBuffers(1, &EBO);

    glBindVertexArray(VAO);

    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(cube_vertices), cube_vertices, GL_STATIC_DRAW);
    //glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(float), vertices.data(), GL_STATIC_DRAW);


    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(cube_indices), cube_indices, GL_STATIC_DRAW);
    //glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.size() * sizeof(unsigned int), indices.data(), GL_STATIC_DRAW);

    // position attribute
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);

    //normal attribute
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)12);
    glEnableVertexAttribArray(1);

    




    //fan

    unsigned int VBO2, VAO2, EBO2;
    glGenVertexArrays(1, &VAO2);
    glGenBuffers(1, &VBO2);
    glGenBuffers(1, &EBO2);

    glBindVertexArray(VAO2);

    glBindBuffer(GL_ARRAY_BUFFER, VBO2);
    glBufferData(GL_ARRAY_BUFFER, sizeof(cylinder_vertix), cylinder_vertix, GL_STATIC_DRAW);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO2);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(cylinder_indices), cylinder_indices, GL_STATIC_DRAW);

    // position attribute
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);

    glBindVertexArray(0);

    // second, configure the light's VAO (VBO stays the same; the vertices are the same for the light object which is also a 3D cube)
    unsigned int lightCubeVAO;
    glGenVertexArrays(1, &lightCubeVAO);
    glBindVertexArray(lightCubeVAO);

    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
    // note that we update the lamp's position attribute's stride to reflect the updated buffer data
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);


    //glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);

    float speed = 0.0;


    //ourShader.use();
    //constantShader.use();

    // render loop
    // -----------
    while (!glfwWindowShouldClose(window))
    {
        // per-frame time logic
        // --------------------
        float currentFrame = static_cast<float>(glfwGetTime());
        deltaTime = currentFrame - lastFrame;
        lastFrame = currentFrame;

        // input
        // -----
        processInput(window);

        // render
        // ------
        glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        // be sure to activate shader when setting uniforms/drawing objects
       // lightingShader.use();
        lightingShader.setVec3("viewPos", basic_camera.eye);

        //point lights set up
        pointlight1.setUpPointLight(lightingShader);
        pointlight2.setUpPointLight(lightingShader);

        //directional light set up
        lightingShader.setVec3("directionalLight.direction", 0.0f, 0.0f, -0.5f);
        lightingShader.setVec3("directionalLight.ambient", 0.1f, 0.1f, 0.1f);
        lightingShader.setVec3("directionalLight.diffuse", 0.8f, 0.8f, 0.8f);
        lightingShader.setVec3("directionalLight.specular", 1.0f, 1.0f, 1.0f);
        lightingShader.setBool("directionLightOn", directionLightOn);

        //spot light set up
        lightingShader.setVec3("spotLight.position", 0.0f, 0.0f, .80f);
        lightingShader.setVec3("spotLight.direction", 0.0f, -1.0f, 0.0f);
        lightingShader.setVec3("spotLight.ambient", 0.5f, 0.5f, 0.5f);
        lightingShader.setVec3("spotLight.diffuse", 0.8f, 0.8f, 0.8f);
        lightingShader.setVec3("spotLight.specular", 1.0f, 1.0f, 1.0f);
        lightingShader.setFloat("spotLight.k_c", 1.0f);
        lightingShader.setFloat("spotLight.k_l", 0.09);
        lightingShader.setFloat("spotLight.k_q", 0.032);
        lightingShader.setFloat("spotLight.cos_theta", glm::cos(glm::radians(60.0f)));
        lightingShader.setBool("spotLightOn", spotLightOn);

        //handle for changes in directional light directly from shedder
        if (glfwGetKey(window, GLFW_KEY_5) == GLFW_PRESS) {
            if (directionLightOn) {
                lightingShader.setBool("ambientLight", !directionalAmbient);
                directionalAmbient = !directionalAmbient;
            }
        }

        if (glfwGetKey(window, GLFW_KEY_6) == GLFW_PRESS) {
            if (directionLightOn) {
                lightingShader.setBool("diffuseLight", !directionalDiffuse);
                directionalDiffuse = !directionalDiffuse;
            }
        }

        if (glfwGetKey(window, GLFW_KEY_7) == GLFW_PRESS) {
            if (directionLightOn) {
                lightingShader.setBool("specularLight", !directionalSpecular);
                directionalSpecular = !directionalSpecular;
            }
        }



       
      
        glm:: mat4 projection =perspectiveProjection(50.0f, (float)SCR_WIDTH / (float)SCR_HEIGHT, 1.0f, 50.0f);
        lightingShader.setMat4("projection", projection);
    

        // camera/view transformation
        glm::mat4  view = basic_camera.createViewMatrix();
        lightingShader.setMat4("view", view);
        //constantShader.setMat4("view", view);





        // Modelling Transformation
        glm::mat4 identityMatrix = glm::mat4(1.0f);
        glm::mat4 translateMatrix, rotateXMatrix, rotateYMatrix, rotateZMatrix, scaleMatrix, model, modelCentered;


        /* model = identityMatrix;

         ourShader.setMat4("model", model);
        // shaderProgram.setVec4("color", glm::vec4(R, G, B, 1.0f));
         glBindVertexArray(VAO3);
         glDrawElements(GL_TRIANGLES, 36, GL_UNSIGNED_INT, 0);*/

         //table

        drawTable(lightingShader, VAO, identityMatrix);

        //tvcabinet
        drawTVCAB(lightingShader, VAO, identityMatrix);

        drawSofa(lightingShader, VAO, identityMatrix);

        drawAlmirah(lightingShader, VAO, identityMatrix);


        float tableHeight = 0.7f; 

        //cone

        glm::mat4 rotationMatrix = myRotate(identityMatrix, glm::radians(180.0), glm::vec3(0.0f, 1.0f, 1.0f));
        scaleMatrix = glm::scale(identityMatrix, glm::vec3(0.1f, 0.1f, 0.1f));
        translateMatrix = glm::translate(identityMatrix, glm::vec3(0.0f,0.0f, .1f));
        model = translateMatrix * scaleMatrix* rotationMatrix;
        drawCylinder(VAO_K, lightingShader, glm::vec3(0.0f, 1.0f, 0.0f), model, indices_k);


        //spheare
        rotationMatrix = myRotate(identityMatrix, glm::radians(180.0), glm::vec3(0.0f, 1.0f, 1.0f));
        scaleMatrix = glm::scale(identityMatrix, glm::vec3(0.1f, 0.1f, 0.1f));
        translateMatrix = glm::translate(identityMatrix, glm::vec3(1.03f, -0.25f, 0.2f));
        model = translateMatrix * scaleMatrix * rotationMatrix;
        drawCylinder(VAO_S, lightingShader, glm::vec3(0.0f, 0.0f, 1.0f), model, indices_s);

        //draw Fan


        rotationMatrix = myRotate(identityMatrix, glm::radians(speed), glm::vec3(0.0f, 0.0f, 1.0f));
        glm::mat4 translationmat = glm::translate(identityMatrix, glm::vec3(0.0f, 0.0f, -1.0f));
        glm::mat4 modelCenter = translationmat * rotationMatrix;
        drawFan(lightingShader, identityMatrix, VAO, VAO2, modelCenter);

        if (f)
        {
            speed += 1;
        }


        //glBindVertexArray(lightCubeVAO);
        //for (unsigned int i = 0; i < 2; i++)
        //{
        //    model = glm::mat4(1.0f);
        //    model = glm::translate(model, pointLightPositions[i]);
        //    model = glm::scale(model, glm::vec3(0.5f)); // Make it a smaller cube

        //    lightingShader.setMat4("model", model);
        //    //ourShader.use();
        //    lightingShader.setVec4("color", glm::vec4(1.0f, 1.0f, 1.0f, 1.0f));
        //    //glDrawElements(GL_TRIANGLES, indices.size(), GL_UNSIGNED_INT, 0);
        //    glDrawElements(GL_TRIANGLES, 36, GL_UNSIGNED_INT, 0);
        //    //glDrawArrays(GL_TRIANGLES, 0, 36);
        //}

        //draw the lamp object(s)
        ourShader.use();
        ourShader.setMat4("projection", projection);
        ourShader.setMat4("view", view);

        //we now draw as many light bulbs as we have point lights.
        glBindVertexArray(lightCubeVAO);

        for (unsigned int i = 0; i < 2; i++)
        {
            translateMatrix = glm::translate(identityMatrix, pointLightPositions[i]);
            scaleMatrix = glm::scale(identityMatrix, glm::vec3(0.2f, -0.2f, 0.2f));
            model = translateMatrix * scaleMatrix;
            ourShader.setMat4("model", model);
            ourShader.setVec4("color", glm::vec4(1.0f, 1.0f, 1.0f, 1.0f));
            glDrawElements(GL_TRIANGLES, 36, GL_UNSIGNED_INT, 0);
        }

        // make sure to initialize matrix to identity matrix first
        //drawCube(ourShader, VAO, identityMatrix, translate_X, translate_Y, translate_Z, rotateAngle_X, rotateAngle_Y, rotateAngle_Z, 2.0f, 0.3f, 2.0f);

        // render boxes
        //for (unsigned int i = 0; i < 10; i++)
        //{
        //    // calculate the model matrix for each object and pass it to shader before drawing
        //    glm::mat4 model = glm::mat4(1.0f); // make sure to initialize matrix to identity matrix first
        //    model = glm::translate(model, cubePositions[i]);
        //    float angle = 20.0f * i;
        //    model = glm::rotate(model, glm::radians(angle), glm::vec3(1.0f, 0.3f, 0.5f));
        //    drawCube(ourShader, VAO, model);
        //}

        // glfw: swap buffers and poll IO events (keys pressed/released, mouse moved etc.)
        // -------------------------------------------------------------------------------
        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    // optional: de-allocate all resources once they've outlived their purpose:
    // ------------------------------------------------------------------------
    glDeleteVertexArrays(1, &VAO);
    glDeleteBuffers(1, &VBO);
    glDeleteBuffers(1, &EBO);

    // glfw: terminate, clearing all previously allocated GLFW resources.
    // ------------------------------------------------------------------
    glfwTerminate();
    return 0;
}


void drawTable(Shader ourShader, unsigned int VAO, glm::mat4 identityMatrix, float posX, float posY, float posz, float scX, float scY, float scZ, float rtX, float rtY, float rtZ) {
    glm::mat4 translateMatrix, rotateXMatrix, rotateYMatrix, rotateZMatrix, scaleMatrix, model, modelCentered;

    scaleMatrix = glm::scale(identityMatrix, glm::vec3(scX, scY, scZ));
    translateMatrix = glm::translate(identityMatrix, glm::vec3(posX, posY, posz));
    rotateYMatrix = myRotate(identityMatrix, glm::radians(rtY), glm::vec3(0.0f, 1.0f, 0.0f));

    model = translateMatrix * scaleMatrix;

    //table
    drawCube(ourShader, VAO, model, -0.25, -0.25, 0.0, 1.5, 1, 0.1, 0.196, 0.275, 0.831);
    drawCube(ourShader, VAO, model, -0.25, -0.25, 0.0, 0.1, 0.1, -0.2);
    drawCube(ourShader, VAO, model, -0.25, 0.20, 0.0, 0.1, 0.1, -0.2);
    drawCube(ourShader, VAO, model, 0.45, 0.20, 0.0, 0.1, 0.1, -0.2);
    drawCube(ourShader, VAO, model, .45, -0.25, 0.0, 0.1, 0.1, -0.2);

    //mat
    drawCube(ourShader, VAO, model, -0.55, -0.55, -0.06, 2.5, 2, 0.01, 0.631, 0.02, 0.02);





}


void drawTVCAB(Shader ourShader, unsigned int VAO, glm::mat4 identityMatrix, float posX, float posY, float posz, float scX, float scY, float scZ, float rtX, float rtY, float rtZ) {
    glm::mat4 translateMatrix, rotateXMatrix, rotateYMatrix, rotateZMatrix, scaleMatrix, model, modelCentered;

    scaleMatrix = glm::scale(identityMatrix, glm::vec3(scX, scY, scZ));
    translateMatrix = glm::translate(identityMatrix, glm::vec3(posX, posY, posz));
    rotateYMatrix = myRotate(identityMatrix, glm::radians(rtY), glm::vec3(0.0f, 1.0f, 0.0f));

    model = scaleMatrix * translateMatrix;


    drawCube(ourShader, VAO, model, -1.3, -0.5, 0.0, .40, 2.0, 0.5, 0.22, 0.157, 0.02);



    drawCube(ourShader, VAO, model, -1.3, -0.5, 0.0, 0.1, 0.1, -0.15, 0.22, 0.157, 0.02);
    drawCube(ourShader, VAO, model, -1.3, 0.45, 0.0, 0.1, 0.1, -0.15, 0.22, 0.157, 0.02);
    drawCube(ourShader, VAO, model, -1.15, 0.45, 0.0, 0.1, 0.1, -.15, 0.22, 0.157, 0.02);
    drawCube(ourShader, VAO, model, -1.15, -0.5, 0.0, 0.1, 0.1, -0.15, 0.22, 0.157, 0.02);



    drawCube(ourShader, VAO, model, -1.3, -0.2, .3, 0.1, 1.0, 0.75, 0.49, 0.482, 0.475);

    drawCube(ourShader, VAO, model, -1.3, 0.025, .25, 0.1, .1, 0.1);



}



void drawSofa(Shader ourShader, unsigned int VAO, glm::mat4 identityMatrix, float posX, float posY, float posz, float scX, float scY, float scZ, float rtX, float rtY, float rtZ) {
    glm::mat4 translateMatrix, rotateXMatrix, rotateYMatrix, rotateZMatrix, scaleMatrix, model, modelCentered;
    glm::vec3 color;
    /*scaleMatrix = glm::scale(identityMatrix, glm::vec3(scX, scY, scZ));
    translateMatrix = glm::translate(identityMatrix, glm::vec3(posX, posY, posz));
    rotateYMatrix = glm::rotate(identityMatrix, glm::radians(rtY), glm::vec3(0.0f, 1.0f, 0.0f));

    model = scaleMatrix * translateMatrix;*/

    ourShader.use();

    //glm::mat4 translateMatrix, rotateXMatrix, rotateYMatrix, rotateZMatrix, scaleMatrix, model, modelCentered;

    scaleMatrix = glm::scale(identityMatrix, glm::vec3(0.65, 1.5, 0.35));
    translateMatrix = glm::translate(identityMatrix, glm::vec3(0.85, -0.35, -0.1));
    model = translateMatrix * scaleMatrix;

    color = glm::vec3(0.702, 0.651, 0.565);

    ourShader.setVec3("material.ambient", color);
    ourShader.setVec3("material.diffuse", color);
    ourShader.setVec3("material.specular", color);
    ourShader.setFloat("material.shininess", 32.0f);

    ourShader.setMat4("model", model);
    //ourShader.setVec4("color", glm::vec4(0.702, 0.651, 0.565, 1.0f));
    glBindVertexArray(VAO);
    glDrawElements(GL_TRIANGLES, 36, GL_UNSIGNED_INT, 0);


    //drawCube(ourShader, VAO, model, 0.85, -0.35, -0.1,0.65,1.5,0.35);

    //drawCube(ourShader, VAO, model, 0.85, -0.35, 0.075, 0.65, 0.25, 0.1);

    scaleMatrix = glm::scale(identityMatrix, glm::vec3(0.65, 0.25, 0.1));
    translateMatrix = glm::translate(identityMatrix, glm::vec3(0.85, -0.35, 0.075));
    model = translateMatrix * scaleMatrix;

    color = glm::vec3(0.0, 0.0, 0.0);

    ourShader.setVec3("material.ambient", color);
    ourShader.setVec3("material.diffuse", color);
    ourShader.setVec3("material.specular", color);
    ourShader.setFloat("material.shininess", 32.0f);

    //ourShader.setVec4("color", glm::vec4(0.0, 0.0, 0.0, 1.0f));
    ourShader.setMat4("model", model);

    glBindVertexArray(VAO);
    glDrawElements(GL_TRIANGLES, 36, GL_UNSIGNED_INT, 0);



    scaleMatrix = glm::scale(identityMatrix, glm::vec3(0.65, 0.30, 0.1));
    translateMatrix = glm::translate(identityMatrix, glm::vec3(0.85, 0.25, 0.075));
    model = translateMatrix * scaleMatrix;
    color = glm::vec3(0.0, 0.0, 0.0);

    ourShader.setVec3("material.ambient", color);
    ourShader.setVec3("material.diffuse", color);
    ourShader.setVec3("material.specular", color);
    ourShader.setFloat("material.shininess", 32.0f);

    ourShader.setMat4("model", model);

    glBindVertexArray(VAO);
    glDrawElements(GL_TRIANGLES, 36, GL_UNSIGNED_INT, 0);



    scaleMatrix = glm::scale(identityMatrix, glm::vec3(0.25, 1.0, 0.1));
    translateMatrix = glm::translate(identityMatrix, glm::vec3(1.05, -0.25, 0.075));
    model = translateMatrix * scaleMatrix;

    ourShader.setMat4("model", model);

    glBindVertexArray(VAO);
    glDrawElements(GL_TRIANGLES, 36, GL_UNSIGNED_INT, 0);


}



void drawAlmirah(Shader ourShader, unsigned int VAO, glm::mat4 identityMatrix, float posX, float posY, float posz, float scX, float scY, float scZ, float rtX, float rtY, float rtZ) {
    glm::mat4 translateMatrix, rotateXMatrix, rotateYMatrix, rotateZMatrix, scaleMatrix, model, modelCentered;
    glm::vec3 color;
    /*scaleMatrix = glm::scale(identityMatrix, glm::vec3(scX, scY, scZ));
    translateMatrix = glm::translate(identityMatrix, glm::vec3(posX, posY, posz));
    rotateYMatrix = glm::rotate(identityMatrix, glm::radians(rtY), glm::vec3(0.0f, 1.0f, 0.0f));

    model = scaleMatrix * translateMatrix;*/

    ourShader.use();

    //glm::mat4 translateMatrix, rotateXMatrix, rotateYMatrix, rotateZMatrix, scaleMatrix, model, modelCentered;

    scaleMatrix = glm::scale(identityMatrix, glm::vec3(2.5, 0.5, 0.5));
    translateMatrix = glm::translate(identityMatrix, glm::vec3(-0.4, 0.6, -0.1));
    model = translateMatrix * scaleMatrix;

    color = glm::vec3(0.388, 0.267, 0.02);

    ourShader.setVec3("material.ambient", color);
    ourShader.setVec3("material.diffuse", color);
    ourShader.setVec3("material.specular", color);
    ourShader.setFloat("material.shininess", 32.0f);

    ourShader.setMat4("model", model);
    //ourShader.setVec4("color", glm::vec4(0.388, 0.267, 0.02, 1.0f));
    glBindVertexArray(VAO);
    glDrawElements(GL_TRIANGLES, 36, GL_UNSIGNED_INT, 0);




    scaleMatrix = glm::scale(identityMatrix, glm::vec3(0.5, 0.01, 0.15));
    translateMatrix = glm::translate(identityMatrix, glm::vec3(0.0, 0.599, 0.01));
    model = translateMatrix * scaleMatrix;

    color = glm::vec3(1.0, 1.0, 1.0);

    ourShader.setVec3("material.ambient", color);
    ourShader.setVec3("material.diffuse", color);
    ourShader.setVec3("material.specular", color);
    ourShader.setFloat("material.shininess", 32.0f);


    ourShader.setMat4("model", model);
    //ourShader.setVec4("color", glm::vec4(1.0, 1.0, 1.0, 1.0f));

    glBindVertexArray(VAO);
    glDrawElements(GL_TRIANGLES, 36, GL_UNSIGNED_INT, 0);



    scaleMatrix = glm::scale(identityMatrix, glm::vec3(0.5, 0.01, 0.15));
    translateMatrix = glm::translate(identityMatrix, glm::vec3(0.28, 0.599, 0.01));
    model = translateMatrix * scaleMatrix;
  
    color = glm::vec3(1.0, 1.0, 1.0);

    ourShader.setVec3("material.ambient", color);
    ourShader.setVec3("material.diffuse", color);
    ourShader.setVec3("material.specular", color);
    ourShader.setFloat("material.shininess", 32.0f);
    ourShader.setMat4("model", model);
    //ourShader.setVec4("color", glm::vec4(1.0, 1.0, 1.0, 1.0f));

    glBindVertexArray(VAO);
    glDrawElements(GL_TRIANGLES, 36, GL_UNSIGNED_INT, 0);


    scaleMatrix = glm::scale(identityMatrix, glm::vec3(0.5, 0.01, 0.15));
    translateMatrix = glm::translate(identityMatrix, glm::vec3(-0.28, 0.599, 0.01));
    model = translateMatrix * scaleMatrix;


    ourShader.setMat4("model", model);
    //ourShader.setVec4("color", glm::vec4(1.0, 1.0, 1.0, 1.0f));

    glBindVertexArray(VAO);
    glDrawElements(GL_TRIANGLES, 36, GL_UNSIGNED_INT, 0);

    scaleMatrix = glm::scale(identityMatrix, glm::vec3(0.5, 0.01, 0.3));
    translateMatrix = glm::translate(identityMatrix, glm::vec3(0.56, 0.599, 0.01));
    model = translateMatrix * scaleMatrix;


    ourShader.setMat4("model", model);
    ourShader.setVec4("color", glm::vec4(1.0, 1.0, 1.0, 1.0f));

    glBindVertexArray(VAO);
    glDrawElements(GL_TRIANGLES, 36, GL_UNSIGNED_INT, 0);




    //wall

    scaleMatrix = glm::scale(identityMatrix, glm::vec3(6.0, 5.0, 0.08));
    translateMatrix = glm::translate(identityMatrix, glm::vec3(-1.5, -1.0, -0.12));
    model = translateMatrix * scaleMatrix;
    color = glm::vec3(0.80, 0.80, 1.0);

    ourShader.setVec3("material.ambient", color);
    ourShader.setVec3("material.diffuse", color);
    ourShader.setVec3("material.specular", color);
    ourShader.setFloat("material.shininess", 32.0f);

    ourShader.setMat4("model", model);
    //ourShader.setVec4("color", glm::vec4(0.80, 0.80, 1.0, 1.0f));

    glBindVertexArray(VAO);
    glDrawElements(GL_TRIANGLES, 36, GL_UNSIGNED_INT, 0);


    scaleMatrix = glm::scale(identityMatrix, glm::vec3(0.08, 5.0, 3.0));
    translateMatrix = glm::translate(identityMatrix, glm::vec3(-1.5, -1.0, -0.1));
    model = translateMatrix * scaleMatrix;
    color = glm::vec3(1.0, 1.0, 1.0);

    ourShader.setVec3("material.ambient", color);
    ourShader.setVec3("material.diffuse", color);
    ourShader.setVec3("material.specular", color);
    ourShader.setFloat("material.shininess", 32.0f);

    ourShader.setMat4("model", model);
    //ourShader.setVec4("color", glm::vec4(1.0, 1.0, 1.0, 1.0f));

    glBindVertexArray(VAO);
    glDrawElements(GL_TRIANGLES, 36, GL_UNSIGNED_INT, 0);


    scaleMatrix = glm::scale(identityMatrix, glm::vec3(6.0, 0.08, 3.0));
    translateMatrix = glm::translate(identityMatrix, glm::vec3(-1.5, 1.5, -0.1));
    model = translateMatrix * scaleMatrix;
    color = glm::vec3(1.0, 1.0, 0.0);

    ourShader.setVec3("material.ambient", color);
    ourShader.setVec3("material.diffuse", color);
    ourShader.setVec3("material.specular", color);
    ourShader.setFloat("material.shininess", 32.0f);

    ourShader.setMat4("model", model);
    //ourShader.setVec4("color", glm::vec4(1.0, 1.0, 0.0, 1.0f));

    glBindVertexArray(VAO);
    glDrawElements(GL_TRIANGLES, 36, GL_UNSIGNED_INT, 0);


    scaleMatrix = glm::scale(identityMatrix, glm::vec3(0.08, 5.0, 3.0));
    translateMatrix = glm::translate(identityMatrix, glm::vec3(1.46, -1.001, -0.1));
    model = translateMatrix * scaleMatrix;
    color = glm::vec3(0.0, 1.0, 1.0);

    ourShader.setVec3("material.ambient", color);
    ourShader.setVec3("material.diffuse", color);
    ourShader.setVec3("material.specular", color);
    ourShader.setFloat("material.shininess", 32.0f);

    ourShader.setMat4("model", model);
    //ourShader.setVec4("color", glm::vec4(0.0, 1.0, 1.0, 1.0f));

    glBindVertexArray(VAO);
    glDrawElements(GL_TRIANGLES, 36, GL_UNSIGNED_INT, 0);

    /*
    scaleMatrix = glm::scale(identityMatrix, glm::vec3(6.0, 5.0, 0.08));
    translateMatrix = glm::translate(identityMatrix, glm::vec3(-1.5, -1.0, 1.40));
    model = translateMatrix * scaleMatrix;

    ourShader.setMat4("model", model);
    ourShader.setVec4("color", glm::vec4(0.80, 0.80, 1.0, 1.0f));

    glBindVertexArray(VAO);
    glDrawElements(GL_TRIANGLES, 36, GL_UNSIGNED_INT, 0);
    */

}

void drawFan(Shader ourShader, glm::mat4 identityMatrix, unsigned int VAO, unsigned int VAO2, glm::mat4 Model_Center)
{
    glm::mat4 translateMatrix, rotationMatrix, scaleMatrix, model;



    ourShader.use();

    translateMatrix = glm::translate(identityMatrix, glm::vec3(-0.025f, -0.025f, 1.1f));
    scaleMatrix = glm::scale(identityMatrix, glm::vec3(0.1, 0.1, 0.6));
    model = translateMatrix * scaleMatrix;

    ourShader.setMat4("model", model);
    ourShader.setVec4("color", glm::vec4(0.631, 0.02, 0.02, 0.70f));
    glBindVertexArray(VAO);
    glDrawElements(GL_TRIANGLES, 36, GL_UNSIGNED_INT, 0);
    glBindVertexArray(0);



    ourShader.use();
    translateMatrix = glm::translate(identityMatrix, glm::vec3(0.0, 0.0, 2.1));
    rotationMatrix = myRotate(identityMatrix, glm::radians(90.0f), glm::vec3(1.0f, 0.0f, 0.0f));
    scaleMatrix = glm::scale(identityMatrix, glm::vec3(0.2, 0.2, 0.2));

    //rotateXMatrix = glm::rotate(identityMatrix, glm::radians(0.0), glm::vec3(1.0f, 0.0f, 0.0f));
    //rotateYMatrix = glm::rotate(identityMatrix, glm::radians(0.0), glm::vec3(0.0f, 1.0f, 0.0f));
    //rotateZMatrix = glm::rotate(identityMatrix, glm::radians(0.0), glm::vec3(0.0f, 0.0f, 1.0f));
    model = Model_Center * translateMatrix * rotationMatrix * scaleMatrix;

    ourShader.setMat4("model", model);
    ourShader.setVec4("color", glm::vec4(0.731, 0.02, 0.02, 1.0f));


    glBindVertexArray(VAO2);
    glDrawElements(GL_TRIANGLE_FAN, 15, GL_UNSIGNED_INT, 0);
    glDrawElements(GL_TRIANGLE_FAN, 15, GL_UNSIGNED_INT, (const void*)(15 * sizeof(unsigned int)));
    glDrawElements(GL_TRIANGLE_FAN, 6, GL_UNSIGNED_INT, (const void*)(30 * sizeof(unsigned int)));
    glDrawElements(GL_TRIANGLE_STRIP, 90, GL_UNSIGNED_INT, (const void*)(36 * sizeof(unsigned int)));
    //glDrawElements(GL_TRIANGLE_FAN, 30, GL_UNSIGNED_INT, 0);
    glBindVertexArray(0);




    ourShader.use();

    translateMatrix = glm::translate(identityMatrix, glm::vec3(-0.1f, 0.0f, 2.0625f));
    scaleMatrix = glm::scale(identityMatrix, glm::vec3(0.4, 1.2, 0.05));
    model = Model_Center * translateMatrix * scaleMatrix;

    ourShader.setMat4("model", model);
    ourShader.setVec4("color", glm::vec4(0.631, 0.02, 0.02, 1.0f));

    glBindVertexArray(VAO);
    glDrawElements(GL_TRIANGLES, 36, GL_UNSIGNED_INT, 0);
    glBindVertexArray(0);





    ourShader.use();

    translateMatrix = glm::translate(identityMatrix, glm::vec3(-0.1f, 0.0f, 2.0625f));
    scaleMatrix = glm::scale(identityMatrix, glm::vec3(0.4, 1.2, 0.05));
    rotationMatrix = myRotate(identityMatrix, glm::radians(120.0f), glm::vec3(0.0f, 0.0f, 1.0f));
    model = Model_Center * rotationMatrix * translateMatrix * scaleMatrix;

    ourShader.setMat4("model", model);
    ourShader.setVec4("color", glm::vec4(0.631, 0.02, 0.02, 1.0f));

    glBindVertexArray(VAO);
    glDrawElements(GL_TRIANGLES, 36, GL_UNSIGNED_INT, 0);
    glBindVertexArray(0);




    ourShader.use();

    translateMatrix = glm::translate(identityMatrix, glm::vec3(-0.1f, 0.0f, 2.0625f));
    scaleMatrix = glm::scale(identityMatrix, glm::vec3(0.4, 1.2, 0.05));
    rotationMatrix = myRotate(identityMatrix, glm::radians(240.0f), glm::vec3(0.0f, 0.0f, 1.0f));
    model = Model_Center * rotationMatrix * translateMatrix * scaleMatrix;

    ourShader.setMat4("model", model);
    ourShader.setVec4("color", glm::vec4(0.631, 0.02, 0.02, 1.0f));
    glBindVertexArray(VAO);
    glDrawElements(GL_TRIANGLES, 36, GL_UNSIGNED_INT, 0);
    glBindVertexArray(0);




}





// process all input: query GLFW whether relevant keys are pressed/released this frame and react accordingly
// ---------------------------------------------------------------------------------------------------------
void processInput(GLFWwindow* window)
{
    if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
        glfwSetWindowShouldClose(window, true);

    if (glfwGetKey(window, GLFW_KEY_I) == GLFW_PRESS) translate_Y += 0.01;
    if (glfwGetKey(window, GLFW_KEY_K) == GLFW_PRESS) translate_Y -= 0.01;
    if (glfwGetKey(window, GLFW_KEY_L) == GLFW_PRESS) translate_X += 0.01;
    if (glfwGetKey(window, GLFW_KEY_J) == GLFW_PRESS) translate_X -= 0.01;
    if (glfwGetKey(window, GLFW_KEY_O) == GLFW_PRESS) translate_Z += 0.01;
    if (glfwGetKey(window, GLFW_KEY_P) == GLFW_PRESS) translate_Z -= 0.01;
    if (glfwGetKey(window, GLFW_KEY_C) == GLFW_PRESS) scale_X += 0.01;
    if (glfwGetKey(window, GLFW_KEY_V) == GLFW_PRESS) scale_X -= 0.01;
    if (glfwGetKey(window, GLFW_KEY_B) == GLFW_PRESS) scale_Y += 0.01;
    if (glfwGetKey(window, GLFW_KEY_N) == GLFW_PRESS) scale_Y -= 0.01;
    if (glfwGetKey(window, GLFW_KEY_M) == GLFW_PRESS) scale_Z += 0.01;
    if (glfwGetKey(window, GLFW_KEY_U) == GLFW_PRESS) scale_Z -= 0.01;

    if (glfwGetKey(window, GLFW_KEY_X) == GLFW_PRESS)
    {
        rotateAngle_X += 1;
    }
    if (glfwGetKey(window, GLFW_KEY_Y) == GLFW_PRESS)
    {
        rotateAngle_Y += 1;
    }
    if (glfwGetKey(window, GLFW_KEY_Z) == GLFW_PRESS)
    {
        rotateAngle_Z += 1;
    }

    if (glfwGetKey(window, GLFW_KEY_H) == GLFW_PRESS)
    {
        eyeX += 2.5 * deltaTime;
        basic_camera.eye = glm::vec3(eyeX, eyeY, eyeZ);
    }
    if (glfwGetKey(window, GLFW_KEY_F) == GLFW_PRESS)
    {
        eyeX -= 2.5 * deltaTime;
        basic_camera.eye = glm::vec3(eyeX, eyeY, eyeZ);
    }
    if (glfwGetKey(window, GLFW_KEY_T) == GLFW_PRESS)
    {
        eyeZ += 2.5 * deltaTime;
        basic_camera.eye = glm::vec3(eyeX, eyeY, eyeZ);
    }
    if (glfwGetKey(window, GLFW_KEY_P) == GLFW_PRESS)
    {
        eyeZ -= 2.5 * deltaTime;
        basic_camera.eye = glm::vec3(eyeX, eyeY, eyeZ);
    }
    if (glfwGetKey(window, GLFW_KEY_Q) == GLFW_PRESS)
    {
        eyeY += 2.5 * deltaTime;
        basic_camera.eye = glm::vec3(eyeX, eyeY, eyeZ);
    }
    if (glfwGetKey(window, GLFW_KEY_E) == GLFW_PRESS)
    {
        eyeY -= 2.5 * deltaTime;
        basic_camera.eye = glm::vec3(eyeX, eyeY, eyeZ);
    }
   /* if (glfwGetKey(window, GLFW_KEY_1) == GLFW_PRESS)
    {
        lookAtX += 2.5 * deltaTime;
        basic_camera.lookAt = glm::vec3(lookAtX, lookAtY, lookAtZ);
    }*/
    /*if (glfwGetKey(window, GLFW_KEY_2) == GLFW_PRESS)
    {
        lookAtX -= 2.5 * deltaTime;
        basic_camera.lookAt = glm::vec3(lookAtX, lookAtY, lookAtZ);
    }*/
    /*if (glfwGetKey(window, GLFW_KEY_3) == GLFW_PRESS)
    {
        lookAtY += 2.5 * deltaTime;
        basic_camera.lookAt = glm::vec3(lookAtX, lookAtY, lookAtZ);
    }
    if (glfwGetKey(window, GLFW_KEY_4) == GLFW_PRESS)
    {
        lookAtY -= 2.5 * deltaTime;
        basic_camera.lookAt = glm::vec3(lookAtX, lookAtY, lookAtZ);
    }*/
    /*if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS) {
        camera.ProcessKeyboard(FORWARD, deltaTime);
    }

    if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS) {
       camera.ProcessKeyboard(BACKWARD, deltaTime);
    }

    if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS) {
        camera.ProcessKeyboard(LEFT, deltaTime);
    }

    if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS) {
        camera.ProcessKeyboard(RIGHT, deltaTime);
    }

    if (glfwGetKey(window, GLFW_KEY_E) == GLFW_PRESS) {
        camera.ProcessKeyboard(UP, deltaTime);
    }

    if (glfwGetKey(window, GLFW_KEY_R) == GLFW_PRESS) {
        camera.ProcessKeyboard(DOWN, deltaTime);
    }

    if (glfwGetKey(window, GLFW_KEY_X) == GLFW_PRESS) {
        camera.ProcessKeyboard(P_UP, deltaTime);
    }

    if (glfwGetKey(window, GLFW_KEY_C) == GLFW_PRESS) {
        camera.ProcessKeyboard(P_DOWN, deltaTime);
    }

    if (glfwGetKey(window, GLFW_KEY_Y) == GLFW_PRESS) {
       camera.ProcessKeyboard(Y_LEFT, deltaTime);
    }

    if (glfwGetKey(window, GLFW_KEY_T) == GLFW_PRESS) {
        camera.ProcessKeyboard(Y_RIGHT, deltaTime);
    }

    if (glfwGetKey(window, GLFW_KEY_Z) == GLFW_PRESS) {
        camera.ProcessKeyboard(R_LEFT, deltaTime);
    }

    if (glfwGetKey(window, GLFW_KEY_V) == GLFW_PRESS) {
       camera.ProcessKeyboard(R_RIGHT, deltaTime);
    }*/

    
    if (glfwGetKey(window, GLFW_KEY_G) == GLFW_PRESS)
    {
        if (f)
            f = false;
        else
            f = true;
    }

    if (glfwGetKey(window, GLFW_KEY_2) == GLFW_PRESS) {
        if (pointlight1.ambientOn > 0 && pointlight1.diffuseOn > 0 && pointlight1.specularOn > 0) {
            pointlight1.turnOff();
            point1 = false;
        }
        else {
            pointlight1.turnOn();
            point1 = true;
        }
    }

    if (glfwGetKey(window, GLFW_KEY_3) == GLFW_PRESS) {
        if (pointlight2.ambientOn > 0 && pointlight2.diffuseOn > 0 && pointlight2.specularOn > 0) {
            pointlight2.turnOff();
            point2 = false;
        }
        else {
            pointlight2.turnOn();
            point2 = true;
        }
    }

    if (glfwGetKey(window, GLFW_KEY_1) == GLFW_PRESS) {
        directionLightOn = !directionLightOn;
    }

    if (glfwGetKey(window, GLFW_KEY_4) == GLFW_PRESS) {
        spotLightOn = !spotLightOn;
    }

    if (glfwGetKey(window, GLFW_KEY_5) == GLFW_PRESS) {
        if (pointlight1.ambientOn > 0 || pointlight2.ambientOn > 0) {
            
                pointlight1.turnAmbientOff();
            
                pointlight2.turnAmbientOff();
        }
        else {
            
                pointlight1.turnAmbientOn();
            
                pointlight2.turnAmbientOn();
        }
    }

    if (glfwGetKey(window, GLFW_KEY_6) == GLFW_PRESS) {
        if (pointlight1.diffuseOn > 0 || pointlight2.diffuseOn > 0) {
           
                pointlight1.turnDiffuseOff();
       
                pointlight2.turnDiffuseOff();
        }
        else {
            
                pointlight1.turnDiffuseOn();
            
                pointlight2.turnDiffuseOn();
        }
    }

    if (glfwGetKey(window, GLFW_KEY_7) == GLFW_PRESS) {
        if (pointlight1.specularOn > 0 || pointlight2.specularOn > 0) {
            
                pointlight1.turnSpecularOff();
            
                pointlight2.turnSpecularOff();
        }
        else {
            
                pointlight1.turnSpecularOn();
            
                pointlight2.turnSpecularOn();
        }
    }

}

// glfw: whenever the window size changed (by OS or user resize) this callback function executes
// ---------------------------------------------------------------------------------------------
void framebuffer_size_callback(GLFWwindow* window, int width, int height)
{
    // make sure the viewport matches the new window dimensions; note that width and
    // height will be significantly larger than specified on retina displays.
    glViewport(0, 0, width, height);
}


// glfw: whenever the mouse moves, this callback is called
// -------------------------------------------------------
void mouse_callback(GLFWwindow* window, double xposIn, double yposIn)
{
    float xpos = static_cast<float>(xposIn);
    float ypos = static_cast<float>(yposIn);

    if (firstMouse)
    {
        lastX = xpos;
        lastY = ypos;
        firstMouse = false;
    }

    float xoffset = xpos - lastX;
    float yoffset = lastY - ypos; // reversed since y-coordinates go from bottom to top

    lastX = xpos;
    lastY = ypos;

    basic_camera.ProcessMouseMovement(xoffset, yoffset);
}

// glfw: whenever the mouse scroll wheel scrolls, this callback is called
// ----------------------------------------------------------------------
void scroll_callback(GLFWwindow* window, double xoffset, double yoffset)
{
    basic_camera.ProcessMouseScroll(static_cast<float>(yoffset));
}



void generateCylinder(float radius, float height, int segments, std::vector<float>& vertices, std::vector<unsigned int>& indices) {
    float angleStep = 2.0f * M_PI / segments;

    // Generate vertices and normals
    for (int i = 0; i <= segments; ++i) {
        float angle = i * angleStep;
        float x = radius * cos(angle);
        float z = radius * sin(angle);

        // Bottom circle vertex
        vertices.push_back(x);
        vertices.push_back(0.0f);
        vertices.push_back(z);

        // Bottom circle normal (pointing down the y-axis)
        vertices.push_back(0.0f);
        vertices.push_back(-1.0f);
        vertices.push_back(0.0f);

        // Top circle vertex
        vertices.push_back(x);
        vertices.push_back(height);
        vertices.push_back(z);

        // Top circle normal (pointing up the y-axis)
        vertices.push_back(0.0f);
        vertices.push_back(1.0f);
        vertices.push_back(0.0f);

        // Side vertex (bottom)
        vertices.push_back(x);
        vertices.push_back(0.0f);
        vertices.push_back(z);

        // Side normal
        vertices.push_back(x);
        vertices.push_back(0.0f);
        vertices.push_back(z);

        // Side vertex (top)
        vertices.push_back(x);
        vertices.push_back(height);
        vertices.push_back(z);

        // Side normal
        vertices.push_back(x);
        vertices.push_back(0.0f);
        vertices.push_back(z);
    }

    // Generate indices for the sides
    for (int i = 0; i < segments; ++i) {
        int bottom1 = 4 * i + 2;
        int top1 = bottom1 + 1;
        int bottom2 = 4 * (i + 1) + 2;
        int top2 = bottom2 + 1;

        // Side triangles
        indices.push_back(bottom1);
        indices.push_back(top1);
        indices.push_back(bottom2);

        indices.push_back(top1);
        indices.push_back(top2);
        indices.push_back(bottom2);
    }

    // Center vertex for the top circle
    int topCenterIndex = vertices.size() / 6;
    vertices.push_back(0.0f);
    vertices.push_back(height);
    vertices.push_back(0.0f);

    // Top center normal (pointing up the y-axis)
    vertices.push_back(0.0f);
    vertices.push_back(1.0f);
    vertices.push_back(0.0f);

    // Generate indices for the top circle
    for (int i = 0; i < segments; ++i) {
        int top1 = 4 * i + 1;
        int top2 = 4 * ((i + 1) % segments) + 1;

        indices.push_back(topCenterIndex);
        indices.push_back(top1);
        indices.push_back(top2);
    }

    // Center vertex for the bottom circle
    int bottomCenterIndex = vertices.size() / 6;
    vertices.push_back(0.0f);
    vertices.push_back(0.0f);
    vertices.push_back(0.0f);

    // Bottom center normal (pointing down the y-axis)
    vertices.push_back(0.0f);
    vertices.push_back(-1.0f);
    vertices.push_back(0.0f);

    // Generate indices for the bottom circle
    for (int i = 0; i < segments; ++i) {
        int bottom1 = 4 * i;
        int bottom2 = 4 * ((i + 1) % segments);

        indices.push_back(bottomCenterIndex);
        indices.push_back(bottom2);
        indices.push_back(bottom1);
    }
}




void generateCone(float radius, float height, int segments, std::vector<float>& vertices, std::vector<unsigned int>& indices) {
    float angleStep = 2.0f * M_PI / segments;

    // Generate vertices and normals for the base circle
    for (int i = 0; i < segments; ++i) {
        float angle = i * angleStep;
        float x = radius * cos(angle);
        float z = radius * sin(angle);

        // Base circle vertex
        vertices.push_back(x);
        vertices.push_back(0.0f); // y = 0 for the base
        vertices.push_back(z);

        // Base circle normal (pointing downwards)
        vertices.push_back(0.0f);
        vertices.push_back(-1.0f);
        vertices.push_back(0.0f);
    }

    // Add the tip vertex of the cone
    int tipIndex = vertices.size() / 6; // Index of the tip
    vertices.push_back(0.0f);
    vertices.push_back(height); // Tip is at y = height
    vertices.push_back(0.0f);

    // Tip normal (pointing upwards)
    vertices.push_back(0.0f);
    vertices.push_back(1.0f);
    vertices.push_back(0.0f);

    // Add the center vertex of the base circle
    int baseCenterIndex = vertices.size() / 6; // Index of the base center
    vertices.push_back(0.0f);
    vertices.push_back(0.0f); // Center is at y = 0
    vertices.push_back(0.0f);

    // Base center normal (pointing downwards)
    vertices.push_back(0.0f);
    vertices.push_back(-1.0f);
    vertices.push_back(0.0f);

    // Generate indices and normals for the sides
    for (int i = 0; i < segments; ++i) {
        int nextIndex = (i + 1) % segments;

        // Side triangle
        indices.push_back(i);
        indices.push_back(nextIndex);
        indices.push_back(tipIndex);

        // Calculate normals for the side vertices
        glm::vec3 v1(vertices[6 * i], vertices[6 * i + 1], vertices[6 * i + 2]);
        glm::vec3 v2(vertices[6 * nextIndex], vertices[6 * nextIndex + 1], vertices[6 * nextIndex + 2]);
        glm::vec3 normal = glm::normalize(glm::cross(v2 - v1, glm::vec3(0.0f, height, 0.0f) - v1));

        // Add normals for the side vertices
        vertices.push_back(v1.x);
        vertices.push_back(v1.y);
        vertices.push_back(v1.z);
        vertices.push_back(normal.x);
        vertices.push_back(normal.y);
        vertices.push_back(normal.z);

        vertices.push_back(v2.x);
        vertices.push_back(v2.y);
        vertices.push_back(v2.z);
        vertices.push_back(normal.x);
        vertices.push_back(normal.y);
        vertices.push_back(normal.z);
    }

    // Generate indices for the base circle
    for (int i = 0; i < segments; ++i) {
        int nextIndex = (i + 1) % segments;

        indices.push_back(baseCenterIndex);
        indices.push_back(i);
        indices.push_back(nextIndex);
    }
}




void generateSphere(float radius, int sectorCount, int stackCount, std::vector<float>& vertices, std::vector<unsigned int>& indices) {
    float x, y, z, xy;                              // vertex position
    float nx, ny, nz, lengthInv = 1.0f / radius;    // vertex normal, lenginv is the inverse of the radius


    float sectorStep = 2 * M_PI / sectorCount;
    float stackStep = M_PI / stackCount;
    float sectorAngle, stackAngle;

    for (int i = 0; i <= stackCount; ++i) {
        stackAngle = M_PI / 2 - i * stackStep;        // starting from pi/2 to -pi/2
        xy = radius * cosf(stackAngle);             // r * cos(u)
        z = radius * sinf(stackAngle);              // r * sin(u)

        for (int j = 0; j <= sectorCount; ++j) {
            sectorAngle = j * sectorStep;           // starting from 0 to 2pi

            // vertex position (x, y, z)
            x = xy * cosf(sectorAngle);             // r * cos(u) * cos(v)
            y = xy * sinf(sectorAngle);             // r * cos(u) * sin(v)
            vertices.push_back(x);
            vertices.push_back(y);
            vertices.push_back(z);

            // normalized vertex normal (nx, ny, nz)
            nx = x * lengthInv;
            ny = y * lengthInv;
            nz = z * lengthInv;
            vertices.push_back(nx);
            vertices.push_back(ny);
            vertices.push_back(nz);
        }
    }

    // generate indices
    int k1, k2;
    for (int i = 0; i < stackCount; ++i) {
        k1 = i * (sectorCount + 1);     // beginning of current stack
        k2 = k1 + sectorCount + 1;      // beginning of next stack

        for (int j = 0; j < sectorCount; ++j, ++k1, ++k2) {
            // 2 triangles per sector excluding first and last stacks
            if (i != 0) {
                indices.push_back(k1);
                indices.push_back(k2);
                indices.push_back(k1 + 1);
            }

            if (i != (stackCount - 1)) {
                indices.push_back(k1 + 1);
                indices.push_back(k2);
                indices.push_back(k2 + 1);
            }
        }
    }
}

void drawCylinder(unsigned int& VAO_C, Shader& lightingShader, glm::vec3 color, glm::mat4 model, std::vector<unsigned int>& indices)
{
    lightingShader.use();

    // setting up materialistic property
    lightingShader.setVec3("material.ambient", color);
    lightingShader.setVec3("material.diffuse", color);
    lightingShader.setVec3("material.specular", color);
    lightingShader.setFloat("material.shininess", 32.0f);
    float emissiveIntensity = 0.05f; // Adjust this value to increase or decrease the intensity
    glm::vec3 emissiveColor = glm::vec3(1.0f, 0.0f, 0.0f) * emissiveIntensity;

    //lightingShader.setVec3("material.emissive", emissiveColor);

    lightingShader.setMat4("model", model);

    glBindVertexArray(VAO_C);
    glDrawElements(GL_TRIANGLES, indices.size(), GL_UNSIGNED_INT, 0);
    glBindVertexArray(0);
}

void drawCube(Shader shaderProgram, unsigned int VAO, glm::mat4 parentTrans, float posX, float posY, float posZ, float scX, float scY, float scZ, float R, float G, float B)
{
    shaderProgram.use();

    glm::mat4 translateMatrix, rotateXMatrix, rotateYMatrix, rotateZMatrix, scaleMatrix, model, modelCentered;
    glm::vec3 color = glm::vec3(R, G, B);

    scaleMatrix = glm::scale(parentTrans, glm::vec3(scX, scY, scZ));
    translateMatrix = glm::translate(parentTrans, glm::vec3(posX, posY, posZ));
    model = translateMatrix * scaleMatrix;

    shaderProgram.setVec3("material.ambient", color);
    shaderProgram.setVec3("material.diffuse", color);
    shaderProgram.setVec3("material.specular", color);
    shaderProgram.setFloat("material.shininess", 32.0f);

    shaderProgram.setMat4("model", model);
    //shaderProgram.setVec4("color", glm::vec4(R, G, B, 1.0f));
    glBindVertexArray(VAO);
    glDrawElements(GL_TRIANGLES, 36, GL_UNSIGNED_INT, 0);
}


// Function to draw a sphere
void drawSphere(unsigned int& VAO_S, Shader& lightingShader, glm::vec3 color, glm::mat4 model, std::vector<unsigned int>& indices)
{
    lightingShader.use();

    // Setting up materialistic property
    lightingShader.setVec3("material.ambient", color);
    lightingShader.setVec3("material.diffuse", color);
    lightingShader.setVec3("material.specular", color);
    lightingShader.setFloat("material.shininess", 32.0f);
    float emissiveIntensity = 0.05f; // Adjust this value to increase or decrease the intensity
    glm::vec3 emissiveColor = glm::vec3(1.0f, 0.0f, 0.0f) * emissiveIntensity;

    // lightingShader.setVec3("material.emissive", emissiveColor);

   

    lightingShader.setMat4("model", model);
  



    glBindVertexArray(VAO_S);
    glDrawElements(GL_TRIANGLES, indices.size(), GL_UNSIGNED_INT, 0);
    glBindVertexArray(0);
}


void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods)
{
    if (glfwGetKey(window, GLFW_KEY_2) == GLFW_PRESS) {
        if (point1) {
            pointlight1.turnOff();
            point1 = false;
        }
        else {
            pointlight1.turnOn();
            point1 = true;
        }
    }

    if (glfwGetKey(window, GLFW_KEY_3) == GLFW_PRESS) {
        if (point2) {
            pointlight2.turnOff();
            point2 = false;
        }
        else {
            pointlight2.turnOn();
            point2 = true;
        }
    }

    if (glfwGetKey(window, GLFW_KEY_1) == GLFW_PRESS) {
        directionLightOn = !directionLightOn;
    }

    if (glfwGetKey(window, GLFW_KEY_4) == GLFW_PRESS) {
        spotLightOn = !spotLightOn;
    }

    if (glfwGetKey(window, GLFW_KEY_5) == GLFW_PRESS) {
        if (pointlight1.ambientOn > 0 || pointlight2.ambientOn > 0) {
            if (point1)
                pointlight1.turnAmbientOff();
            if (point2)
                pointlight2.turnAmbientOff();
        }
        else {
            if (point1)
                pointlight1.turnAmbientOn();
            if (point2)
                pointlight2.turnAmbientOn();
        }
    }

    if (glfwGetKey(window, GLFW_KEY_6) == GLFW_PRESS) {
        if (pointlight1.diffuseOn > 0 || pointlight2.diffuseOn > 0) {
            if (point1)
                pointlight1.turnDiffuseOff();
            if (point2)
                pointlight2.turnDiffuseOff();
        }
        else {
            if (point1)
                pointlight1.turnDiffuseOn();
            if (point2)
                pointlight2.turnDiffuseOn();
        }
    }

    if (glfwGetKey(window, GLFW_KEY_7) == GLFW_PRESS) {
        if (pointlight1.specularOn > 0 || pointlight2.specularOn > 0) {
            if (point1)
                pointlight1.turnSpecularOff();
            if (point2)
                pointlight2.turnSpecularOff();
        }
        else {
            if (point1)
                pointlight1.turnSpecularOn();
            if (point2)
                pointlight2.turnSpecularOn();
        }
    }

}

