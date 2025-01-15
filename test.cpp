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


glm:: mat4 projection =perspectiveProjection(50.0f, (float)SCR_WIDTH / (float)SCR_HEIGHT, 1.0f, 50.0f);
lightingShader.setMat4("projection", projection);


import numpy as np

def perspective_matrix(left, right, bottom, top, near, far):
  
    # Ensure near and far planes are positive
    if near <= 0 or far <= 0:
        raise ValueError("Near and far clipping planes must be positive.")
    if near >= far:
        raise ValueError("The near plane must be closer than the far plane.")

    # Initialize the perspective projection matrix
    matrix = np.zeros((4, 4), dtype=np.float32)

    # Fill in the matrix elements
    matrix[0, 0] = 2 * near / (right - left)
    matrix[1, 1] = 2 * near / (top - bottom)
    matrix[0, 2] = (right + left) / (right - left)
    matrix[1, 2] = (top + bottom) / (top - bottom)
    matrix[2, 2] = -(far + near) / (far - near)
    matrix[2, 3] = -2 * far * near / (far - near)
    matrix[3, 2] = -1

    return matrix

# Example usage:
left = -1.0
right = 1.0
bottom = -1.0
top = 1.0
near = 1.0
far = 100.0

perspective_proj_matrix = perspective_matrix(left, right, bottom, top, near, far)
print("Perspective Projection Matrix:")
print(perspective_proj_matrix)




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

scaleMatrix = glm::scale(identityMatrix, glm::vec3(scX, scY, scZ));
translateMatrix = glm::translate(identityMatrix, glm::vec3(posX, posY, posz));
rotateYMatrix = myRotate(identityMatrix, glm::radians(rtY), glm::vec3(0.0f, 1.0f, 0.0f));


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




    glm::mat4 lookAt(const glm::vec3& eye, const glm::vec3& center, const glm::vec3& up) {
    // Calculate forward, right, and up vectors
    glm::vec3 forward = glm::normalize(center - eye); // Forward direction
    glm::vec3 right = glm::normalize(glm::cross(forward, glm::normalize(up))); // Right vector
    glm::vec3 upVector = glm::cross(right, forward); // Recalculate orthogonal up vector

    // Create the rotation part of the matrix
    glm::mat4 rotation(1.0f);
    rotation[0][0] = right.x;
    rotation[1][0] = right.y;
    rotation[2][0] = right.z;
    rotation[0][1] = upVector.x;
    rotation[1][1] = upVector.y;
    rotation[2][1] = upVector.z;
    rotation[0][2] = -forward.x;
    rotation[1][2] = -forward.y;
    rotation[2][2] = -forward.z;

    // Create the translation part of the matrix
    glm::mat4 translation(1.0f);
    translation[3][0] = -eye.x;
    translation[3][1] = -eye.y;
    translation[3][2] = -eye.z;

    // Combine rotation and translation
    return rotation * translation;
}

    glm::mat4 viewMatrix = lookAt(eye, center, up);


//smallest cylinder making code ever exist
float r = 1;
for (float tx = -r; tx <= r; tx += 0.02)
{
    float tz = sqrtf(r * r - tx * tx);
    drawCube(ourShader, VAO, identityMatrix, translate_X + tx, translate_Y, translate_Z+tz, rotateAngle_X, rotateAngle_Y, rotateAngle_Z, scale_X, scale_Y, scale_Z);
    tz = -tz;
    drawCube(ourShader, VAO, identityMatrix, translate_X+tx, translate_Y, translate_Z+tz, rotateAngle_X, rotateAngle_Y, rotateAngle_Z, scale_X, scale_Y, scale_Z);
}