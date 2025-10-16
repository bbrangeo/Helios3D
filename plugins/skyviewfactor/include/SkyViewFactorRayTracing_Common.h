/** \file "SkyViewFactorRayTracing_Common.h" Common definitions for sky view factor ray tracing.

    Copyright (C) 2025 Boris Dufour

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, version 2.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

*/

#ifndef SKYVIEWFACTORRAYTRACING_COMMON_H
#define SKYVIEWFACTORRAYTRACING_COMMON_H

#include <stdint.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

typedef unsigned int uint;

// Common structures and types that can be used in both C++ and CUDA code
struct SkyViewFactorPayload {
    bool visible;        ///< Whether the ray hits the sky (no obstacles)
    float distance;      ///< Distance to first intersection
    uint primitiveID;    ///< ID of intersected primitive
    float hit_point[3];  ///< Hit point coordinates (as array for compatibility)
    float normal[3];     ///< Surface normal at hit point (as array for compatibility)
    float weight;        ///< Ray weight (cos²(θ))
};

// Helper functions for sky view factor calculation (CPU implementations)
inline float calculateRayWeight(const float direction[3]) {
    // Calculate cos²(θ) weight for the ray
    // direction[2] is cos(θ) since we're working in hemisphere (z > 0)
    float cosTheta = direction[2];
    return cosTheta * cosTheta;
}

inline bool isRayVisible(const SkyViewFactorPayload& payload) {
    // Ray is visible if it doesn't hit any obstacles
    return payload.visible;
}

#endif //SKYVIEWFACTORRAYTRACING_COMMON_H
