/** \file "SkyViewFactorRayTracing.h" Header file for sky view factor ray tracing functionality.

    Copyright (C) 2025 Boris Dufour

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, version 2.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

*/

#ifndef SKYVIEWFACTORRAYTRACING_H
#define SKYVIEWFACTORRAYTRACING_H

#include <stdint.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

typedef unsigned int uint;

// Only include OptiX-specific code when CUDA/OptiX is available
#ifdef CUDA_AVAILABLE
// Launch parameters for sky view factor calculation
rtDeclareVariable(rtObject, top_object, , );
rtDeclareVariable(unsigned int, random_seed, , );
rtDeclareVariable(unsigned int, launch_offset, , );
rtDeclareVariable(unsigned int, launch_face, , );
rtDeclareVariable(unsigned int, Nrays_launch, , );
rtDeclareVariable(unsigned int, Nrays_global, , );
rtBuffer<bool, 1> ray_launch_flag;

// Ray types for sky view factor calculation
rtDeclareVariable(unsigned int, skyview_ray_type, , );

rtDeclareVariable(uint3, launch_index, rtLaunchIndex, );
rtDeclareVariable(uint3, launch_dim, rtLaunchDim, );

// Sky view factor specific buffers
rtBuffer<float3, 1> ray_origins;             ///< Ray origin points
rtBuffer<float3, 1> ray_directions;          ///< Ray direction vectors
rtBuffer<float, 1> ray_weights;              ///< Ray weights (cos²(θ))
rtBuffer<bool, 1> ray_visibility;            ///< Ray visibility results
rtBuffer<float, 1> sky_view_factors;         ///< Calculated sky view factors

// Primitive data buffers
rtBuffer<float3, 1> primitive_vertices;      ///< Primitive vertex data
rtBuffer<uint3, 1> primitive_triangles;      ///< Primitive triangle indices
rtBuffer<uint, 1> primitive_materials;       ///< Primitive material IDs
rtBuffer<uint, 1> primitive_primitiveIDs;    ///< Primitive IDs

// Texture-related buffers (inherited from radiation plugin)
rtBuffer<bool, 3> maskdata;
rtBuffer<int2, 1> masksize;
rtBuffer<int, 1> maskID;
rtBuffer<float2, 2> uvdata;
rtBuffer<int, 1> uvID;
rtBuffer<float3, 2> normaldata;
rtBuffer<int, 1> normalID;
rtBuffer<float3, 2> colordata;
rtBuffer<int, 1> colorID;
rtBuffer<float, 2> alphadata;
rtBuffer<int, 1> alphaID;
rtBuffer<float, 2> roughnessdata;
rtBuffer<int, 1> roughnessID;
rtBuffer<float, 2> transmittancedata;
rtBuffer<int, 1> transmittanceID;

// Material properties
rtBuffer<float3, 1> material_diffuse;
rtBuffer<float3, 1> material_specular;
rtBuffer<float, 1> material_roughness;
rtBuffer<float, 1> material_transmittance;
rtBuffer<float, 1> material_emission;

// Sky view factor calculation parameters
rtDeclareVariable(float, max_ray_length, , );
rtDeclareVariable(float3, sample_point, , );
rtDeclareVariable(uint, num_rays, , );

// Ray payload for sky view factor calculation
struct SkyViewFactorPayload {
    bool visible;        ///< Whether the ray hits the sky (no obstacles)
    float distance;      ///< Distance to first intersection
    uint primitiveID;    ///< ID of intersected primitive
    float3 hit_point;    ///< Hit point coordinates
    float3 normal;       ///< Surface normal at hit point
    float weight;        ///< Ray weight (cos²(θ))
};

// Ray generation program for sky view factor calculation
RT_PROGRAM void skyViewFactorRayGeneration() {
    // Implementation will be in .cu file
}

// Ray hit program for sky view factor calculation
RT_PROGRAM void skyViewFactorRayHit() {
    // Implementation will be in .cu file
}

// Ray miss program for sky view factor calculation
RT_PROGRAM void skyViewFactorRayMiss() {
    // Implementation will be in .cu file
}

// Primitive intersection program for sky view factor calculation
RT_PROGRAM void skyViewFactorPrimitiveIntersection(int primIdx) {
    // Implementation will be in .cu file
}

// Bounding box program for sky view factor calculation
RT_PROGRAM void skyViewFactorBoundingBox(int primIdx, float result[6]) {
    // Implementation will be in .cu file
}

// Helper functions for sky view factor calculation
__device__ __forceinline__ float3 generateHemisphereRay(uint index, uint totalRays, uint seed) {
    // Generate uniform random direction on hemisphere
    // Implementation will be in .cu file
    return make_float3(0.0f, 0.0f, 1.0f);
}

__device__ __forceinline__ float calculateRayWeight(const float3& direction) {
    // Calculate cos²(θ) weight for the ray
    // Implementation will be in .cu file
    return 1.0f;
}

__device__ __forceinline__ bool isRayVisible(const SkyViewFactorPayload& payload) {
    // Check if ray is visible (no obstacles)
    // Implementation will be in .cu file
    return payload.visible;
}

#endif // CUDA_AVAILABLE

#endif //SKYVIEWFACTORRAYTRACING_H
