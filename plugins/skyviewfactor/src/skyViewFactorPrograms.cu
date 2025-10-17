/** \file "skyViewFactorPrograms.cu" OptiX programs for sky view factor calculation.

    Copyright (C) 2025 Boris Dufour

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, version 2.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

*/

#include "SkyViewFactorRayTracing.h"

// Only compile OptiX code if both CUDA and OptiX are available
#if defined(CUDA_AVAILABLE) && defined(OPTIX_AVAILABLE)

using namespace optix;

// Closest hit program for sky view factor calculation
RT_PROGRAM void skyViewFactorClosestHit() {
    // Get hit information
    const float3 world_ray_dir = optixGetWorldRayDirection();
    const float3 world_ray_origin = optixGetWorldRayOrigin();
    const float t = optixGetRayTmax();
    const unsigned int primitive_id = optixGetPrimitiveIndex();
    
    // Calculate hit point
    const float3 hit_point = world_ray_origin + t * world_ray_dir;
    
    // Get primitive normal (simplified - assumes triangle)
    const float3* vertices = (float3*)optixGetPrimitiveVertexPointer();
    const unsigned int* indices = (unsigned int*)optixGetPrimitiveIndexPointer();
    
    // Calculate triangle normal
    float3 v0 = vertices[indices[0]];
    float3 v1 = vertices[indices[1]];
    float3 v2 = vertices[indices[2]];
    
    float3 edge1 = v1 - v0;
    float3 edge2 = v2 - v0;
    float3 normal = cross(edge1, edge2);
    normal = normalize(normal);
    
    // Update payload
    SkyViewFactorPayload& payload = optixGetPayload_0<SkyViewFactorPayload>();
    payload.visible = false;  // Ray hit something, so it's not visible to sky
    payload.distance = t;
    payload.primitiveID = primitive_id;
    payload.hit_point = hit_point;
    payload.normal = normal;
}

// Any hit program for sky view factor calculation
RT_PROGRAM void skyViewFactorAnyHit() {
    // For sky view factor, we only need to know if there's any hit
    // We don't need to continue tracing after the first hit
    SkyViewFactorPayload& payload = optixGetPayload_0<SkyViewFactorPayload>();
    payload.visible = false;  // Ray hit something
    payload.distance = optixGetRayTmax();
    
    // Terminate ray tracing after first hit
    optixTerminateRay();
}

// Miss program for sky view factor calculation
RT_PROGRAM void skyViewFactorMiss() {
    // Ray didn't hit anything, so it's visible to sky
    SkyViewFactorPayload& payload = optixGetPayload_0<SkyViewFactorPayload>();
    payload.visible = true;
    payload.distance = optixGetRayTmax();
    payload.primitiveID = 0;
    payload.hit_point = make_float3(0.0f, 0.0f, 0.0f);
    payload.normal = make_float3(0.0f, 0.0f, 1.0f);
}

#endif // CUDA_AVAILABLE && OPTIX_AVAILABLE
