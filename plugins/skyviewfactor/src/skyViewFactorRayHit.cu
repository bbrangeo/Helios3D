/** \file "skyViewFactorRayHit.cu" CUDA ray hit handling for sky view factor calculation.

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
#include <optix.h>
#include <optixu/optixu_math_namespace.h>
#include <optixu/optixu_vector_types.h>

using namespace optix;

// Ray hit program for sky view factor calculation
RT_PROGRAM void skyViewFactorRayHit() {
    // Get hit information
    float3 hitPoint = rtTransformPoint(RT_OBJECT_TO_WORLD, rtGetPoint(rtIntersectionDistance));
    float3 normal = rtTransformNormal(RT_OBJECT_TO_WORLD, rtGetNormal());
    normal = normalize(normal);
    
    // Get primitive information
    uint primitiveID = rtGetPrimitiveIndex();
    uint materialID = primitive_materials[primitiveID];
    
    // Get payload
    SkyViewFactorPayload& payload = rtGetPayload<SkyViewFactorPayload>();
    
    // Update payload with hit information
    payload.visible = false;  // Ray hit an obstacle, so sky is not visible
    payload.distance = rtIntersectionDistance;
    payload.primitiveID = primitiveID;
    payload.hit_point = hitPoint;
    payload.normal = normal;
    
    // Store hit information for analysis
    // (This could be used for more detailed analysis of what was hit)
}

// Ray miss program for sky view factor calculation
RT_PROGRAM void skyViewFactorRayMiss() {
    // Get payload
    SkyViewFactorPayload& payload = rtGetPayload<SkyViewFactorPayload>();
    
    // Ray missed all objects, so sky is visible
    payload.visible = true;
    payload.distance = max_ray_length;
    payload.primitiveID = 0;
    payload.hit_point = make_float3(0.0f, 0.0f, 0.0f);
    payload.normal = make_float3(0.0f, 0.0f, 1.0f);
}

// Exception program for error handling
RT_PROGRAM void skyViewFactorException() {
    // Get payload
    SkyViewFactorPayload& payload = rtGetPayload<SkyViewFactorPayload>();
    
    // Set error state
    payload.visible = false;
    payload.distance = 0.0f;
    payload.primitiveID = 0;
    payload.hit_point = make_float3(0.0f, 0.0f, 0.0f);
    payload.normal = make_float3(0.0f, 0.0f, 1.0f);
    
    // Log error (in a real implementation, this would use proper logging)
    rtPrintf("SkyViewFactor: Ray tracing exception occurred\n");
}
