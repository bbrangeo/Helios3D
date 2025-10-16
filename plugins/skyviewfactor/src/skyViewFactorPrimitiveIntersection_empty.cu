/** \file "skyViewFactorPrimitiveIntersection_empty.cu" Empty CUDA file for when OptiX is not available.

    Copyright (C) 2025 Boris Dufour

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, version 2.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

*/

#include "SkyViewFactorRayTracing_Common.h"

// This file is used when OptiX is not available to provide empty implementations
// The actual OptiX code is in skyViewFactorPrimitiveIntersection.cu but only compiled when OptiX is available

// Only compile if CUDA is available but OptiX is not
#if defined(CUDA_AVAILABLE) && !defined(OPTIX_AVAILABLE)

// Empty implementations for when OptiX is not available
// These functions will be called but do nothing

#endif // CUDA_AVAILABLE && !OPTIX_AVAILABLE
