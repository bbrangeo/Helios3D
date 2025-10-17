/** \file "SkyViewFactorModel.h" Primary header file for sky view factor calculation model.

    Copyright (C) 2025 Boris Dufour

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, version 2.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

*/

#ifndef SKYVIEWFACTORMODEL_H
#define SKYVIEWFACTORMODEL_H

#include "Context.h"
#include "SkyViewFactorRayTracing_Common.h"
#include <vector>
#include <string>

namespace helios {

    /** \class SkyViewFactorModel
     * \brief Main class for calculating sky view factor using ray tracing
     * 
     * The sky view factor (SVF) measures the fraction of the sky hemisphere 
     * visible from a given point. It ranges from 0 (completely enclosed) to 1 (completely open).
     * 
     * Mathematical definition:
     * f_sky = (1/π) ∫ V(θ,φ) cos²(θ) dω
     * 
     * Where:
     * - V(θ,φ) is the visibility function (1 if sky is visible, 0 if occluded)
     * - θ is the zenith angle
     * - dω is the solid angle element
     */
    class SkyViewFactorModel {
    private:
        
        Context* context;                    ///< Pointer to the HELIOS context
        
        // Ray tracing parameters
        uint rayCount_default;               ///< Default number of rays for SVF calculation
        uint rayCount;                       ///< Current number of rays for SVF calculation
        
        // OptiX/CUDA related
        bool cuda_flag;                      ///< Flag indicating if CUDA is available
        bool optix_flag;                     ///< Flag indicating if OptiX is available
        
        // Ray generation parameters
        float maxRayLength;                  ///< Maximum ray length for intersection testing
        bool message_flag;                   ///< Flag for console output
        
        // Internal data structures
        std::vector<float> skyViewFactors;   ///< Storage for calculated sky view factors
        std::vector<vec3> samplePoints;      ///< Points where SVF is calculated
        
        // CUDA/OptiX buffers and contexts
        void* cuda_context;                  ///< CUDA context
        void* optix_context;                 ///< OptiX context
        void* optix_module;                  ///< OptiX module
        void* optix_program_groups;          ///< OptiX program groups
        void* optix_pipeline;                ///< OptiX pipeline
        
        // Ray generation data
        void* ray_generation_data;           ///< Data for ray generation
        void* primitive_data;                ///< Primitive data for intersection testing
        
        // Private methods
        void initializeOptiX();              ///< Initialize OptiX context and modules
        void cleanupOptiX();                 ///< Clean up OptiX resources
        void generateRays(const vec3& point, std::vector<vec3>& rayDirections, std::vector<float>& rayWeights); ///< Generate rays for SVF calculation
        float calculateSkyViewFactorGPU(const vec3& point); ///< GPU-based SVF calculation
        float calculateSkyViewFactorOptimized(const vec3& point, const std::vector<std::vector<helios::vec3>>& primitiveVertices); ///< Optimized CPU-based SVF calculation
        
    public:
        
        /** \brief Constructor
         * \param context_a Pointer to HELIOS context
         */
        SkyViewFactorModel(Context* context_a);
        
        /** \brief Destructor */
        ~SkyViewFactorModel();
        
        /** \brief Calculate sky view factor for a single point
         * \param point 3D point where to calculate SVF
         * \return Sky view factor value (0-1)
         */
        float calculateSkyViewFactor(const vec3& point);
        
        /** \brief Calculate sky view factor for a single point using CPU implementation
         * \param point 3D point where to calculate SVF
         * \return Sky view factor value (0-1)
         */
        float calculateSkyViewFactorCPU(const vec3& point);
        
        /** \brief Calculate sky view factors for multiple points
         * \param points Vector of 3D points
         * \return Vector of sky view factor values
         */
        std::vector<float> calculateSkyViewFactors(const std::vector<vec3>& points);
        
        /** \brief Calculate sky view factor for all primitive centers
         * \return Vector of sky view factor values for each primitive
         */
        std::vector<float> calculateSkyViewFactorsForPrimitives();
        
        /** \brief Set the number of rays for SVF calculation
         * \param N Number of rays to use
         */
        void setRayCount(uint N);
        
        /** \brief Get the current number of rays
         * \return Number of rays
         */
        uint getRayCount() const;
        
        /** \brief Set maximum ray length for intersection testing
         * \param length Maximum ray length
         */
        void setMaxRayLength(float length);
        
        /** \brief Get maximum ray length
         * \return Maximum ray length
         */
        float getMaxRayLength() const;
        
        /** \brief Enable/disable console output
         * \param flag True to enable messages, false to disable
         */
        void setMessageFlag(bool flag);
        
        /** \brief Check if CUDA is available
         * \return True if CUDA is available
         */
        bool isCudaAvailable() const;
        
        /** \brief Check if OptiX is available
         * \return True if OptiX is available
         */
        bool isOptiXAvailable() const;
        
        /** \brief Get the last calculated sky view factors
         * \return Vector of sky view factor values
         */
        std::vector<float> getSkyViewFactors() const;
        
        /** \brief Export sky view factors to file
         * \param filename Output filename
         * \return True if successful
         */
        bool exportSkyViewFactors(const std::string& filename) const;
        
        /** \brief Load sky view factors from file
         * \param filename Input filename
         * \return True if successful
         */
        bool loadSkyViewFactors(const std::string& filename);
        
        /** \brief Reset all calculated data
         */
        void reset();
        
        /** \brief Get statistics about the last calculation
         * \return String containing calculation statistics
         */
        std::string getStatistics() const;
    };

}

#endif //SKYVIEWFACTORMODEL_H
