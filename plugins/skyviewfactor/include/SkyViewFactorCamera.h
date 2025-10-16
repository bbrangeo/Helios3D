/** \file "SkyViewFactorCamera.h" Header file for sky view factor camera functionality.

    Copyright (C) 2025 Boris Dufour

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, version 2.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

*/

#ifndef SKYVIEWFACTORCAMERA_H
#define SKYVIEWFACTORCAMERA_H

#include "Context.h"
#include <vector>
#include <string>

namespace helios {

    /** \class SkyViewFactorCamera
     * \brief Camera class for sky view factor visualization and analysis
     * 
     * This class provides functionality to visualize sky view factors
     * using camera-based rendering techniques.
     */
    class SkyViewFactorCamera {
    private:
        
        Context* context;                    ///< Pointer to the HELIOS context
        
        // Camera parameters
        vec3 position;                       ///< Camera position
        vec3 target;                         ///< Camera target point
        vec3 up;                             ///< Camera up vector
        float fov;                           ///< Field of view in degrees
        uint2 resolution;                    ///< Image resolution (width, height)
        
        // Ray tracing parameters
        uint rayCount;                       ///< Number of rays per pixel
        float maxRayLength;                  ///< Maximum ray length
        
        // Output data
        std::vector<float> skyViewFactorImage; ///< Sky view factor values for each pixel
        std::vector<vec3> hitPoints;         ///< Hit points for each ray
        std::vector<bool> visibilityMask;    ///< Visibility mask for each ray
        
        // Internal methods
        void generateCameraRays(std::vector<vec3>& rayOrigins, std::vector<vec3>& rayDirections); ///< Generate camera rays
        float calculatePixelSkyViewFactor(const vec3& rayOrigin, const vec3& rayDirection); ///< Calculate SVF for a single pixel
        
    public:
        
        /** \brief Constructor
         * \param context_a Pointer to HELIOS context
         */
        SkyViewFactorCamera(Context* context_a);
        
        /** \brief Destructor */
        ~SkyViewFactorCamera();
        
        /** \brief Set camera position
         * \param pos Camera position
         */
        void setPosition(const vec3& pos);
        
        /** \brief Set camera target
         * \param target Camera target point
         */
        void setTarget(const vec3& target);
        
        /** \brief Set camera up vector
         * \param up Camera up vector
         */
        void setUp(const vec3& up);
        
        /** \brief Set field of view
         * \param fov_deg Field of view in degrees
         */
        void setFieldOfView(float fov_deg);
        
        /** \brief Set image resolution
         * \param width Image width in pixels
         * \param height Image height in pixels
         */
        void setResolution(uint width, uint height);
        
        /** \brief Set number of rays per pixel
         * \param count Number of rays per pixel
         */
        void setRayCount(uint count);
        
        /** \brief Set maximum ray length
         * \param length Maximum ray length
         */
        void setMaxRayLength(float length);
        
        /** \brief Get camera position
         * \return Camera position
         */
        vec3 getPosition() const;
        
        /** \brief Get camera target
         * \return Camera target
         */
        vec3 getTarget() const;
        
        /** \brief Get camera up vector
         * \return Camera up vector
         */
        vec3 getUp() const;
        
        /** \brief Get field of view
         * \return Field of view in degrees
         */
        float getFieldOfView() const;
        
        /** \brief Get image resolution
         * \return Image resolution (width, height)
         */
        uint2 getResolution() const;
        
        /** \brief Get number of rays per pixel
         * \return Number of rays per pixel
         */
        uint getRayCount() const;
        
        /** \brief Get maximum ray length
         * \return Maximum ray length
         */
        float getMaxRayLength() const;
        
        /** \brief Render sky view factor image
         * \return True if successful
         */
        bool render();
        
        /** \brief Get sky view factor image
         * \return Vector of SVF values for each pixel
         */
        std::vector<float> getSkyViewFactorImage() const;
        
        /** \brief Export sky view factor image to file
         * \param filename Output filename
         * \return True if successful
         */
        bool exportImage(const std::string& filename) const;
        
        /** \brief Get sky view factor value at specific pixel
         * \param x Pixel x coordinate
         * \param y Pixel y coordinate
         * \return Sky view factor value
         */
        float getPixelSkyViewFactor(uint x, uint y) const;
        
        /** \brief Get hit point at specific pixel
         * \param x Pixel x coordinate
         * \param y Pixel y coordinate
         * \return Hit point coordinates
         */
        vec3 getPixelHitPoint(uint x, uint y) const;
        
        /** \brief Get visibility at specific pixel
         * \param x Pixel x coordinate
         * \param y Pixel y coordinate
         * \return True if sky is visible
         */
        bool getPixelVisibility(uint x, uint y) const;
        
        /** \brief Reset camera data
         */
        void reset();
        
        /** \brief Get rendering statistics
         * \return String containing rendering statistics
         */
        std::string getStatistics() const;
    };

}

#endif //SKYVIEWFACTORCAMERA_H
