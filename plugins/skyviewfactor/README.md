# Sky View Factor Plugin

## Description

Le plugin Sky View Factor (SVF) calcule le facteur de vue du ciel pour des points donnés dans une scène 3D. Le facteur de vue du ciel mesure la fraction de la voûte céleste visible depuis un point donné, variant de 0 (complètement confiné) à 1 (complètement dégagé).

## Définition mathématique

Le sky view factor est défini comme :

```
f_sky = (1/π) ∫ V(θ,φ) cos²(θ) dω
```

Où :
- V(θ,φ) est la fonction de visibilité (1 si le ciel est visible, 0 si masqué)
- θ est l'angle zénithal
- dω est l'élément d'angle solide

## Architecture

### Classes principales

- **SkyViewFactorModel** : Classe principale pour le calcul du SVF
- **SkyViewFactorCamera** : Classe pour la visualisation et l'analyse du SVF
- **SkyViewFactorRayTracing** : Fonctionnalités de ray tracing CUDA/OptiX

### Fichiers

#### Headers (.h)
- `SkyViewFactorModel.h` : Interface principale du modèle SVF
- `SkyViewFactorCamera.h` : Interface de la caméra SVF
- `SkyViewFactorRayTracing.h` : Définitions CUDA/OptiX

#### Sources C++ (.cpp)
- `SkyViewFactorModel.cpp` : Implémentation du modèle SVF
- `SkyViewFactorCamera.cpp` : Implémentation de la caméra SVF

#### Sources CUDA (.cu)
- `skyViewFactorRayGeneration.cu` : Génération de rayons pour le SVF
- `skyViewFactorRayHit.cu` : Gestion des intersections de rayons
- `skyViewFactorPrimitiveIntersection.cu` : Intersections primitives-rayons

#### Tests
- `selfTest.cpp` : Tests unitaires
- `TestMain.cpp` : Suite de tests complète

## Utilisation

### Calcul de base

```cpp
#include "SkyViewFactorModel.h"

// Créer un contexte HELIOS
Context context;

// Créer le modèle SVF
SkyViewFactorModel svfModel(&context);

// Calculer le SVF pour un point
vec3 point(0.0f, 0.0f, 0.0f);
float svf = svfModel.calculateSkyViewFactor(point);

// Calculer le SVF pour plusieurs points
std::vector<vec3> points = {vec3(0,0,0), vec3(1,0,0), vec3(0,1,0)};
std::vector<float> svfs = svfModel.calculateSkyViewFactors(points);
```

### Configuration

```cpp
// Définir le nombre de rayons
svfModel.setRayCount(1000);

// Définir la longueur maximale des rayons
svfModel.setMaxRayLength(1000.0f);

// Activer/désactiver les messages
svfModel.setMessageFlag(true);
```

### Visualisation avec caméra

```cpp
#include "SkyViewFactorCamera.h"

// Créer une caméra SVF
SkyViewFactorCamera camera(&context);

// Configurer la caméra
camera.setPosition(vec3(0, 0, 10));
camera.setTarget(vec3(0, 0, 0));
camera.setResolution(512, 512);
camera.setRayCount(100);

// Rendre l'image
camera.render();

// Exporter l'image
camera.exportImage("skyviewfactor.ppm");
```

## Compilation

Le plugin utilise CMake et nécessite :
- CUDA Toolkit
- OptiX (optionnel, pour l'accélération GPU)
- Compilateur C++ compatible C++11

```bash
mkdir build
cd build
cmake ..
make
```

## Tests

Exécuter les tests :

```bash
# Tests unitaires
./skyviewfactor_tests

# Tests complets
./TestMain
```

## Performance

- **CPU** : Implémentation de référence avec traçage de rayons CPU
- **GPU** : Accélération CUDA/OptiX pour des calculs rapides
- **Rayons** : Nombre configurable (défaut : 1000)
- **Précision** : Améliorée avec plus de rayons

## Limitations

- Actuellement optimisé pour les primitives triangulaires
- L'implémentation GPU nécessite OptiX
- La précision dépend du nombre de rayons utilisés

## Développement

### Ajout de nouvelles primitives

Pour supporter de nouveaux types de primitives, modifier :
- `skyViewFactorPrimitiveIntersection.cu`
- `SkyViewFactorModel.cpp` (fonction `calculateSkyViewFactorCPU`)

### Optimisations

- Utiliser plus de rayons pour une meilleure précision
- Implémenter l'accélération GPU complète
- Ajouter des techniques de sampling avancées

## Licence

GNU General Public License v2.0

## Auteur

Boris Dufour - 2025
