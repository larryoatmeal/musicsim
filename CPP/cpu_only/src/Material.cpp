#include "Material.h"
Vector3f Material::shade(const Ray &ray,
    const Hit &hit,
    const Vector3f &dirToLight,
    const Vector3f &lightIntensity)
{
    // TODO implement
    // return Vector3f(1, 1, 1);

    // BEGIN SOLN 
    //float lightdist = dirToLight.abs();
    float ndotl = Vector3f::dot(hit.normal, dirToLight.normalized());
    // FIXME falloff computation is part of light source color :<
    //ndotl /= (lightdist * lightdist);
    if (ndotl < 0.0f) {
        ndotl = 0.0f;
    }
    else {
        float foo = ndotl;
    }
    Vector3f diffuse = ndotl * lightIntensity * _diffuseColor;

    Vector3f E = ray.getDirection();
    Vector3f R = E - 2 * hit.normal * Vector3f::dot(hit.normal, E);
    float ldotr = Vector3f::dot(R, dirToLight.normalized());
    if (ldotr < 0.0f) {
        ldotr = 0.0f;
    }
    Vector3f specular = powf(ldotr, _shininess)* lightIntensity * _specularColor;

    // END SOLN
    return diffuse + specular;
}
