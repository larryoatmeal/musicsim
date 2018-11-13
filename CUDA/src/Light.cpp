#include "Light.h"
    void DirectionalLight::getIllumination(const Vector3f &p, 
                                 Vector3f &tolight, 
                                 Vector3f &intensity, 
                                 float &distToLight) const
    {
        // the direction to the light is the opposite of the
        // direction of the directional light source
        // BEGIN SOLUTION
        tolight = -_direction;
        intensity  = _color;
        distToLight = std::numeric_limits<float>::max();
        // END SOLUTION
    }
    void PointLight::getIllumination(const Vector3f &p, 
                                 Vector3f &tolight, 
                                 Vector3f &intensity, 
                                 float &distToLight) const
    {
        // FIXME THIS SHOULD NOT BE STARTER CODE

        // IF YOU DON'T IMPLEMENT THIS, YOU DON'T KNOW HOW SHADING WORKS
        // ALSO, What is 'falloff', and why is it zero.

        // the direction to the light is the opposite of the
        // direction of the directional light source
        tolight = _position - p;
        distToLight = tolight.abs();
        tolight.normalize();
        intensity = _color / (_falloff * distToLight * distToLight);
        // lukas: what does the 1.0f in the denominator do?
    }

