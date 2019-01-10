#include "Renderer.h"

#include "ArgParser.h"
#include "Camera.h"
#include "Image.h"
#include "Ray.h"
#include "VecUtils.h"

#include <limits>


Renderer::Renderer(const ArgParser &args) :
    _args(args),
    _scene(args.input_file)
{
}

void
Renderer::Render()
{
    // TODO: IMPLEMENT 
    int w = _args.width;
    int h = _args.height;

    Image image(w, h);
    Image nimage(w, h);
    Image dimage(w, h);
    // loop through all the pixels in the image
    // generate all the samples

    // BEGIN SOLN
    // SHOULD WE PROVIDE THIS?
    Camera* cam = _scene.getCamera();
    for (int y = 0; y < h; ++y) {
        float ndcy = 2 * (y / (h - 1.0f)) - 1.0f;
        for (int x = 0; x < w; ++x) {
            float ndcx = 2 * (x / (w - 1.0f)) - 1.0f;
            Ray r = cam->generateRay(Vector2f(ndcx, ndcy));
            Hit h;
            Vector3f color = traceRay(r, cam->getTMin(), _args.bounces, h);

            image.setPixel(x, y, color);
            nimage.setPixel(x, y, (h.getNormal() + 1.0f) / 2.0f);
            float range = (_args.depth_max - _args.depth_min);
            if (range) {
                dimage.setPixel(x, y, Vector3f((h.t - _args.depth_min) / range));
            }
        }
    }
    // END SOLN

    // save the files 
    if (_args.output_file.size()) {
        image.savePNG(_args.output_file);
    }
    if (_args.depth_file.size()) {
        dimage.savePNG(_args.depth_file);
    }
    if (_args.normals_file.size()) {
        nimage.savePNG(_args.normals_file);
    }
}



Vector3f
Renderer::traceRay(const Ray &r,
    float tmin,
    int bounces,
    Hit &h) const
{
    // TODO: IMPLEMENT 

    // BEGIN SOLN 
    //hit = Hit(std::numeric_limits<float>::max(), NULL, Vector3f(0, 0, 0));
    bool intersect = _scene.getGroup()->intersect(r, tmin, h);
    if (intersect) {
        // HIT
        Vector3f col;
        Vector3f P = r.pointAtParameter(h.t);

        for (Light* light : _scene.lights) {
            Vector3f to_light;
            Vector3f light_intensity;
            float dtol;
            light->getIllumination(P, to_light, light_intensity, dtol);

            // shadow ray
            Hit h2 = h;
            h2.t = FLT_MAX;
            Ray r2(P + to_light * 0.01f, to_light);
            bool light_intersect = _scene.getGroup()->intersect(r2, tmin, h2);
            if (!_args.shadows || !light_intersect || h2.t > dtol) {
                // not in shadow
                col += h.getMaterial()->shade(r, h, to_light, light_intensity);
            }
            else {
                float t = h2.t;

            }
        }
        if (h.getMaterial()->getSpecularColor().abs() > 0.1 && bounces) {
            Hit h2;
            Vector3f E = r.getDirection();
            Vector3f R = E - 2 * h.normal * Vector3f::dot(h.normal, E);
            Ray r2(P + 0.01f * R, R);
            col += h.getMaterial()->getSpecularColor() *  traceRay(r2, tmin, bounces - 1, h2);
        }
        col += _scene.getAmbientLight() * h.getMaterial()->getDiffuseColor();
        return col;
    }
    else {
        return _scene.getBackgroundColor(r.getDirection());
    }
    // END SOLN 
}

