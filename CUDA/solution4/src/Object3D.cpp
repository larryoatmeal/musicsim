#include "Object3D.h"

bool Sphere::intersect(const Ray &r, float tmin, Hit &h) const
{
    // BEGIN WE PROVIDE ?

    // Locate intersection point ( 2 pts )
    const Vector3f &rayOrigin = r.getOrigin(); //Ray origin in the world coordinate
    const Vector3f &dir = r.getDirection();

    Vector3f origin = rayOrigin - _center;      //Ray origin in the sphere coordinate

    float a = dir.absSquared();
    float b = 2 * Vector3f::dot(dir, origin);
    float c = origin.absSquared() - _radius * _radius;

    // no intersection
    if (b * b - 4 * a * c < 0) {
        return false;
    }

    float d = sqrt(b * b - 4 * a * c);

    float tplus = (-b + d) / (2.0f*a);
    float tminus = (-b - d) / (2.0f*a);

    // the two intersections are at the camera back
    if ((tplus < tmin) && (tminus < tmin)) {
        return false;
    }

    float t = 10000;
    // the two intersections are at the camera front
    if (tminus > tmin) {
        t = tminus;
    }

    // one intersection at the front. one at the back 
    if ((tplus > tmin) && (tminus < tmin)) {
        t = tplus;
    }

    if (t < h.getT()) {
        Vector3f normal = r.pointAtParameter(t) - _center;
        normal = normal.normalized();
        h.set(t, this->material, normal);
        return true;
    }
    // END WE PROVIDE
    return false;
}

// Add object to group
void Group::addObject(Object3D *obj) {
    // BEGIN SOLN
    m_members.push_back(obj);
    // END SOLN
}

// Return number of objects in group
int Group::getGroupSize() const {
    // BEGIN SOLN
    return (int)m_members.size();
    // END SOLN
}

bool Group::intersect(const Ray &r, float tmin, Hit &h) const
{
    // TODO: implement and fill hit
    bool hit = false;
    // BEGIN SOLN
    for (Object3D* o : m_members) {
        if (o->intersect(r, tmin, h)) {
            hit = true;
        }
    }
    // END SOLN
    return hit;
}


Plane::Plane(const Vector3f &normal, float d, Material *m) : Object3D(m) {
    // BEGIN SOLN
    p[0] = normal[0];
    p[1] = normal[1];
    p[2] = normal[2];
    p[3] = d;
    // END SOLN
}
bool Plane::intersect(const Ray &r, float tmin, Hit &h) const
{
    // return false;  // TODO: implement
    // BEGIN SOLN
    float t = (p[3] - Vector3f::dot(r.getOrigin(), p.xyz())) / Vector3f::dot(r.getDirection(), p.xyz());
    if (t < tmin) {
        return false;
    }
    if (t < h.getT()) {
        h.set(t, material, p.xyz());
        return true;
    } else {
        return false;
    }
    // END SOLN
}
bool Triangle::intersect(const Ray &r, float tmin, Hit &h) const 
{
    // BEGIN SOLN
    Vector3f O = _v[0];
    Vector3f v1 = _v[1] - _v[0];
    Vector3f v2 = _v[2] - _v[0];
    Vector3f n = Vector3f::cross(v1, v2).normalized();

    Hit th = h;

    Plane pl(n, Vector3f::dot(O, n), material);
    if (!pl.intersect(r, tmin, th)) {
        return false;
    } 
    Vector3f b = r.pointAtParameter(th.t) - O;

    // Ax = b
    Matrix3f A(v1, v2, n);
    Vector3f x = A.inverse() * b;
    if (fabsf(x[2]) > 1e-3   || 
        x[0] < 0 || x[0] > 1 ||
        x[1] < 0 || x[1] > 1 ||
        x[0] + x[1] > 1 ||
        h.t < th.t /*||
        Vector3f::dot(n, r.getDirection()) > 0 // back facing*/
        ) {
        return false;
    } else {
        h = th;
        float A = 1 -x[0] - x[1];
        float B = x[0];
        float C = x[1];
        h.normal = A * _normals[0] + B * _normals[1] + C * _normals[2];
        h.normal.normalize();
        return true;
    }
     // END SOLN
}


Transform::Transform(const Matrix4f &m,
    Object3D *obj) : _object(obj) {
    // BEGIN SOLN
    M = m.inverse();
    Mit = m.getSubmatrix3x3(0, 0).inverse().transposed();
    // END SOLN
}
bool Transform::intersect(const Ray &r, float tmin, Hit &h) const
{
    // return _object->intersect(r, tmin, h);  // TODO: implement correctly
    // BEGIN SOLN 
    Vector4f O = M * Vector4f(r.getOrigin(), 1);
    O = O / O.w();
    Vector3f D = M.getSubmatrix3x3(0, 0) * r.getDirection();
    Ray rlocal(O.xyz(), D);
    bool ret = _object->intersect(rlocal, tmin, h);
    if (ret) {
        h.normal = Mit * h.normal;
        h.normal.normalize();
    }
    return ret;
    // END SOLN
}