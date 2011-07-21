#ifndef RAY_TRACE_HPP
#define RAY_TRACE_HPP

#include "element_collection.hpp"
#include "helper.hpp"
#include "porting.hpp"

#define LIGHT_NUM (1)
#define OBJECT_NUM (1+13)
#define REFLECT_NUM (5)

namespace gtc
{

typedef RGBA8U RGBA;
class Primitive;

static int frame_count = 0;

struct Ray
{
    Coord origin;
    Vector direction;
    float strong;

    FUNC_DECL
    Ray(void)
        : origin(), direction(), strong(0.0f) {}
    
    FUNC_DECL
    Ray(const Coord& _origin, const Vector& _direction, float _strong)
        : origin(_origin), direction(_direction.normalize()), strong(_strong) {}

    FUNC_DECL
    bool operator==(const Ray & rhs)
    {
        if (this->origin == rhs.origin &&
            this->direction == rhs.direction &&
            this->strong == rhs.strong)
        {
            return true;
        }
        else
        {
            return false;
        }
    }
};

struct Intersect
{
    const Primitive * primitive;
    float reflection;
    float distance;
    Coord coord;
    Vector normal;
    Ray ray;
   
    FUNC_DECL
    Intersect(void)
        : primitive(NULL), reflection(0.0f), distance(helper::make_inf()), coord(), normal(), ray() {}
    
    FUNC_DECL
    Intersect(const Primitive * _primitive, float _reflection, float _distance)
        : primitive(_primitive), reflection(_reflection), distance(_distance), coord(), normal(), ray() {}
    
    FUNC_DECL
    Intersect(const Primitive * _primitive, float _reflection, float _distance,
              const Coord & _coord, const Vector & _normal, const Ray & _ray)
        : primitive(_primitive), reflection(_reflection), distance(_distance), coord(_coord), normal(_normal), ray(_ray) {}

    FUNC_DECL
    bool operator==(const Intersect & rhs)
    {
        if (this->primitive == rhs.primitive &&
            this->reflection == rhs.reflection && 
            this->distance == rhs.distance &&
            this->coord == rhs.coord &&
            this->normal == rhs.normal &&
            this->ray == rhs.ray)
        {
            return true;
        }
        else
        {
            return false;
        }
    }
};

struct Material
{
    RGBA color;
    float reflection;

    FUNC_DECL
    Material(void)
        : color(), reflection(0) {}
    
    FUNC_DECL
    Material(const RGBA & _color, float _reflection)
        : color(_color), reflection(_reflection) {}
};

class Primitive
{
protected:
    Material material_;
    const RGBA invisible_color_;

public:    
    FUNC_DECL
    Primitive(void)
        : material_(), invisible_color_(RGBA(0, 0, 0)) {}

    FUNC_DECL
    Primitive(const Material& material)
        : material_(material), invisible_color_(RGBA(0, 0, 0)) {}

    /* calculate intersect between ray and itself */
    FUNC_DECL 
    virtual Intersect intersect(const Ray& ray) const = 0;
    
    /* calculate shading at intersect coord */
    FUNC_DECL 
    virtual RGBA shading(Coord * lights, unsigned int light_num,
                          Primitive ** primitives, unsigned int primitive_num,
                          const Intersect & isect) const = 0;
};

class BackGround : public Primitive
{
public:
    FUNC_DECL
    BackGround(void)
        : Primitive() {}
    
    FUNC_DECL 
    virtual Intersect intersect(const Ray& ray) const
    {
        return Intersect(this, 0.0f, helper::make_max<float>());
    }
    
    FUNC_DECL 
    virtual RGBA shading(Coord * lights, unsigned int light_num,
                          Primitive ** primitives, unsigned int primitive_num,
                          const Intersect & isect) const
    {
        return invisible_color_;
    }
};

class Triangle : public Primitive
{
private:
    Coord v0_;
    Coord v1_;
    Coord v2_;

    Vector normal_;

    FUNC_DECL
    Vector calc_normal()
    {
        Vector v01 = v1_-v0_;
        Vector v02 = v2_-v0_;
        return v02.cross(v01).normalize();
    }

public:
    FUNC_DECL
    Triangle(void)
        : Primitive(), v0_(), v1_(), v2_() {}

    FUNC_DECL
    Triangle(const Coord & v0,
             const Coord & v1, 
             const Coord & v2)
           : Primitive(), v0_(v0), v1_(v1), v2_(v2)
    {
        normal_ = calc_normal();
    }

    FUNC_DECL
    Triangle(const Material & material,
             const Coord & v0,
             const Coord & v1, 
             const Coord & v2)
           : Primitive(material), v0_(v0), v1_(v1), v2_(v2)
    {
        normal_ = calc_normal();
    }
    
    FUNC_DECL 
    virtual Intersect intersect(const Ray& ray) const
    {
        float nume = (v0_ - ray.origin).dot(normal_);
        float deno = ray.direction.dot(normal_);

        if (-0.0001f <= deno)
        {
            return Intersect();
        }
        
        float t = nume / deno;

        if (t < 0)
        {
            return Intersect();
        }
        
        Coord p = ray.origin + ray.direction * t;
        
        Vector d0p = p - v0_;
        Vector d01 = v1_ - v0_;
        if (d0p.cross(d01).dot(normal_) < 0)
        {
            return Intersect();
        }
        
        Vector d1p = p - v1_;
        Vector d12 = v2_ - v1_;
        if (d1p.cross(d12).dot(normal_) < 0)
        {
            return Intersect();
        }

        Vector d2p = p - v2_;
        Vector d20 = v0_ - v2_;
        if (d2p.cross(d20).dot(normal_) < 0)
        {
            return Intersect();
        }
        
        float reflet = 2.0f * (ray.direction.dot(normal_));
        Ray new_ray(p, ray.direction - (normal_ * reflet), ray.strong * material_.reflection);
        
        return Intersect(this, ray.strong, t, p, normal_, new_ray);
    }

    FUNC_DECL 
    virtual RGBA shading(Coord * lights, unsigned int light_num,
                         Primitive ** primitives, unsigned int primitive_num,
                         const Intersect & isect) const
    { 
        RGBA pixel = material_.color;

        for (unsigned int i=0; i<light_num; ++i)
        {
            const Coord & light = lights[i];
            
            if (isect.normal.dot(light-isect.coord) <= 0.0f)
            {
                return Primitive::invisible_color_;
            }
 
            Ray ray(light, isect.coord - light, 1.0);

            Intersect my_isect = this->intersect(ray);

            for (unsigned int j=0; j<primitive_num; ++j)
            {
                Intersect other_isect;
                const Primitive * primitive = primitives[j];

                if (NULL == primitive)
                {
                    continue;
                }

                other_isect = primitive->intersect(ray);
                
                if (NULL != other_isect.primitive && 
                    other_isect.distance < my_isect.distance)
                {
                    return Primitive::invisible_color_;
                }
            }
            
            float lambert = fabs(ray.direction.dot(isect.normal));
            pixel = material_.color * lambert * isect.reflection;
        }

        return pixel;
    }
};

class Sphere : public Primitive
{
private:
    Coord center_;
    float radius_;

public:
    FUNC_DECL
    Sphere(void)
        : Primitive(), center_(), radius_(0) {}
    
    FUNC_DECL
    Sphere(const Coord & center,
           float radius)
        : Primitive(), center_(center), radius_(radius) {}
    
    FUNC_DECL
    Sphere(const Material & material,
           const Coord & center,
           float radius)
        : Primitive(material), center_(center), radius_(radius) {}

    FUNC_DECL
    virtual Intersect intersect(const Ray& ray) const
    {
        float a = ray.direction.self_dot();
        float b = ray.direction.dot((ray.origin - center_));
        float c = (ray.origin - center_).self_dot() - (radius_*radius_);

        float d = b * b - a * c;

        if (0 > d) 
        {
            return Intersect();
        }

        float tn = -b - sqrtf(d);
        float tp = -b + sqrtf(d);

        if (tn < 0 && tp < 0)
        {
            return Intersect();
        }

        float t = MIN(tn, tp);
        Coord p = ray.origin + ray.direction * t;
        Vector n = (p - center_).normalize();
        
        float reflet = 2.0f * (ray.direction.dot(n));
        Ray new_ray(p, ray.direction - (n * reflet), ray.strong * material_.reflection);
        
        return Intersect(this, ray.strong, t, p, n, new_ray);
    }
    
    FUNC_DECL 
    virtual RGBA shading(Coord * lights, unsigned int light_num,
                         Primitive ** primitives, unsigned int primitive_num,
                         const Intersect & isect) const
    {
        RGBA pixel = material_.color;

        for (unsigned int i=0; i<light_num; ++i)
        {
            const Coord & light = lights[i];

            if (isect.normal.dot(light-isect.coord) <= 0.0f)
            {
                return Primitive::invisible_color_;
            }

            Ray ray(light, isect.coord - light, 1.0);

            Intersect my_isect = this->intersect(ray);

            for (unsigned int j=0; j<primitive_num; ++j)
            {
                Intersect other_isect;
                const Primitive * primitive = primitives[j];

                if (NULL == primitive)
                {
                    continue;
                }

                other_isect = primitive->intersect(ray);
                
                if (NULL != other_isect.primitive && 
                    other_isect.distance < my_isect.distance)
                {
                    return Primitive::invisible_color_;
                }
            }

            float lambert = fabs(ray.direction.dot(isect.normal));
            pixel = material_.color * lambert * isect.reflection;
        }

        return pixel;
    }
};

class Scene
{
private:
    int width_;
    int height_;
    
    Coord view_point_;
    Coord screen_;
    float screen_width_;
    float screen_height_;
    
    Coord lights_[LIGHT_NUM];

    Primitive * primitives_[OBJECT_NUM];
   
public:
    
    FUNC_DECL
    Scene(int width, int height)
        : width_(width), height_(height), 
          view_point_(0, 0, 1.0)
    {
        screen_ = Coord(-1.0, -1.0, 0.0);
        screen_width_ = 2.0;
        screen_height_ = 2.0;
                
        lights_[0] = Coord(0.0, 5.0, 1.0);
        
        primitives_[0] = new BackGround();
#if 1
        primitives_[1] = new Sphere(Material(RGBA(255, 255, 0), 0.5f), 
                              Coord(-0.7f, -0.6f, -3.0f), 0.7f);

        primitives_[2] = new Sphere(Material(RGBA(0, 255, 255), 0.2f), 
                              Coord(+0.7f, -0.6f, -3.0f), 0.7f);
        
        primitives_[3] = new Sphere(Material(RGBA(255, 255, 255), 1.0f), 
                              Coord(+0.0f, +0.6f, -3.0f), 0.7f);

        /* Floor */
        primitives_[4] = new Triangle(Material(RGBA(128, 128, 128), 0.2f),
                                Coord(-4.0f, -4.0f, +8.0f), 
                                Coord(-4.0f, -4.0f, -8.0f),
                                Coord(+4.0f, -4.0f, +8.0f));

        primitives_[5] = new Triangle(Material(RGBA(128, 128, 128), 0.2f),
                                Coord(-4.0f, -4.0f, -8.0f), 
                                Coord(+4.0f, -4.0f, -8.0f),
                                Coord(+4.0f, -4.0f, +8.0f));

        /* Loof */
        primitives_[6] = new Triangle(Material(RGBA(64, 64, 64), 0.2f),
                                Coord(-4.0f, +4.0f, -8.0f), 
                                Coord(-4.0f, +4.0f, +8.0f),
                                Coord(+4.0f, +4.0f, +8.0f));

        primitives_[7] = new Triangle(Material(RGBA(64, 64, 64), 0.2f),
                                Coord(+4.0f, +4.0f, +8.0f), 
                                Coord(+4.0f, +4.0f, -8.0f),
                                Coord(-4.0f, +4.0f, +8.0f));
 
        /* Front Wall */
        primitives_[8] = new Triangle(Material(RGBA(255, 255, 255), 0.2f),
                                Coord(-4.0f, -4.0f, -8.0f), 
                                Coord(-4.0f, +4.0f, -8.0f),
                                Coord(+4.0f, -4.0f, -8.0f));

        primitives_[9] = new Triangle(Material(RGBA(255, 255, 255), 0.2f),
                                Coord(+4.0f, +4.0f, -8.0f), 
                                Coord(+4.0f, -4.0f, -8.0f),
                                Coord(-4.0f, +4.0f, -8.0f));

        /* Left Wall */
        primitives_[10] = new Triangle(Material(RGBA(255, 0, 0), 0.2f),
                                 Coord(-4.0f, -4.0f, +8.0f), 
                                 Coord(-4.0f, +4.0f, +8.0f),
                                 Coord(-4.0f, -4.0f, -8.0f));

        primitives_[11] = new Triangle(Material(RGBA(255, 0, 0), 0.2f),
                                 Coord(-4.0f, +4.0f, -8.0f), 
                                 Coord(-4.0f, -4.0f, -8.0f),
                                 Coord(-4.0f, +4.0f, +8.0f));
        
        /* Right Wall */
        primitives_[12] = new Triangle(Material(RGBA(0, 255, 0), 0.2f),
                                Coord(+4.0f, -4.0f, -8.0f), 
                                Coord(+4.0f, +4.0f, -8.0f),
                                Coord(+4.0f, -4.0f, +8.0f));

        primitives_[13] = new Triangle(Material(RGBA(0, 255, 0), 0.2f),
                                Coord(+4.0f, +4.0f, +8.0f), 
                                Coord(+4.0f, -4.0f, +8.0f),
                                Coord(+4.0f, +4.0f, -8.0f));


#elif 0
        primitives_[1] = new Triangle(Material(RGBA(255, 255, 255), 1.0), 
                                Coord(-20.0, -20.0, -1.0),
                                Coord(+20.0, -20.0, -1.0),
                                Coord(+20.0, +20.0, -1.0));
        primitives_[2] = new Triangle(Material(RGBA(255, 255, 255), 1.0), 
                                Coord(-20.0, -20.0, -1.0),
                                Coord(-20.0, +20.0, -1.0),
                                Coord(+20.0, +20.0, -1.0));
        primitives_[3] = NULL;
        primitives_[4] = NULL;
        primitives_[5] = NULL;
#else
        primitives_[1] = new Sphere(Material(RGBA(255, 255, 0), 0.5), 
                              Coord(0.0, 0.0, -3.0), 1.0);
        primitives_[2] = NULL;
        primitives_[3] = NULL;
        primitives_[4] = NULL;
        primitives_[5] = NULL;
#endif
    
    }

    FUNC_DECL
    ~Scene()
    {
        for (int i=0; i<OBJECT_NUM; ++i)
        {
            delete primitives_[i];
        }
    }

    FUNC_DECL
    RGBA8U render(int x, int y)
    {

        Coord screen_coord = screen_+
            Coord(screen_width_*static_cast<float>(x)/static_cast<float>(width_),
                  screen_height_*static_cast<float>(y)/static_cast<float>(height_),
                  0.0);

        Vector direction = screen_coord - view_point_;
        Ray ray(view_point_, direction, 1.0);
        Intersect isects[REFLECT_NUM];

        RGBA8U pixel;
       
        unsigned int reflect_count = 0;

        do
        {
            for (unsigned int i=0; i<OBJECT_NUM; ++i)
            {
                Intersect isect;

                if (NULL == primitives_[i])
                {
                    continue;
                }

                if (0 < reflect_count && isects[reflect_count-1].primitive == primitives_[i])
                {
                    continue;
                }

                isect = primitives_[i]->intersect(ray);
                
                if (NULL != isect.primitive)
                {
                    if (isect.distance < isects[reflect_count].distance)
                    {
                        isects[reflect_count] = isect;
                    }
                }
            }

            ray = isects[reflect_count].ray;
                       
            reflect_count++;

        } while (NULL != isects[reflect_count-1].primitive && 
                 0.0f < isects[reflect_count-1].reflection &&
                 reflect_count < REFLECT_NUM);
       
        
        for (unsigned int i=0; i<reflect_count; ++i)
        {
            if (NULL != isects[i].primitive)
            {
                pixel = pixel.add_sat(isects[i].primitive->shading(&lights_[0], 
                                                             LIGHT_NUM, 
                                                             &primitives_[0], 
                                                             OBJECT_NUM, 
                                                             isects[i]));
            }
        }
       
        return pixel;
    }

    FUNC_DECL
    void displace_view(const Vector& displacement)
    {
        view_point_ = view_point_ + displacement;
    }

    FUNC_DECL
    void displace_light(const Vector& displacement)
    {
        lights_[0] = lights_[0] + displacement;
    }

};

} /* namespace gtc */

#endif  /* RAY_TRACe_HPP */
