#ifndef HELPER_HPP
#define HELPER_HPP

#define DEF_TYPE(sign_type, value_type) \
    typedef sign_type SignType; \
    typedef value_type ValueType; 

#include "porting.hpp"

namespace gtc
{
    namespace helper
    {
        struct Signed;
        struct Unsigned;
        struct Integer;
        struct Floating;
        
        template<typename T>
        struct TypeClass
        {
        };
    
        template<>
        struct TypeClass<signed char>
        {
            DEF_TYPE(Signed, Integer)
        };

        template<>
        struct TypeClass<signed short>
        {
            DEF_TYPE(Signed, Integer)
        };

        template<>
        struct TypeClass<signed int>
        {
            DEF_TYPE(Signed, Integer)
        };

        template<>
        struct TypeClass<signed long long>
        {
            DEF_TYPE(Signed, Integer)
        };
        
        template<>
        struct TypeClass<unsigned char>
        {
            DEF_TYPE(Unsigned, Integer)
        };

        template<>
        struct TypeClass<unsigned short>
        {
            DEF_TYPE(Unsigned, Integer)
        };

        template<>
        struct TypeClass<unsigned int>
        {
            DEF_TYPE(Unsigned, Integer)
        };

        template<>
        struct TypeClass<unsigned long long>
        {
            DEF_TYPE(Unsigned, Integer)
        };
        
        template<>
        struct TypeClass<float>
        {
            DEF_TYPE(Signed, Floating)
        };

        template<>
        struct TypeClass<double>
        {
            DEF_TYPE(Signed, Floating)
        };

        FUNC_DECL
        float make_inf(void)
        {
            union {
                float f;
                unsigned int i;
            } u;
            u.i = 0x7f800000;
            return u.f;
        }

        template<typename ST>
        struct MakeMax
        {
        };
         
        template<>
        struct MakeMax<Unsigned>
        {
            template<typename T>
            static
            FUNC_DECL
            T op()
            {
                T max = static_cast<T>(-1);
                return max;
            }
        };

        template<>
        struct MakeMax<Signed>
        {
            template<typename T>
            static 
            FUNC_DECL
            T op()
            {
                T max = static_cast<T>(~(1<<(sizeof(T)*8-1)));
                return max;
            }
        };
        
        template<typename T>
        FUNC_DECL
        T make_max(void)
        {
            return MakeMax<typename TypeClass<T>::SignType>::template op<T>();
        }
               
        template<>
        FUNC_DECL
        float make_max<float>(void)
        {
            union {
                float f;
                unsigned int i;
            } u;
            u.i = 0x7f7fffff;
            return u.f;
        }
 
        template<>
        FUNC_DECL
        double make_max<double>(void)
        {
            union {
                double f;
                unsigned long long i;
            } u;
            u.i = 0x7fefffffffffffff;
            return u.f;
        }

        template<typename T>
        FUNC_DECL
        T make_min(void)
        {
            return ~make_max<T>();
        }
        
        template<>
        FUNC_DECL
        float make_min<float>(void)
        {
            union {
                float f;
                unsigned int i;
            } u;
            u.i = 0x8f7fffff;
            return u.f;
        }
 
        template<>
        FUNC_DECL
        double make_min<double>(void)
        {
            union {
                double f;
                unsigned long long i;
            } u;
            u.i = 0x8fefffffffffffff;
            return u.f;
        }

        template<typename ST>
        struct AddSat
        {
        };

        template<>
        struct AddSat<Unsigned>
        {
            template<typename T>
            static 
            FUNC_DECL
            T op(T val1, T val2)
            {
                T result = val1 + val2;
                if (result < val1)
                {
                    return make_max<T>();
                }
                else
                {
                    return result;
                }
            }
        };

        template<>
        struct AddSat<Signed>
        {
            template<typename T>
            static 
            FUNC_DECL
            T op(T val1, T val2)
            {
                T result = val1 + val2;
                if (static_cast<T>(0) < val1 && static_cast<T>(0) < val2)
                {
                    return (result < val1) ? make_max<T>() : result;
                }
                else if (val1 < static_cast<T>(0) && val2 < static_cast<T>(0))
                {
                    return (val1 < result) ? make_min<T>() : result;
                }
                else
                {
                    return result;
                }
            }
        };

        template<typename T>
        FUNC_DECL
        T add_sat(T val1, T val2)
        {
            return AddSat<typename TypeClass<T>::SignType>::op(val1, val2);
        }

        template<typename T>
        FUNC_DECL
        T mul_sat(T val1, float val2)
        {
            float result = static_cast<float>(val1) * val2;
            T ret;

            if (static_cast<float>(make_max<T>()) < result)
            {
                ret = make_max<T>();
            }
            else
            {
                ret = static_cast<T>(result);
            }
            
            return ret;
        }

    } /* namespace helper */

} /* namespace gtc */

#endif /* HELPER_HPP */
