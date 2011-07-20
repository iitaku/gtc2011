#ifndef ELEMENT_COLLECTION_HPP
#define ELEMENT_COLLECTION_HPP

#include <cmath>

#include "porting.hpp"
#include "helper.hpp"

#define SQ(val) (val*val)
#define MAX(val1, val2) (val1<val2 ? val2 : val1)
#define MIN(val1, val2) (val1<val2 ? val1 : val2)

namespace gtc
{

    template<typename T>
    class ThreeElement
    {
        private:
            T v1_;
            T v2_;
            T v3_;

        public:
            FUNC_DECL
            ThreeElement(void)
                : v1_(0), v2_(0), v3_(0)
            {}

            FUNC_DECL
            ThreeElement(T v1, T v2, T v3)
                : v1_(v1), v2_(v2), v3_(v3)
            {}

            FUNC_DECL
            ThreeElement(const ThreeElement<T>& other)
                : v1_(other.v1_), v2_(other.v2_), v3_(other.v3_)
            {} 

            FUNC_DECL
            T V1(void) const { return this->v1_; }
            FUNC_DECL
            T V2(void) const { return this->v2_; }
            FUNC_DECL
            T V3(void) const { return this->v3_; }

            /* compare */
            FUNC_DECL
            bool operator==(const ThreeElement<T> & rhs) const
            {
                if (this->v1_ == rhs.v1_ &&
                    this->v2_ == rhs.v2_ &&
                    this->v3_ == rhs.v3_ )
                {
                    return true;
                }
                else
                {
                    return false;
                }
            }

            /* substitute */
            template<typename S>
            FUNC_DECL
            ThreeElement<T>& operator=(const ThreeElement<S>& rhs) 
            {
                this->v1_ = static_cast<S>(rhs.V1());
                this->v2_ = static_cast<S>(rhs.V2());
                this->v3_ = static_cast<S>(rhs.V3());
                return *this;
            }
            
            /* add substitute */
            FUNC_DECL
            ThreeElement<T>& operator+=(const ThreeElement<T>& rhs) 
            {
                this->v1_ += rhs.V1();
                this->v2_ += rhs.V2();
                this->v3_ += rhs.V3();
                return *this;
            }
            
            /* add */
            FUNC_DECL
            ThreeElement<T> operator+(const ThreeElement<T>& rhs) const
            {
                return ThreeElement<T>(this->v1_+rhs.v1_,
                                       this->v2_+rhs.v2_,
                                       this->v3_+rhs.v3_);
            }

            template<typename S>
            FUNC_DECL
            ThreeElement<T> operator+(S s) const
            {
                return ThreeElement<T>(static_cast<T>(static_cast<S>(this->v1_)+s),
                                       static_cast<T>(static_cast<S>(this->v2_)+s),
                                       static_cast<T>(static_cast<S>(this->v3_)+s));
            }

            /* sub */
            FUNC_DECL
            ThreeElement<T> operator-(const ThreeElement<T>& rhs) const
            {
                return ThreeElement<T>(this->v1_-rhs.v1_,
                                       this->v2_-rhs.v2_,
                                       this->v3_-rhs.v3_);
            }

            template<typename S>
            FUNC_DECL
            ThreeElement<T> operator-(S s) const
            {
                return ThreeElement<T>(static_cast<T>(static_cast<S>(this->v1_)-s), 
                                       static_cast<T>(static_cast<S>(this->v2_)-s), 
                                       static_cast<T>(static_cast<S>(this->v3_)-s));
            }

            /* mul */
            FUNC_DECL
            ThreeElement<T> operator*(const ThreeElement<T>& rhs) const
            {
                return ThreeElement<T>(this->v1_*rhs.v1_, 
                                       this->v2_*rhs.v2_, 
                                       this->v3_*rhs.v3_);
            }

            template<typename S>
            FUNC_DECL
            ThreeElement<T> operator*(S s) const
            {
                return ThreeElement<T>(static_cast<T>(static_cast<S>(this->v1_)*s), 
                                       static_cast<T>(static_cast<S>(this->v2_)*s), 
                                       static_cast<T>(static_cast<S>(this->v3_)*s));
            }
            
            /* saturating add */
            FUNC_DECL
            ThreeElement<T> add_sat(const ThreeElement<T>& rhs) const
            {
                return ThreeElement<T>(helper::add_sat<T>(this->v1_, rhs.v1_),
                                       helper::add_sat<T>(this->v2_, rhs.v2_),
                                       helper::add_sat<T>(this->v3_, rhs.v3_));
            }
            
            /* dot product */
            FUNC_DECL
            float dot(const ThreeElement<T>& rhs) const
            {
                return (this->v1_*rhs.v1_+
                        this->v2_*rhs.v2_+
                        this->v3_*rhs.v3_);
            }

            FUNC_DECL
            float self_dot(void) const
            {
                return (this->v1_*this->v1_+
                        this->v2_*this->v2_+
                        this->v3_*this->v3_);
            }

            /* exterior product */
            FUNC_DECL
            ThreeElement<T> cross(const ThreeElement<T>& rhs) const
            {
                return ThreeElement<T>(this->v2_*rhs.v3_-this->v3_*rhs.v2_,
                                       this->v3_*rhs.v1_-this->v1_*rhs.v3_,
                                       this->v1_*rhs.v2_-this->v2_*rhs.v1_);
            }


            /* normalize  */
            FUNC_DECL
            ThreeElement<T> normalize(void) const
            {
                float d = sqrt(SQ(this->v1_)+
                               SQ(this->v2_)+
                               SQ(this->v3_));

                return ThreeElement<T>(this->v1_/d, 
                                       this->v2_/d, 
                                       this->v3_/d);
            }
     };


    template<typename T>
    class FourElement
    {
        private:
            T v1_;
            T v2_;
            T v3_;
            T v4_;

        public:
            FUNC_DECL
            FourElement(void)
                : v1_(0), v2_(0), v3_(0), v4_(0)
            {}

            FUNC_DECL
            FourElement(T v1, T v2, T v3)
                : v1_(v1), v2_(v2), v3_(v3), v4_(0)
            {}

            FUNC_DECL
            FourElement(T v1, T v2, T v3, T v4)
                : v1_(v1), v2_(v2), v3_(v3), v4_(v4)
            {}

            FUNC_DECL
            FourElement(const FourElement<T>& other)
                : v1_(other.v1_), v2_(other.v2_), v3_(other.v3_), v4_(other.v4_)
            {} 
            
            FUNC_DECL
            T V1(void) const { return this->v1_; }
            FUNC_DECL
            T V2(void) const { return this->v2_; }
            FUNC_DECL
            T V3(void) const { return this->v3_; }
            FUNC_DECL
            T V4(void) const { return this->v4_; }
 
            /* compare */
            FUNC_DECL
            bool operator==(const FourElement<T> & rhs) const
            {
                if (this->v1_ == rhs.v1_ &&
                    this->v2_ == rhs.v2_ &&
                    this->v3_ == rhs.v3_ &&
                    this->v4_ == rhs.v4_ )
                {
                    return true;
                }
                else
                {
                    return false;
                }
            }
            
            /* substitute */
            template<typename S>
            FUNC_DECL
            FourElement<T>& operator=(const FourElement<S>& rhs) 
            {
                this->v1_ = static_cast<T>(rhs.V1());
                this->v2_ = static_cast<T>(rhs.V2());
                this->v3_ = static_cast<T>(rhs.V3());
                this->v4_ = static_cast<T>(rhs.V4());
                return *this;
            }

            /* add substitute */
            FUNC_DECL
            FourElement<T>& operator+=(const FourElement<T>& rhs) 
            {
                this->v1_ += rhs.V1();
                this->v2_ += rhs.V2();
                this->v3_ += rhs.V3();
                this->v3_ += rhs.V4();
                return *this;
            }


            /* add */
            FUNC_DECL
            FourElement<T> operator+(const FourElement<T>& rhs) const
            {
                return FourElement<T>(this->v1_+rhs.v1_, 
                                      this->v2_+rhs.v2_, 
                                      this->v3_+rhs.v3_,
                                      this->v4_+rhs.v4_);
            }

            template<typename S>
            FUNC_DECL
            FourElement<T> operator+(S s) const
            {
                return FourElement<T>(static_cast<T>(static_cast<S>(this->v1_)+s), 
                                      static_cast<T>(static_cast<S>(this->v2_)+s), 
                                      static_cast<T>(static_cast<S>(this->v3_)+s), 
                                      static_cast<T>(static_cast<S>(this->v4_)+s));
            }

            /* sub */
            FUNC_DECL
            FourElement<T> operator-(const FourElement<T>& rhs) const
            {
                return FourElement<T>(this->v1_-rhs.v1_, 
                                      this->v2_-rhs.v2_, 
                                      this->v3_-rhs.v3_, 
                                      this->v4_-rhs.v4_);
            }

            template<typename S>
            FUNC_DECL
            FourElement<T> operator-(S s) const
            {
                return FourElement<T>(static_cast<T>(static_cast<S>(this->v1_)-s), 
                                      static_cast<T>(static_cast<S>(this->v2_)-s), 
                                      static_cast<T>(static_cast<S>(this->v3_)-s), 
                                      static_cast<T>(static_cast<S>(this->v4_)-s));
            }

            /* mul */
            FUNC_DECL
            FourElement<T> operator*(const FourElement<T>& rhs) const
            {
                return FourElement<T>(this->v1_*rhs.v1_, 
                                      this->v2_*rhs.v2_, 
                                      this->v3_*rhs.v3_,
                                      this->v4_*rhs.v4_);
            }

            template<typename S>
            FUNC_DECL
            FourElement<T> operator*(S s) const
            {
                return FourElement<T>(static_cast<T>(static_cast<S>(this->v1_)*s), 
                                      static_cast<T>(static_cast<S>(this->v2_)*s), 
                                      static_cast<T>(static_cast<S>(this->v3_)*s),
                                      static_cast<T>(static_cast<S>(this->v4_)*s));
            }
 
            /* saturating add */
            FUNC_DECL
            FourElement<T> add_sat(const FourElement<T>& rhs) const
            {
                return FourElement<T>(helper::add_sat<T>(this->v1_, rhs.v1_),
                                      helper::add_sat<T>(this->v2_, rhs.v2_),
                                      helper::add_sat<T>(this->v3_, rhs.v3_),
                                      helper::add_sat<T>(this->v4_, rhs.v4_));
            }

            /* dot product */
            FUNC_DECL
            float dot(const FourElement<T>& rhs) const
            {
                return (this->v1_*rhs.v1_+
                        this->v2_*rhs.v2_+
                        this->v3_*rhs.v3_+
                        this->v4_*rhs.v4_);
            }

            FUNC_DECL
            float self_dot(void) const
            {
                return (this->v1_*this->v1_+
                        this->v2_*this->v2_+
                        this->v3_*this->v3_+
                        this->v4_*this->v4_);
            }

            /* normalize */
            FUNC_DECL
            FourElement<T> normalize (void) const
            {
                float d = sqrt(SQ(this->v1_)+
                               SQ(this->v2_)+
                               SQ(this->v3_)+
                               SQ(this->v4_));

                return FourElement<T>(this->v1_/d, 
                                      this->v2_/d, 
                                      this->v3_/d, 
                                      this->v4_/d);
            }
     };

    typedef ThreeElement<float> Coord;
    typedef ThreeElement<float> Vector;
    typedef FourElement<unsigned char> RGBA8U;
    typedef FourElement<unsigned short> RGBA16U;
    typedef FourElement<float> RGBA32F;
    typedef FourElement<double> RGBA64F;

} /* namespace gtc */

#endif /* ELEMENT_COLLECTION_HPP */
