#ifndef MD5_H
#define MD5_H

#include <arm_neon.h>  // 引入NEON指令集支持
#include <cstdint>
#include <string>

using namespace std;

typedef uint8_t Byte;
typedef uint32_t bit32;

// MD5的一系列参数
#define s11 7
#define s12 12
#define s13 17
#define s14 22
#define s21 5
#define s22 9
#define s23 14
#define s24 20
#define s31 4
#define s32 11
#define s33 16
#define s34 23
#define s41 6
#define s42 10
#define s43 15
#define s44 21

// SIMD优化后的MD5函数
inline void FF_SIMD(uint32x4_t &va, uint32x4_t vb, uint32x4_t vc, uint32x4_t vd, uint32x4_t vx, int s, uint32x4_t vac) {
    va = vaddq_u32(va, vbslq_u32(vandq_u32(vb, vc), vb, vandq_u32(vmvnq_u32(va), vd)));
    va = vaddq_u32(va, vaddq_u32(vx, vac));
    va = vorrq_u32(vshlq_n_u32(va, s), vshrq_n_u32(va, 32 - s));
    va = vaddq_u32(va, vb);
}

inline void GG_SIMD(uint32x4_t &va, uint32x4_t vb, uint32x4_t vc, uint32x4_t vd, uint32x4_t vx, int s, uint32x4_t vac) {
    va = vaddq_u32(va, vbslq_u32(vandq_u32(vb, vd), vb, vandq_u32(vc, vmvnq_u32(va))));
    va = vaddq_u32(va, vaddq_u32(vx, vac));
    va = vorrq_u32(vshlq_n_u32(va, s), vshrq_n_u32(va, 32 - s));
    va = vaddq_u32(va, vb);
}

inline void HH_SIMD(uint32x4_t &va, uint32x4_t vb, uint32x4_t vc, uint32x4_t vd, uint32x4_t vx, int s, uint32x4_t vac) {
    va = vaddq_u32(va, vorrq_u32(vorrq_u32(vb, vc), vd));
    va = vaddq_u32(va, vaddq_u32(vx, vac));
    va = vorrq_u32(vshlq_n_u32(va, s), vshrq_n_u32(va, 32 - s));
    va = vaddq_u32(va, vb);
}

inline void II_SIMD(uint32x4_t &va, uint32x4_t vb, uint32x4_t vc, uint32x4_t vd, uint32x4_t vx, int s, uint32x4_t vac) {
    va = vaddq_u32(va, vbslq_u32(vorrq_u32(vc, vmvnq_u32(vd)), vb, vdupq_n_u32(0)));
    va = vaddq_u32(va, vaddq_u32(vx, vac));
    va = vorrq_u32(vshlq_n_u32(va, s), vshrq_n_u32(va, 32 - s));
    va = vaddq_u32(va, vb);
}

// 声明四参数版本的MD5Hash函数
void MD5Hash(const string& input1, const string& input2, const string& input3, const string& input4, bit32 *state);

#endif // MD5_H