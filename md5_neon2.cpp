#include "md5_neon2.h"
#include <cstring>
#include <iostream>
#include <cassert>

using namespace std;

// 处理四个字符串并进行 MD5 填充
Byte* StringProcess(const string& input1, const string& input2, const string& input3, const string& input4, int* n_byte) {
    // 合并四个字符串
    string combined = input1 + input2 + input3 + input4;
    const Byte* input_data = reinterpret_cast<const Byte*>(combined.data());
    int length = combined.length();

    int bitLength = length * 8;
    int paddingBits = (448 - (bitLength % 512)) % 512;
    if (paddingBits < 0) paddingBits += 512;

    int paddingBytes = paddingBits / 8;
    int totalLength = length + paddingBytes + 8;

    Byte* paddedMessage = new Byte[totalLength];
    memcpy(paddedMessage, input_data, length);

    paddedMessage[length] = 0x80;
    if (paddingBytes > 1) {
        memset(paddedMessage + length + 1, 0, paddingBytes - 1);
    }

    uint64_t bitLength64 = static_cast<uint64_t>(bitLength);
    for (int i = 0; i < 8; ++i) {
        paddedMessage[length + paddingBytes + i] = static_cast<Byte>((bitLength64 >> (i * 8)) & 0xFF);
    }

    *n_byte = totalLength;
    return paddedMessage;
}

// 修改后的 MD5Hash 函数，处理四个字符串
void MD5Hash(const string& input1, const string& input2, const string& input3, const string& input4, bit32 *state) {
    int messageLength;
    Byte* paddedMessage = StringProcess(input1, input2, input3, input4, &messageLength);
    int n_blocks = messageLength / 64;

    state[0] = 0x67452301;
    state[1] = 0xefcdab89;
    state[2] = 0x98badcfe;
    state[3] = 0x10325476;

    alignas(16) bit32 x[16];  // 确保数据对齐到16字节边界

    for (int i = 0; i < n_blocks; ++i) {
        const Byte* block = paddedMessage + i * 64;

        // 将输入块转换为32位整数数组
        for (int j = 0; j < 16; ++j) {
            x[j] = static_cast<bit32>(block[j * 4]) |
                   (static_cast<bit32>(block[j * 4 + 1]) << 8) |
                   (static_cast<bit32>(block[j * 4 + 2]) << 16) |
                   (static_cast<bit32>(block[j * 4 + 3]) << 24);
        }

        // 加载状态变量到NEON向量
        uint32x4_t va = vdupq_n_u32(state[0]);
        uint32x4_t vb = vdupq_n_u32(state[1]);
        uint32x4_t vc = vdupq_n_u32(state[2]);
        uint32x4_t vd = vdupq_n_u32(state[3]);

        // Round 1: 批量加载并并行化处理
        for (int j = 0; j < 16; j += 4) {
            uint32x4_t vx = vld1q_u32(&x[j]);
            uint32x4_t vac = vdupq_n_u32(0xd76aa478); // 示例常量
            FF_SIMD(va, vb, vc, vd, vx, s11, vac);
        }

        // Round 2
        for (int j = 0; j < 16; j += 4) {
            uint32x4_t vx = vld1q_u32(&x[j]);
            uint32x4_t vac = vdupq_n_u32(0xf61e2562); // 示例常量
            GG_SIMD(va, vb, vc, vd, vx, s21, vac);
        }

        // Round 3
        for (int j = 0; j < 16; j += 4) {
            uint32x4_t vx = vld1q_u32(&x[j]);
            uint32x4_t vac = vdupq_n_u32(0xfffa3942); // 示例常量
            HH_SIMD(va, vb, vc, vd, vx, s31, vac);
        }

        // Round 4
        for (int j = 0; j < 16; j += 4) {
            uint32x4_t vx = vld1q_u32(&x[j]);
            uint32x4_t vac = vdupq_n_u32(0xf4292244); // 示例常量
            II_SIMD(va, vb, vc, vd, vx, s41, vac);
        }

        // 更新状态
        state[0] += vgetq_lane_u32(va, 0);
        state[1] += vgetq_lane_u32(vb, 0);
        state[2] += vgetq_lane_u32(vc, 0);
        state[3] += vgetq_lane_u32(vd, 0);
    }

    // 处理字节序转换
    for (int i = 0; i < 4; i++) {
        uint32_t value = state[i];
        state[i] = ((value & 0x000000FF) << 24) |
                   ((value & 0x0000FF00) << 8)  |
                   ((value & 0x00FF0000) >> 8)  |
                   ((value & 0xFF000000) >> 24);
    }

    delete[] paddedMessage;
}