#ifndef SWAP_H
#define SWAP_H

#include <inttypes.h>

int32_t swap_int32(int32_t n)
{
    return (n & 0xFF000000) >> 24
         | (n & 0x00FF0000) >> 8
         | (n & 0x0000FF00) << 8
         | (n & 0x000000FF) << 24;
}

#endif // SWAP_H