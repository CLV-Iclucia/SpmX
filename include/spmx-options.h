//
// Created by creeper on 23-3-9.
//

#ifndef SPMX_OPTIONS_H
#define SPMX_OPTIONS_H

namespace spmx
{
    struct Options
    {
        static bool auto_eliminate_zeros_enabled;
        static bool auto_eliminate_duplicates_enabled;
    };
    inline bool IsAutoEliminateZerosEnabled() { return Options::auto_eliminate_zeros_enabled; }
    inline bool IsAutoEliminateDuplicatesEnabled() { return Options::auto_eliminate_duplicates_enabled; }
    inline void EnableAutoEliminateZeros() { Options::auto_eliminate_zeros_enabled = true; }
    inline void DisableAutoEliminateZeros() { Options::auto_eliminate_zeros_enabled = false; }
    inline void EnableAutoEliminateDuplicates() { Options::auto_eliminate_duplicates_enabled = true; }
    inline void DisableAutoEliminateDuplicates() { Options::auto_eliminate_duplicates_enabled = false; }
}

#endif //SPMX_OPTIONS_H
