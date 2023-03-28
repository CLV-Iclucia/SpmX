//
// Created by creeper on 23-3-9.
//

#ifndef SPMX_OPTIONS_H
#define SPMX_OPTIONS_H

namespace SpmX
{
    struct Options
    {
        static bool auto_eliminate_zeros_enabled;
        static bool auto_eliminate_duplicates_enabled;
    };
    inline bool isAutoEliminateZerosEnabled() { return Options::auto_eliminate_zeros_enabled; }
    inline bool isAutoEliminateDuplicatesEnabled() { return Options::auto_eliminate_duplicates_enabled; }
    inline void enableAutoEliminateZeros() { Options::auto_eliminate_zeros_enabled = true; }
    inline void disableAutoEliminateZeros() { Options::auto_eliminate_zeros_enabled = false; }
    inline void enableAutoEliminateDuplicates() { Options::auto_eliminate_duplicates_enabled = true; }
    inline void disableAutoEliminateDuplicates() { Options::auto_eliminate_duplicates_enabled = false; }
}

#endif //SPMX_OPTIONS_H
