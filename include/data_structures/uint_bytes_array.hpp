/**
 * part of LukasNalbach/move-data-structure
 *
 * MIT License
 *
 * Copyright (c) Lukas Nalbach
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#ifndef MOVE_DATA_STRUCTURE_DATA_STRUCTURES_UINT_BYTES_ARRAY_HPP
#define MOVE_DATA_STRUCTURE_DATA_STRUCTURES_UINT_BYTES_ARRAY_HPP

#include <cstdint>
#include <cstring>
#include <vector>

#include <misc/sort.hpp>
#include <misc/utils.hpp>

// the largest value that an array with a width chosen at runtime stores with 32 bits (see bytes_per_value); a build
// can set it lower to test the 40-bit values (see MOVE_DATA_STRUCTURE_MAX_SHORT_VALUE in CMakeLists.txt)
#ifndef MOVE_DATA_STRUCTURE_MAX_SHORT_VALUE
#define MOVE_DATA_STRUCTURE_MAX_SHORT_VALUE ((uint64_t { 1 } << 32) - 1)
#endif

inline constexpr uint64_t max_short_value = MOVE_DATA_STRUCTURE_MAX_SHORT_VALUE;

// returns the number of bytes per value (4, 5 or 6) of an array with a width chosen at runtime for values in [0, max_value]
inline constexpr uint8_t bytes_per_value(uint64_t max_value)
{
    return max_value <= max_short_value ? 4 : (wide_values(max_value) ? 6 : 5);
}

/**
 * @brief an array of groups of num_fields unsigned values (e.g., pairs for num_fields = 2), each value stored with
 *        4, 5 or 6 bytes (the fewest that hold the largest value, see bytes_per_value); a value is read by loading
 *        8 bytes and masking them
 * @tparam num_fields number of values per group
 */
template <uint8_t num_fields = 1>
class uint_bytes_array {
private:
    /* the values: for i in [0..size()-1] and f in [0..num_fields-1], value f of group i is stored little endian
       in the bytes [(i * num_fields + f) * w .. (i * num_fields + f + 1) * w - 1], where w = bytes_per_value(); followed by
       sizeof(uint64_t) bytes that are not initialized (get loads 8 bytes) */
    no_init_vector<uint8_t> _data;
    uint64_t _size = 0; // number of groups
    uint8_t _width = 0; // w, the number of bytes per value
    uint64_t _mask = 0; // the low 8 * w bits of a 64-bit word

    // writes the low w bytes of value to dst
    inline void store(uint8_t* dst, uint64_t value) const;

    // sorts the groups by their field-th value, as records of num_fields * value_bytes bytes
    template <uint8_t field, uint8_t value_bytes>
    void sort_records_by(uint64_t num_threads);

public:
    uint_bytes_array() = default;

    /**
     * @brief creates an array of groups (not initialized)
     * @param size number of groups
     * @param max_value the largest value
     */
    uint_bytes_array(uint64_t size, uint64_t max_value) { reset(size, max_value); }

    /**
     * @brief resizes the array; its values are lost (not initialized)
     * @param size number of groups
     * @param max_value the largest value
     */
    void reset(uint64_t size, uint64_t max_value);

    // frees the array
    void clear();

    // returns the number of groups
    inline uint64_t size() const { return _size; }

    // returns whether the array has no groups
    inline bool empty() const { return _size == 0; }

    // returns the number of bytes per value
    inline uint8_t bytes_per_value() const { return _width; }

    // returns the bytes of the groups (size_in_bytes() many)
    inline char* data() { return reinterpret_cast<char*>(_data.data()); }
    inline const char* data() const { return reinterpret_cast<const char*>(_data.data()); }
    inline uint64_t size_in_bytes() const { return _size * num_fields * _width; }

    // returns the field-th value of the i-th group
    template <uint8_t field = 0>
    inline uint64_t get(uint64_t i) const;

    // sets the field-th value of the i-th group (threads may write different groups at the same time)
    template <uint8_t field = 0>
    inline void set(uint64_t i, uint64_t value) { store(_data.data() + (i * num_fields + field) * _width, value); }

    // sorts the groups by their field-th value (see radix_sort)
    template <uint8_t field = 0>
    void sort_by(uint64_t num_threads);
};

#include "uint_bytes_array.tpp"

#endif // MOVE_DATA_STRUCTURE_DATA_STRUCTURES_UINT_BYTES_ARRAY_HPP
