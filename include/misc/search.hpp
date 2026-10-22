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

#ifndef MOVE_DATA_STRUCTURE_MISC_SEARCH_HPP
#define MOVE_DATA_STRUCTURE_MISC_SEARCH_HPP

#include <cstdint>

#include <misc/utils.hpp>

/**
 * @brief binary-searches the largest index in [left, right] whose value is <= value (the value_at sequence
 *        must be ascending); assumes value_at(left) <= value
 * @tparam fnc_t type of the value_at function
 * @param value the value to compare against
 * @param left left search boundary (inclusive)
 * @param right right search boundary (inclusive)
 * @param value_at function mapping an index to its value
 * @return the largest index in [left, right] with value_at(index) <= value
 */
template <typename fnc_t>
inline static uint64_t bin_search_max_leq(uint64_t value, uint64_t left, uint64_t right, fnc_t value_at)
{
    uint64_t middle;

    while (left != right) {
        middle = left + (right - left) / 2 + 1;
        uint64_t middle_value = value_at(middle);

        if (middle_value <= value) {
            left = middle;
        } else {
            right = middle - 1;
        }
    }

    return left;
}

/**
 * @brief binary-searches the smallest index in [left, right] whose value is >= value (the value_at sequence
 *        must be ascending); assumes value_at(right) >= value
 * @tparam fnc_t type of the value_at function
 * @param value the value to compare against
 * @param left left search boundary (inclusive)
 * @param right right search boundary (inclusive)
 * @param value_at function mapping an index to its value
 * @return the smallest index in [left, right] with value_at(index) >= value
 */
template <typename fnc_t>
inline static uint64_t bin_search_min_geq(uint64_t value, uint64_t left, uint64_t right, fnc_t value_at)
{
    uint64_t middle;

    while (left != right) {
        middle = left + (right - left) / 2;
        uint64_t middle_value = value_at(middle);

        if (value <= middle_value) {
            right = middle;
        } else {
            left = middle + 1;
        }
    }

    return left;
}

/**
 * @brief binary-searches the largest index in [left, right] whose value is < value (the value_at sequence
 *        must be ascending); assumes value_at(left) < value
 * @tparam fnc_t type of the value_at function
 * @param value the value to compare against
 * @param left left search boundary (inclusive)
 * @param right right search boundary (inclusive)
 * @param value_at function mapping an index to its value
 * @return the largest index in [left, right] with value_at(index) < value
 */
template <typename fnc_t>
inline static uint64_t bin_search_max_lt(uint64_t value, uint64_t left, uint64_t right, fnc_t value_at)
{
    uint64_t middle;

    while (left != right) {
        middle = left + (right - left) / 2 + 1;

        if (value_at(middle) < value) {
            left = middle;
        } else {
            right = middle - 1;
        }
    }

    return left;
}

/**
 * @brief binary-searches the smallest index in [left, right] whose value is > value (the value_at sequence
 *        must be ascending); assumes value_at(right) > value
 * @tparam fnc_t type of the value_at function
 * @param value the value to compare against
 * @param left left search boundary (inclusive)
 * @param right right search boundary (inclusive)
 * @param value_at function mapping an index to its value
 * @return the smallest index in [left, right] with value_at(index) > value
 */
template <typename fnc_t>
inline static uint64_t bin_search_min_gt(uint64_t value, uint64_t left, uint64_t right, fnc_t value_at)
{
    uint64_t middle;

    while (left != right) {
        middle = left + (right - left) / 2;

        if (value < value_at(middle)) {
            right = middle;
        } else {
            left = middle + 1;
        }
    }

    return left;
}

/**
 * @brief exponential search for the largest index in [left, right] whose value is <= value, starting at one end of
 *        [left, right] (the value_at sequence must be strictly increasing); assumes value_at(left) <= value and, for
 *        LEFT, value < value_at(right)
 * @tparam search_dir direction of the search (LEFT starts at right, RIGHT starts at left)
 * @tparam fnc_t type of the value_at function
 * @param value the value to compare against
 * @param left left search boundary (inclusive)
 * @param right right search boundary (inclusive)
 * @param value_at function mapping an index to its value
 * @return the largest index in [left, right] with value_at(index) <= value
 */
template <direction_t search_dir, typename fnc_t>
inline static uint64_t exp_search_max_leq(uint64_t value, uint64_t left, uint64_t right, fnc_t value_at)
{
    if (left == right) [[unlikely]] return left;
    uint64_t cur_step_size = 1;

    if constexpr (search_dir == LEFT) {
        right -= cur_step_size;

        while (value < value_at(right)) {
            cur_step_size *= 2;

            if (right < left + cur_step_size) {
                cur_step_size = right - left;
                right = left;
                break;
            }

            right -= cur_step_size;
        }

        return bin_search_max_leq(value, right, right + cur_step_size - 1, value_at);
    } else {
        left += cur_step_size;

        while (value_at(left) < value) {
            cur_step_size *= 2;

            if (right < cur_step_size || right - cur_step_size < left) {
                cur_step_size = right - left;
                left = right;
                break;
            }

            left += cur_step_size;
        }

        return bin_search_max_leq(value, left - cur_step_size, left, value_at);
    }
}

#endif // MOVE_DATA_STRUCTURE_MISC_SEARCH_HPP
