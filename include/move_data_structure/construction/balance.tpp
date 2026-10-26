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

#ifndef MOVE_DATA_STRUCTURE_MOVE_DATA_STRUCTURE_CONSTRUCTION_BALANCE_TPP
#define MOVE_DATA_STRUCTURE_MOVE_DATA_STRUCTURE_CONSTRUCTION_BALANCE_TPP

#include <move_data_structure/construction/construction.hpp>

// ############################# BALANCING #############################

template <move_pos_encoding_t enc, move_mode_t mode>
template <typename cursor_t, typename other_cursor_t>
inline uint64_t move_data_structure<enc, mode>::construction::count_other_starts(
    const cursor_t& cursor, other_cursor_t& other, no_init_vector<uint64_t>& split_pos) const
{
    uint64_t piece_beg = cursor.beg();
    uint64_t piece_end = cursor.end();
    while (other.end() <= piece_beg) other.next();

    // num_starts = number of starting positions of pieces of other in [piece_beg, piece_end) found so far; the
    // (a+1)-th, (2a+1)-th, ... of them are stored in split_pos
    uint64_t num_starts = other.beg() == piece_beg ? 1 : 0;
    uint64_t next_split_count = a + 1;
    split_pos.clear();

    while (other.end() < piece_end) {
        other.next();

        if (++num_starts == next_split_count) {
            split_pos.push_back(other.beg());
            next_split_count += a;
        }
    }

    return num_starts;
}

template <move_pos_encoding_t enc, move_mode_t mode>
inline void move_data_structure<enc, mode>::construction::split_heavy_output_piece(
    const output_cursor& oc, input_cursor& ic, std::vector<split_t>& found, no_init_vector<uint64_t>& split_pos) const
{
    uint64_t num_starts = count_other_starts(oc, ic, split_pos);
    if (num_starts < 2 * a) return;

    // split at the first num_splits = num_starts / a - 1 positions in split_pos: each resulting piece contains a of
    // the starting positions, the last one num_starts - num_splits * a (in [a, 2a))
    uint64_t num_splits = num_starts / a - 1;
    uint64_t pair = pair_of_output(oc.y);
    for (uint64_t i = 0; i < num_splits; i++) found.push_back({ pair, oc.y, split_pos[i] - oc.q });
}

template <move_pos_encoding_t enc, move_mode_t mode>
inline void move_data_structure<enc, mode>::construction::split_heavy_input_piece(
    const input_cursor& ic, output_cursor& oc, std::vector<split_t>& found, no_init_vector<uint64_t>& split_pos) const
{
    uint64_t num_starts = count_other_starts(ic, oc, split_pos);
    if (num_starts < 2 * a) return;
    uint64_t num_splits = num_starts / a - 1;
    uint64_t y = rho_of(ic.x);
    for (uint64_t i = 0; i < num_splits; i++) found.push_back({ ic.x, y, split_pos[i] - ic.p });
}

template <move_pos_encoding_t enc, move_mode_t mode>
void move_data_structure<enc, mode>::construction::balance()
{
    log_phase_start(log, time, "balancing");

    std::vector<std::vector<split_t>> found(num_threads); // the splits found by each thread
    std::vector<no_init_vector<uint64_t>> split_pos(num_threads); // buffer of each thread (see split_heavy_output_piece)
    uint64_t chunks = num_threads == 1 ? 1 : chunks_per_thread * num_threads;
    step_begin();

    // first round: check all output interval pieces (the output intervals split at the multiples of l_max)
    #pragma omp parallel for num_threads(num_threads) schedule(dynamic, 1)
    for (uint64_t c = 0; c < chunks; c++) {
        uint64_t y_beg = (c * k) / chunks;
        uint64_t y_end = ((c + 1) * k) / chunks;
        if (y_beg == y_end) continue;
        uint64_t thread = omp_get_thread_num();
        output_cursor oc { this };
        oc.go_to_output(y_beg);
        input_cursor ic { this };
        ic.seek(oc.beg());

        while (oc.y < y_end) {
            split_heavy_output_piece(oc, ic, found[thread], split_pos[thread]);
            oc.next();
        }
    }

    // and all input interval pieces
    if constexpr (has_inverse) {
        #pragma omp parallel for num_threads(num_threads) schedule(dynamic, 1)
        for (uint64_t c = 0; c < chunks; c++) {
            uint64_t x_beg = (c * k) / chunks;
            uint64_t x_end = ((c + 1) * k) / chunks;
            if (x_beg == x_end) continue;
            uint64_t thread = omp_get_thread_num();
            input_cursor ic { this };
            ic.go_to_pair(x_beg);
            output_cursor oc { this };
            oc.seek(ic.beg());

            while (ic.x < x_end) {
                split_heavy_input_piece(ic, oc, found[thread], split_pos[thread]);
                ic.next();
            }
        }
    }

    step_end("step_round1");
    num_rounds = 1;
    no_init_vector<split_t> new_splits;
    no_init_vector<piece_t> out_pieces, in_pieces;
    uint64_t num_splits_first = 0;

    for (;;) {
        // collect the splits found in the round, sort them by (x, o) and remove duplicates
        step_begin();
        std::vector<uint64_t> found_beg(num_threads + 1, 0);
        for (uint64_t i = 0; i < num_threads; i++) found_beg[i + 1] = found_beg[i] + found[i].size();
        resize_no_init(new_splits, found_beg[num_threads]);

        #pragma omp parallel for num_threads(num_threads)
        for (uint64_t i = 0; i < num_threads; i++) {
            std::copy(found[i].begin(), found[i].end(), new_splits.begin() + found_beg[i]);
            found[i].clear();
        }

        if (new_splits.empty()) break;
        mds_sort_unique(new_splits, num_threads, mds_split_key {}, mds_split_offset {});
        if (num_rounds == 1) num_splits_first = new_splits.size();
        if (mf != nullptr) *mf << " cuts_round_" << num_rounds << "=" << new_splits.size();
        step_end("step_sort_cuts");
        step_begin();
        // every split offset adds an interval
        mds_check_num_intervals(k + splits.offsets.size() + new_splits.size());
        splits.merge(new_splits, num_threads);
        step_end("step_merge");
        step_begin();
        index_splits();
        step_end("step_index_cuts");
        step_begin();

        // a split of the pair x at offset o adds the input interval starting position p_x + o and the output
        // interval starting position q_x + o; find the output interval piece containing p_x + o and (if
        // has_inverse) the input interval piece containing q_x + o
        resize_no_init(out_pieces, new_splits.size());
        if constexpr (has_inverse) resize_no_init(in_pieces, new_splits.size());

        #pragma omp parallel for num_threads(num_threads)
        for (uint64_t i = 0; i < new_splits.size(); i++) {
            const split_t& s = new_splits[i];
            output_cursor oc { this };
            oc.seek(p_of(s.x) + s.o);
            out_pieces[i] = { oc.y, oc.pieces.beg };

            if constexpr (has_inverse) {
                input_cursor ic { this };
                ic.seek(q_of_output(s.y) + s.o);
                in_pieces[i] = { ic.x, ic.pieces.beg };
            }
        }

        step_end("step_locate");
        step_begin();
        mds_sort_unique(out_pieces, num_threads, mds_piece_key {}, mds_piece_offset {});
        if constexpr (has_inverse) mds_sort_unique(in_pieces, num_threads, mds_piece_key {}, mds_piece_offset {});
        if (mf != nullptr) *mf << " rechecks_round_" << num_rounds + 1 << "=" << out_pieces.size() + in_pieces.size();
        step_end("step_sort_pieces");
        step_begin();

        // check these pieces
        #pragma omp parallel for num_threads(num_threads) schedule(dynamic, dynamic_chunk_length(out_pieces.size()))
        for (uint64_t i = 0; i < out_pieces.size(); i++) {
            uint64_t thread = omp_get_thread_num();
            output_cursor oc { this };
            oc.seek(out_pieces[i].interval, q_of_output(out_pieces[i].interval) + out_pieces[i].offset);
            input_cursor ic { this };
            ic.seek(oc.beg());
            split_heavy_output_piece(oc, ic, found[thread], split_pos[thread]);
        }

        if constexpr (has_inverse) {
            #pragma omp parallel for num_threads(num_threads) schedule(dynamic, dynamic_chunk_length(in_pieces.size()))
            for (uint64_t i = 0; i < in_pieces.size(); i++) {
                uint64_t thread = omp_get_thread_num();
                input_cursor ic { this };
                ic.seek(in_pieces[i].interval, in_pieces[i].offset);
                output_cursor oc { this };
                oc.seek(ic.beg());
                split_heavy_input_piece(ic, oc, found[thread], split_pos[thread]);
            }
        }

        step_end("step_recheck");
        num_rounds++;
    }

    if (mf != nullptr) *mf << " rounds=" << num_rounds << " cuts=" << splits.size() << " cuts_first_round=" << num_splits_first;
    write_steps();
    if (log) log_out() << " (" << num_rounds << (num_rounds == 1 ? " round)" : " rounds)") << std::flush;
    log_phase_end(log, time, mf, "time_balance");
}

#endif // MOVE_DATA_STRUCTURE_MOVE_DATA_STRUCTURE_CONSTRUCTION_BALANCE_TPP
