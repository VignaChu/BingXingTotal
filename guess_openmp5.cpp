#include "PCFG.h"
#include <omp.h>
#include <vector>
#include <iterator>
using namespace std;

void PriorityQueue::CalProb(PT &pt)
{
    pt.prob = pt.preterm_prob;
    int index = 0;

    for (int idx : pt.curr_indices)
    {
        if (pt.content[index].type == 1)
        {
            pt.prob *= m.letters[m.FindLetter(pt.content[index])].ordered_freqs[idx];
            pt.prob /= m.letters[m.FindLetter(pt.content[index])].total_freq;
        }
        if (pt.content[index].type == 2)
        {
            pt.prob *= m.digits[m.FindDigit(pt.content[index])].ordered_freqs[idx];
            pt.prob /= m.digits[m.FindDigit(pt.content[index])].total_freq;
        }
        if (pt.content[index].type == 3)
        {
            pt.prob *= m.symbols[m.FindSymbol(pt.content[index])].ordered_freqs[idx];
            pt.prob /= m.symbols[m.FindSymbol(pt.content[index])].total_freq;
        }
        index += 1;
    }
}

void PriorityQueue::init()
{
    for (PT pt : m.ordered_pts)
    {
        for (segment seg : pt.content)
        {
            if (seg.type == 1)
            {
                pt.max_indices.emplace_back(m.letters[m.FindLetter(seg)].ordered_values.size());
            }
            if (seg.type == 2)
            {
                pt.max_indices.emplace_back(m.digits[m.FindDigit(seg)].ordered_values.size());
            }
            if (seg.type == 3)
            {
                pt.max_indices.emplace_back(m.symbols[m.FindSymbol(seg)].ordered_values.size());
            }
        }
        pt.preterm_prob = float(m.preterm_freq[m.FindPT(pt)]) / m.total_preterm;
        CalProb(pt);
        priority.emplace_back(pt);
    }
}

void PriorityQueue::PopNext()
{
    Generate(priority.front());
    vector<PT> new_pts = priority.front().NewPTs();
    
    for (PT pt : new_pts)
    {
        CalProb(pt);
        for (auto iter = priority.begin(); iter != priority.end(); iter++)
        {
            if (iter != priority.end() - 1 && iter != priority.begin())
            {
                if (pt.prob <= iter->prob && pt.prob > (iter + 1)->prob)
                {
                    priority.emplace(iter + 1, pt);
                    break;
                }
            }
            if (iter == priority.end() - 1)
            {
                priority.emplace_back(pt);
                break;
            }
            if (iter == priority.begin() && iter->prob < pt.prob)
            {
                priority.emplace(iter, pt);
                break;
            }
        }
    }
    priority.erase(priority.begin());
}

vector<PT> PT::NewPTs()
{
    vector<PT> res;
    if (content.size() == 1)
    {
        return res;
    }
    else
    {
        int init_pivot = pivot;
        for (int i = pivot; i < curr_indices.size() - 1; i += 1)
        {
            curr_indices[i] += 1;
            if (curr_indices[i] < max_indices[i])
            {
                pivot = i;
                res.emplace_back(*this);
            }
            curr_indices[i] -= 1;
        }
        pivot = init_pivot;
        return res;
    }
    return res;
}

void PriorityQueue::Generate(PT pt)
{
    CalProb(pt);

    if (pt.content.size() == 1)
    {
        segment* a;
        if (pt.content[0].type == 1) {
            a = &m.letters[m.FindLetter(pt.content[0])];
        } else if (pt.content[0].type == 2) {
            a = &m.digits[m.FindDigit(pt.content[0])];
        } else {
            a = &m.symbols[m.FindSymbol(pt.content[0])];
        }

        // 真正并行的实现
        const int max_idx = pt.max_indices[0];
        vector<vector<string>> thread_guesses(4); // 4个线程的存储

        #pragma omp parallel num_threads(4)
        {
            const int tid = omp_get_thread_num();
            #pragma omp for schedule(dynamic, 512)
            for (int i = 0; i < max_idx; ++i) {
                thread_guesses[tid].emplace_back(a->ordered_values[i]);
            }
        }

        // 合并结果
        size_t total = 0;
        for (auto& vec : thread_guesses) total += vec.size();
        guesses.reserve(guesses.size() + total);
        
        for (auto& vec : thread_guesses) {
            guesses.insert(guesses.end(),
                         make_move_iterator(vec.begin()),
                         make_move_iterator(vec.end()));
        }
        total_guesses += total;
    }
    else
    {
        string base_guess;
        int seg_idx = 0;
        for (int idx : pt.curr_indices)
        {
            if (pt.content[seg_idx].type == 1)
            {
                base_guess += m.letters[m.FindLetter(pt.content[seg_idx])].ordered_values[idx];
            }
            if (pt.content[seg_idx].type == 2)
            {
                base_guess += m.digits[m.FindDigit(pt.content[seg_idx])].ordered_values[idx];
            }
            if (pt.content[seg_idx].type == 3)
            {
                base_guess += m.symbols[m.FindSymbol(pt.content[seg_idx])].ordered_values[idx];
            }
            seg_idx += 1;
            if (seg_idx == pt.content.size() - 1) {
                break;
            }
        }

        segment* a;
        const int last_seg_idx = pt.content.size() - 1;
        if (pt.content[last_seg_idx].type == 1) {
            a = &m.letters[m.FindLetter(pt.content[last_seg_idx])];
        } else if (pt.content[last_seg_idx].type == 2) {
            a = &m.digits[m.FindDigit(pt.content[last_seg_idx])];
        } else {
            a = &m.symbols[m.FindSymbol(pt.content[last_seg_idx])];
        }

        // 真正并行的实现
        const int max_idx = pt.max_indices[last_seg_idx];
        vector<vector<string>> thread_guesses(4); // 4个线程的存储

        #pragma omp parallel num_threads(4)
        {
            const int tid = omp_get_thread_num();
            #pragma omp for schedule(dynamic, 512)
            for (int i = 0; i < max_idx; ++i) {
                thread_guesses[tid].emplace_back(base_guess + a->ordered_values[i]);
            }
        }

        // 合并结果
        size_t total = 0;
        for (auto& vec : thread_guesses) total += vec.size();
        guesses.reserve(guesses.size() + total);
        
        for (auto& vec : thread_guesses) {
            guesses.insert(guesses.end(),
                         make_move_iterator(vec.begin()),
                         make_move_iterator(vec.end()));
        }
        total_guesses += total;
    }
}