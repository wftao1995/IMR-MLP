// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#ifndef _IMRMLP_PROC_STR_H_
#define _IMRMLP_PROC_STR_H_
#include <string>
#include <string_view>
#include <vector>
#include <cstring>

// todo string_view trim

void trim_all(std::string &instr);
void trim_indent(char *);
char *get_indent(char *);
void trim_comment(std::string &instr, const char *leadingchars);
// may have problems for general use, I haven't fully tested it yet.
void str_split(const std::string &instr, std::vector<std::string> &splitstr);
// Note that string will be destroyed
int str_split(char *instr, char **splite);
int str_split(char *instr, char **splite, int count);
// simple string to bool converter
bool str_to_bool(const std::string &str);
bool str_to_bool(const char *str);
int str_split(char *instr, char **split, char *token);

unsigned int bkdr_hash(const char *opt);

constexpr unsigned int _bkdr_hash(const char *opt)
{
    unsigned int seed = 131; // 31 131 1313 13131 131313 etc..
    unsigned int hash = 0;

    while (*opt)
    {
        hash = hash * seed + (*opt++);
    }

    return (hash & 0x7FFFFFFF);
}

#endif //_IMRMLP_PROC_STR_H_
