// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#include "procstr.h"
#include <regex>
#include <sstream>
#include <cstring>

void trim_all(std::string &instr)
{
    std::regex re_ls("\\s+");
    std::regex re_rs("^\\s+");
    std::regex re_ts("\\s+$");
    instr = std::regex_replace(instr, re_rs, "");
    instr = std::regex_replace(instr, re_ts, "");
    instr = std::regex_replace(instr, re_ls, " ");
}

void trim_indent(char *str)
{
    int i = 0; 
    while (str[i] == ' ' || str[i] == '\t')
    {
        ++i;
    }
    for (int j = 0; j < strlen(str) - i; ++j)
    {
        str[j] = str[i+j];
    }
}

char *get_indent(char *str)
{
    int i = 0; 
    while (str[i] == ' ' || str[i] == '\t')
    {
        ++i;
    }
    return str+i;
}

void trim_comment(std::string &instr, const char *headers)
{
    for(int i = 0; i < strlen(headers); ++i)
    {
        char st[4];
        st[0] = headers[i];
        st[1] = '.';
        st[2] = '*';
        st[3] = '\0';
        std::regex re_comment(st);
        instr = std::regex_replace(instr, re_comment, "");
    }
}

bool str_to_bool(const std::string &str)
{
    if (str[0] == 'T' || str[0] == 't' || str[0] == 'Y' || str[0] == 'y' || str[0] == '1')
        return true;
    return false;
}
bool str_to_bool(const char * str)
{
    if (str[0] == 'T' || str[0] == 't' || str[0] == 'Y' || str[0] == 'y' || str[0] == '1')
        return true;
    if (str[0] == '.')
    {
        if (str[1] == 'T' || str[1] == 't')
            return true;
    }
    return false;
}

void str_split(const std::string &instr, std::vector<std::string> &splitstr)
{
    splitstr.clear();
    std::istringstream ss1(instr);
    std::string word;
    while (ss1 >> word)
        splitstr.emplace_back(word);
}

int str_split(char *instr, char **split)
{
    char *p = instr;
    int num = 0;
    while (*p) {
        while (*p == ' ' || *p == '\n' || *p == '\t' || *p == '\r') ++p;
        if (*p == '\0') break;
        split[num++] = p;
        while (*p && *p != ' ' && *p != '\n' && *p != '\t' && *p != '\r') ++p;
        if (*p) *p++ = '\0';
    }
    return num;
}

int str_split(char *instr, char **split, char *token)
{
    char *p = instr;
    int num = 0;
    while (*p) {
        while (*p && strchr(token, *p)) ++p;
        if (*p == '\0') break;
        split[num++] = p;
        while (*p && !strchr(token, *p)) ++p;
        if (*p) *p++ = '\0';
    }
    return num;
}

int str_split(char *instr, char **split, int count)
{
    char *p = instr;
    int num = 0;
    while (*p && num < count) {
        while (*p == ' ' || *p == '\n' || *p == '\t' || *p == '\r') ++p;
        if (*p == '\0') break;
        split[num++] = p;
        while (*p && *p != ' ' && *p != '\n' && *p != '\t' && *p != '\r') ++p;
        if (*p) *p++ = '\0';
    }
    return num;
}

unsigned int bkdr_hash(const char *opt)
{
    unsigned int seed = 131; // 31 131 1313 13131 131313 etc..
    unsigned int hash = 0;

    while (*opt)
    {
        hash = hash * seed + (*opt++);
    }

    return (hash & 0x7FFFFFFF);
}
// just for fun, do not call this
unsigned int kmp_search(char *string, char *pattern)
{
    return 0;
}