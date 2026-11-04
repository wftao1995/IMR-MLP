// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT
#include "potentials/model_elements.h"

#include "tools/gpm_container.h"
#include "tools/mtp_metadata.h"
#include "utils/compressed_stream.h"

#include <algorithm>
#include <string>
#include <utility>
#include <vector>

namespace Potentials
{

namespace
{

std::vector<std::string> probe_gpm(const std::string &path)
{
    GPM::Reader reader(path);
    const auto &species = GPM::array(GPM::at(reader.manifest, "species"));
    std::vector<std::string> names(species.size());
    std::vector<bool> seen(species.size(), false);
    for (const auto &entry : species)
    {
        const auto &object = GPM::object(entry);
        const int id = GPM::integer(GPM::at(object, "id"));
        if (id < 0 || id >= static_cast<int>(names.size()) || seen[id])
            return {};
        const auto symbol = object.find("symbol");
        if (symbol == object.end() || !symbol->second.is<std::string>())
            return {};
        names[id] = GPM::string(symbol->second);
        seen[id] = true;
    }
    if (std::find(seen.begin(), seen.end(), false) != seen.end())
        return {};
    return names;
}

std::vector<std::string> probe_mtp(const std::string &path)
{
    auto stream = open_ifile(path);
    if (!stream)
        return {};
    const auto header = MTPMetadata::read_header(*stream);
    return header.tag.elements();
}

std::vector<std::string> probe_hybrid(const GPSettings &pot_settings)
{
    std::vector<std::string> result;
    for (int index = 1;; ++index)
    {
        const auto child = pot_settings.ExtractSubSettings("pot" + std::to_string(index));
        if (child["pottype"].empty())
            break;
        auto names = probe_model_elements(child);
        if (names.empty())
            continue;
        if (result.empty())
        {
            result = std::move(names);
        }
        else if (result != names)
        {
            return {};
        }
    }
    return result;
}

} // namespace

std::vector<std::string> probe_model_elements(const GPSettings &pot_settings)
{
    try
    {
        const std::string pottype = pot_settings["pottype"];
        if (pottype == "hybrid" || pottype == "hybrid_scale")
            return probe_hybrid(pot_settings);

        const std::string potfile = pot_settings["potfile"];
        if (potfile.empty())
            return {};
        if (GPM::is_file(potfile) || GPM::extension(potfile))
            return probe_gpm(potfile);
        if (pottype.empty() || pottype == "auto" || pottype == "mtp" || pottype == "mtp_m" ||
            pottype == "mtp_les" || pottype == "mtpnn1")
            return probe_mtp(potfile);
    }
    catch (const std::exception &)
    {
        // The authoritative model load reports malformed files. Probing is
        // best-effort so an absent or unusual format keeps the old map path.
        return {};
    }
    return {};
}

} // namespace Potentials
