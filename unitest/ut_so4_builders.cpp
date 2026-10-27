#include <catch2/catch_all.hpp>
#include "descriptors/descriptor_BispectrumSO4.h"
#include "maxvol/b_matrix_builder.h"
#include "unitest/pair_cell_fixture.h"
#include <sstream>

TEST_CASE("SO4 B-matrix settings survive clone and serialization", "[descriptor][so4][builder]")
{
    const std::string mode = GENERATE(std::string("nbh"), std::string("cfg"));
    GPSettings settings;
    settings.Modify("biso4:twojmax", "2");
    settings.Modify("biso4:cutoff", "3.2");
    settings.Modify("biso4:weights", "0.7 1.3");
    settings.Modify("biso4:selfweights", "0.4 0.8");
    settings.Modify("biso4:quaternion_convention", "q2");
    settings.Modify("biso4:radial_angle_map", "atan");
    settings.Modify("biso4:contraction_backend", "snap");
    auto builder = make_builder("biso4", "", "", settings, mode, 1., 0., 0., 0.);
    REQUIRE(builder);
    auto atoms = PairCellTest::geometry(true);
    BMatrixBuilder::RowMatrix actual;
    REQUIRE(builder->build_rows(atoms, actual, 1., 0.) == (mode == "nbh" ? atoms.size() : 1));
    Descriptor_BispectrumSO4 descriptor(settings.ExtractSubSettings("biso4"));
    Neighborhoods sites(atoms, descriptor.cutoff_);
    BMatrixBuilder::RowMatrix local(atoms.size(), descriptor.feature_size());
    for (int i = 0; i < atoms.size(); ++i) {
        Eigen::VectorXd value;
        descriptor.compute(sites[i], value);
        local.row(i) = value.transpose();
    }
    if (mode == "nbh") REQUIRE(actual.isApprox(local, 1e-12));
    else REQUIRE(actual.row(0).isApprox(local.colwise().sum(), 1e-12));
    auto cloned = builder->clone();
    REQUIRE(cloned);
    std::stringstream stream;
    builder->save_config(stream);
    auto restored = BMatrixBuilder::load_config(stream);
    REQUIRE(restored);
    for (auto *copy : {cloned.get(), restored.get()}) {
        BMatrixBuilder::RowMatrix result;
        copy->build_rows(atoms, result, 1., 0.);
        REQUIRE(result.isApprox(actual, 1e-12));
    }
}
