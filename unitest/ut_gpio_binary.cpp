// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#include <catch2/catch_all.hpp>
#include "io/gpio.h"
#include "io/gpio_binary.h"
#include <sstream>

using Catch::Approx;

TEST_CASE("GPIO recognizes documented vasprun.xml format name", "[gpio][vasprun]")
{
    REQUIRE(GPIO::format2enum("vasprun.xml") == GPIO::VASPRUN);
    REQUIRE(GPIO::format2enum("vasprun") == GPIO::VASPRUN);
    REQUIRE(GPIO::format2enum("xml") == GPIO::VASPRUN);
}

static Atoms make_binary_io_atom()
{
    Atoms atom;
    atom.cell_ << 4.0, 0.2, 0.1,
                  0.0, 5.0, 0.3,
                  0.0, 0.0, 6.0;
    atom.pos_ = {{0.1, 0.2, 0.3}, {1.1, 1.2, 1.3}, {2.1, 2.2, 2.3}};
    atom.chemmap_ = {0, 1, 0};
    atom.pbc_ = {{true, false, true}};
    atom.forces_ = {{0.01, 0.02, 0.03}, {0.11, 0.12, 0.13}, {0.21, 0.22, 0.23}};
    atom.velocities_ = {{1.0, 2.0, 3.0}, {1.1, 2.1, 3.1}, {1.2, 2.2, 3.2}};
    atom.site_energies_ = {-1.0, -1.1, -1.2};
    atom.charges_ = {0.1, -0.2, 0.3};
    atom.magmom_scalar_ = {1.0, 0.0, -1.0};
    atom.ids_ = {10, 20, 30};
    atom.energy_ = -12.5;
    atom.have_energy = true;
    atom.fenergy_ = -12.25;
    atom.have_fenergy = true;
    atom.stress_ << 1.0, 0.1, 0.2,
                    0.1, 2.0, 0.3,
                    0.2, 0.3, 3.0;
    atom.have_stress = true;

    atom.features_["label"] = "roundtrip";
    atom.global_attribs_d_["temperature"] = 300.0;
    atom.global_attribs_v_["box_origin"].resize(3);
    atom.global_attribs_v_["box_origin"] << 1.0, 2.0, 3.0;
    atom.global_attribs_i_["step"] = 42;
    atom.global_attribs_vi_["shape"].resize(3);
    atom.global_attribs_vi_["shape"] << 4, 5, 6;

    atom.local_attribs_d_["weight"].resize(3);
    atom.local_attribs_d_["weight"] << 0.5, 0.6, 0.7;
    atom.local_attribs_i_["group"].resize(3);
    atom.local_attribs_i_["group"] << 1, 2, 1;
    atom.local_attribs_v_["descriptor"].resize(2, 3);
    atom.local_attribs_v_["descriptor"] << 1.0, 2.0, 3.0,
                                           4.0, 5.0, 6.0;
    atom.local_attribs_vi_["neighbors"].resize(2, 3);
    atom.local_attribs_vi_["neighbors"] << 1, 2, 3,
                                           4, 5, 6;
    return atom;
}

TEST_CASE("GPIO binary mlpcfg round-trips Atoms fields", "[gpio][binary]")
{
    Atoms input = make_binary_io_atom();
    std::stringstream buffer(std::ios::in | std::ios::out | std::ios::binary);

    GPIO writer;
    REQUIRE(writer.write_one(input, buffer, "bin"));

    buffer.seekg(0);
    Atoms output;
    GPIO reader;
    REQUIRE(reader.read_one(output, buffer, "bin"));

    REQUIRE(output.size() == input.size());
    REQUIRE(output.cell_.isApprox(input.cell_));
    REQUIRE(output.chemmap_ == input.chemmap_);
    REQUIRE(output.pbc_ == input.pbc_);
    for (int i = 0; i < input.size(); ++i) {
        REQUIRE(output.pos_[i].isApprox(input.pos_[i]));
        REQUIRE(output.forces_[i].isApprox(input.forces_[i]));
        REQUIRE(output.velocities_[i].isApprox(input.velocities_[i]));
    }
    REQUIRE(output.site_energies_ == input.site_energies_);
    REQUIRE(output.charges_ == input.charges_);
    REQUIRE(output.magmom_scalar_ == input.magmom_scalar_);
    REQUIRE(output.ids_ == input.ids_);
    REQUIRE(output.energy_ == Approx(input.energy_));
    REQUIRE(output.fenergy_ == Approx(input.fenergy_));
    REQUIRE(output.stress_.isApprox(input.stress_));
    REQUIRE(output.features_ == input.features_);
    REQUIRE(output.global_attribs_d_.at("temperature") == Approx(300.0));
    REQUIRE(output.global_attribs_v_.at("box_origin").isApprox(input.global_attribs_v_.at("box_origin")));
    REQUIRE(output.global_attribs_i_.at("step") == 42);
    REQUIRE(output.global_attribs_vi_.at("shape") == input.global_attribs_vi_.at("shape"));
    REQUIRE(output.local_attribs_d_.at("weight").isApprox(input.local_attribs_d_.at("weight")));
    REQUIRE(output.local_attribs_i_.at("group") == input.local_attribs_i_.at("group"));
    REQUIRE(output.local_attribs_v_.at("descriptor").isApprox(input.local_attribs_v_.at("descriptor")));
    REQUIRE(output.local_attribs_vi_.at("neighbors") == input.local_attribs_vi_.at("neighbors"));
}


TEST_CASE("GPIO binary version 2 defaults to fully periodic", "[gpio][binary][pbc]")
{
    using namespace gpio_binary;
    std::stringstream buffer(std::ios::in|std::ios::out|std::ios::binary);
    const std::uint32_t old_version=2;
    const std::int32_t natoms=1;
    REQUIRE(write_span(buffer,MAGIC,sizeof(MAGIC)));
    REQUIRE(write_value(buffer,old_version));
    REQUIRE(write_value(buffer,natoms));
    const Eigen::Matrix3d cell=5.0*Eigen::Matrix3d::Identity();
    const Eigen::RowVector3d position(1.0,2.0,3.0);
    const int type=0;
    REQUIRE(write_tag(buffer,Tag::Cell));
    REQUIRE(write_span(buffer,cell.data(),9));
    REQUIRE(write_tag(buffer,Tag::Positions));
    REQUIRE(write_span(buffer,position.data(),3));
    REQUIRE(write_tag(buffer,Tag::Types));
    REQUIRE(write_value(buffer,type));
    REQUIRE(write_tag(buffer,Tag::End));
    buffer.seekg(0);
    Atoms output;
    GPIO reader;
    REQUIRE(reader.read_one(output,buffer,"bin"));
    REQUIRE(output.pbc_==std::array<bool,3>{{true,true,true}});
}

TEST_CASE("BIN preserves declared full Hessians and HVP-only labels", "[gpio][binary][hessian]") {
    auto atom = make_binary_io_atom();
    atom.hessian_ = Eigen::MatrixXd::Identity(9, 9);
    atom.have_hessian = true;
    GPIO io;
    std::stringstream rejected(std::ios::in | std::ios::out | std::ios::binary);
    REQUIRE(io.write_one(atom, rejected, "bin"));
    Atoms dense;
    REQUIRE(io.read_one(dense,rejected,"bin"));
    REQUIRE(dense.have_hessian);
    REQUIRE(dense.hessian_.isApprox(atom.hessian_));
    REQUIRE(atom.have_hessian);
    REQUIRE(atom.hessian_.isApprox(Eigen::MatrixXd::Identity(9, 9)));

    // The flag declares whether a dense label is present; stale storage alone
    // does not prevent writing an explicitly HVP-only record.
    atom.have_hessian = false;
    atom.hvp_directions_ = Eigen::MatrixXd::Ones(9, 1);
    atom.hvp_targets_ = 2 * atom.hvp_directions_;
    std::stringstream supported(std::ios::in | std::ios::out | std::ios::binary);
    REQUIRE(io.write_one(atom, supported, "bin"));
    Atoms restored;
    REQUIRE(io.read_one(restored, supported, "bin"));
    REQUIRE_FALSE(restored.have_hessian);
    REQUIRE(restored.hvp_directions_.isApprox(atom.hvp_directions_));
    REQUIRE(restored.hvp_targets_.isApprox(atom.hvp_targets_));
}
