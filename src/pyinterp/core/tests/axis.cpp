// Copyright (c) 2019 CNES
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.
#include "pyinterp/detail/axis.hpp"
#include <gtest/gtest.h>

namespace detail = pyinterp::detail;

TEST(axis, default_constructor) {
  // undefined axis
  detail::Axis axis;
  EXPECT_TRUE(std::isnan(axis.front()));
  EXPECT_TRUE(std::isnan(axis.back()));
  EXPECT_TRUE(std::isnan(axis.min_value()));
  EXPECT_TRUE(std::isnan(axis.max_value()));
  double inc;
  EXPECT_THROW(inc = axis.increment(), std::logic_error);
  EXPECT_FALSE(axis.is_circle());
  EXPECT_TRUE(axis.is_ascending());
  EXPECT_TRUE(std::isnan(axis.front()));
  EXPECT_TRUE(std::isnan(axis.back()));
  EXPECT_THROW(inc = axis.increment(), std::logic_error);
  EXPECT_EQ(axis.is_regular(), false);
  EXPECT_EQ(axis.size(), 0);
  EXPECT_THROW(inc = axis.coordinate_value(0), std::out_of_range);
  EXPECT_EQ(axis.find_index(360, true), -1);
  EXPECT_EQ(axis.find_index(360, false), -1);
  auto indexes = axis.find_indexes(360);
  EXPECT_FALSE(indexes.has_value());
  EXPECT_EQ(static_cast<std::string>(axis),
            "Axis([nan], is_circle=false, is_radian=false)");
}

TEST(axis, singleton) {
  // axis with one value
  detail::Axis axis(0, 1, 1, 1e-6, false, false);
  EXPECT_EQ(axis.find_index(0, false), 0);
  EXPECT_EQ(axis.find_index(1, false), -1);
  EXPECT_EQ(axis.find_index(1, true), 0);
  auto indexes = axis.find_indexes(0);
  EXPECT_FALSE(indexes.has_value());
  EXPECT_TRUE(axis.is_ascending());
  EXPECT_EQ(axis.front(), 0);
  EXPECT_EQ(axis.back(), 0);
  EXPECT_EQ(axis.min_value(), 0);
  EXPECT_EQ(axis.max_value(), 0);
  EXPECT_EQ(axis.increment(), 1);
  EXPECT_FALSE(axis.is_circle());
  EXPECT_EQ(axis.is_regular(), true);
  EXPECT_EQ(axis.size(), 1);
  EXPECT_EQ(axis.coordinate_value(0), 0);
  double value;
  EXPECT_THROW(value = axis.coordinate_value(1), std::exception);
  EXPECT_EQ(static_cast<std::string>(axis),
            "Axis([0], is_circle=false, is_radian=false)");
}

TEST(axis, binary) {
  // axis with two values
  detail::Axis axis(0, 1, 2, 1e-6, false, false);
  auto indexes = axis.find_indexes(0);
  ASSERT_TRUE(indexes);
  EXPECT_EQ(std::get<0>(*indexes), 0);
  EXPECT_EQ(std::get<1>(*indexes), 1);
  indexes = axis.find_indexes(1);
  ASSERT_TRUE(indexes);
  EXPECT_EQ(std::get<0>(*indexes), 0);
  EXPECT_EQ(std::get<1>(*indexes), 1);
  EXPECT_FALSE(axis.find_indexes(-0.1));
  EXPECT_FALSE(axis.find_indexes(+1.1));
  indexes = axis.find_indexes(0.4);
  ASSERT_TRUE(indexes);
  EXPECT_EQ(std::get<0>(*indexes), 0);
  EXPECT_EQ(std::get<1>(*indexes), 1);
  indexes = axis.find_indexes(0.6);
  ASSERT_TRUE(indexes);
  EXPECT_EQ(std::get<0>(*indexes), 0);
  EXPECT_EQ(std::get<1>(*indexes), 1);
  EXPECT_EQ(axis.front(), 0);
  EXPECT_EQ(axis.back(), 1);
  EXPECT_EQ(axis.min_value(), 0);
  EXPECT_EQ(axis.max_value(), 1);
  EXPECT_EQ(axis.increment(), 1);
  EXPECT_FALSE(axis.is_circle());
  EXPECT_TRUE(axis.is_ascending());
  EXPECT_EQ(axis.is_regular(), true);
  EXPECT_EQ(axis.size(), 2);
  auto value = axis.coordinate_value(0);
  EXPECT_EQ(value, 0);
  value = axis.coordinate_value(1);
  EXPECT_EQ(value, 1);
  EXPECT_THROW(value = axis.coordinate_value(2), std::exception);
  EXPECT_EQ(static_cast<std::string>(axis),
            "Axis([0, 1], is_circle=false, is_radian=false)");
}

TEST(axis, wrap_longitude) {
  // axis representing a circle
  int64_t i1;
  detail::Axis a1(0, 359, 360, 1e-6, true, false);

  EXPECT_EQ(a1.front(), 0);
  EXPECT_EQ(a1.increment(), 1);
  EXPECT_TRUE(a1.is_circle());
  EXPECT_TRUE(a1.is_regular());
  EXPECT_TRUE(a1.is_ascending());
  EXPECT_EQ(a1.front(), 0);
  EXPECT_EQ(a1.back(), 359);
  EXPECT_EQ(a1.min_value(), 0);
  EXPECT_EQ(a1.max_value(), 359);
  EXPECT_EQ(a1.size(), 360);
  EXPECT_EQ(a1.coordinate_value(0), 0);
  EXPECT_EQ(a1.coordinate_value(180), 180);
  double value;
  EXPECT_THROW(value = a1.coordinate_value(520), std::exception);
  i1 = a1.find_index(0, false);
  EXPECT_EQ(i1, 0);
  i1 = a1.find_index(360, true);
  EXPECT_EQ(i1, 0);
  i1 = a1.find_index(360, false);
  EXPECT_EQ(i1, 0);
  auto indexes = a1.find_indexes(360);
  ASSERT_TRUE(indexes);
  EXPECT_EQ(std::get<0>(*indexes), 0);
  EXPECT_EQ(std::get<1>(*indexes), 1);
  indexes = a1.find_indexes(370);
  ASSERT_TRUE(indexes);
  EXPECT_EQ(std::get<0>(*indexes), 10);
  EXPECT_EQ(std::get<1>(*indexes), 11);
  indexes = a1.find_indexes(-9.5);
  ASSERT_TRUE(indexes);
  EXPECT_EQ(std::get<0>(*indexes), 350);
  EXPECT_EQ(std::get<1>(*indexes), 351);
  EXPECT_EQ(static_cast<std::string>(a1),
            "Axis([0, 1, 2, ..., 358, 359], is_circle=true, is_radian=false)");
  a1.flip();
  EXPECT_EQ(a1.front(), 359);
  EXPECT_EQ(a1.increment(), -1);
  EXPECT_TRUE(a1.is_circle());
  EXPECT_TRUE(a1.is_regular());
  EXPECT_FALSE(a1.is_ascending());
  EXPECT_EQ(a1.front(), 359);
  EXPECT_EQ(a1.back(), 0);
  EXPECT_EQ(a1.min_value(), 0);
  EXPECT_EQ(a1.max_value(), 359);
  EXPECT_EQ(a1.size(), 360);
  EXPECT_EQ(a1.coordinate_value(0), 359);
  EXPECT_EQ(a1.coordinate_value(180), 179);
  EXPECT_THROW(value = a1.coordinate_value(520), std::exception);
  i1 = a1.find_index(0, false);
  EXPECT_EQ(i1, 359);
  i1 = a1.find_index(359, true);
  EXPECT_EQ(i1, 0);
  i1 = a1.find_index(359, false);
  EXPECT_EQ(i1, 0);
  indexes = a1.find_indexes(359.5);
  ASSERT_TRUE(indexes);
  EXPECT_EQ(std::get<0>(*indexes), 359);
  EXPECT_EQ(std::get<1>(*indexes), 0);
  indexes = a1.find_indexes(370);
  ASSERT_TRUE(indexes);
  EXPECT_EQ(std::get<0>(*indexes), 349);
  EXPECT_EQ(std::get<1>(*indexes), 350);
  indexes = a1.find_indexes(-9.5);
  ASSERT_TRUE(indexes);
  EXPECT_EQ(std::get<0>(*indexes), 9);
  EXPECT_EQ(std::get<1>(*indexes), 8);
  EXPECT_EQ(
      static_cast<std::string>(a1),
      "Axis([359, 358, 357, ..., 1, 0], is_circle=true, is_radian=false)");
  detail::Axis a2(-180, 179, 360, 1e-6, true, false);
  EXPECT_EQ(a2.front(), -180);
  EXPECT_EQ(a2.increment(), 1);
  EXPECT_TRUE(a2.is_circle());
  EXPECT_TRUE(a2.is_regular());
  EXPECT_TRUE(a2.is_ascending());
  indexes = a2.find_indexes(370);
  ASSERT_TRUE(indexes);
  EXPECT_EQ(std::get<0>(*indexes), 190);
  EXPECT_EQ(std::get<1>(*indexes), 191);
  EXPECT_EQ(a2.coordinate_value(190), 10);
  EXPECT_EQ(a2.front(), -180);
  EXPECT_EQ(a2.back(), 179);
  EXPECT_EQ(a2.min_value(), -180);
  EXPECT_EQ(a2.max_value(), 179);
  EXPECT_EQ(a2.coordinate_value(0), -180);
  EXPECT_EQ(a2.coordinate_value(180), 0);
  EXPECT_NE(a1, a2);

  a2 = detail::Axis(180, -179, 360, 1e-6, true, false);
  EXPECT_EQ(a2.front(), 180);
  EXPECT_EQ(a2.increment(), -1);
  EXPECT_TRUE(a2.is_circle());
  EXPECT_TRUE(a2.is_regular());
  EXPECT_FALSE(a2.is_ascending());
  indexes = a2.find_indexes(370.2);
  ASSERT_TRUE(indexes);
  EXPECT_TRUE(a2(std::get<0>(*indexes)) <= a2.normalize_coordinate(370.2) &&
              a2.normalize_coordinate(370.2) <= a2(std::get<1>(*indexes)));
  EXPECT_EQ(a2.coordinate_value(190), -10);
  EXPECT_EQ(a2.min_value(), -179);
  EXPECT_EQ(a2.max_value(), 180);
  EXPECT_EQ(a2.front(), 180);
  EXPECT_EQ(a2.back(), -179);
  EXPECT_EQ(a2.coordinate_value(0), 180);
  EXPECT_EQ(a2.coordinate_value(180), 0);
  EXPECT_NE(a1, a2);

  a2.flip();
  EXPECT_EQ(a2.front(), -179);
  EXPECT_EQ(a2.increment(), 1);
  EXPECT_TRUE(a2.is_circle());
  EXPECT_TRUE(a2.is_regular());
  EXPECT_TRUE(a2.is_ascending());
  indexes = a2.find_indexes(370.2);
  ASSERT_TRUE(indexes);
  EXPECT_TRUE(a2(std::get<0>(*indexes)) <= a2.normalize_coordinate(370.2) &&
              a2.normalize_coordinate(370.2) <= a2(std::get<1>(*indexes)));
  EXPECT_EQ(a2.coordinate_value(190), 11);
  EXPECT_EQ(a2.min_value(), -179);
  EXPECT_EQ(a2.max_value(), 180);
  EXPECT_EQ(a2.front(), -179);
  EXPECT_EQ(a2.back(), 180);
  EXPECT_EQ(a2.coordinate_value(0), -179);
  EXPECT_EQ(a2.coordinate_value(180), 1);
}

TEST(axis, radians) {
  // axis representing a circle
  int64_t i1;
  auto inc = detail::math::pi<double>() / 360.0;
  detail::Axis a1(0, detail::math::pi<double>() - inc, 360, 1e-6, true, true);

  EXPECT_EQ(a1.front(), 0);
  EXPECT_EQ(a1.increment(), detail::math::pi<double>() / 360.0);
  EXPECT_TRUE(a1.is_circle());
  EXPECT_TRUE(a1.is_regular());
  EXPECT_TRUE(a1.is_ascending());
  EXPECT_EQ(a1.front(), 0);
  EXPECT_EQ(a1.back(), detail::math::pi<double>() - inc);
  EXPECT_EQ(a1.min_value(), 0);
  EXPECT_EQ(a1.max_value(), detail::math::pi<double>() - inc);
  EXPECT_EQ(a1.size(), 360);
  EXPECT_EQ(a1.coordinate_value(0), 0);
  EXPECT_EQ(a1.coordinate_value(180), detail::math::pi<double>() * 0.5);
  double value;
  EXPECT_THROW(value = a1.coordinate_value(520), std::exception);
  i1 = a1.find_index(0, false);
  EXPECT_EQ(i1, 0);
  i1 = a1.find_index(detail::math::pi<double>(), true);
  EXPECT_EQ(i1, 0);
  i1 = a1.find_index(detail::math::pi<double>(), false);
  EXPECT_EQ(i1, 0);
  auto indexes = a1.find_indexes(detail::math::pi<double>());
  ASSERT_TRUE(indexes);
  EXPECT_EQ(std::get<0>(*indexes), 0);
  EXPECT_EQ(std::get<1>(*indexes), 1);
  auto angle = detail::math::radians(370.0);
  indexes = a1.find_indexes(angle);
  ASSERT_TRUE(indexes);
  EXPECT_TRUE(a1(std::get<0>(*indexes)) <= a1.normalize_coordinate(angle) &&
              a1.normalize_coordinate(angle) <= a1(std::get<1>(*indexes)));
  angle = detail::math::radians(-9.5);
  indexes = a1.find_indexes(angle);
  ASSERT_TRUE(indexes);
  EXPECT_TRUE(a1(std::get<0>(*indexes)) <= a1.normalize_coordinate(angle) &&
              a1.normalize_coordinate(angle) <= a1(std::get<1>(*indexes)));
  EXPECT_EQ(static_cast<std::string>(a1),
            "Axis([0, 0.00872665, 0.0174533, ..., 3.12414, 3.13287], "
            "is_circle=true, is_radian=true)");
}

TEST(axis, irregular) {
  // axis with irregular pitch between values
  int64_t i1;
  std::vector<double> values;
  values.push_back(-89.000000);
  values.push_back(-88.908818);
  values.push_back(-88.809323);
  values.push_back(-88.700757);
  values.push_back(-88.582294);
  values.push_back(-88.453032);
  values.push_back(-88.311987);
  values.push_back(-88.158087);
  values.push_back(-87.990161);
  values.push_back(-87.806932);
  values.push_back(-87.607008);
  values.push_back(-87.388869);
  values.push_back(-87.150861);
  values.push_back(-86.891178);
  values.push_back(-86.607851);
  values.push_back(-86.298736);
  values.push_back(-85.961495);
  values.push_back(-85.593582);
  values.push_back(-85.192224);
  values.push_back(-84.754402);
  values.push_back(-84.276831);
  values.push_back(-83.755939);
  values.push_back(-83.187844);
  values.push_back(-82.568330);
  values.push_back(-81.892820);
  values.push_back(-81.156357);
  values.push_back(-80.353575);
  values.push_back(-79.478674);
  values.push_back(-78.525397);
  values.push_back(-77.487013);
  values.push_back(-76.356296);
  values.push_back(-75.125518);
  values.push_back(-73.786444);
  values.push_back(-72.330344);
  values.push_back(-70.748017);
  values.push_back(-69.029837);
  values.push_back(-67.165823);
  values.push_back(-65.145744);
  values.push_back(-62.959262);
  values.push_back(-60.596124);
  values.push_back(-58.046413);
  values.push_back(-55.300856);
  values.push_back(-52.351206);
  values.push_back(-49.190700);
  values.push_back(-45.814573);
  values.push_back(-42.220632);
  values.push_back(-38.409866);
  values.push_back(-34.387043);
  values.push_back(-30.161252);
  values.push_back(-25.746331);
  values.push_back(-21.161107);
  values.push_back(-16.429384);
  values.push_back(-11.579629);
  values.push_back(-6.644331);
  values.push_back(-1.659041);
  values.push_back(3.338836);
  values.push_back(8.311423);
  values.push_back(13.221792);
  values.push_back(18.035297);
  values.push_back(22.720709);
  values.push_back(27.251074);
  values.push_back(31.604243);
  values.push_back(35.763079);
  values.push_back(39.715378);
  values.push_back(43.453560);
  values.push_back(46.974192);
  values.push_back(50.277423);
  values.push_back(53.366377);
  values.push_back(56.246554);
  values.push_back(58.925270);
  values.push_back(61.411164);
  values.push_back(63.713764);
  values.push_back(65.843134);
  values.push_back(67.809578);
  values.push_back(69.623418);
  values.push_back(71.294813);
  values.push_back(72.833637);
  values.push_back(74.249378);
  values.push_back(75.551083);
  values.push_back(76.747318);
  values.push_back(77.846146);
  values.push_back(78.855128);
  values.push_back(79.781321);
  values.push_back(80.631294);
  values.push_back(81.411149);
  values.push_back(82.126535);
  values.push_back(82.782681);
  values.push_back(83.384411);
  values.push_back(83.936179);
  values.push_back(84.442084);
  values.push_back(84.905904);
  values.push_back(85.331111);
  values.push_back(85.720897);
  values.push_back(86.078198);
  values.push_back(86.405707);
  values.push_back(86.705898);
  values.push_back(86.981044);
  values.push_back(87.233227);
  values.push_back(87.464359);
  values.push_back(87.676195);
  values.push_back(87.870342);
  values.push_back(88.048275);
  values.push_back(88.211348);
  values.push_back(88.360799);
  values.push_back(88.497766);
  values.push_back(88.623291);
  values.push_back(88.738328);
  values.push_back(88.843755);
  values.push_back(88.940374);

  detail::Axis axis(Eigen::Map<Eigen::VectorXd>(values.data(), values.size()),
                    1e-6, false, false);
  EXPECT_EQ(axis.front(), -89);
  double inc;
  EXPECT_THROW(inc = axis.increment(), std::logic_error);
  EXPECT_FALSE(axis.is_circle());
  EXPECT_TRUE(axis.is_ascending());
  EXPECT_EQ(axis.is_regular(), false);
  EXPECT_EQ(axis.min_value(), -89);
  EXPECT_EQ(axis.max_value(), 88.940374);
  EXPECT_EQ(axis.front(), -89);
  EXPECT_EQ(axis.back(), 88.940374);
  EXPECT_EQ(axis.size(), values.size());
  EXPECT_EQ(axis.coordinate_value(0), -89);
  EXPECT_EQ(axis.coordinate_value(108), 88.940374);
  double value;
  EXPECT_THROW(value = axis.coordinate_value(360), std::exception);
  i1 = axis.find_index(-1.659041, false);
  EXPECT_EQ(i1, 54);
  i1 = axis.find_index(-88.700757, false);
  EXPECT_EQ(i1, 3);
  i1 = axis.find_index(88.497766, false);
  EXPECT_EQ(i1, 104);
  i1 = axis.find_index(0, false);
  EXPECT_EQ(i1, 54);
  i1 = axis.find_index(-90, false);
  EXPECT_EQ(i1, -1);
  i1 = axis.find_index(-90, true);
  EXPECT_EQ(i1, 0);
  i1 = axis.find_index(90, false);
  EXPECT_EQ(i1, -1);
  i1 = axis.find_index(90, true);
  EXPECT_EQ(i1, 108);
  auto indexes = axis.find_indexes(60);
  ASSERT_TRUE(indexes);
  EXPECT_EQ(std::get<0>(*indexes), 69);
  EXPECT_EQ(std::get<1>(*indexes), 70);

  axis.flip();
  EXPECT_EQ(axis.front(), 88.940374);
  EXPECT_THROW(inc = axis.increment(), std::logic_error);
  EXPECT_FALSE(axis.is_circle());
  EXPECT_EQ(axis.is_regular(), false);
  EXPECT_EQ(axis.min_value(), -89);
  EXPECT_EQ(axis.max_value(), 88.940374);
  EXPECT_EQ(axis.size(), values.size());

  i1 = axis.find_index(-1.659041, false);
  EXPECT_EQ(i1, 54);
  i1 = axis.find_index(-88.700757, false);
  EXPECT_EQ(i1, 105);
  i1 = axis.find_index(88.497766, false);
  EXPECT_EQ(i1, 4);
  i1 = axis.find_index(0, false);
  EXPECT_EQ(i1, 54);
  i1 = axis.find_index(-90, false);
  EXPECT_EQ(i1, -1);
  i1 = axis.find_index(-90, true);
  EXPECT_EQ(i1, 108);
  i1 = axis.find_index(90, false);
  EXPECT_EQ(i1, -1);
  i1 = axis.find_index(90, true);
  EXPECT_EQ(i1, 0);

  indexes = axis.find_indexes(60);
  ASSERT_TRUE(indexes);
  EXPECT_TRUE(axis(std::get<0>(*indexes)) <= 60 &&
              60 <= axis(std::get<1>(*indexes)));
}

TEST(axis, search_indexes) {
  // search for indexes around a value on an axis
  detail::Axis axis(0, 359, 360, 1e-6, true, false);
  auto indexes = axis.find_indexes(359.4);

  ASSERT_TRUE(indexes.has_value());
  EXPECT_EQ(std::get<0>(*indexes), 359);
  EXPECT_EQ(std::get<1>(*indexes), 0);

  indexes = axis.find_indexes(359.6);
  ASSERT_TRUE(indexes.has_value());
  EXPECT_EQ(std::get<0>(*indexes), 359);
  EXPECT_EQ(std::get<1>(*indexes), 0);

  indexes = axis.find_indexes(-0.1);
  ASSERT_TRUE(indexes.has_value());
  EXPECT_EQ(std::get<0>(*indexes), 359);
  EXPECT_EQ(std::get<1>(*indexes), 0);

  indexes = axis.find_indexes(359.9);
  ASSERT_TRUE(indexes.has_value());
  EXPECT_EQ(std::get<0>(*indexes), 359);
  EXPECT_EQ(std::get<1>(*indexes), 0);

  indexes = axis.find_indexes(0.01);
  ASSERT_TRUE(indexes.has_value());
  EXPECT_EQ(std::get<0>(*indexes), 0);
  EXPECT_EQ(std::get<1>(*indexes), 1);

  indexes = axis.find_indexes(358.9);
  ASSERT_TRUE(indexes.has_value());
  EXPECT_EQ(std::get<0>(*indexes), 358);
  EXPECT_EQ(std::get<1>(*indexes), 359);

  axis = detail::Axis(10, 20, 1, 1e-6, true, false);
  EXPECT_FALSE(axis.find_indexes(20.01).has_value());
  EXPECT_FALSE(axis.find_indexes(9.9).has_value());
}

TEST(axis, search_window) {
  // search for indexes that frame a value around a window
  std::vector<int64_t> indexes;
  detail::Axis axis(-180, 179, 360, 1e-6, true, false);

  indexes = axis.find_indexes(0, 1, detail::Axis::kUndef);
  ASSERT_EQ(indexes.size(), 2);
  EXPECT_EQ(indexes[0], 180);
  EXPECT_EQ(indexes[1], 181);

  EXPECT_THROW(indexes = axis.find_indexes(0, 0, detail::Axis::kUndef),
               std::invalid_argument);

  indexes = axis.find_indexes(0, 5, detail::Axis::kUndef);
  ASSERT_EQ(indexes.size(), 10);
  EXPECT_EQ(indexes[0], 176);
  EXPECT_EQ(indexes[1], 177);
  EXPECT_EQ(indexes[2], 178);
  EXPECT_EQ(indexes[3], 179);
  EXPECT_EQ(indexes[4], 180);
  EXPECT_EQ(indexes[5], 181);
  EXPECT_EQ(indexes[6], 182);
  EXPECT_EQ(indexes[7], 183);
  EXPECT_EQ(indexes[8], 184);
  EXPECT_EQ(indexes[9], 185);

  indexes = axis.find_indexes(-180, 5, detail::Axis::kUndef);
  ASSERT_EQ(indexes.size(), 10);
  EXPECT_EQ(indexes[0], 356);
  EXPECT_EQ(indexes[1], 357);
  EXPECT_EQ(indexes[2], 358);
  EXPECT_EQ(indexes[3], 359);
  EXPECT_EQ(indexes[4], 0);
  EXPECT_EQ(indexes[5], 1);
  EXPECT_EQ(indexes[6], 2);
  EXPECT_EQ(indexes[7], 3);
  EXPECT_EQ(indexes[8], 4);
  EXPECT_EQ(indexes[9], 5);

  indexes = axis.find_indexes(179, 5, detail::Axis::kUndef);
  ASSERT_EQ(indexes.size(), 10);
  EXPECT_EQ(indexes[0], 354);
  EXPECT_EQ(indexes[1], 355);
  EXPECT_EQ(indexes[2], 356);
  EXPECT_EQ(indexes[3], 357);
  EXPECT_EQ(indexes[4], 358);
  EXPECT_EQ(indexes[5], 359);
  EXPECT_EQ(indexes[6], 0);
  EXPECT_EQ(indexes[7], 1);
  EXPECT_EQ(indexes[8], 2);
  EXPECT_EQ(indexes[9], 3);

  indexes = axis.find_indexes(179.4, 5, detail::Axis::kUndef);
  ASSERT_EQ(indexes.size(), 10);
  EXPECT_EQ(indexes[0], 355);
  EXPECT_EQ(indexes[1], 356);
  EXPECT_EQ(indexes[2], 357);
  EXPECT_EQ(indexes[3], 358);
  EXPECT_EQ(indexes[4], 359);
  EXPECT_EQ(indexes[5], 0);
  EXPECT_EQ(indexes[6], 1);
  EXPECT_EQ(indexes[7], 2);
  EXPECT_EQ(indexes[8], 3);
  EXPECT_EQ(indexes[9], 4);

  indexes = axis.find_indexes(179.6, 5, detail::Axis::kUndef);
  ASSERT_EQ(indexes.size(), 10);
  EXPECT_EQ(indexes[0], 355);
  EXPECT_EQ(indexes[1], 356);
  EXPECT_EQ(indexes[2], 357);
  EXPECT_EQ(indexes[3], 358);
  EXPECT_EQ(indexes[4], 359);
  EXPECT_EQ(indexes[5], 0);
  EXPECT_EQ(indexes[6], 1);
  EXPECT_EQ(indexes[7], 2);
  EXPECT_EQ(indexes[8], 3);
  EXPECT_EQ(indexes[9], 4);

  axis = detail::Axis(0, 9, 10, 1e-6, false, false);
  indexes = axis.find_indexes(5, 4, detail::Axis::kUndef);
  ASSERT_EQ(indexes.size(), 8);
  EXPECT_EQ(indexes[0], 2);
  EXPECT_EQ(indexes[1], 3);
  EXPECT_EQ(indexes[2], 4);
  EXPECT_EQ(indexes[3], 5);
  EXPECT_EQ(indexes[4], 6);
  EXPECT_EQ(indexes[5], 7);
  EXPECT_EQ(indexes[6], 8);
  EXPECT_EQ(indexes[7], 9);

  indexes = axis.find_indexes(-1, 4, detail::Axis::kUndef);
  EXPECT_EQ(indexes.empty(), true);
  indexes = axis.find_indexes(10, 4, detail::Axis::kUndef);
  EXPECT_EQ(indexes.empty(), true);

  indexes = axis.find_indexes(1, 4, detail::Axis::kSym);
  ASSERT_EQ(indexes.size(), 8);
  EXPECT_EQ(indexes[0], 2);
  EXPECT_EQ(indexes[1], 1);
  EXPECT_EQ(indexes[2], 0);
  EXPECT_EQ(indexes[3], 1);
  EXPECT_EQ(indexes[4], 2);
  EXPECT_EQ(indexes[5], 3);
  EXPECT_EQ(indexes[6], 4);
  EXPECT_EQ(indexes[7], 5);

  indexes = axis.find_indexes(9, 4, detail::Axis::kSym);
  ASSERT_EQ(indexes.size(), 8);
  EXPECT_EQ(indexes[0], 5);
  EXPECT_EQ(indexes[1], 6);
  EXPECT_EQ(indexes[2], 7);
  EXPECT_EQ(indexes[3], 8);
  EXPECT_EQ(indexes[4], 9);
  EXPECT_EQ(indexes[5], 8);
  EXPECT_EQ(indexes[6], 7);
  EXPECT_EQ(indexes[7], 6);

  indexes = axis.find_indexes(1, 4, detail::Axis::kWrap);
  ASSERT_EQ(indexes.size(), 8);
  EXPECT_EQ(indexes[0], 8);
  EXPECT_EQ(indexes[1], 9);
  EXPECT_EQ(indexes[2], 0);
  EXPECT_EQ(indexes[3], 1);
  EXPECT_EQ(indexes[4], 2);
  EXPECT_EQ(indexes[5], 3);
  EXPECT_EQ(indexes[6], 4);
  EXPECT_EQ(indexes[7], 5);

  indexes = axis.find_indexes(9, 4, detail::Axis::kWrap);
  ASSERT_EQ(indexes.size(), 8);
  EXPECT_EQ(indexes[0], 5);
  EXPECT_EQ(indexes[1], 6);
  EXPECT_EQ(indexes[2], 7);
  EXPECT_EQ(indexes[3], 8);
  EXPECT_EQ(indexes[4], 9);
  EXPECT_EQ(indexes[5], 0);
  EXPECT_EQ(indexes[6], 1);
  EXPECT_EQ(indexes[7], 2);

  indexes = axis.find_indexes(1, 4, detail::Axis::kExpand);
  ASSERT_EQ(indexes.size(), 8);
  EXPECT_EQ(indexes[0], 0);
  EXPECT_EQ(indexes[1], 0);
  EXPECT_EQ(indexes[2], 0);
  EXPECT_EQ(indexes[3], 1);
  EXPECT_EQ(indexes[4], 2);
  EXPECT_EQ(indexes[5], 3);
  EXPECT_EQ(indexes[6], 4);
  EXPECT_EQ(indexes[7], 5);

  indexes = axis.find_indexes(9, 4, detail::Axis::kExpand);
  ASSERT_EQ(indexes.size(), 8);
  EXPECT_EQ(indexes[0], 5);
  EXPECT_EQ(indexes[1], 6);
  EXPECT_EQ(indexes[2], 7);
  EXPECT_EQ(indexes[3], 8);
  EXPECT_EQ(indexes[4], 9);
  EXPECT_EQ(indexes[5], 9);
  EXPECT_EQ(indexes[6], 9);
  EXPECT_EQ(indexes[7], 9);

  indexes = axis.find_indexes(1, 4, detail::Axis::kUndef);
  ASSERT_TRUE(indexes.empty());
  indexes = axis.find_indexes(9, 4, detail::Axis::kUndef);
  ASSERT_TRUE(indexes.empty());
}
