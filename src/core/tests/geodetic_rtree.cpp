#include "pyinterp/detail/geodetic/rtree.hpp"
#include <gtest/gtest.h>

namespace geodetic = pyinterp::detail::geodetic;

TEST(geodetic, rtree) {
  auto lon = Eigen::RowVectorXd::LinSpaced(11, -179, 179);
  auto lat = Eigen::RowVectorXd::LinSpaced(11, -80, 80);

  auto mlon = Eigen::Matrix<double, -1, -1>();
  auto mlat = Eigen::Matrix<double, -1, -1>();

  mlon.resize(lon.size(), lat.size());
  mlat.resize(lon.size(), lat.size());

  for (auto ix = 0; ix < lon.size(); ++ix) {
    mlon.row(ix) = lon.transpose();
  }
  for (auto ix = 0; ix < lat.size(); ++ix) {
    mlat.col(ix) = lat;
  }

  auto coordinates = Eigen::Matrix<double, -1, -1>();
  coordinates.resize(mlon.size(), 2);
  coordinates.col(0) = Eigen::Map<Eigen::VectorXd>(mlon.data(), mlon.size());
  coordinates.col(1) = Eigen::Map<Eigen::VectorXd>(mlat.data(), mlat.size());

  auto rtree = geodetic::RTree<double, int>({});
  EXPECT_TRUE(rtree.empty());
  rtree.packing(coordinates,
                Eigen::RowVectorXi::LinSpaced(coordinates.rows(), 0,
                                              coordinates.rows() - 1));
  ASSERT_EQ(rtree.size(), coordinates.rows());
  EXPECT_FALSE(rtree.empty());

  auto bounds = rtree.equatorial_bounds();
  ASSERT_TRUE(bounds);
  EXPECT_EQ(boost::geometry::get<0>(bounds->min_corner()), -179);
  EXPECT_EQ(boost::geometry::get<1>(bounds->min_corner()), -80);
  EXPECT_NEAR(boost::geometry::get<2>(bounds->min_corner()), 0, 1e-8);
  EXPECT_EQ(boost::geometry::get<0>(bounds->max_corner()), 179);
  EXPECT_EQ(boost::geometry::get<1>(bounds->max_corner()), 80);
  EXPECT_NEAR(boost::geometry::get<2>(bounds->max_corner()), 0, 1e-8);

  auto nearest = rtree.query({0, 0}, 1);
  ASSERT_TRUE(nearest.size());
  EXPECT_EQ(nearest[0].first, 0);
  EXPECT_EQ(nearest[0].second, 60);

  nearest = rtree.query_ball({0, 0}, 1);
  ASSERT_TRUE(nearest.size());
  EXPECT_EQ(nearest[0].first, 0);
  EXPECT_EQ(nearest[0].second, 60);

  nearest = rtree.query_within({0, 0}, 1);
  ASSERT_TRUE(nearest.size());
  EXPECT_EQ(nearest[0].first, 0);
  EXPECT_EQ(nearest[0].second, 60);

  nearest = rtree.query_within({0, 90}, 1);
  EXPECT_EQ(nearest.size(), 0);

  rtree.clear();
  EXPECT_EQ(rtree.size(), 0);
  EXPECT_TRUE(rtree.empty());

  rtree.insert(coordinates, Eigen::RowVectorXi::LinSpaced(
                                coordinates.rows(), 0, coordinates.rows() - 1));
  ASSERT_EQ(rtree.size(), coordinates.rows());
  EXPECT_FALSE(rtree.empty());

  auto nearests = rtree.query(coordinates, 4, false, 0);
  for (auto ix = 0ULL; ix < coordinates.rows(); ++ix) {
    for (auto jx = 0ULL; jx < 4; ++jx) {
      // EXPECT_EQ(std::get<0>(nearests)(ix, jx), 0);
    }
  }
}
