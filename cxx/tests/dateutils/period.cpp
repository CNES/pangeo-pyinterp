// Copyright (c) 2026 CNES.
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.

#include "pyinterp/period.hpp"

#include <gtest/gtest.h>

namespace pyinterp {

TEST(PeriodTest, ConstructionAndProperties) {
  // [10, 20)
  Period p1(10, 20, false);
  EXPECT_EQ(p1.begin, 10);
  EXPECT_EQ(p1.last, 19);
  EXPECT_EQ(p1.end(), 20);
  EXPECT_EQ(p1.duration(), 10);
  EXPECT_FALSE(p1.is_null());

  // [10, 19]
  Period p2(10, 19, true);
  EXPECT_EQ(p2.begin, 10);
  EXPECT_EQ(p2.last, 19);
  EXPECT_EQ(p2.end(), 20);
  EXPECT_EQ(p1, p2);

  // Null period
  Period null_p(20, 10, false);
  EXPECT_TRUE(null_p.is_null());
}

TEST(PeriodTest, ContainsAndPosition) {
  Period p(10, 20);  // [10, 20] -> [10, 21)

  // Point containment
  EXPECT_TRUE(p.contains(10));
  EXPECT_TRUE(p.contains(20));
  EXPECT_FALSE(p.contains(9));
  EXPECT_FALSE(p.contains(21));

  // Period containment
  EXPECT_TRUE(p.contains(Period(12, 18)));
  EXPECT_TRUE(p.contains(p));
  EXPECT_FALSE(p.contains(Period(9, 15)));

  // Relative position
  EXPECT_TRUE(p.is_after(9));
  EXPECT_FALSE(p.is_after(10));
  EXPECT_TRUE(p.is_before(21));
  EXPECT_FALSE(p.is_before(20));
}

TEST(PeriodTest, IntersectionAndMerge) {
  Period p1(10, 20);  // [10, 20]
  Period p2(15, 25);  // [15, 25]
  Period p3(21, 30);  // [21, 30] (Adjacent to p1)
  Period p4(30, 40);  // [30, 40] (Disjoint)

  // Intersects
  EXPECT_TRUE(p1.intersects(p2));
  EXPECT_FALSE(p1.intersects(p3));
  EXPECT_FALSE(p1.intersects(p4));

  // Intersection
  Period i12 = p1.intersection(p2);
  EXPECT_EQ(i12.begin, 15);
  EXPECT_EQ(i12.last, 20);

  EXPECT_TRUE(p1.intersection(p3).is_null());

  // Adjacent
  EXPECT_TRUE(p1.is_adjacent(p3));
  EXPECT_FALSE(p1.is_adjacent(p2));  // Overlapping, not just adjacent

  // Merge
  Period m12 = p1.merge(p2);  // [10, 25]
  EXPECT_EQ(m12.begin, 10);
  EXPECT_EQ(m12.last, 25);

  Period m13 = p1.merge(p3);  // [10, 30]
  EXPECT_EQ(m13.begin, 10);
  EXPECT_EQ(m13.last, 30);

  EXPECT_TRUE(p1.merge(p4).is_null());
}

TEST(PeriodTest, ExtendAndShift) {
  Period p(10, 20);  // [10, 20]

  // Extend with point inside - no change
  Period e1 = p.extend(15);
  EXPECT_EQ(e1.begin, 10);
  EXPECT_EQ(e1.last, 20);

  // Extend with point before - expands begin
  Period e2 = p.extend(5);
  EXPECT_EQ(e2.begin, 5);
  EXPECT_EQ(e2.last, 20);

  // Extend with point after - expands last
  Period e3 = p.extend(25);
  EXPECT_EQ(e3.begin, 10);
  EXPECT_EQ(e3.last, 25);

  // Shift forward
  Period s1 = p.shift(10);
  EXPECT_EQ(s1.begin, 20);
  EXPECT_EQ(s1.last, 30);
  EXPECT_EQ(s1.duration(), p.duration());

  // Shift backward
  Period s2 = p.shift(-5);
  EXPECT_EQ(s2.begin, 5);
  EXPECT_EQ(s2.last, 15);
  EXPECT_EQ(s2.duration(), p.duration());

  // Shift by zero
  Period s3 = p.shift(0);
  EXPECT_EQ(s3.begin, p.begin);
  EXPECT_EQ(s3.last, p.last);
}

TEST(PeriodTest, IsClose) {
  Period p(100, 200);  // [100, 200]

  // Exact boundaries
  EXPECT_TRUE(p.is_close(100, 0));
  EXPECT_TRUE(p.is_close(200, 0));

  // Inside period
  EXPECT_TRUE(p.is_close(150, 0));

  // Within tolerance before begin
  EXPECT_TRUE(p.is_close(90, 15));
  EXPECT_FALSE(p.is_close(90, 5));

  // Within tolerance after last
  EXPECT_TRUE(p.is_close(210, 15));
  EXPECT_FALSE(p.is_close(210, 5));

  // Outside tolerance
  EXPECT_FALSE(p.is_close(50, 10));
  EXPECT_FALSE(p.is_close(250, 10));
}

TEST(PeriodTest, NullPeriodBehavior) {
  Period null_p;
  EXPECT_TRUE(null_p.is_null());
  EXPECT_EQ(null_p.begin, 0);
  EXPECT_EQ(null_p.last, -1);
  EXPECT_EQ(null_p.end(), 0);
  EXPECT_EQ(null_p.duration(), 0);  // end() - begin = 0 - 0 = 0

  Period valid_p(10, 20);

  // Null period operations
  EXPECT_FALSE(null_p.contains(0));
  EXPECT_FALSE(null_p.contains(valid_p));
  EXPECT_FALSE(null_p.is_after(5));
  EXPECT_FALSE(null_p.is_before(5));
  EXPECT_FALSE(null_p.intersects(valid_p));
  EXPECT_TRUE(null_p.intersection(valid_p).is_null());
  EXPECT_TRUE(null_p.merge(valid_p).is_null());
}

TEST(PeriodTest, NegativeValues) {
  // Period with negative values
  Period p(-100, -50);  // [-100, -50]
  EXPECT_EQ(p.begin, -100);
  EXPECT_EQ(p.last, -50);
  EXPECT_EQ(p.end(), -49);
  EXPECT_EQ(p.duration(), 51);
  EXPECT_FALSE(p.is_null());

  // Contains negative point
  EXPECT_TRUE(p.contains(-75));
  EXPECT_FALSE(p.contains(-25));

  // Operations with negative periods
  Period p2(-75, -25);
  EXPECT_TRUE(p.intersects(p2));
  Period i = p.intersection(p2);
  EXPECT_EQ(i.begin, -75);
  EXPECT_EQ(i.last, -50);
}

TEST(PeriodTest, ComparisonOperators) {
  Period p1(10, 20);
  Period p2(10, 20);
  Period p3(10, 25);
  Period p4(15, 20);

  // Equality
  EXPECT_EQ(p1, p2);
  EXPECT_NE(p1, p3);
  EXPECT_NE(p1, p4);

  // Ordering (lexicographic on begin, then last)
  EXPECT_LT(p1, p3);
  EXPECT_LT(p1, p4);
  EXPECT_GT(p3, p1);
  EXPECT_LE(p1, p2);
  EXPECT_GE(p1, p2);
}

TEST(PeriodTest, EdgeCases) {
  // Single point period [5, 5]
  Period single(5, 5);
  EXPECT_EQ(single.begin, 5);
  EXPECT_EQ(single.last, 5);
  EXPECT_EQ(single.duration(), 1);
  EXPECT_TRUE(single.contains(5));
  EXPECT_FALSE(single.contains(6));

  // Adjacent periods
  Period p1(0, 10);   // [0, 10]
  Period p2(11, 20);  // [11, 20]
  EXPECT_TRUE(p1.is_adjacent(p2));
  EXPECT_TRUE(p2.is_adjacent(p1));
  EXPECT_FALSE(p1.intersects(p2));

  // Merge adjacent creates continuous period
  Period merged = p1.merge(p2);
  EXPECT_EQ(merged.begin, 0);
  EXPECT_EQ(merged.last, 20);
  EXPECT_FALSE(merged.is_null());
}

TEST(PeriodListTest, StorageAndSorting) {
  PeriodList list;
  list.push_back(Period(10, 15));
  list.push_back(Period(1, 5));
  list.push_back(Period(20, 25));

  EXPECT_EQ(list.size(), 3);
  EXPECT_FALSE(list.is_sorted_and_disjoint());

  list.sort();
  EXPECT_EQ(list[0].begin, 1);
  EXPECT_EQ(list[1].begin, 10);
  EXPECT_EQ(list[2].begin, 20);
  EXPECT_TRUE(list.is_sorted_and_disjoint());

  list.clear();
  EXPECT_EQ(list.size(), 0);
}

TEST(PeriodListTest, MergeInPlace) {
  PeriodList list1;
  list1.push_back(Period(10, 20));
  list1.push_back(Period(40, 50));

  PeriodList list2;
  list2.push_back(Period(15, 25));
  list2.push_back(Period(60, 70));

  // Merge list2 into list1 (modifies list1 in place)
  list1.merge(list2);

  // Should have merged overlapping periods
  EXPECT_GE(list1.size(), 3);  // At least 3 periods after merge

  // Verify merged list contains all ranges
  EXPECT_NE(list1.find_containing(15), nullptr);
  EXPECT_NE(list1.find_containing(45), nullptr);
  EXPECT_NE(list1.find_containing(65), nullptr);
}

TEST(PeriodListTest, MergeAdjacentPeriods) {
  PeriodList list1;
  list1.push_back(Period(10, 20));
  list1.push_back(Period(21, 30));  // Adjacent to first

  PeriodList list2;
  list2.push_back(Period(31, 40));  // Adjacent to second

  list1.merge(list2);

  // Should merge all adjacent periods into one
  EXPECT_EQ(list1.size(), 1);
  EXPECT_EQ(list1[0].begin, 10);
  EXPECT_EQ(list1[0].last, 40);
}

TEST(PeriodListTest, MergeDisjointPeriods) {
  PeriodList list1;
  list1.push_back(Period(10, 20));

  PeriodList list2;
  list2.push_back(Period(50, 60));

  list1.merge(list2);

  // Should keep both periods separate
  EXPECT_EQ(list1.size(), 2);
  EXPECT_EQ(list1[0].begin, 10);
  EXPECT_EQ(list1[1].begin, 50);
}

TEST(PeriodListTest, MergeOverlappingPeriods) {
  PeriodList list1;
  list1.push_back(Period(10, 30));
  list1.push_back(Period(50, 70));

  PeriodList list2;
  list2.push_back(Period(20, 40));  // Overlaps with first
  list2.push_back(Period(45, 55));  // Overlaps with second

  list1.merge(list2);

  // Should merge overlapping periods
  EXPECT_EQ(list1.size(), 2);
  EXPECT_EQ(list1[0].begin, 10);
  EXPECT_EQ(list1[0].last, 40);
  EXPECT_EQ(list1[1].begin, 45);
  EXPECT_EQ(list1[1].last, 70);
}

TEST(PeriodListTest, MergeEmptyLists) {
  PeriodList list1;
  PeriodList list2;

  list1.merge(list2);
  EXPECT_EQ(list1.size(), 0);

  list1.push_back(Period(10, 20));
  list1.merge(list2);
  EXPECT_EQ(list1.size(), 1);
  EXPECT_EQ(list1[0].begin, 10);
}

TEST(PeriodListTest, MergeComplexScenario) {
  PeriodList list1;
  list1.push_back(Period(10, 20));
  list1.push_back(Period(30, 40));
  list1.push_back(Period(60, 70));

  PeriodList list2;
  list2.push_back(Period(15, 35));  // Bridges first two periods
  list2.push_back(Period(80, 90));  // New period after all

  list1.merge(list2);

  // Should merge first two periods into one, keep others separate
  EXPECT_EQ(list1.size(), 3);
  EXPECT_EQ(list1[0].begin, 10);
  EXPECT_EQ(list1[0].last, 40);
  EXPECT_EQ(list1[1].begin, 60);
  EXPECT_EQ(list1[1].last, 70);
  EXPECT_EQ(list1[2].begin, 80);
  EXPECT_EQ(list1[2].last, 90);
}

TEST(PeriodListTest, JoinAdjacentPeriods) {
  PeriodList list;
  list.push_back(Period(10, 20));
  list.push_back(Period(22, 30));  // Gap of 1 (22 - 20 = 1, since last = 20)
  list.push_back(Period(32, 40));  // Gap of 1

  // Note: join_adjacent_periods returns void in the updated implementation
  // So we need to check the return type first
  auto result = list.join_adjacent_periods(5);  // Epsilon = 5, should join all

  EXPECT_EQ(result.size(), 1);
  EXPECT_EQ(result[0].begin, 10);
  EXPECT_EQ(result[0].last, 40);
}

TEST(PeriodListTest, JoinAdjacentPeriodsStrictEpsilon) {
  PeriodList list;
  list.push_back(Period(10, 20));
  list.push_back(Period(22, 30));  // Gap of 1
  list.push_back(Period(35, 40));  // Gap of 4

  auto result =
      list.join_adjacent_periods(2);  // Epsilon = 2, joins first two only

  EXPECT_EQ(result.size(), 2);
  EXPECT_EQ(result[0].begin, 10);
  EXPECT_EQ(result[0].last, 30);
  EXPECT_EQ(result[1].begin, 35);
  EXPECT_EQ(result[1].last, 40);
}

TEST(PeriodListTest, JoinAdjacentPeriodsEmptyAndSingle) {
  PeriodList empty;
  auto result_empty = empty.join_adjacent_periods(5);
  EXPECT_EQ(result_empty.size(), 0);

  PeriodList single;
  single.push_back(Period(10, 20));
  auto result_single = single.join_adjacent_periods(5);
  EXPECT_EQ(result_single.size(), 1);
  EXPECT_EQ(result_single[0].begin, 10);
  EXPECT_EQ(result_single[0].last, 20);
}

TEST(PeriodListTest, FindAndClose) {
  PeriodList list;  // ns default
  list.push_back(Period(100, 200));
  list.push_back(Period(300, 400));
  list.sort();

  // Find containing
  EXPECT_EQ(list.find_containing_index(150), 0);
  EXPECT_EQ(list.find_containing_index(300), 1);
  EXPECT_EQ(list.find_containing_index(250), -1);

  // Is close
  EXPECT_TRUE(list.is_close(90, 15));  // 90 is within 15 of 100
  EXPECT_FALSE(list.is_close(90, 5));
}

TEST(PeriodListTest, Duration) {
  PeriodList list;
  EXPECT_EQ(list.duration(), 0);  // Empty list

  list.push_back(Period(10, 20));  // [10, 20]
  EXPECT_EQ(list.duration(), 11);  // Single element: 20 - 10 + 1 = 11

  list.push_back(Period(30, 40));  // [30, 40]
  list.sort();
  // Duration from first begin to last end: 40 - 10 + 1 = 31
  EXPECT_EQ(list.duration(), 31);

  list.push_back(Period(50, 60));  // [50, 60]
  list.sort();
  // Duration from first begin to last end: 60 - 10 + 1 = 51
  EXPECT_EQ(list.duration(), 51);
}

TEST(PeriodListTest, AggregateDuration) {
  PeriodList list;
  EXPECT_EQ(list.aggregate_duration(), 0);  // Empty list

  list.push_back(Period(0, 10));   // duration = 11
  list.push_back(Period(20, 30));  // duration = 11
  list.push_back(Period(40, 45));  // duration = 6

  EXPECT_EQ(list.aggregate_duration(), 28);  // 11 + 11 + 6
}

TEST(PeriodListTest, AggregateDurationWithOverlap) {
  PeriodList list;
  list.push_back(Period(0, 10));   // [0, 10]
  list.push_back(Period(5, 15));   // [5, 15] - overlaps with first
  list.push_back(Period(20, 30));  // [20, 30] - disjoint
  list.sort();

  // Aggregate should merge overlapping periods: [0, 15] + [20, 30] = 16 + 11 =
  // 27
  EXPECT_EQ(list.aggregate_duration(), 27);
}

TEST(PeriodListTest, DurationVsAggregateDuration) {
  PeriodList list;
  list.push_back(Period(0, 10));   // [0, 10]
  list.push_back(Period(50, 60));  // [50, 60]
  list.sort();

  // duration() gives span from first to last
  EXPECT_EQ(list.duration(), 61);  // 60 - 0 + 1 = 61

  // aggregate_duration() gives sum of individual durations
  EXPECT_EQ(list.aggregate_duration(), 22);  // 11 + 11 = 22
}

TEST(PeriodListTest, OverlappingPeriods) {
  PeriodList list;
  list.push_back(Period(10, 20));
  list.push_back(Period(15, 25));  // Overlaps with first

  EXPECT_FALSE(list.is_sorted_and_disjoint());

  // After sorting, still overlapping
  list.sort();
  EXPECT_FALSE(list.is_sorted_and_disjoint());
}

TEST(PeriodListTest, FindContainingEdgeCases) {
  PeriodList list;
  list.push_back(Period(10, 20));
  list.push_back(Period(30, 40));
  list.push_back(Period(50, 60));
  list.sort();

  // Find at boundaries
  EXPECT_EQ(list.find_containing_index(10), 0);   // Begin of first
  EXPECT_EQ(list.find_containing_index(20), 0);   // Last of first
  EXPECT_EQ(list.find_containing_index(21), -1);  // Gap
  EXPECT_EQ(list.find_containing_index(60), 2);   // Last of last

  // Find in gaps
  EXPECT_EQ(list.find_containing_index(25), -1);
  EXPECT_EQ(list.find_containing_index(45), -1);

  // Find before all periods
  EXPECT_EQ(list.find_containing_index(5), -1);

  // Find after all periods
  EXPECT_EQ(list.find_containing_index(65), -1);

  // Verify find_containing returns nullptr for non-existent
  EXPECT_EQ(list.find_containing(25), nullptr);
  EXPECT_NE(list.find_containing(30), nullptr);
}

TEST(PeriodListTest, EmptyList) {
  PeriodList list;

  EXPECT_TRUE(list.is_sorted_and_disjoint());  // Empty is trivially sorted
  EXPECT_EQ(list.aggregate_duration(), 0);
  EXPECT_EQ(list.find_containing_index(100), -1);
  EXPECT_EQ(list.find_containing(100), nullptr);
  EXPECT_FALSE(list.is_close(100, 10));
}

TEST(PeriodListTest, SingleElement) {
  PeriodList list;
  list.push_back(Period(10, 20));

  EXPECT_TRUE(list.is_sorted_and_disjoint());
  EXPECT_EQ(list.aggregate_duration(), 11);
  EXPECT_EQ(list.find_containing_index(15), 0);
  EXPECT_EQ(list.find_containing_index(5), -1);
}

TEST(PeriodListTest, FilterContained) {
  PeriodList list;
  list.push_back(Period(10, 20));
  list.push_back(Period(30, 40));
  list.push_back(Period(50, 60));
  list.push_back(Period(70, 80));

  // Filter by a period that contains some but not all
  Period container(25, 65);
  auto filtered = list.filter_contained(container);

  EXPECT_EQ(filtered.size(), 2);
  EXPECT_EQ(filtered[0].begin, 30);
  EXPECT_EQ(filtered[0].last, 40);
  EXPECT_EQ(filtered[1].begin, 50);
  EXPECT_EQ(filtered[1].last, 60);
}

TEST(PeriodListTest, FilterContainedNone) {
  PeriodList list;
  list.push_back(Period(10, 20));
  list.push_back(Period(30, 40));

  Period container(50, 60);  // Contains none
  auto filtered = list.filter_contained(container);

  EXPECT_EQ(filtered.size(), 0);
}

TEST(PeriodListTest, FilterMinDuration) {
  PeriodList list;
  list.push_back(Period(10, 12));   // duration = 3
  list.push_back(Period(20, 30));   // duration = 11
  list.push_back(Period(40, 45));   // duration = 6
  list.push_back(Period(50, 100));  // duration = 51

  auto filtered = list.filter_min_duration(10);

  EXPECT_EQ(filtered.size(), 2);
  EXPECT_EQ(filtered[0].begin, 20);
  EXPECT_EQ(filtered[1].begin, 50);
}

TEST(PeriodListTest, FilterMinDurationNone) {
  PeriodList list;
  list.push_back(Period(10, 12));  // duration = 3
  list.push_back(Period(20, 22));  // duration = 3

  auto filtered = list.filter_min_duration(10);

  EXPECT_EQ(filtered.size(), 0);
}

TEST(PeriodListTest, CrossAPeriod) {
  PeriodList list;
  list.push_back(Period(10, 20));
  list.push_back(Period(30, 40));
  list.push_back(Period(50, 60));

  Vector<int64_t> dates(6);
  dates << 5, 15, 25, 35, 45, 55;

  auto flags = list.cross_a_period(dates);

  EXPECT_EQ(flags.size(), 6);
  // Date 5: before first period, next period exists -> true
  EXPECT_TRUE(flags(0));
  // Date 15: in first period -> true
  EXPECT_TRUE(flags(1));
  // Date 25: between periods, next period exists -> true
  EXPECT_TRUE(flags(2));
  // Date 35: in second period -> true
  EXPECT_TRUE(flags(3));
  // Date 45: between periods, next period exists -> true
  EXPECT_TRUE(flags(4));
  // Date 55: in last period -> all true from optimization
  EXPECT_TRUE(flags(5));
}

TEST(PeriodListTest, BelongToAPeriod) {
  PeriodList list;
  list.push_back(Period(10, 20));
  list.push_back(Period(30, 40));
  list.push_back(Period(50, 60));

  Vector<int64_t> dates(6);
  dates << 5, 15, 25, 35, 45, 55;

  auto flags = list.belong_to_a_period(dates);

  EXPECT_EQ(flags.size(), 6);
  EXPECT_FALSE(flags(0));  // 5: before all periods
  EXPECT_TRUE(flags(1));   // 15: in first period [10, 20]
  EXPECT_FALSE(flags(2));  // 25: gap between periods
  EXPECT_TRUE(flags(3));   // 35: in second period [30, 40]
  EXPECT_FALSE(flags(4));  // 45: gap between periods
  EXPECT_TRUE(flags(5));   // 55: in third period [50, 60]
}

TEST(PeriodListTest, BelongToAPeriodAllOutside) {
  PeriodList list;
  list.push_back(Period(10, 20));
  list.push_back(Period(30, 40));

  Vector<int64_t> dates(3);
  dates << 5, 25, 45;

  auto flags = list.belong_to_a_period(dates);

  EXPECT_EQ(flags.size(), 3);
  EXPECT_FALSE(flags(0));
  EXPECT_FALSE(flags(1));
  EXPECT_FALSE(flags(2));
}

TEST(PeriodListTest, BelongToAPeriodAllInside) {
  PeriodList list;
  list.push_back(Period(0, 100));

  Vector<int64_t> dates(5);
  dates << 10, 25, 50, 75, 90;

  auto flags = list.belong_to_a_period(dates);

  EXPECT_EQ(flags.size(), 5);
  for (int i = 0; i < 5; ++i) {
    EXPECT_TRUE(flags(i));
  }
}

TEST(PeriodListTest, BelongToAPeriodBoundaries) {
  PeriodList list;
  list.push_back(Period(10, 20));
  list.push_back(Period(30, 40));

  Vector<int64_t> dates(4);
  dates << 10, 20, 30, 40;  // All on boundaries

  auto flags = list.belong_to_a_period(dates);

  EXPECT_EQ(flags.size(), 4);
  EXPECT_TRUE(flags(0));  // 10: begin of first period
  EXPECT_TRUE(flags(1));  // 20: last of first period
  EXPECT_TRUE(flags(2));  // 30: begin of second period
  EXPECT_TRUE(flags(3));  // 40: last of second period
}

TEST(PeriodListTest, CrossAPeriodEdgeCases) {
  PeriodList list;
  list.push_back(Period(10, 20));
  list.push_back(Period(30, 40));

  Vector<int64_t> dates(5);
  dates << 5, 25, 35, 45, 50;

  auto flags = list.cross_a_period(dates);

  EXPECT_EQ(flags.size(), 5);
  // The algorithm finds the period containing or after the last date (50).
  // Since 50 is after all periods, it stops processing dates when it reaches
  // the period that would be after the last date.
  EXPECT_TRUE(flags(0));  // 5: before first period, has periods after
  EXPECT_TRUE(flags(1));  // 25: between periods, has periods after
  EXPECT_TRUE(flags(2));  // 35: in second period
  // Once we reach dates that are after all periods or at the last_index,
  // the algorithm stops marking them as true
  EXPECT_FALSE(flags(3));  // 45: after all periods
  EXPECT_FALSE(flags(4));  // 50: after all periods (last date)
}

TEST(PeriodListTest, CrossAPeriodLastDateInPeriod) {
  PeriodList list;
  list.push_back(Period(10, 20));
  list.push_back(Period(30, 50));

  Vector<int64_t> dates(3);
  dates << 5, 25, 40;  // Last date (40) is in second period

  auto flags = list.cross_a_period(dates);

  EXPECT_EQ(flags.size(), 3);
  // If last date is in a period, all should be true (optimization)
  EXPECT_TRUE(flags(0));
  EXPECT_TRUE(flags(1));
  EXPECT_TRUE(flags(2));
}

TEST(PeriodListTest, CrossAPeriodEmpty) {
  PeriodList list;  // Empty list

  Vector<int64_t> dates(3);
  dates << 5, 10, 15;

  auto flags = list.cross_a_period(dates);

  EXPECT_EQ(flags.size(), 3);
  EXPECT_FALSE(flags(0));
  EXPECT_FALSE(flags(1));
  EXPECT_FALSE(flags(2));
}

TEST(PeriodListTest, BelongToAPeriodEmpty) {
  PeriodList list;  // Empty list

  Vector<int64_t> dates(3);
  dates << 5, 10, 15;

  auto flags = list.belong_to_a_period(dates);

  EXPECT_EQ(flags.size(), 3);
  EXPECT_FALSE(flags(0));
  EXPECT_FALSE(flags(1));
  EXPECT_FALSE(flags(2));
}

TEST(PeriodListTest, JoinAdjacentPeriodsNoJoin) {
  PeriodList list;
  list.push_back(Period(10, 20));
  list.push_back(Period(30, 40));  // Gap of 9
  list.push_back(Period(50, 60));  // Gap of 9

  auto result = list.join_adjacent_periods(5);  // Epsilon too small

  EXPECT_EQ(result.size(), 3);  // No periods joined
  EXPECT_EQ(result[0].begin, 10);
  EXPECT_EQ(result[1].begin, 30);
  EXPECT_EQ(result[2].begin, 50);
}

TEST(PeriodListTest, IsSortedAndDisjointVariousCases) {
  PeriodList list;
  EXPECT_TRUE(list.is_sorted_and_disjoint());  // Empty

  list.push_back(Period(10, 20));
  EXPECT_TRUE(list.is_sorted_and_disjoint());  // Single element

  list.push_back(Period(30, 40));
  EXPECT_TRUE(list.is_sorted_and_disjoint());  // Two disjoint

  list.push_back(Period(25, 35));  // Out of order
  EXPECT_FALSE(list.is_sorted_and_disjoint());

  list.clear();
  list.push_back(Period(10, 20));
  list.push_back(Period(15, 25));  // Overlapping
  EXPECT_FALSE(list.is_sorted_and_disjoint());
}

TEST(PeriodListTest, SortPreservesData) {
  PeriodList list;
  list.push_back(Period(50, 60));
  list.push_back(Period(10, 20));
  list.push_back(Period(30, 40));

  list.sort();

  EXPECT_EQ(list.size(), 3);
  EXPECT_EQ(list[0].begin, 10);
  EXPECT_EQ(list[0].last, 20);
  EXPECT_EQ(list[1].begin, 30);
  EXPECT_EQ(list[1].last, 40);
  EXPECT_EQ(list[2].begin, 50);
  EXPECT_EQ(list[2].last, 60);
  EXPECT_TRUE(list.is_sorted_and_disjoint());
}

}  // namespace pyinterp
