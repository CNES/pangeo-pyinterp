# Copyright (c) 2025 CNES.
#
# This software is distributed by the CNES under a proprietary license.
# It is not public and cannot be redistributed or used without permission.
"""Merge multiple xUnit1 XML files into one."""

import argparse
import xml.etree.ElementTree as ET
import sys
import os
from dataclasses import dataclass


@dataclass
class Stats:
    """Class to hold statistics for merged xUnit XML files."""

    #: Total number of tests
    tests: int = 0
    #: Total number of failures
    failures: int = 0
    #: Total number of errors
    errors: int = 0
    #: Total number of skipped tests
    skipped: int = 0
    #: Total time taken for tests
    time: float = 0.0

    def update(self, element: ET.Element) -> None:
        """Update statistics dictionary with values from an XML element."""
        self.tests += int(element.get("tests", 0))
        self.failures += int(element.get("failures", 0))
        self.errors += int(element.get("errors", 0))
        self.skipped += int(element.get("skipped", 0))
        self.time += float(element.get("time", 0.0))


def merge_xunit_files(input_files: list[str], output_file: str) -> None:
    """Merge xUnit XML files into a single output file."""
    merged_root = ET.Element("testsuites")

    stats = Stats()

    process_input_files(input_files, merged_root, stats)
    update_merged_root_stats(merged_root, stats)
    write_merged_output(merged_root, output_file, stats)


def process_input_files(
    input_files: list[str], merged_root: ET.Element, stats: Stats
) -> None:
    """Process each input XML file and update merged_root and statistics."""
    for infile_path in input_files:
        if not os.path.exists(infile_path):
            print(
                f"Warning: Input file not found: {infile_path}",
                file=sys.stderr,
            )
            continue

        try:
            root = parse_xml_file(infile_path)
            if root is None:
                continue

            if root.tag == "testsuite":
                merged_root.append(root)
                stats.update(root)
            elif root.tag == "testsuites":
                process_testsuites(root, merged_root, stats)
            else:
                print(
                    f"Warning: Unexpected root element '{root.tag}' in "
                    f"{infile_path}. Skipping.",
                    file=sys.stderr,
                )
        except Exception as e:
            print(
                f"Warning: An error occurred processing {infile_path}: {e}",
                file=sys.stderr,
            )


def parse_xml_file(file_path: str) -> ET.Element | None:
    """Parse an XML file and return its root element."""
    try:
        tree = ET.parse(file_path)
        return tree.getroot()
    except ET.ParseError as e:
        print(
            f"Warning: Could not parse XML file {file_path}: {e}",
            file=sys.stderr,
        )
        return None


def process_testsuites(
    root: ET.Element, merged_root: ET.Element, stats: Stats
) -> None:
    """Handle testsuites element and append its children to merged_root."""
    for testsuite in root.findall("testsuite"):
        merged_root.append(testsuite)
        stats.update(testsuite)


def update_merged_root_stats(merged_root: ET.Element, stats: Stats) -> None:
    """Update the merged root element with aggregated statistics."""
    merged_root.set("tests", str(stats.tests))
    merged_root.set("failures", str(stats.failures))
    merged_root.set("errors", str(stats.errors))
    merged_root.set("skipped", str(stats.skipped))
    merged_root.set("time", f"{stats.time:.3f}")


def write_merged_output(
    merged_root: ET.Element, output_file: str, stats: Stats
) -> None:
    """Write the merged XML tree to the output file."""
    try:
        merged_tree = ET.ElementTree(merged_root)
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        merged_tree.write(output_file, encoding="utf-8", xml_declaration=True)

        print(f"Successfully merged results into {output_file}")
        print(f"  Total Tests: {stats.tests}")
        print(f"  Total Failures: {stats.failures}")
        print(f"  Total Errors: {stats.errors}")
        print(f"  Total Skipped: {stats.skipped}")
        print(f"  Total Time: {stats.time:.3f}s")
    except Exception as e:
        print(f"Error writing output file {output_file}: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Merge multiple xUnit1 XML files into one."
    )
    parser.add_argument(
        "folder",
        help="Folder containing the xUnit XML files to merge. The script "
        "will search for all .xml files in this folder and its subfolders.",
    )
    parser.add_argument(
        "-o",
        "--output",
        help="Path to the output merged xUnit XML file.",
        default="merged_results.xml",
    )

    args = parser.parse_args()

    input_files: list[str] = []
    for root, _, files in os.walk(args.folder):
        input_files += [
            os.path.join(root, file) for file in files if file.endswith(".xml")
        ]

    if not input_files:
        print(f"No xUnit XML files found in {args.folder}.", file=sys.stderr)
        sys.exit(1)

    merge_xunit_files(input_files, os.path.abspath(args.output))
