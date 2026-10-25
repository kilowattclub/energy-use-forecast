"""Build the bundled DefaultPredictor profile from the Low Carbon London dataset.

Source: UK Power Networks, "SmartMeter Energy Consumption Data in London
Households" (half-hourly readings from 5,567 homes, Nov 2011 - Feb 2014):
https://data.london.gov.uk/dataset/smartmeter-energy-consumption-data-in-london-households-vqm0d

The download (LCL-FullData.zip) uses Deflate64 compression, which Python's
zipfile cannot read, so stream the CSV out with 7-Zip instead of unpacking it:

    7z e -so LCL-FullData.zip | python scripts/build_default_profile.py - src/default_predictor/default_profile.json

Only flat-rate ("Std") households are used; the time-of-use group was asked to
shift its demand. Every valid reading is averaged into one of 48 half-hour
slots of the day, and each slot's average power in kW is written at the middle
of the slot, so the predictor's linear interpolation stays centred on it.

The timestamps are UTC, like the predictor: readings form an unbroken
half-hour grid through both clock changes each year. Each is taken to mark the
start of its half-hour.
"""

import argparse
import csv
import sys
from datetime import datetime

HALF_HOURS_PER_DAY = 48
HOURS_PER_HALF_HOUR = 0.5
FLAT_RATE_TARIFF = "Std"
MAXIMUM_PLAUSIBLE_KWH = 50.0  # A reading above this is a meter fault: 50 kWh in half an hour is a 100 kW draw.


def find_column(header: list[str], name: str) -> int:
    stripped_names = [column_name.strip() for column_name in header]
    return stripped_names.index(name)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("dataset_csv", help="CC_LCL-FullData.csv, or - to read it from stdin")
    parser.add_argument("output_json", help="where to write the profile")
    return parser.parse_args()


def parse_csv(dataset_csv: str) -> tuple[list[float], list[int]]:
    """Total energy and reading count for each half-hour slot of the day."""
    energy_sums_kwh = [0.0] * HALF_HOURS_PER_DAY
    reading_counts = [0] * HALF_HOURS_PER_DAY
    with (
        sys.stdin
        if dataset_csv == "-"
        else open(dataset_csv, encoding="utf-8", newline="")
    ) as csv_file:
        rows = csv.reader(csv_file)
        header = next(rows)
        tariff_column = find_column(header, "stdorToU")
        time_column = find_column(header, "DateTime")
        energy_column = find_column(header, "KWH/hh (per half hour)")

        for row in rows:
            if row[tariff_column] != FLAT_RATE_TARIFF:
                continue
            try:
                energy_kwh = float(row[energy_column])
                reading_time = datetime.fromisoformat(row[time_column][:19])  # Timestamps look like "2012-10-12 00:30:00.0000000".
            except ValueError:  # Missing readings are written as "Null".
                continue
            if not 0.0 <= energy_kwh <= MAXIMUM_PLAUSIBLE_KWH:
                continue

            slot_index = reading_time.hour * 2 + reading_time.minute // 30
            energy_sums_kwh[slot_index] += energy_kwh
            reading_counts[slot_index] += 1

    return energy_sums_kwh, reading_counts


def main() -> None:
    arguments = parse_args()
    energy_sums_kwh, reading_counts = parse_csv(arguments.dataset_csv)

    def profile_line(slot_index: int) -> str:
        """One JSON entry: the slot's average power, at the middle of the slot."""
        average_energy_kwh = energy_sums_kwh[slot_index] / reading_counts[slot_index]
        average_power_kw = average_energy_kwh / HOURS_PER_HALF_HOUR
        slot_middle_minutes = slot_index * 30 + 15
        time = f"{slot_middle_minutes // 60:02}:{slot_middle_minutes % 60:02}"
        return f'  {{ "time": "{time}", "kw": {average_power_kw:.3f} }}'

    lines = [profile_line(slot_index) for slot_index in range(HALF_HOURS_PER_DAY)]
    with open(arguments.output_json, "w", encoding="utf-8", newline="\n") as output_file:
        output_file.write("[\n" + ",\n".join(lines) + "\n]\n")


if __name__ == "__main__":
    main()
