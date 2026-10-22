#!/usr/bin/env python3
"""
Build a practice observed-binary catalog from Will Grundy's "Orbit Status
of Known Binary TNOs" pages (http://74.208.185.217/tnbs/status.html).

For every system listed on the status page, takes the earliest row of its
"Astrometric observations" table (the first detection) -- by default the
earliest HST row, for a homogeneous sample; --all-instruments takes the
earliest row from any instrument -- and writes one row per system with
the columns emcee_walker's reference_pop csv needs:

    Name, x, y, z, H, sep, pa, dm

plus provenance/QA columns. x/y/z are the primary's geocentric J2000
ecliptic position [km] at the observation time, from JPL HORIZONS
(astrometric, i.e. light-time corrected, geocenter 500). Systems with no
astrometry table (special cases, possible binaries, discovery-only
entries, or no HST rows under the default filter) are skipped and listed
on stdout.

Conventions:
  * Delta-x is Delta-RA cos(Dec) (east), Delta-y is Delta-Dec (north), in
    arcsec, secondary minus primary.
  * sep = hypot(dx, dy) [arcsec].
  * pa matches Population.compute_separation(): degrees, North through
    East, with North the *ecliptic* pole projected onto the sky -- so the
    equatorial (dx, dy) offset is rotated into that frame. pa_eq_ne is
    the same angle in equatorial N-through-E, and pa_atan2_dy_dx the
    literal atan2(dy, dx) (counterclockwise from east), for reference.
  * H is the primary's own H_V: the pages give combined-light H_V, so
    H = H_combined + 2.5 log10(1 + 10^(-0.4 dm)).

Usage (needs astroquery, which the simulator itself does not):
    python fetch_tnbs_catalog.py [--all-instruments] [--out OUT.csv]
                                 [--cache tnbs_cache]
"""
import argparse
import html
import os
import re
import time
import urllib.request

import numpy as np
import pandas as pd
from astropy import units as u
from astropy.time import Time

BASE_URL = "http://74.208.185.217/tnbs/"
AU_KM = 149597870.7
# IAU 2006 obliquity of the ecliptic at J2000, as HORIZONS uses for its
# "ecliptic" reference plane.
OBLIQUITY = np.radians(84381.406 / 3600.0)


def fetch(page, cache_dir):
    """Return page's HTML, downloading it into cache_dir on first use."""
    path = os.path.join(cache_dir, page)
    if not os.path.exists(path):
        with urllib.request.urlopen(BASE_URL + page, timeout=30) as resp:
            data = resp.read()
        with open(path, "wb") as f:
            f.write(data)
        time.sleep(0.3)  # be polite to the server
    with open(path, encoding="utf-8", errors="replace") as f:
        return f.read()


def _cells(row_html):
    """Text of each <TD> cell in one table row."""
    cells = re.findall(r"<TD>(.*?)</TD>", row_html, re.S)
    return [html.unescape(re.sub(r"<.*?>", " ", c)).strip() for c in cells]


def parse_status(status_html):
    """(page, display name, orbit status) for every system on the page."""
    rows = re.findall(r"<TR bgcolor=#[0-9A-F]+>\s*<TD><A HREF=([^>]+)>"
                      r"(.*?)</A></TD>(.*?)</TR>", status_html, re.S)
    return [(page, html.unescape(name), _cells(rest)[-1])
            for page, name, rest in rows]


def parse_time(text):
    """'2002/06/30  19.88030' (UTC date, decimal hours) -> astropy Time."""
    date, hours = text.split()
    return (Time(date.replace("/", "-"), scale="utc")
            + float(hours) * u.hour)


def parse_page(page_html, hst_only=True):
    """
    Combined H_V, delta-mag and the earliest astrometry row of one system
    page -- the earliest HST row if `hst_only` -- or None for the
    astrometry if the page has no (HST) rows.
    """
    hv = re.search(r"H<sub>V</sub>\s*=\s*([-\d.]+)", page_html)
    dm = re.search(r"&Delta;<sub>mag</sub>\s*=\s*([-\d.]+)", page_html)
    table = re.search(r"Astrometric observations</H2>(.*?)</TABLE>",
                      page_html, re.S)
    rows = []
    if table:
        for tr in re.findall(r"<TR[^>]*>(.*?)(?=<TR|$)", table.group(1),
                             re.S):
            c = _cells(tr)
            # Skips the header and "plus proprietary data" rows.
            if len(c) >= 9 and re.match(r"\d{4}/\d\d/\d\d", c[0]):
                if not hst_only or c[1].startswith("HST/"):
                    rows.append(c)
    first = min(rows, key=lambda c: parse_time(c[0]).jd) if rows else None
    return (float(hv.group(1)) if hv else np.nan,
            float(dm.group(1)) if dm else np.nan, first)


def horizons_id(name):
    """
    HORIZONS small-body id: the MPC number if the system has one, else
    its provisional designation. "Huya (38628 2000 EB173)" -> "38628",
    "2002 VF130" -> "2002 VF130".
    """
    # "26308 (1998 SM165)" or "... (38628 2000 EB173)"; the year of a
    # bare "(1998 SM165)" must not be mistaken for a number.
    m = (re.match(r"(\d+) \(", name)
         or re.search(r"\((\d+) \d{4} [A-Z]{2}", name))
    if m:
        return m.group(1) + ";"
    m = re.search(r"\(?(\d{4} [A-Z]{2}\d*)\)?$", name)
    return m.group(1)


def geocentric_ecliptic(hid, t):
    """
    Geocentric J2000 ecliptic position [km] of small body `hid` at time
    `t`, astrometric (light-time corrected) from JPL HORIZONS.
    """
    from astroquery.jplhorizons import Horizons
    obj = Horizons(id=hid, id_type="smallbody", location="500",
                   epochs=[t.tdb.jd])
    vec = obj.vectors(refplane="ecliptic", aberrations="astrometric")
    return np.array([vec["x"][0], vec["y"][0], vec["z"][0]]) * AU_KM


def radec_to_vec(ra, dec):
    """'+13:32:09.58', '-00:05:44.7' -> equatorial unit vector."""
    def sexa(s):
        sign = -1.0 if s.strip().startswith("-") else 1.0
        d, m, x = (abs(float(v)) for v in s.split(":"))
        return sign * (d + m / 60.0 + x / 3600.0)
    a, d = np.radians(sexa(ra) * 15.0), np.radians(sexa(dec))
    return np.array([np.cos(d) * np.cos(a), np.cos(d) * np.sin(a),
                     np.sin(d)])


def eq_to_ecl(v):
    """Rotate a J2000 equatorial vector into the J2000 ecliptic frame."""
    c, s = np.cos(OBLIQUITY), np.sin(OBLIQUITY)
    return np.array([v[0], c * v[1] + s * v[2], -s * v[1] + c * v[2]])


def _north_east(los, pole):
    """Sky-plane north (toward `pole`) and east unit vectors at `los`."""
    north = pole - los * np.dot(pole, los)
    north /= np.linalg.norm(north)
    east = np.cross(north, los)
    return north, east / np.linalg.norm(east)


def ecliptic_pa(los_ecl, dx, dy):
    """
    Position angle [deg], North through East with North the ecliptic pole,
    of the equatorial sky offset (dx east, dy north) at line of sight
    `los_ecl` (ecliptic unit vector) -- the same convention as
    Population.compute_separation().
    """
    eq_pole = eq_to_ecl(np.array([0.0, 0.0, 1.0]))
    n_eq, e_eq = _north_east(los_ecl, eq_pole)
    n_ecl, e_ecl = _north_east(los_ecl, np.array([0.0, 0.0, 1.0]))
    offset = dx * e_eq + dy * n_eq
    return np.degrees(np.arctan2(np.dot(offset, e_ecl),
                                 np.dot(offset, n_ecl))) % 360.0


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--all-instruments", dest="hst_only",
                        action="store_false",
                        help="use every instrument's astrometry, not just "
                             "HST's (default: HST only)")
    parser.add_argument("--out", default=None,
                        help="output csv (default: runs/TNBS/"
                             "tnbs_first_detections_{hst,all}.csv)")
    parser.add_argument("--cache", default="tnbs_cache",
                        help="directory for downloaded HTML pages")
    args = parser.parse_args()
    if args.out is None:
        args.out = os.path.join(
            "runs", "TNBS", "tnbs_first_detections_"
            + ("hst" if args.hst_only else "all") + ".csv")
    os.makedirs(args.cache, exist_ok=True)

    rows, skipped = [], []
    for page, name, status in parse_status(fetch("status.html", args.cache)):
        h_comb, dm, first = parse_page(fetch(page, args.cache),
                                       hst_only=args.hst_only)
        if first is None:
            skipped.append((name, status))
            continue
        t = parse_time(first[0])
        dx, dy = float(first[5]), float(first[7])
        hid = horizons_id(name)
        xyz = geocentric_ecliptic(hid, t)
        los = xyz / np.linalg.norm(xyz)
        # QA: angle between HORIZONS's direction and the page's RA/Dec.
        page_los = eq_to_ecl(radec_to_vec(first[2], first[3]))
        offset = np.degrees(np.arccos(np.clip(np.dot(los, page_los),
                                              -1.0, 1.0))) * 3600.0
        rows.append({
            "Name": name,
            "x": xyz[0], "y": xyz[1], "z": xyz[2],
            "H": h_comb + 2.5 * np.log10(1.0 + 10.0 ** (-0.4 * dm)),
            "sep": np.hypot(dx, dy),
            "pa": ecliptic_pa(los, dx, dy),
            "dm": dm,
            "H_combined": h_comb,
            "pa_eq_ne": np.degrees(np.arctan2(dx, dy)) % 360.0,
            "pa_atan2_dy_dx": np.degrees(np.arctan2(dy, dx)) % 360.0,
            "dx": dx, "dy": dy,
            "obs_time_utc": t.isot,
            "instrument": first[1],
            "orbit_status": status,
            "horizons_id": hid,
            "page_delta_au": float(first[4]),
            "horizons_delta_au": np.linalg.norm(xyz) / AU_KM,
            "radec_offset_arcsec": offset,
            "page": BASE_URL + page,
        })
        print(f"{name}: sep {rows[-1]['sep']:.3f}\" dm {dm:.2f} "
              f"(HORIZONS vs page RA/Dec {offset:.1f}\")")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    pd.DataFrame(rows).to_csv(args.out, index=False)
    print(f"\nWrote {len(rows)} systems to {args.out}")
    print(f"Skipped {len(skipped)} with no "
          f"{'HST ' if args.hst_only else ''}astrometry:")
    for name, status in skipped:
        print(f"  {name} ({status})")


if __name__ == "__main__":
    main()
