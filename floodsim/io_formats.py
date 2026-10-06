"""นำเข้า/ส่งออกเส้นแม่น้ำ: KML, KMZ, GPX, GeoJSON, CSV"""
from __future__ import annotations

import csv
import io
import json
import math
import os
import xml.etree.ElementTree as ET
import zipfile

import numpy as np

from .geometry import EARTH_R, River, Section

SUPPORTED = (("ไฟล์แม่น้ำ", "*.kml *.kmz *.gpx *.geojson *.json *.csv *.txt"),
             ("KML/KMZ", "*.kml *.kmz"), ("GPX", "*.gpx"), ("GeoJSON", "*.geojson *.json"),
             ("CSV", "*.csv *.txt"), ("ทุกไฟล์", "*.*"))


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _length_geo(lon, lat) -> float:
    lat0 = math.radians(np.mean(lat))
    return float(np.sum(np.hypot(np.diff(lon) * math.cos(lat0), np.diff(lat)))) * EARTH_R * math.pi / 180


# ---------------------------------------------------------------- parsers
def _parse_kml_text(text: str):
    root = ET.fromstring(text.encode("utf-8") if isinstance(text, str) else text)
    lines, points = [], []
    doc_name = None
    for pm in root.iter():
        if _local(pm.tag) == "Document" and doc_name is None:
            for ch in pm:
                if _local(ch.tag) == "name" and ch.text:
                    doc_name = ch.text.strip()
        if _local(pm.tag) != "Placemark":
            continue
        name = next((c.text.strip() for c in pm if _local(c.tag) == "name" and c.text), None)
        role = next(((d.findtext("{*}value") or d.findtext("value")) for d in pm.iter()
                     if _local(d.tag) == "Data" and d.get("name") == "floodsim_role"), None)
        for el in pm.iter():
            tag = _local(el.tag)
            if tag == "LineString":
                coords = next((c.text for c in el.iter() if _local(c.tag) == "coordinates"), "")
                pts = []
                for tok in coords.split():
                    v = tok.split(",")
                    if len(v) >= 2:
                        pts.append((float(v[0]), float(v[1]), float(v[2]) if len(v) > 2 and v[2] else np.nan))
                if len(pts) >= 2:
                    lines.append((name, pts, role))
            elif tag == "Track":
                pts = []
                for c in el:
                    if _local(c.tag) == "coord" and c.text:
                        v = c.text.split()
                        pts.append((float(v[0]), float(v[1]), float(v[2]) if len(v) > 2 else np.nan))
                if len(pts) >= 2:
                    lines.append((name, pts, role))
            elif tag == "Point":
                coords = next((c.text for c in el.iter() if _local(c.tag) == "coordinates"), "")
                v = coords.strip().split(",")
                if len(v) >= 2:
                    points.append((float(v[0]), float(v[1]), float(v[2]) if len(v) > 2 and v[2] else np.nan))
    if lines:
        return [(n or doc_name, p, r) for n, p, r in lines], ""
    if len(points) >= 2:
        return [(doc_name, points, None)], f"ไม่พบ LineString — ต่อจุด {len(points)} จุดตามลำดับ"
    raise ValueError("ไม่พบ LineString หรือจุดในไฟล์ KML")


def _parse_kmz(path):
    with zipfile.ZipFile(path) as z:
        names = [n for n in z.namelist() if n.lower().endswith(".kml")]
        if not names:
            raise ValueError("ไม่พบไฟล์ .kml ภายใน KMZ")
        main = "doc.kml" if "doc.kml" in names else names[0]
        return _parse_kml_text(z.read(main).decode("utf-8", errors="replace"))


def _parse_gpx(text):
    root = ET.fromstring(text.encode("utf-8"))
    pts, name = [], None
    for el in root.iter():
        t = _local(el.tag)
        if t == "name" and name is None and el.text:
            name = el.text.strip()
        if t in ("trkpt", "rtept"):
            ele = next((c.text for c in el if _local(c.tag) == "ele"), None)
            pts.append((float(el.get("lon")), float(el.get("lat")), float(ele) if ele else np.nan))
    if len(pts) < 2:
        raise ValueError("ไม่พบ trkpt/rtept ในไฟล์ GPX")
    return [(name, pts, None)], ""


def _parse_geojson(text):
    gj = json.loads(text)
    feats = gj.get("features") if gj.get("type") == "FeatureCollection" else [gj]
    cands = []
    for f in feats:
        geom = f.get("geometry", f) if f.get("type") == "Feature" else f
        name = (f.get("properties") or {}).get("name") if f.get("type") == "Feature" else None
        t, c = geom.get("type"), geom.get("coordinates")
        parts = [c] if t == "LineString" else (c if t == "MultiLineString" else [])
        for p in parts:
            pts = [(q[0], q[1], q[2] if len(q) > 2 else np.nan) for q in p]
            if len(pts) >= 2:
                cands.append((name, pts, (f.get("properties") or {}).get("floodsim_role")))
    if not cands:
        raise ValueError("ไม่พบ LineString ใน GeoJSON")
    return cands, ""


def _parse_csv(text):
    sample = text[:4096]
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",;\t ")
    except csv.Error:
        dialect = csv.excel
    rows = [r for r in csv.reader(io.StringIO(text), dialect) if any(c.strip() for c in r)]
    header = None
    try:
        [float(c) for c in rows[0][:2]]
    except ValueError:
        header = [c.strip().lower() for c in rows[0]]
        rows = rows[1:]

    def col(names, default):
        if header:
            for i, h in enumerate(header):
                if h in names:
                    return i
        return default

    ix = col(("lon", "longitude", "lng", "x", "easting"), 0)
    iy = col(("lat", "latitude", "y", "northing"), 1)
    iz = col(("z", "elev", "elevation", "alt", "altitude", "ele", "bed", "zb"), 2)
    data = []
    for r in rows:
        try:
            a, b = float(r[ix]), float(r[iy])
        except (ValueError, IndexError):
            continue
        try:
            c = float(r[iz])
        except (ValueError, IndexError):
            c = np.nan
        data.append((a, b, c))
    if len(data) < 2:
        raise ValueError("CSV ต้องมีอย่างน้อย 2 แถวของ x,y (หรือ lon,lat) และ z (ไม่บังคับ)")
    arr = np.array(data)
    geo = np.all(np.abs(arr[:, 0]) <= 180) and np.all(np.abs(arr[:, 1]) <= 90)
    if header and header[ix] in ("x", "easting"):
        geo = False
    return [(None, data, None)], ("" if geo else "พิกัดเป็นเมตร (ไม่ใช่ละติจูด/ลองจิจูด)"), geo


def read_river_file(path: str, pick: int | None = None) -> dict:
    """คืนค่า dict: name, x, y (เมตร), z, lat0, lon0, info, n_points, candidates
    ถ้าไฟล์มีหลายเส้น: ใช้เส้นที่ระบุด้วย pick, เส้นที่ติดป้าย centerline, หรือเส้นที่ยาวที่สุด"""
    ext = os.path.splitext(path)[1].lower()
    geo = True
    if ext == ".kmz":
        lines, info = _parse_kmz(path)
    else:
        with open(path, "r", encoding="utf-8-sig", errors="replace") as f:
            text = f.read()
        if ext == ".kml" or text.lstrip().startswith("<?xml") and "<kml" in text[:500]:
            lines, info = _parse_kml_text(text)
        elif ext == ".gpx" or "<gpx" in text[:500]:
            lines, info = _parse_gpx(text)
        elif ext in (".geojson", ".json"):
            lines, info = _parse_geojson(text)
        else:
            lines, info, geo = _parse_csv(text)
    lengths = []
    for _, p, _ in lines:
        a = np.asarray(p, float)
        if geo:
            lengths.append(_length_geo(a[:, 0], a[:, 1]))
        else:
            lengths.append(float(np.sum(np.hypot(np.diff(a[:, 0]), np.diff(a[:, 1])))))
    tagged = [i for i, l in enumerate(lines) if l[2] == "centerline"]
    if pick is None:
        pick = tagged[0] if tagged else int(np.argmax(lengths))
    if len(lines) > 1:
        info = (info + " " if info else "") + f"พบเส้น {len(lines)} เส้น — ใช้เส้นที่ {pick + 1}"
    name, pts, _ = lines[pick]
    a = np.asarray(pts, float)
    if geo:
        lat0, lon0 = float(a[:, 1].mean()), float(a[:, 0].mean())
        k = EARTH_R * math.pi / 180
        x = (a[:, 0] - lon0) * k * math.cos(math.radians(lat0))
        y = (a[:, 1] - lat0) * k
    else:
        lat0 = lon0 = float("nan")
        x, y = a[:, 0] - a[0, 0], a[:, 1] - a[0, 1]
    cands = [(n or f"เส้นที่ {i + 1}", L / 1000) for i, ((n, _, _), L) in enumerate(zip(lines, lengths))]
    return dict(name=name or os.path.splitext(os.path.basename(path))[0], x=x, y=y, z=a[:, 2],
                lat0=lat0, lon0=lon0, info=info, n_points=len(a), geo=geo, candidates=cands,
                picked=pick, path=path)


# ---------------------------------------------------------------- export
def _to_lonlat(river: River, x, y):
    lat0 = river.lat0 if math.isfinite(river.lat0) else 14.35
    lon0 = river.lon0 if math.isfinite(river.lon0) else 100.55
    k = EARTH_R * math.pi / 180
    X, Y = np.asarray(x) + river.x_off, np.asarray(y) + river.y_off
    return lon0 + X / (k * math.cos(math.radians(lat0))), lat0 + Y / k


def export_kml(river: River, sec: Section, path: str, max_pts: int = 3000):
    step = max(1, river.N // max_pts)
    idx = np.r_[np.arange(0, river.N, step), river.N - 1]
    idx = np.unique(idx)
    x, y, zb = river.x[idx], river.y[idx], river.zb[idx]
    nx, ny = river.nx[idx], river.ny[idx]

    def coords(xx, yy, zz):
        lon, lat = _to_lonlat(river, xx, yy)
        return " ".join(f"{a:.7f},{b:.7f},{c:.2f}" for a, b, c in zip(lon, lat, zz))

    hw = sec.Tb / 2
    fp = hw + sec.Bf
    desc = (f"ความยาวลำน้ำ {river.L/1000:.2f} กม. | ระยะทางเส้นตรง {river.D/1000:.2f} กม. | "
            f"ความคดเคี้ยว {river.sigma:.2f} | B={sec.B} ม. Hb={sec.Hb} ม. n={sec.n}")
    pm = []
    for nm, xx, yy, zz, color, role in (
        ("แนวร่องน้ำ (ท้องน้ำ)", x, y, zb, "ffff8a2a", "centerline"),
        ("ตลิ่งซ้าย", x + nx * hw, y + ny * hw, zb + sec.Hb, "ff2a6a8a", "bank"),
        ("ตลิ่งขวา", x - nx * hw, y - ny * hw, zb + sec.Hb, "ff2a6a8a", "bank"),
        ("ขอบที่ราบน้ำท่วมซ้าย", x + nx * fp, y + ny * fp, zb + sec.Hb, "ff2aa02a", "floodplain"),
        ("ขอบที่ราบน้ำท่วมขวา", x - nx * fp, y - ny * fp, zb + sec.Hb, "ff2aa02a", "floodplain"),
    ):
        if "ที่ราบ" in nm and sec.Bf <= 0:
            continue
        pm.append(f"""  <Placemark><name>{nm}</name>
   <ExtendedData><Data name="floodsim_role"><value>{role}</value></Data></ExtendedData><Style><LineStyle><color>{color}</color><width>2</width></LineStyle></Style>
   <LineString><tessellate>1</tessellate><altitudeMode>clampToGround</altitudeMode>
    <coordinates>{coords(xx, yy, zz)}</coordinates></LineString></Placemark>""")
    kml = f"""<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2">
<Document><name>{river.name}</name><description>{desc}</description>
{chr(10).join(pm)}
</Document></kml>
"""
    with open(path, "w", encoding="utf-8") as f:
        f.write(kml)
