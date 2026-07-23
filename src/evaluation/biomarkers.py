"""Clinical biomarker extraction from cardiac segmentations.

Volumes are reported in millilitres, masses in grams, lengths in millimetres.
Voxel volume is computed from the preprocessed in-plane spacing (1.5 mm) and
the patient's original slice thickness, so physical size is preserved despite
resampling.
"""

import numpy as np

TARGET_SPACING = 1.5          # mm, in plane after preprocessing
MYOCARDIUM_DENSITY = 1.05     # g/cm3, standard value for myocardial tissue

LABEL_RV, LABEL_MYO, LABEL_LV = 1, 2, 3


def voxel_volume_ml(spacing_z):
    """Volume of one voxel in millilitres. 1 mL = 1000 mm3."""
    return (TARGET_SPACING * TARGET_SPACING * spacing_z) / 1000.0


def structure_volume_ml(mask, label, spacing_z):
    """Volume of one labelled structure in millilitres."""
    return float((mask == label).sum()) * voxel_volume_ml(spacing_z)


def ejection_fraction(edv, esv):
    """Ejection fraction as a percentage."""
    if edv <= 0:
        return np.nan
    return 100.0 * (edv - esv) / edv


def body_surface_area(height_cm, weight_kg):
    """Mosteller formula, in square metres."""
    return float(np.sqrt(height_cm * weight_kg / 3600.0))


def wall_thickness_mm(mask, spacing_z):
    """Mean and maximum LV wall thickness across slices.

    Each slice is approximated as concentric circles: the endocardial radius
    from the LV cavity area, the epicardial radius from cavity plus myocardium
    area, and thickness as the difference. This standard approximation avoids
    the instability of pixel-wise distance measures on thin structures.
    """
    per_slice = []
    pixel_area = TARGET_SPACING ** 2
    for i in range(mask.shape[0]):
        lv = (mask[i] == LABEL_LV).sum() * pixel_area
        myo = (mask[i] == LABEL_MYO).sum() * pixel_area
        if myo <= 0:
            continue
        r_endo = np.sqrt(lv / np.pi)
        r_epi = np.sqrt((lv + myo) / np.pi)
        per_slice.append(r_epi - r_endo)
    if not per_slice:
        return np.nan, np.nan
    return float(np.mean(per_slice)), float(np.max(per_slice))


def compute_biomarkers(ed_mask, es_mask, spacing_z, height_cm, weight_kg):
    """Full biomarker set for one patient from ED and ES masks."""
    lv_edv = structure_volume_ml(ed_mask, LABEL_LV, spacing_z)
    lv_esv = structure_volume_ml(es_mask, LABEL_LV, spacing_z)
    rv_edv = structure_volume_ml(ed_mask, LABEL_RV, spacing_z)
    rv_esv = structure_volume_ml(es_mask, LABEL_RV, spacing_z)

    myo_ed_ml = structure_volume_ml(ed_mask, LABEL_MYO, spacing_z)
    lv_mass_g = myo_ed_ml * MYOCARDIUM_DENSITY   # 1 mL = 1 cm3

    bsa = body_surface_area(height_cm, weight_kg)
    wt_mean, wt_max = wall_thickness_mm(ed_mask, spacing_z)

    return {
        "lv_edv_ml": lv_edv,
        "lv_esv_ml": lv_esv,
        "lv_sv_ml": lv_edv - lv_esv,
        "lv_ef_pct": ejection_fraction(lv_edv, lv_esv),
        "rv_edv_ml": rv_edv,
        "rv_esv_ml": rv_esv,
        "rv_sv_ml": rv_edv - rv_esv,
        "rv_ef_pct": ejection_fraction(rv_edv, rv_esv),
        "lv_mass_g": lv_mass_g,
        "wall_thickness_mean_mm": wt_mean,
        "wall_thickness_max_mm": wt_max,
        "lv_edv_index": lv_edv / bsa if bsa > 0 else np.nan,
        "lv_esv_index": lv_esv / bsa if bsa > 0 else np.nan,
        "rv_edv_index": rv_edv / bsa if bsa > 0 else np.nan,
        "lv_mass_index": lv_mass_g / bsa if bsa > 0 else np.nan,
        "lv_rv_edv_ratio": lv_edv / rv_edv if rv_edv > 0 else np.nan,
        "mass_to_volume": lv_mass_g / lv_edv if lv_edv > 0 else np.nan,
        "bsa_m2": bsa,
    }


if __name__ == "__main__":
    print("VOLUME MATHS")
    m = np.zeros((5, 256, 256), np.uint8)
    m[:, 100:140, 100:140] = LABEL_LV   # 60 x 60 mm over 5 slices of 10 mm
    got = structure_volume_ml(m, LABEL_LV, 10.0)
    print(f"  cuboid 60x60x50 mm -> {got:.2f} mL (expect 180.00)")
    assert abs(got - 180.0) < 1e-6
    got5 = structure_volume_ml(m, LABEL_LV, 5.0)
    print(f"  same shape at 5 mm slices -> {got5:.2f} mL (expect 90.00)")
    assert abs(got5 - 90.0) < 1e-6

    print("\nEJECTION FRACTION")
    print(f"  EDV 100, ESV 40 -> {ejection_fraction(100, 40):.1f}% (expect 60.0)")
    assert abs(ejection_fraction(100, 40) - 60.0) < 1e-9

    print("\nBODY SURFACE AREA (Mosteller)")
    print(f"  180 cm, 80 kg -> {body_surface_area(180, 80):.3f} m2 (expect 2.000)")
    assert abs(body_surface_area(180, 80) - 2.0) < 1e-9

    print("\nWALL THICKNESS")
    grid = np.zeros((1, 256, 256), np.uint8)
    yy, xx = np.mgrid[0:256, 0:256]
    r = np.sqrt((yy - 128) ** 2 + (xx - 128) ** 2) * TARGET_SPACING
    grid[0][r < 20] = LABEL_LV
    grid[0][(r >= 20) & (r < 30)] = LABEL_MYO
    mean_t, _ = wall_thickness_mm(grid, 10.0)
    print(f"  annulus 20->30 mm -> mean {mean_t:.2f} mm (expect ~10.00)")
    assert abs(mean_t - 10.0) < 0.5

    print("\nall assertions passed")
