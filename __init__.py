# RNAtools/__init__.py
from pymol import cmd
from Bio.PDB import Superimposer
import numpy as np
import os

# 和pipline同款原子集合
BACKBONE_ATOMS = ["C1'", "C2'", "C3'", "C4'", "C5'",
                  "O2'", "O3'", "O4'", "O5'", "OP1", "OP2", "P"]
HEAVY_ATOMS    = ["C2", "C4", "C5", "C6", "C8",
                  "N1", "N2", "N3", "N4", "N6", "N7", "N9",
                  "O2", "O4", "O6"]
ALL_ATOMS = BACKBONE_ATOMS + HEAVY_ATOMS


# ---------- 残基映射相关 ----------

def _default_mapfile():
    """插件目录里的 residues.txt"""
    return os.path.join(os.path.dirname(__file__), "residues.txt")


def _load_res_map(filename=None):
    """
    读取类似:
      RG5 G
      RG  G
    的映射文件，忽略第二列为 '-' 的行。
    """
    if filename is None:
        filename = _default_mapfile()

    mapping = {}
    with open(filename) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            old, new1 = parts[0], parts[1]
            if new1 == "-":
                continue  # 比如离子/配体等不处理
            mapping[old] = new1
    return mapping


def normalize_resn(obj="all", mapfile=None):
    """
    normalize_resn obj, mapfile=None

    按 residues.txt 把非常规核苷名统一成 A/C/G/U。
    obj     : 要处理的对象或选择（默认 all）
    mapfile : 映射文件路径，默认用插件目录里的 residues.txt
    """
    mapping = _load_res_map(mapfile)
    print(f"[normalize_resn] loaded {len(mapping)} mappings")

    for old, new in mapping.items():
        sel = f"({obj}) and resn {old}"
        n = cmd.count_atoms(sel)
        if n == 0:
            continue
        print(f"[normalize_resn] {old} -> {new}, atoms: {n}")
        cmd.alter(sel, f"resn='{new}'")

    cmd.sort()


# ---------- Biopython super 相关 ----------

def _build_res_dict(model):
    """按 (chain, resi) 分组原子"""
    res_dict = {}
    for a in model.atom:
        key = (a.chain, a.resi)
        res_dict.setdefault(key, []).append(a)
    return res_dict


def _collect_pairs_allatoms(ref_obj, mob_obj, sele="all"):
    """
    模拟你脚本的配对逻辑：
    - 对公共 (chain, resi) 里
    - 只用 ALL_ATOMS 并按原子名配对
    """
    m_ref = cmd.get_model(f"{ref_obj} and ({sele})")
    m_mob = cmd.get_model(f"{mob_obj} and ({sele})")

    ref_res = _build_res_dict(m_ref)
    mob_res = _build_res_dict(m_mob)

    common_keys = sorted(
        set(ref_res.keys()) & set(mob_res.keys()),
        key=lambda k: (k[0], int(k[1]) if k[1].isdigit() else k[1])
    )

    ref_xyz = []
    mob_xyz = []

    for key in common_keys:
        ref_atoms = ref_res[key]
        mob_atoms = mob_res[key]
        for atom_name in ALL_ATOMS:
            a_ref = next((a for a in ref_atoms if a.name == atom_name), None)
            a_mob = next((a for a in mob_atoms if a.name == atom_name), None)
            if a_ref is None or a_mob is None:
                continue
            ref_xyz.append([a_ref.x, a_ref.y, a_ref.z])
            mob_xyz.append([a_mob.x, a_mob.y, a_mob.z])

    if not ref_xyz or not mob_xyz:
        print(f"[Biopy] {ref_obj} vs {mob_obj}: no common ALL_ATOMS.")
        return None, None

    ref_xyz = np.array(ref_xyz, dtype=float)
    mob_xyz = np.array(mob_xyz, dtype=float)
    print(f"[Biopy] {ref_obj} vs {mob_obj}: use {ref_xyz.shape[0]} atom pairs.")
    return ref_xyz, mob_xyz


def biopy_super_to_ref(mobile, reference, sele="all"):
    """
    biopy_super_to_ref mobile, reference, sele=all

    用 Biopython.Superimposer，按 ALL_ATOMS 刚体对齐 mobile -> reference，
    返回 RMSD。
    """
    ref_xyz, mob_xyz = _collect_pairs_allatoms(reference, mobile, sele=sele)
    if ref_xyz is None:
        return None

    sup = Superimposer()
    sup.set_atoms(ref_xyz, mob_xyz)
    rot, tran = sup.rotran

    cmd.transform_object(
        mobile,
        rot.flatten().tolist(),
        tran.tolist()
    )

    print(f"[Biopy] {mobile} -> {reference}, RMSD = {sup.rms:.3f} Å")
    return sup.rms


# ---------- 一键命令（可选） ----------

def rna_cleanup_and_super(mobile, reference, mapfile=None, sele="all"):
    """
    rna_cleanup_and_super mobile, reference, mapfile=None, sele=all

    先对 mobile 做残基名清洗，再用 Biopython super 对齐到 reference。
    """
    normalize_resn(obj=mobile, mapfile=mapfile)
    return biopy_super_to_ref(mobile, reference, sele=sele)


# ---------- 插件入口：注册命令 ----------

def __init_plugin__(app=None):
    """
    PyMOL 插件入口。被加载时调用。
    """
    cmd.extend("normalize_resn", normalize_resn)
    cmd.extend("biopy_super_to_ref", biopy_super_to_ref)
    cmd.extend("rna_cleanup_and_super", rna_cleanup_and_super)
    print("[RNAtools] commands registered: normalize_resn, biopy_super_to_ref, rna_cleanup_and_super")
