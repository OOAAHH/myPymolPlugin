# RNARMSDPlugin/__init__.py
#
# 依赖：
#   - PyMOL
#   - Biopython (from Bio.PDB import Superimposer)
#   - numpy
#
# 功能：
#   使用 residues.txt / atoms.txt 实现：
#     - 残基名标准化
#     - 原子名标准化 + 允许/忽略原子集合
#     - 用 backbone + heavy 原子做 Biopython.Superimposer RMSD，对齐 mobile -> reference

from pymol import cmd
import os

# Biopython & numpy
try:
    from Bio.PDB import Superimposer
except ImportError as e:
    raise ImportError(
        "[RNARMSDPlugin] Biopython 未安装或当前 Python 环境不可用。\n"
        "请在 PyMOL 使用的 Python 里安装：pip install biopython"
    ) from e

try:
    import numpy as np
except ImportError as e:
    raise ImportError(
        "[RNARMSDPlugin] numpy 未安装或当前 Python 环境不可用。\n"
        "请在 PyMOL 使用的 Python 里安装：pip install numpy"
    ) from e


# ===================== 路径工具 =====================

def _plugin_dir():
    return os.path.dirname(__file__)


def _default_residue_mapfile():
    return os.path.join(_plugin_dir(), "residues.txt")


def _default_atom_mapfile():
    return os.path.join(_plugin_dir(), "atoms.txt")


# ===================== residues.txt：残基映射 =====================

def _load_residue_map(filename=None):
    """
    解析 residues.txt:
      old_resname  new_code
    new_code 为 '-' 的行忽略。
    返回: dict { old_resn -> new_resn }
    """
    if filename is None:
        filename = _default_residue_mapfile()

    mapping = {}
    with open(filename) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            old, new = parts[0], parts[1]
            if new == "-":
                continue
            mapping[old] = new
    return mapping


def normalize_resn(obj="all", mapfile=None):
    """
    normalize_resn obj, mapfile=None

    使用 residues.txt 批量重命名残基：
      obj     : 要处理的对象/选择（默认 all）
      mapfile : residues.txt 路径，默认插件目录里的 residues.txt
    """
    mapping = _load_residue_map(mapfile)
    print(f"[RNARMSD] loaded {len(mapping)} residue mappings")

    for old, new in mapping.items():
        sel = f"({obj}) and resn {old}"
        n = cmd.count_atoms(sel)
        if n == 0:
            continue
        print(f"[RNARMSD] resn {old} -> {new}, atoms: {n}")
        # 用双引号包名字，避免 C1' 类问题（虽然这里一般是 A/C/G/U）
        cmd.alter(sel, f'resn="{new}"')

    cmd.sort()


# ===================== atoms.txt：原子映射 =====================

def _load_atom_map(filename=None):
    """
    解析 atoms.txt，识别：
      - backbone 区段原子（参与 RMSD）
      - heavy 区段原子（参与 RMSD）
      - 原子别名映射（C1* -> C1'，O1P -> OP1 等）
      - 忽略原子（non canonical / ignore 区段 或 显式映射为 '-'）

    返回:
      rename_map : dict { old_atom_name -> canonical_atom_name }
      allowed    : set  { canonical_atom_name }  # 参与 RMSD 的原子（backbone+heavy）
      ignored    : set  { old_atom_name }        # 应忽略的原子名
    """
    if filename is None:
        filename = _default_atom_mapfile()

    rename_map = {}
    allowed = set()
    ignored = set()

    section = None  # 'heavy', 'backbone', 'ignore', 'dna', None

    with open(filename) as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue

            if line.startswith("#"):
                header = line.lstrip("#").strip().lower()
                if "backbone" in header:
                    section = "backbone"
                elif "heavy" in header:
                    section = "heavy"
                elif "ignore" in header:
                    section = "ignore"
                elif "non canonical" in header or "noncanonical" in header:
                    section = "ignore"
                elif "dna" in header:
                    section = "dna"
                else:
                    section = None
                continue

            parts = line.split()
            if len(parts) < 2:
                continue

            old, new = parts[0], parts[1]

            # DNA 区段交给 residues.txt 去处理，这里略过
            if section == "dna":
                continue

            # new 为 '-'：忽略原子
            if new == "-":
                ignored.add(old)
                continue

            # 其它情况：old -> new 归一化
            if old != new:
                rename_map[old] = new

            # backbone/heavy 区段，new 为 canonical atom 名字，加入 allowed
            if section in ("backbone", "heavy"):
                allowed.add(new)

    if not allowed:
        print("[RNARMSD] WARNING: atoms.txt 中未识别到 backbone/heavy 原子，"
              "RMSD 将为空，请检查 atoms.txt。")

    return rename_map, allowed, ignored


def normalize_atom_names(obj="all", atomfile=None):
    """
    normalize_atom_names obj, atomfile=None

    使用 atoms.txt 中的原子别名映射，对原子 name 做标准化（例如 C1* -> C1'）。
    只修改 old != new 的条目，忽略 '-' 和 DNA 区段。
    """
    rename_map, allowed, ignored = _load_atom_map(atomfile)
    print(f"[RNARMSD] loaded {len(rename_map)} atom rename rules")

    for old, new in rename_map.items():
        sel = f"({obj}) and name {old}"
        n = cmd.count_atoms(sel)
        if n == 0:
            continue
        print(f"[RNARMSD] atom name {old} -> {new}, atoms: {n}")
        # 这里必须用双引号，否则 C1' 会导致字符串截断
        cmd.alter(sel, f'name="{new}"')

    cmd.sort()


# ===================== 原子配对 & Biopython Superimposer =====================

def _build_res_dict(model, rename_map, allowed, ignored):
    """
    把 PyMOL model.atom 按 residue 分组并应用原子规则。

    key: (chain, resi, resn)
    val: dict { canonical_atom_name -> [x, y, z] }

    只保留：
      - 不在 ignored 集合里的原子
      - 归一化后名字在 allowed 集合里的原子
    """
    res_dict = {}

    for a in model.atom:
        # 原子名规范化
        name = rename_map.get(a.name, a.name)

        # 忽略掉不需要参与 RMSD 的原子
        if name in ignored:
            continue
        if allowed and (name not in allowed):
            continue

        key = (a.chain, a.resi, a.resn)
        # 有的 PyMOL 版本有 coord 属性，有的只有 x/y/z
        if hasattr(a, "coord"):
            coords = [a.coord[0], a.coord[1], a.coord[2]]
        else:
            coords = [a.x, a.y, a.z]

        bucket = res_dict.setdefault(key, {})
        # 同一个 residue + atom name 多 conformer 时，简单覆盖
        bucket[name] = coords

    return res_dict


def _collect_atom_pairs(reference, mobile, sele="all", atomfile=None):
    """
    按 atoms.txt 的规则，从 reference / mobile 中抽取完全对应的原子对：
      - residue 按 (chain, resi, resn) 对齐
      - residue 内按 canonical atom name 对齐
      - 只用 allowed 集合中的原子
    返回：
      ref_xyz: numpy (N, 3)
      mob_xyz: numpy (N, 3)
    """
    rename_map, allowed, ignored = _load_atom_map(atomfile)

    m_ref = cmd.get_model(f"{reference} and ({sele})")
    m_mob = cmd.get_model(f"{mobile} and ({sele})")

    ref_res = _build_res_dict(m_ref, rename_map, allowed, ignored)
    mob_res = _build_res_dict(m_mob, rename_map, allowed, ignored)

    # residue 对应：公共 (chain, resi, resn)
    common_keys = sorted(
        set(ref_res.keys()) & set(mob_res.keys()),
        key=lambda k: (k[0], int(k[1]) if str(k[1]).isdigit() else str(k[1]))
    )

    ref_xyz = []
    mob_xyz = []
    pair_count = 0

    for key in common_keys:
        ref_atoms = ref_res[key]
        mob_atoms = mob_res[key]

        common_atoms = sorted(set(ref_atoms.keys()) & set(mob_atoms.keys()))
        for aname in common_atoms:
            ref_xyz.append(ref_atoms[aname])
            mob_xyz.append(mob_atoms[aname])
            pair_count += 1

    if pair_count == 0:
        print(f"[RNARMSD] {mobile} vs {reference}: 找不到任何匹配原子对（按 atoms.txt 规则）。")
        return None, None

    ref_xyz = np.array(ref_xyz, dtype=float)
    mob_xyz = np.array(mob_xyz, dtype=float)

    print(f"[RNARMSD] {mobile} vs {reference}: 使用 {pair_count} 对原子进行 RMSD 拟合。")
    return ref_xyz, mob_xyz


def biopy_super_to_ref(mobile, reference, sele="all",
                       atomfile=None, apply_transform=1):
    """
    biopy_super_to_ref mobile, reference, sele=all, atomfile=None, apply_transform=1

    使用 atoms.txt 选 backbone+heavy 原子，并用 Biopython.Superimposer
    做刚体最小二乘，对齐 mobile -> reference：

      src_atoms = reference
      trg_atoms = mobile
      sup.set_atoms(src_atoms, trg_atoms)

    在 PyMOL 中默认会对 mobile 调用 transform_object（apply_transform=1）。
    返回：
      RMSD (float) 或 None
    """
    ref_xyz, mob_xyz = _collect_atom_pairs(reference, mobile, sele=sele, atomfile=atomfile)
    if ref_xyz is None:
        return None

    sup = Superimposer()
    # 对应你 rmsd() 里的调用顺序：sup.set_atoms(src_atoms, trg_atoms)
    sup.set_atoms(ref_xyz, mob_xyz)
    rot, tran = sup.rotran

    if apply_transform:
        cmd.transform_object(
            mobile,
            rot.flatten().tolist(),  # 9 元素 list
            tran.tolist()            # 3 元素 list
        )

    rmsd_val = float(sup.rms)
    print(f"[RNARMSD] {mobile} -> {reference}, RMSD = {rmsd_val:.3f} Å")
    return rmsd_val


def biopy_rmsd2_dists(mobile, reference, sele="all", atomfile=None):
    """
    biopy_rmsd2_dists mobile, reference, sele=all, atomfile=None

    类似 rmsd2():
      - 用同样的原子集合做 Superimposer 拟合
      - 不修改 PyMOL 中的对象
      - 返回每一对原子在拟合后的距离列表 ds

    返回：
      Python list[float] 或 None
    """
    ref_xyz, mob_xyz = _collect_atom_pairs(reference, mobile, sele=sele, atomfile=atomfile)
    if ref_xyz is None:
        return None

    sup = Superimposer()
    sup.set_atoms(ref_xyz, mob_xyz)
    rot, tran = sup.rotran

    # aligned = mob @ rot + tran
    mob_fit = np.dot(mob_xyz, rot) + tran

    ds = np.linalg.norm(ref_xyz - mob_fit, axis=1)
    print(f"[RNARMSD] biopy_rmsd2_dists: {len(ds)} distances, RMSD={float(sup.rms):.3f} Å")
    return ds.tolist()


def rna_cleanup_and_super(mobile, reference, sele="all",
                          resfile=None, atomfile=None):
    """
    rna_cleanup_and_super mobile, reference, sele=all, resfile=None, atomfile=None

    一键流程：
      1) 对 reference / mobile 都做残基名标准化（residues.txt）
      2) 对 reference / mobile 都做原子名标准化（atoms.txt）
      3) 用 atoms.txt 里 backbone+heavy 原子集合做 Biopython super，对齐 mobile -> reference

    返回：
      RMSD (float) 或 None
    """
    print(f"[RNARMSD] cleanup residues for {reference} & {mobile}")
    normalize_resn(obj=reference, mapfile=resfile)
    normalize_resn(obj=mobile,    mapfile=resfile)

    print(f"[RNARMSD] cleanup atom names for {reference} & {mobile}")
    normalize_atom_names(obj=reference, atomfile=atomfile)
    normalize_atom_names(obj=mobile,    atomfile=atomfile)

    print(f"[RNARMSD] super {mobile} -> {reference}")
    return biopy_super_to_ref(mobile, reference, sele=sele, atomfile=atomfile, apply_transform=1)


def rna_super_all_to(reference, sele="all", resfile=None, atomfile=None):
    """
    rna_super_all_to reference, sele=all, resfile=None, atomfile=None

    对当前 session 里的所有 object：
      1) 清洗残基名
      2) 清洗原子名
      3) 用 Biopython super 对齐到 reference
    """
    names = cmd.get_names("objects")
    for obj in names:
        if obj == reference:
            continue
        print(f"[RNARMSD] === {obj} -> {reference} ===")
        rna_cleanup_and_super(obj, reference,
                              sele=sele,
                              resfile=resfile,
                              atomfile=atomfile)


# ===================== PyMOL 插件入口 =====================

def __init_plugin__(app=None):
    """
    PyMOL 插件入口：注册命令。
    """
    cmd.extend("normalize_resn",        normalize_resn)
    cmd.extend("normalize_atom_names",  normalize_atom_names)
    cmd.extend("biopy_super_to_ref",    biopy_super_to_ref)
    cmd.extend("biopy_rmsd2_dists",     biopy_rmsd2_dists)
    cmd.extend("rna_cleanup_and_super", rna_cleanup_and_super)
    cmd.extend("rna_super_all_to",      rna_super_all_to)

    print("[RNARMSDPlugin] 已注册命令：")
    print("  normalize_resn obj, mapfile=None")
    print("  normalize_atom_names obj, atomfile=None")
    print("  biopy_super_to_ref mobile, reference, sele=all, atomfile=None, apply_transform=1")
    print("  biopy_rmsd2_dists mobile, reference, sele=all, atomfile=None")
    print("  rna_cleanup_and_super mobile, reference, sele=all, resfile=None, atomfile=None")
    print("  rna_super_all_to reference, sele=all, resfile=None, atomfile=None")
