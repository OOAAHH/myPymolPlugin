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
#     - 链 ID 标准化：RNA 链在前，DNA 链其后，蛋白链再后，最后其它链；
#       每个 object 从 A/B/C… 重新编号
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
        # 必须双引号，否则 C1' 之类会截断字符串
        cmd.alter(sel, f'name="{new}"')

    cmd.sort()


# ===================== 链类型判断 & 链 ID 归一化 =====================

_PROTEIN_RESN = {
    "ALA", "ARG", "ASN", "ASP", "CYS",
    "GLU", "GLN", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO",
    "SER", "THR", "TRP", "TYR", "VAL",
    "SEC", "PYL",
    "HSD", "HSE", "HSP"
}

def _classify_chain_type(obj, chain_id, residue_map):
    """
    粗略分类链类型：
      - 根据 residue_map 判断核酸（RNA / DNA）
      - 根据 _PROTEIN_RESN 判断蛋白
    返回: 'rna' / 'dna' / 'protein' / 'other'
    """
    if chain_id:
        sel = f"{obj} and chain {chain_id}"
    else:
        sel = f"{obj} and chain ''"

    model = cmd.get_model(sel)
    if not model.atom:
        return "other"

    resn_counts = {}
    for a in model.atom:
        resn = a.resn
        resn_counts[resn] = resn_counts.get(resn, 0) + 1

    protein = dna = rna = 0

    for resn, count in resn_counts.items():
        if resn in _PROTEIN_RESN:
            protein += count

        code = residue_map.get(resn)
        if code in ("A", "C", "G", "U"):
            rna += count
        elif code in ("T", "D"):
            dna += count

        if code is None:
            if resn in ("A", "C", "G", "U", "I"):
                rna += count
            elif resn in ("DA", "DC", "DG", "DT", "DI", "T"):
                dna += count

    max_score = max(rna, dna, protein)
    if max_score == 0:
        return "other"
    if max_score == rna:
        return "rna"
    if max_score == dna:
        return "dna"
    if max_score == protein:
        return "protein"
    return "other"


def normalize_chain_ids(obj, resfile=None):
    """
    normalize_chain_ids obj, resfile=None

    对单个 object 的链 ID 重新编号：
      - 先按类型分组：RNA -> DNA -> protein -> other
      - 每组内保持原始顺序
      - 然后整体重新编号为 A, B, C, ...（即 C/D 变成 A/B）

    使用 segi 临时存储原始链 ID，避免冲突。
    """
    residue_map = _load_residue_map(resfile)

    chains = list(cmd.get_chains(obj))
    if not chains:
        return

    type_map = {}
    for ch in chains:
        ctype = _classify_chain_type(obj, ch, residue_map)
        type_map[ch] = ctype

    ordered = []
    for t in ("rna", "dna", "protein", "other"):
        ordered.extend([ch for ch in chains if type_map.get(ch) == t])

    for ch in chains:
        if ch not in ordered:
            ordered.append(ch)

    chain_chars = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")
    if len(ordered) > len(chain_chars):
        print(f"[RNARMSD] WARNING: 对象 {obj} 的链数量超过 36，超出重命名范围。")

    mapping = {}
    for i, ch in enumerate(ordered):
        if i >= len(chain_chars):
            new_id = chain_chars[-1]
        else:
            new_id = chain_chars[i]
        mapping[ch] = new_id

    print(f"[RNARMSD] normalize_chain_ids for {obj}: {mapping}")

    cmd.alter(obj, "segi=chain")
    for old, new in mapping.items():
        if old:
            sel = f"({obj}) and segi '{old}'"
        else:
            sel = f"({obj}) and segi ''"
        cmd.alter(sel, f'chain="{new}"')
    cmd.alter(obj, 'segi=""')
    cmd.sort()


# ===================== 原子配对 & Biopython Superimposer =====================

def _build_res_dict(model, rename_map, allowed, ignored):
    """
    把 PyMOL model.atom 按 residue 分组并应用原子规则。

    key: (chain, resi, resn)
    val: dict { canonical_atom_name -> [x, y, z] }
    """
    res_dict = {}

    for a in model.atom:
        name = rename_map.get(a.name, a.name)

        if name in ignored:
            continue
        if allowed and (name not in allowed):
            continue

        key = (a.chain, a.resi, a.resn)

        if hasattr(a, "coord"):
            coords = [a.coord[0], a.coord[1], a.coord[2]]
        else:
            coords = [a.x, a.y, a.z]

        bucket = res_dict.setdefault(key, {})
        bucket[name] = coords

    return res_dict


def _collect_atom_pairs(reference, mobile, sele="all", atomfile=None):
    """
    按 atoms.txt 的规则，从 reference / mobile 中抽取完全对应的原子对。
    返回：
      ref_xyz: numpy (N, 3)
      mob_xyz: numpy (N, 3)
    """
    rename_map, allowed, ignored = _load_atom_map(atomfile)

    m_ref = cmd.get_model(f"{reference} and ({sele})")
    m_mob = cmd.get_model(f"{mobile} and ({sele})")

    ref_res = _build_res_dict(m_ref, rename_map, allowed, ignored)
    mob_res = _build_res_dict(m_mob, rename_map, allowed, ignored)

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


# --- 关键修复：给 Biopython.Superimposer 一个带 get_coord 的“伪 Atom” ---

class _CoordAtom:
    """简单包装一个坐标，让 Biopython.Superimposer 可以调用 get_coord()."""
    def __init__(self, coord):
        self._coord = np.array(coord, dtype=float)
    def get_coord(self):
        return self._coord


def biopy_super_to_ref(mobile, reference, sele="all",
                       atomfile=None, apply_transform=1):
    """
    biopy_super_to_ref mobile, reference, sele=all, atomfile=None, apply_transform=1

    使用 atoms.txt 选 backbone+heavy 原子，并用 Biopython.Superimposer
    做刚体最小二乘，对齐 mobile -> reference。
    """
    # 1. 根据 atoms.txt 收集原子对（只用于求 rotran）
    ref_xyz, mob_xyz = _collect_atom_pairs(reference, mobile, sele=sele, atomfile=atomfile)
    if ref_xyz is None:
        return None

    # 2. 用 Biopython Superimposer 求解旋转/平移
    sup = Superimposer()
    fixed_atoms  = [_CoordAtom(c) for c in ref_xyz]   # reference
    moving_atoms = [_CoordAtom(c) for c in mob_xyz]   # mobile
    sup.set_atoms(fixed_atoms, moving_atoms)
    rot, tran = sup.rotran      # 注意：Biopython 约定是 mob @ rot + tran

    # 3. 把这个刚体变换真正应用到 PyMOL 里的 mobile 对象坐标上
    if apply_transform:
        # 取出 mobile 当前 state=1 的所有坐标（包含对象矩阵）
        coords = cmd.get_coords(mobile, state=1)
        if coords is None:
            print(f"[RNARMSD] WARNING: get_coords({mobile}) 返回空，无法应用变换。")
        else:
            # 和 Biopython 示例完全一致：moving_on_fixed = moving @ rot + tran
            coords_fit = np.dot(coords, rot) + tran

            # 回写到 mobile（同一个 state）
            # load_coords 的坐标顺序和 get_coords 一致
            cmd.load_coords(coords_fit.tolist(), mobile, state=1)

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
    """
    ref_xyz, mob_xyz = _collect_atom_pairs(reference, mobile, sele=sele, atomfile=atomfile)
    if ref_xyz is None:
        return None

    sup = Superimposer()
    fixed_atoms  = [_CoordAtom(c) for c in ref_xyz]
    moving_atoms = [_CoordAtom(c) for c in mob_xyz]
    sup.set_atoms(fixed_atoms, moving_atoms)
    rot, tran = sup.rotran

    mob_fit = np.dot(mob_xyz, rot) + tran
    ds = np.linalg.norm(ref_xyz - mob_fit, axis=1)
    print(f"[RNARMSD] biopy_rmsd2_dists: {len(ds)} distances, RMSD={float(sup.rms):.3f} Å")
    return ds.tolist()


def rna_cleanup_and_super(mobile, reference, sele="all",
                          resfile=None, atomfile=None):
    """
    rna_cleanup_and_super mobile, reference, sele=all, resfile=None, atomfile=None

    一键流程：
      1) 对 reference / mobile 都做链 ID 标准化（RNA→DNA→蛋白→其它）
      2) 对 reference / mobile 都做残基名标准化（residues.txt）
      3) 对 reference / mobile 都做原子名标准化（atoms.txt）
      4) 用 backbone+heavy 原子集合做 Biopython super，对齐 mobile -> reference
    """
    print(f"[RNARMSD] normalize chains for {reference} & {mobile}")
    normalize_chain_ids(reference, resfile=resfile)
    normalize_chain_ids(mobile,    resfile=resfile)

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
      1) 标准化链 ID（RNA 链在前，DNA、蛋白在后，链从 A/B/C... 编号）
      2) 清洗残基名
      3) 清洗原子名
      4) 用 Biopython super 对齐到 reference
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
    cmd.extend("normalize_chain_ids",   normalize_chain_ids)
    cmd.extend("biopy_super_to_ref",    biopy_super_to_ref)
    cmd.extend("biopy_rmsd2_dists",     biopy_rmsd2_dists)
    cmd.extend("rna_cleanup_and_super", rna_cleanup_and_super)
    cmd.extend("rna_super_all_to",      rna_super_all_to)

    print("[RNARMSDPlugin] 已注册命令：")
    print("  normalize_resn obj, mapfile=None")
    print("  normalize_atom_names obj, atomfile=None")
    print("  normalize_chain_ids obj, resfile=None")
    print("  biopy_super_to_ref mobile, reference, sele=all, atomfile=None, apply_transform=1")
    print("  biopy_rmsd2_dists mobile, reference, sele=all, atomfile=None")
    print("  rna_cleanup_and_super mobile, reference, sele=all, resfile=None, atomfile=None")
    print("  rna_super_all_to reference, sele=all, resfile=None, atomfile=None")
