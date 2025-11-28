# myPymolPlugin

## 清洗某个对象的残基名 基于residue.txt
normalize_resn mobile_obj

## 清洗某个对象的原子名 基于atom.txt
normalize_atom_names obj

## 用 Biopython super 对齐，用biopython而不是pymol的实现来执行刚体最小问题
biopy_super_to_ref mobile_obj, ref_obj

## 对残基序号重新计数
renumber_residues obj

## 一键：先清洗 mobile，再对齐到 ref
rna_cleanup_and_super mobile_obj, ref_obj

## 用rnapuzzles assesment的第二种RMSD计算方式实现刚体最小问题的求解
biopy_rmsd2_dists mobile, reference

## 把ref之外的所有的model都执行归一化，并用biopython而不是pymol的实现来执行刚体最小问题
rna_super_all_to reference
