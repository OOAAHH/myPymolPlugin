# myPymolPlugin

## 清洗某个对象的残基名
normalize_resn mobile_obj

## 用 Biopython super 对齐
biopy_super_to_ref mobile_obj, ref_obj

## 一键：先清洗 mobile，再对齐到 ref
rna_cleanup_and_super mobile_obj, ref_obj
