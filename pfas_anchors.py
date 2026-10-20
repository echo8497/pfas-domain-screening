"""
锚点面板唯一定义（single source of truth）。

背景：在此之前 `screen_domain.load_reference()`（18 条）与
`screen_alternatives.REFERENCE`（20 条）各维护一份锚点清单，且其中
ADONA / F-53B / 6:2 FTS / 4:2 FTS 四条结构有误，直接导致稿件中
"18 个锚点" 与 "20 个锚点"、16/18 与 20 条新面板的计数矛盾。

本模块是唯一真源。所有脚本必须 `from pfas_anchors import ANCHORS, ...`，
不得再各自硬编码。

每条锚点字段：
    name     展示名
    group    "legacy"（受监管/前体 PFAA） 或 "alternative"（已商品化替代物）
    smiles   结构（分子式已用 RDKit 逐一核对）
    formula  期望分子式（用于交叉校验）
    casrn    CAS 登记号
    note     语境注释
"""

_CF2 = "C(F)(F)"


# ---------------------------------------------------------------- 结构构造器
def pfca(n):
    """全氟羧酸 F(CF2)n-1COOH"""
    return "OC(=O)" + _CF2 * (n - 1) + "F"


def pfsa(n):
    """全氟磺酸 F(CF2)nSO3H"""
    return "OS(=O)(=O)" + _CF2 * n + "F"


def samide(n):
    """全氟磺酰胺 F(CF2)nSO2NH2"""
    return "NS(=O)(=O)" + _CF2 * n + "F"


def fts(n):
    """氟调聚物磺酸 F(CF2)nCH2CH2SO3H。磺酰基与乙基不含全氟碳，
    故全氟链需 n 个 CF 单元（末位 CF2 由 F 封端，即 CF3）。"""
    return "OS(=O)(=O)CC" + _CF2 * n + "F"


ADONA = "OC(=O)C(F)(F)C(OC(F)(F)C(F)(F)C(F)(F)OC(F)(F)F)F"
F53B = "OS(=O)(=O)" + _CF2 * 2 + "O" + _CF2 * 6 + "Cl"
GENX = "OC(=O)C(F)(C(F)(F)F)OC(F)(F)C(F)(F)C(F)(F)F"
PFMOAA = "OC(=O)C(F)(OC(F)(F)F)F"
NETFOSAA = "CCN(CC(=O)O)S(=O)(=O)" + _CF2 * 8 + "F"


# ---------------------------------------------------------------- 唯一面板
ANCHORS = [
    # name, group, smiles, formula, casrn, note
    ("PFOA", "legacy", pfca(8), "C8HF15O2", "335-67-1", "C8 羧酸 · 已列 POPs"),
    ("PFNA", "legacy", pfca(9), "C9HF17O2", "375-95-1", "C9 羧酸 · 已列 POPs"),
    ("PFDA", "legacy", pfca(10), "C10HF19O2", "335-76-2", "C10 羧酸"),
    ("PFHxA", "legacy", pfca(6), "C6HF11O2", "307-24-4", "C6 羧酸 · 短链"),
    ("PFHpA", "legacy", pfca(7), "C7HF13O2", "375-85-9", "C7 羧酸"),
    ("PFBA", "legacy", pfca(4), "C4HF7O2", "375-22-4", "C4 羧酸 · 短链"),
    ("PFPeA", "legacy", pfca(5), "C5HF9O2", "2706-90-3", "C5 羧酸 · 短链"),
    ("PFOS", "legacy", pfsa(8), "C8HF17O3S", "1763-23-1", "C8 磺酸 · 已列 POPs"),
    ("PFHxS", "legacy", pfsa(6), "C6HF13O3S", "355-46-4", "C6 磺酸 · 短链"),
    ("PFBS", "legacy", pfsa(4), "C4HF9O3S", "375-73-5", "C4 磺酸 · 短链"),
    ("PFDS", "legacy", pfsa(10), "C10HF21O3S", "335-77-3", "C10 磺酸"),
    ("PFOSA", "legacy", samide(8), "C8H2F17NO2S", "754-91-6", "磺酰胺 · 前体物"),
    ("N-EtFOSAA", "legacy", NETFOSAA, "C12H8F17NO4S", "2991-50-6",
     "磺酰胺乙酸 · 前体物"),
    ("PFHxSA", "legacy", samide(6), "C6H2F13NO2S", "41997-13-1", "磺酰胺 · 短链"),
    ("GenX / HFPO-DA", "alternative", GENX, "C6HF11O3", "13252-13-6",
     "醚型羧酸 · 已商品化替代物"),
    ("ADONA", "alternative", ADONA, "C7H2F12O4", "919005-14-4",
     "醚型羧酸 · 已商品化替代物"),
    ("F-53B (6:2 Cl-PFESA)", "alternative", F53B, "C8HClF16O4S", "73606-19-6",
     "氯代醚磺酸 · 中国在用替代物"),
    ("6:2 FTS", "alternative", fts(6), "C8H5F13O3S", "27619-97-2",
     "氟调聚物磺酸 · 替代物"),
    ("4:2 FTS", "alternative", fts(4), "C6H5F9O3S", "757124-72-4",
     "氟调聚物磺酸 · 替代物"),
    ("PFMOAA", "alternative", PFMOAA, "C3HF5O3", "674-13-5",
     "全氟醚羧酸 · 中国氟化工侧产物"),
]

# 已经商品化的替代物（不能算 emerging）
KNOWN_SUBSTITUTES = {a[0] for a in ANCHORS if a[1] == "alternative"}

# 早期版本中被证实写错的结构（供报告列出「已修正」项）
SUPERSEDED = {
    "ADONA": "OC(=O)C(F)(F)OC(F)(F)C(F)(F)OC(F)(F)C(F)(F)C(F)(F)F"
             "（C7HF13O4：多 1 个 F、少 1 个 H）",
    "F-53B (6:2 Cl-PFESA)": "OS(=O)(=O)" + _CF2 * 2 + _CF2 * 6 + "Cl"
                            "（漏掉醚氧，实为全氟氯代磺酸）",
    "6:2 FTS": "OS(=O)(=O)" + _CF2 * 6 + "CC"
               "（CH2CH2 错接在全氟链末端，实为 C8H6F12O3S）",
    "4:2 FTS": "OS(=O)(=O)" + _CF2 * 4 + "CCCC"
               "（碳数 9，非 4:2 FTS）",
}

# 相对早期 18 条面板新增的两条
ADDED_VS_EARLIER_PANEL = ["PFPeA", "PFMOAA"]

FIELDS = ("name", "group", "smiles", "formula", "casrn", "note")


def as_dicts():
    return [dict(zip(FIELDS, a)) for a in ANCHORS]


def reference_tuples():
    """(name, note, smiles) —— 兼容 screen_alternatives.REFERENCE 的旧签名。"""
    return [(a[0], a[5], a[2]) for a in ANCHORS]


def load_reference():
    """兼容 screen_domain.load_reference() 的 DataFrame 接口（含扩展列）。"""
    import pandas as pd
    return pd.DataFrame(as_dicts()).rename(
        columns={"name": "ref_name", "note": "ref_note"}
    )[["ref_name", "ref_note", "group", "smiles", "formula", "casrn"]]


def anchor_names(group=None):
    return [a[0] for a in ANCHORS if group is None or a[1] == group]


if __name__ == "__main__":
    from collections import Counter
    from rdkit import Chem
    from rdkit.Chem import rdMolDescriptors

    c = Counter(a[1] for a in ANCHORS)
    print(f"锚点面板：共 {len(ANCHORS)} 条 —— "
          f"legacy {c['legacy']} + alternative {c['alternative']}")
    bad = 0
    for a in ANCHORS:
        m = Chem.MolFromSmiles(a[2])
        f = rdMolDescriptors.CalcMolFormula(m)
        ok = (f == a[3])
        bad += (not ok)
        print(f"  {a[0]:24s} {f:14s} {'OK ' if ok else 'MISMATCH(exp ' + a[3] + ')'}"
              f"  CAS {a[4]}")
    print(f"分子式核对：{len(ANCHORS) - bad}/{len(ANCHORS)} 通过")
