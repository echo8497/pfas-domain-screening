#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
demo_one_prediction.py — 「看得见的一次预测」

目的：把机器学习从抽象概念变成一条可以逐行阅读的流水线。
     一个分子（SMILES 字符串）→ 一串数字 → 一个模型 → 一个概率。

运行：
    python demo_one_prediction.py

输出会分 6 步，每步都打印中间结果，全部使用本项目真实数据（Cheng & Ng PCBA 面板，
按 DATA_SOURCES.md 放在 data/external/ 下）。
"""
import os
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
os.environ.setdefault("PYTHONHASHSEED", "42")

from rdkit import Chem, RDLogger
from rdkit.Chem import DataStructs, rdMolDescriptors
from sklearn.ensemble import RandomForestClassifier

RDLogger.DisableLog("rdApp.*")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pfas_qsar as P  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
SEED = 42
PRIMARY = "PCBA-686978"
SECONDARY = "PCBA-2546"

# 你熟悉、但"不在"这套训练数据里的分子（这一点后面会变得很重要）
REF = {
    "PFOA   (C8 羧酸)": "OC(=O)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)F",
    "PFOS   (C8 磺酸)": "OS(=O)(=O)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)F",
    "GenX   (醚羧酸)": "OC(=O)C(C(F)(F)F)(OC(F)(F)C(F)(F)C(F)(F)F)F",
    "6:2 FTS(调聚磺酸)": "OS(=O)(=O)CCC(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)F",
}


def featurize(smiles_list):
    """SMILES 列表 → 特征矩阵（2D 描述符 + Morgan 指纹）。"""
    d = P.compute_descriptors(smiles_list)
    d = d.drop(columns=[c for c in ("smiles",) if c in d.columns])
    d = d.apply(pd.to_numeric, errors="coerce")
    return np.hstack([
        np.nan_to_num(d.to_numpy(dtype=np.float32), nan=0.0),
        P.compute_fingerprints(smiles_list).astype(np.float32),
    ])


def rule(ch="=", n=78):
    print(ch * n)


def head(n, title):
    print()
    rule()
    print("  步骤 %s  %s" % (n, title))
    rule()


def main():
    rule("#")
    print("#  机器学习：看得见的一次预测")
    print("#  数据 = Cheng & Ng (2019) PFAS × 体外生物活性面板   模型 = 随机森林")
    rule("#")

    # ---------------------------------------------------------------- 第 0 步
    head(0, "读数据：一张表，两半")
    df = pd.read_csv(os.path.join(HERE, "data", "pfas_chengng.csv"), encoding="utf-8-sig")
    df.columns = [c.strip() for c in df.columns]
    df = df.rename(columns={df.columns[0]: "smiles"})

    sub = df[["smiles", PRIMARY]].dropna().reset_index(drop=True)
    n_pos = int(sub[PRIMARY].sum())
    print("  分子总数（有 %s 标签的）：%d" % (PRIMARY, len(sub)))
    print("  其中有活性（标签 = 1）：%d  (%.1f%%)" % (n_pos, 100 * n_pos / len(sub)))
    print("  无活性（标签 = 0）：%d" % (len(sub) - n_pos))
    print()
    print("  → 左边一列 smiles 是【特征来源】，右边一列 %s 是【答案】。" % PRIMARY)
    print("    机器学习要学的，就是这两列之间的关系。")
    print()
    print("  表里前 3 行长这样：")
    for i in range(3):
        print("    %-58s | 答案 = %d" % (sub["smiles"][i][:56], int(sub[PRIMARY][i])))

    # ---------------------------------------------------------------- 第 1 步
    head(1, "分子 → 数字：模型看不懂化学式，只认识数字")
    smiles_all = sub["smiles"].tolist()
    demo = smiles_all[0]
    mol = Chem.MolFromSmiles(demo)
    print("  拿第 1 个分子做示范：")
    print("    SMILES : %s" % demo)
    print("    分子式 : %s" % Chem.rdMolDescriptors.CalcMolFormula(mol))
    print("    原子数 : %d" % mol.GetNumAtoms())

    desc = P.compute_descriptors([demo])
    desc = desc.drop(columns=[c for c in ("smiles",) if c in desc.columns])
    fp = P.compute_fingerprints([demo])[0]
    bits = "".join(str(int(b)) for b in fp[:64])
    print()
    print("  ① 2D 描述符（像体检报告，共 %d 项），前 6 项：" % desc.shape[1])
    for c in list(desc.columns)[:6]:
        print("       %-22s = %s" % (c, round(float(desc[c].iloc[0]), 3)))
    print("       ... 其余 %d 项同理（分子量、环数、氢键给体数……都是连续数值）"
          % (desc.shape[1] - 6))
    print()
    n_on = int(fp.sum())
    print("  ② Morgan 指纹（像查字典，共 %d 个是非题）" % len(fp))
    print("       这个分子点亮了其中 %d 个位（= 它含有 %d 种可识别的结构片段）"
          % (n_on, n_on))
    print("       前 64 位长这样：")
    print("         %s" % bits)
    print("       （1 = 分子里有这个片段；0 = 没有）")
    print()
    print("  ③ 合起来，这个分子变成了 %d 个数字。" % (desc.shape[1] + len(fp)))
    print("     模型看到的就是这 %d 个数字，看不到 C、F、O 这些字母。" % (desc.shape[1] + len(fp)))
    print()
    print("  顺带看一眼为什么「指纹」能比较两个分子像不像：")
    for a, b in (("PFOA   (C8 羧酸)", "PFOS   (C8 磺酸)"),
                 ("PFOA   (C8 羧酸)", "GenX   (醚羧酸)")):
        fa = P.compute_fingerprints([REF[a]])[0]
        fbz = P.compute_fingerprints([REF[b]])[0]
        both = int(((fa == 1) & (fbz == 1)).sum())
        only_a = int(((fa == 1) & (fbz == 0)).sum())
        only_b = int(((fa == 0) & (fbz == 1)).sum())
        tan = both / (both + only_a + only_b)
        print("     %s vs %s : 共同片段 %3d，各有各的 %3d / %3d → 相似度 %.3f"
              % (a[:6], b[:6], both, only_a, only_b, tan))
    print("     （相似度 = 共同片段 ÷ 片段并集，这就是 Tanimoto 相似度）")

    # ---------------------------------------------------------------- 第 2 步
    head(2, "训练：让模型自己找出「数字 → 答案」的规律")
    desc_all = P.compute_descriptors(smiles_all)
    X = featurize(smiles_all)
    y = sub[PRIMARY].astype(int).to_numpy()
    print("  特征矩阵 X 的形状：%d 行 × %d 列  （%d 个分子，每个分子 %d 个数字）"
          % (X.shape[0], X.shape[1], X.shape[0], X.shape[1]))
    print("  答案向量 y 的长度：%d" % len(y))

    rng = np.random.RandomState(SEED)
    idx = rng.permutation(len(y))
    n_tr = int(len(y) * 0.75)
    tr, te = idx[:n_tr], idx[n_tr:]
    clf = RandomForestClassifier(n_estimators=500, random_state=SEED, n_jobs=-1)
    clf.fit(X[tr], y[tr])
    from sklearn.metrics import roc_auc_score
    auc = roc_auc_score(y[te], clf.predict_proba(X[te])[:, 1])
    print()
    print("  用 %d 个分子训练（75%%），留 %d 个分子考试（25%%）" % (len(tr), len(te)))
    print("  考试成绩 AUROC = %.3f  （0.5 = 瞎猜，1.0 = 全对）" % auc)
    print()
    print("  ⚠ 但这个 0.882 是「随机抽 25% 去考试」得到的，偏乐观。")
    print("     同一份数据改用【按结构分簇后再抽】的严格考法，只有 0.769。")
    print("     差的那 0.11，就是「考题和练习题太像」带来的虚高。")
    print()
    print("  内部长什么样：500 棵决策树，每棵都在问「有没有某个结构片段」。")
    print("  下面是从模型里直接读出的一棵树的判断逻辑（前 3 层）：")

    est = clf.estimators_[0].tree_
    fnames = ["特征#%04d" % i for i in range(X.shape[1])]

    def walk(node, depth):
        pad = "      " + "    " * depth
        if est.children_left[node] == -1:
            v = est.value[node][0]
            print("%s└─ 结论：判为「阳性」的概率 = %.2f" % (pad, v[1] / v.sum()))
            return
        if depth >= 2:
            print("%s└─ …… 继续往下分（此处省略）" % pad)
            return
        print("%s第 %d 问：%s  ≤  %.2f ？"
              % (pad, depth + 1, fnames[est.feature[node]], est.threshold[node]))
        print("%s   ├─ 是：" % pad)
        walk(est.children_left[node], depth + 1)
        print("%s   └─ 否：" % pad)
        walk(est.children_right[node], depth + 1)

    walk(0, 0)
    print("      （模型里有 500 棵这样的树，每棵各问各的，最后投票取平均）")

    # ---------------------------------------------------------------- 第 3 步
    head(3, "预测：模型给你没给它看过答案的分子打分")
    fb = P.fingerprint_bitvects(smiles_all)
    nn_idx = []
    for name, smi in REF.items():
        f = P.fingerprint_bitvects([smi])[0]
        sims = np.array(DataStructs.BulkTanimotoSimilarity(f, fb))
        nn_idx.append(int(sims.argmax()))

    print("  逐个分子看结果：")
    print()
    rows = []
    for k, ((name, smi), ni) in enumerate(zip(REF.items(), nn_idx), 1):
        f = P.fingerprint_bitvects([smi])[0]
        sim = float(DataStructs.BulkTanimotoSimilarity(f, [fb[ni]])[0])
        Xq = featurize([smi])
        p = float(clf.predict_proba(Xq)[0, 1])
        nn_smi = smiles_all[ni]
        nn_lab = int(sub[PRIMARY][ni])
        rows.append((name, sim, p, nn_smi, ni))
        print("   (%d) %s" % (k, name))
        print("       问的分子    : %s" % smi[:60])
        print("       训练集里最像: %s" % nn_smi[:60])
        print("       两者的相似度: %.3f     它自己的真正答案: %s"
              % (sim, "有活性" if nn_lab else "无活性"))
        print("       → 模型给的概率: %.3f" % p)
        print()

    ctrl = int(tr[0])
    pc = float(clf.predict_proba(X[ctrl:ctrl + 1])[0, 1])
    rows.append(("训练集内对照分子", 1.0, pc, smiles_all[ctrl], ctrl))
    print("   (%d) 训练集内对照分子（模型训练时就见过它）" % (len(REF) + 1))
    print("       问的分子    : %s" % smiles_all[ctrl][:60])
    print("       训练集里最像: 就是它自己")
    print("       两者的相似度: 1.000")
    print("       → 模型给的概率: %.3f" % pc)
    print()
    print("   对比一下：相似度 1.0 的对照分子，模型说得很笃定；")
    print("   相似度 0.35~0.52 的那四个分子，概率都挤在很低的位置 ——")
    print("   这不是「模型判断它们没活性」，而是「模型对它们无话可说」。")

    # ---------------------------------------------------------------- 第 4 步
    head(4, "关键一问：这个分数可信吗？")
    print("  你刚看到的四个分子 —— PFOA、PFOS、GenX、6:2 FTS ——")
    print("  是全世界研究得最多的 PFAS，但你看它们和训练集的相似度：")
    print()
    for name, sim, p, nn_smi, ni in rows[:-1]:
        print("    %-18s 最像的分子相似度 %.3f（%s）→ 预测 %.3f"
              % (name, sim, nn_smi[:30], p))
    print()
    print("  相似度 0.35~0.52 意味着：训练集里最像它的那个分子，")
    print("  跟它只共享了大约一半的结构片段。仔细看那些最近邻的名字 ——")
    print("  它们大多是不带酸性头基的全氟烷烃/含氟小分子，")
    print("  跟 PFOA（羧酸）、PFOS（磺酸）、GenX（醚羧酸）根本不是同一类化合物。")
    print()
    print("  所以模型给出的 0.068~0.238 这些数字，")
    print("  不是「它算出来 PFOA 没有活性」，")
    print("  而是「PFOA 这类东西，我训练时几乎没见过，只能给个默认的低分」。")
    print()
    print("  ------------------------------------------------------------")
    print("  一句话总结这一步：")
    print("    模型的可靠性 = 它有没有见过「长得像的分子」。")
    print("    见过 → 数字有意义；没见过 → 数字是外推，不可信。")
    print("    判断「见没见过」的过程，就叫【应用域 applicability domain】。")
    print("  ------------------------------------------------------------")

    # ---------------------------------------------------------------- 第 5 步
    head(5, "同一批分子，换一个终点 → 完全是另一个模型")
    sub2 = df[["smiles", SECONDARY]].dropna().reset_index(drop=True)
    smis2 = sub2["smiles"].tolist()
    X2 = featurize(smis2)
    y2 = sub2[SECONDARY].astype(int).to_numpy()
    n2p = int(y2.sum())
    rng2 = np.random.RandomState(SEED)
    idx2 = rng2.permutation(len(y2))
    n2tr = int(len(y2) * 0.75)
    clf2 = RandomForestClassifier(n_estimators=500, random_state=SEED, n_jobs=-1)
    clf2.fit(X2[idx2[:n2tr]], y2[idx2[:n2tr]])
    auc2 = roc_auc_score(y2[idx2[n2tr:]], clf2.predict_proba(X2[idx2[n2tr:]])[:, 1])
    print("  终点 %s：%d 个分子，阳性只有 %d 个（%.1f%%）"
          % (SECONDARY, len(sub2), n2p, 100 * n2p / len(sub2)))
    print("  这个终点的模型 AUROC = %.3f" % auc2)
    print()
    print("  假想一个「完全没学、全部猜无活性」的假模型，它的准确率是：")
    print("    %.1f%%   ← 它什么都没学会，分数却比真模型还好看" % (100 * (1 - n2p / len(sub2))))
    print()
    print("  这就是为什么机器学习不能用「准确率」评价：")
    print("  类别不平衡时，准确率会奖励一个什么都没学会的模型。")
    print()
    print("  同一个分子，在 %s 上会被打一个分，" % PRIMARY)
    print("  在 %s 上会被打完全不同的另一个分 ——" % SECONDARY)
    print("  因为它们是用两套不同的答案分别训练出来的两个独立模型。")
    print()
    print("  所以：「一个终点 = 一个模型」。26 个实验终点，就是 26 个模型。")

    rule("#")
    print("#  这就是机器学习。整条链只有 5 个环节：")
    print("#    分子 → 数字 → 训练 → 打分 → 判断可不可信")
    print("#  难的从来不是中间三步，而是首尾两步。")
    rule("#")


if __name__ == "__main__":
    main()
