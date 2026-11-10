import torch
import torch.nn.functional as F


def updateproto_(
    support_feats,
    query_feats,
    support_labels,
    n_way=5,
    temp=15.0
):
    """
    基于查询集加权均值的直推式原型修正。

    处理流程：
    1. 对 support 和 query 特征进行 L2 归一化；
    2. 根据 support 标签计算各类别锚点原型；
    3. 计算每个 query 对各类别的软分配权重；
    4. 使用软分配权重计算各类别的查询加权均值；
    5. 将支持集原型与查询加权均值相加并归一化。

    修正公式：
        query_center_c =
            sum_i(w_ic * q_i) / sum_i(w_ic)

        updated_proto_c =
            Normalize(anchor_proto_c + query_center_c)

    由于 anchor_proto 和 query_center 在融合前均经过 L2 归一化，
    该写法等价于：
        Normalize(
            0.5 * anchor_proto_c
            + 0.5 * query_center_c
        )

    参数：
        support_feats:
            支持集特征，形状为 [N_support, feat_dim]。

        query_feats:
            查询集特征，形状为 [N_query, feat_dim]。

        support_labels:
            episode 内支持集局部标签，形状为 [N_support]，
            标签取值应为 0, 1, ..., n_way - 1。

        n_way:
            episode 的类别数量。

        temp:
            Softmax 温度锐化系数。

    返回：
        updated_prototypes:
            修正后的类别原型，形状为 [n_way, feat_dim]。
    """

    # =========================================================
    # 0. 基础输入检查
    # =========================================================
    if support_feats.ndim != 2:
        raise ValueError(
            f"support_feats 应为二维张量，"
            f"当前形状为 {tuple(support_feats.shape)}"
        )

    if query_feats.ndim != 2:
        raise ValueError(
            f"query_feats 应为二维张量，"
            f"当前形状为 {tuple(query_feats.shape)}"
        )

    if support_labels.ndim != 1:
        raise ValueError(
            f"support_labels 应为一维张量，"
            f"当前形状为 {tuple(support_labels.shape)}"
        )

    if support_feats.size(0) != support_labels.size(0):
        raise ValueError(
            "支持集特征数量与支持集标签数量不一致："
            f"{support_feats.size(0)} != "
            f"{support_labels.size(0)}"
        )

    if support_feats.size(1) != query_feats.size(1):
        raise ValueError(
            "支持集与查询集的特征维度不一致："
            f"{support_feats.size(1)} != "
            f"{query_feats.size(1)}"
        )

    if query_feats.size(0) == 0:
        raise ValueError("query_feats 中没有查询样本。")

    if temp <= 0:
        raise ValueError(
            f"temp 必须大于 0，当前值为 {temp}"
        )

    # 确保标签位于同一设备，并转换为整数类型
    support_labels = support_labels.to(
        device=support_feats.device,
        dtype=torch.long
    )

    # =========================================================
    # 1. 将所有特征投影到单位超球面
    # =========================================================
    support_feats = F.normalize(
        support_feats,
        p=2,
        dim=-1,
        eps=1e-12
    )

    query_feats = F.normalize(
        query_feats,
        p=2,
        dim=-1,
        eps=1e-12
    )

    # =========================================================
    # 2. 使用支持集计算初始锚点原型
    # =========================================================
    anchor_prototypes = []

    for cls in range(n_way):
        class_mask = support_labels == cls

        if not class_mask.any():
            raise ValueError(
                f"类别 {cls} 没有支持样本。"
                "请检查 support_labels 是否为 episode 局部标签，"
                "即标签应为 0 到 n_way-1。"
            )

        # 计算当前类别的支持集均值原型
        class_proto = support_feats[
            class_mask
        ].mean(
            dim=0,
            keepdim=True
        )

        # 将类别原型重新投影到单位超球面
        class_proto = F.normalize(
            class_proto,
            p=2,
            dim=-1,
            eps=1e-12
        )

        anchor_prototypes.append(
            class_proto.squeeze(0)
        )

    # [n_way, feat_dim]
    anchor_prototypes = torch.stack(
        anchor_prototypes,
        dim=0
    )

    # =========================================================
    # 3. 查询样本对各类别锚点原型进行软分配
    # =========================================================
    # support/query 和原型均已归一化，因此矩阵乘法等价于余弦相似度
    similarity_matrix = torch.matmul(
        query_feats,
        anchor_prototypes.transpose(0, 1)
    )

    # [N_query, n_way]
    soft_assign = F.softmax(
        similarity_matrix * temp,
        dim=-1
    )

    # =========================================================
    # 4. 计算查询加权均值并修正类别原型
    # =========================================================
    updated_prototypes = []

    for cls in range(n_way):
        # 当前类别获得的所有查询软权重
        # 形状：[N_query, 1]
        query_weights = soft_assign[
            :, cls:cls + 1
        ]

        # 当前类别的权重总和
        weight_sum = query_weights.sum().clamp_min(
            1e-8
        )

        # 查询集加权均值，而不是原来的加权和
        query_center = (
            query_feats * query_weights
        ).sum(
            dim=0
        ) / weight_sum

        # 将查询软中心归一化
        query_center = F.normalize(
            query_center,
            p=2,
            dim=-1,
            eps=1e-12
        )

        # 支持集锚点原型 + 查询集加权均值
        beta = 0.9
        updated_proto = (
                (1.0 - beta) * anchor_prototypes[cls]
                + beta * query_center
        )

        # 将最终原型重新投影到单位超球面
        updated_proto = F.normalize(
            updated_proto,
            p=2,
            dim=-1,
            eps=1e-12
        )

        updated_prototypes.append(
            updated_proto
        )

    # [n_way, feat_dim]
    updated_prototypes = torch.stack(
        updated_prototypes,
        dim=0
    )

    return updated_prototypes