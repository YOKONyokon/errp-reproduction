# =============================================================
# ERRP 复现项目 · Docker 镜像定义（阶段 2 交付物）
# 对应论文：Yasemin et al. (2023), JNE 20:016015
#
# 在具备权限的环境（普通 VM / 已授权的 Docker 主机）构建：
#   docker build -t errp:latest .
# 运行（数据与结果通过挂载，不打进镜像）：
#   docker run --rm -v "$PWD/third_party:/app/third_party" \
#     -v "$PWD/results:/app/results" errp:latest \
#     python run_reproduction.py --quick
# 仅做环境自检：
#   docker run --rm errp:latest python /app/scripts/verify_env.py
# =============================================================
FROM python:3.12-slim

# ---- 运行期环境变量 ----
# PYTHONNOUSERSITE：禁用用户级 site-packages，保证只用镜像内依赖
# MPLBACKEND=Agg：无显示环境下使用非交互后端
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONNOUSERSITE=1 \
    MPLBACKEND=Agg \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# ---- 先装依赖（单独一层，利用构建缓存）----
COPY requirements.txt ./
RUN python -m pip install --upgrade pip && \
    python -m pip install -r requirements.txt

# ---- 复制项目代码与入口脚本 ----
COPY config.py ./
COPY src ./src
COPY run_reproduction.py run_experiments.py run_morphology.py run_regression.py ./
COPY scripts ./scripts

# ---- 数据 / 结果挂载点（镜像内只建空目录，实际内容由 -v 注入）----
RUN mkdir -p third_party results
VOLUME ["/app/third_party", "/app/results"]

# ---- 默认：环境自检，确认容器可运行且依赖路径正确 ----
ENTRYPOINT ["python"]
CMD ["/app/scripts/verify_env.py"]
