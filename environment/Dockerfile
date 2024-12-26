# ベースイメージとしてPythonを使用
FROM python:3.12.8

# 作業ディレクトリを設定
WORKDIR /app

# 依存関係をコピー
COPY requirements.txt .

# 依存関係をインストール
RUN pip install -r requirements.txt

# Install CPLEX
# https://www.ibm.com/docs/ja/icos/22.1.2?topic=2212-installing-cplex-optimization-studio
# https://www.ibm.com/docs/ja/icos/22.1.2?topic=2212-silent-installation-cplex-optimization-studio

# for preview version
ARG CPLEX_INSTALLER="cos_installer_preview-22.1.2.R4-M0N99ML-linux-arm64.bin"
ARG CPLEX_PATH="/opt/ibm/ILOG/CPLEX_Studio_Community2212"

# build CPLEX
COPY cplex/${CPLEX_INSTALLER} /cplex/${CPLEX_INSTALLER}
COPY cplex/response.properties /cplex/response.properties
RUN chmod u+x /cplex/${CPLEX_INSTALLER}
RUN /cplex/${CPLEX_INSTALLER} -f /cplex/response.properties
RUN rm -rf /cplex

# pathの設定
ENV PATH="$PATH:/opt/ibm/ILOG/CPLEX_Studio_Community2212/cplex/bin/arm64_linux"
ENV PYTHONPATH="/app/script:$PYTHONPATH"

CMD ["/bin/bash"]
