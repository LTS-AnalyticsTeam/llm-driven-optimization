# 環境構築
下記箇所に`cos_installer_preview-22.1.2.R4-M0N99ML-linux-arm64.bin`を配置する。
```
.
└─ environment
   └─ cplex
      ├─ cos_installer_preview-22.1.2.R4-M0N99ML-linux-arm64.bin
      └─ response.properties
```

次のコマンドを実行することで、dockerイメージとコンテナが作成できる。
```sh
cd environment
bash run.sh 
```
