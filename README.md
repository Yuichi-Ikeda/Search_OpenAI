# RAG（Retrieval-Augmented Generation）システムの調査用ツール

## 概要

このプロジェクトは、Azure AI Search と Azure OpenAI を使用してユーザーの質問に回答する RAG（Retrieval-Augmented Generation）システムを構築する際の調査用コマンドラインツールです。

### 主な機能

- **ハイブリッド検索**: キーワード検索、ベクトル検索、セマンティック検索を組み合わせた高精度な文書検索
- **自動回答生成**: 検索結果を基にGPT-4が適切な回答を生成
- **トークン管理**: 120,000トークンの制限内で最適な検索結果を選定
- **スコアリング**: セマンティックランカーによる検索結果の品質評価

## システム構成

```mermaid
graph LR
    A[ユーザーの質問] --> B[Azure AI Search]
    B --> C[検索結果]
    C --> D[OpenAI]
    D --> E[最終回答]
```

## 必要な環境

- Python 3.11+
- Azure OpenAI サービス
- Azure AI Search サービス

## セットアップ

### 1. 環境変数の設定

プロジェクトルートに `.env` ファイルを作成し、以下の環境変数を設定してください：

```env
AZURE_OPENAI_KEY=your_azure_openai_key
AZURE_OPENAI_ENDPOINT=your_azure_openai_endpoint
AZURE_SEARCH_KEY=your_azure_search_key
AZURE_SEARCH_ENDPOINT=your_azure_search_endpoint
AZURE_SEARCH_INDEX=your_search_index_name
```
### 2. Python 仮想環境の作成

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
```

### 3. 依存関係のインストール

```bash
pip install -r requirements.txt
```

## 使用方法

### 基本的な使用方法

```cmd
python main.py "質問内容"
```

### 使用例

```cmd
python main.py "有給休暇の申請方法を教えてください"
```

## ファイル構成

```
Search_OpenAI/
├── main.py              # メインプログラム
├── system_prompt.py     # システムプロンプトの定義
├── requirements.txt     # 依存パッケージ一覧
├── .env                # 環境変数（要作成）
└── README.md           # このファイル
```

## 主要な関数

### `main()`
- コマンドライン引数から質問を取得
- 検索から回答生成までの一連の処理を実行

### `search(query, target_index)`
- Azure AI Search を使用したハイブリッド検索
- キーワード、ベクトル、セマンティック検索を組み合わせて実行
- 上位100件の結果を取得し、トークン制限内で最適化

### `openai(system_template, query)`
- GPT-4 を使用した最終回答の生成
- 検索結果をコンテキストとして活用

### `to_vectorize(query)`
- OpenAI Embedding API を使用したテキストのベクトル化
- text-embedding-3-large モデルを使用（3072次元）

### `calc_token(content)`
- テキストのトークン数計算
- トークン制限の管理に使用

## 技術仕様

- **使用モデル**: GPT-4.1
- **埋め込みモデル**: text-embedding-3-large
- **検索方式**: ハイブリッド検索（キーワード + ベクトル + セマンティック）
- **トークン制限**: 120,000トークンに設定（カスタマイズ可能）
- **検索結果数**: 最大100件（カスタマイズ可能）

## 出力情報

実行時に以下の情報が表示されます：

- 検索結果の詳細（ID、スコア、ファイルパス、キャプション）
- 各文書のトークン数
- 総トークン数
- LLM処理時間
- 最終回答

## 開発者向け情報

### システムプロンプト

`system_prompt.py` で OpenAI モデルに与えるシステムプロンプトを管理しています。回答の品質や形式を調整したい場合は、このファイルを編集してください。

### カスタマイズ

- 検索結果の上位件数: `main.py` の `top` パラメータを変更
- トークン制限: `main.py` の `120000` の値を変更
- セマンティック設定: 検索クエリの `semantic_configuration_name` を変更