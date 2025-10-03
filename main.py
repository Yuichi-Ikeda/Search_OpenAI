import os, sys, datetime
import tiktoken
from dotenv import load_dotenv
from openai import AzureOpenAI
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.models import QueryType
from system_prompt import AZURE_OPENAI_SYSTEM_PROMPT_CREATE_ANSWER

load_dotenv()
AZURE_OPENAI_KEY = os.getenv('AZURE_OPENAI_KEY')
AZURE_OPENAI_ENDPOINT = os.getenv('AZURE_OPENAI_ENDPOINT')

AZURE_SEARCH_KEY = os.getenv('AZURE_SEARCH_KEY')
AZURE_SEARCH_ENDPOINT = os.getenv('AZURE_SEARCH_ENDPOINT')
AZURE_SEARCH_INDEX = os.getenv('AZURE_SEARCH_INDEX')

client = AzureOpenAI(
        api_key = AZURE_OPENAI_KEY,  
        api_version = "2025-04-01-preview",
        azure_endpoint =AZURE_OPENAI_ENDPOINT 
    )


def main():
    """
    メイン関数：コマンドライン引数から質問文を取得し、
    1. Azure Searchで検索
    2. OpenAI GPTで回答生成の一連の処理を実行
    """
    args = sys.argv
    if 1 == len(args):
        print("Usage: python main.py [query]")
        return
    
    # 第一引数から質問文を取得
    query = args[1]

    # 質問文を用いて社内文書を検索
    context = search(query, AZURE_SEARCH_INDEX)

    # 検索結果を引用文としてシステムテンプレートを作成
    system_template = answer_prompt(context)

    # システムテンプレートと質問文を用いて最終回答を生成
    openai(system_template, query)


def search(query: str, target_index) -> list[dict]:
    """
    Azure AI Searchを使用してハイブリッド検索（キーワード+ベクトル+セマンティック）を実行
    
    Args:
        query (str): 検索クエリ
        target_index (str): 検索対象のインデックス名
        
    Returns:
        list[dict]: 検索結果のリスト（ファイルパス、キャプション、スコア、コンテンツを含む）
    """
    search_client = SearchClient(
            endpoint=AZURE_SEARCH_ENDPOINT,
            index_name=target_index,
            credential=AzureKeyCredential(AZURE_SEARCH_KEY)
        )

    search_results = search_client.search(
        search_text=query, #キーワード検索
        query_language="ja-jp",
        vector=to_vectorize(query), #ベクトル検索
        vector_fields='text_vector', #ベクトル検索の対象フィールド
        semantic_configuration_name='rag-1759107491727-semantic-configuration', #セマンティック設定
        query_type=QueryType.SEMANTIC, #セマンティックランカー
        query_caption="extractive",
        top=100 #上位n件
    )

    query_tokens = calc_token(query)

    count = 0
    arr_results = []
    list_filepasth = []
    for result in search_results:
        filepath = result.get('title', 'None')
        content = result.get('chunk', 'None')
        if result.get('@search.captions') != None:
            caption = result.get('@search.captions')[0].text
        else:
            caption = "None"
        reranker_score = result.get('@search.reranker_score', 0) #範囲：0.00 ~ 4.00
        id = result.get('chunk_id', 'None')

        list_filepasth.append(filepath)
        arr_results.append(
            {
                "filepath": filepath,
                "caption": caption,
                "reranker_score":reranker_score,
                "content": content
            }
        )

        # 120,000トークンを超える場合は、処理を中断
        system_tokens = calc_token(answer_prompt(arr_results))
        if(system_tokens + query_tokens > 120000):
            arr_results.pop()
            break

        # trace information
        count += 1
        print(f"({count}). id: {id}")
        print(f"reranker_score: {reranker_score}")
        print(f"filepath: {filepath}")
        print(f"caption: {caption}")
        #print(f"content: {content}")
        print(f"content_token: {calc_token(content)}")
        print("--------------------------------------------------")

    system_tokens = calc_token(answer_prompt(arr_results))
    print(f"total_token : {system_tokens + query_tokens}\n")
    return arr_results


def openai(system_template, query):
    """
    OpenAI を使用して最終回答を生成する関数
    
    Args:
        system_template (str): システムプロンプトテンプレート（検索結果を含む）
        query (str): ユーザーからの質問文
    """
    # OpenAI による最終回答の生成
    time_start = datetime.datetime.now()
    print("LLM - Start time: ", time_start)

    response = client.chat.completions.create(
        model="gpt-4.1", 
        messages=[
        {"role": "system", "content": system_template},
        {"role": "user", "content": query}
        ]
    )

    time_end = datetime.datetime.now()
    print("LLM - End time: ", time_end)

    duration = time_end- time_start
    print("Duration: ", duration)

    print("\n########### 最終回答 ############")
    print(response.choices[0].message.content)
    del response


def to_vectorize(query) -> list[float]:
    """
    テキストをベクトル化する関数（OpenAI Embedding APIを使用）
    
    Args:
        query (str): ベクトル化したいテキスト
        
    Returns:
        list[float]: ベクトル化されたテキスト（3072次元）
    """
    response = client.embeddings.create(
        input = query,
        model= "text-embedding-3-large"
    )

    embeddings = response.data[0].embedding

    return embeddings

def answer_prompt(context):
  """
  検索結果をシステムプロンプトテンプレートに組み込んで最終的なプロンプトを作成
  
  Args:
      context (list): 検索結果のリスト
      
  Returns:
      str: システムプロンプトと検索結果を組み合わせた最終プロンプト
  """
  prompt_template = AZURE_OPENAI_SYSTEM_PROMPT_CREATE_ANSWER
  custom_template = f'''{prompt_template}{context}'''

  return custom_template


def calc_token(content):
    """
    テキストのトークン数を計算する関数
    
    Args:
        content (str): トークン数を計算したいテキスト
        
    Returns:
        int: テキストのトークン数
    """
    encoding = tiktoken.get_encoding("o200k_base")
    num_tokens = len(encoding.encode(content))
    return num_tokens

main()