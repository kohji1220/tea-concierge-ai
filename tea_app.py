import json
import os
import requests
import base64
import time
import hashlib
import streamlit as st

# 本格的なベクトルデータベース・エンジンの導入
import chromadb

# ==========================================
# 1. 初期設定
# ==========================================
st.set_page_config(page_title="AIティーコンシェルジュ (ChromaDB搭載)", page_icon="☕", layout="centered")

st.markdown("""
<style>
    .chat-bubble { padding: 1rem; border-radius: 0.5rem; margin-bottom: 1rem; }
    .uploaded-image { border-radius: 8px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); margin-top: 8px; }
</style>
""", unsafe_allow_html=True)

# 【重要】ご自身のAPIキーに書き換えてください！
API_KEY = "AIzaSyDsqZA3A_tDi3hOQqupEVNtpQa7_LMRydw"

# ChromaDBの保存先ディレクトリ
CHROMA_DB_PATH = "./tea_chroma_db"

# ==========================================
# 2. ベクトルデータベース (ChromaDB) エンジン
# ==========================================

def get_embedding(text):
    """Gemini APIを使用してテキストを768次元のベクトルに変換"""
    url = f"https://generativelanguage.googleapis.com/v1beta/models/text-embedding-004:embedContent?key={API_KEY}"
    headers = {'Content-Type': 'application/json'}
    payload = {
        "model": "models/text-embedding-004",
        "content": {"parts": [{"text": text}]}
    }
    try:
        res = requests.post(url, headers=headers, json=payload, timeout=10)
        if res.status_code == 200:
            return res.json()['embedding']['values']
        return None
    except Exception:
        return None

def generate_id(text):
    """テキストから一意のハッシュIDを生成（重複登録を防ぐため）"""
    return hashlib.md5(text.encode('utf-8')).hexdigest()

@st.cache_resource(show_spinner=False)
def init_chromadb():
    """ChromaDBを初期化し、未登録のデータがあれば自動でベクトル化して追加する"""
    file_path = "web_tea_knowledge.jsonl"
    
    # ChromaDBの永続化クライアント（ローカルフォルダにDBを構築）
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    
    # コレクション（テーブル）の取得または作成
    collection = client.get_or_create_collection(name="tea_knowledge_collection")
    
    if not os.path.exists(file_path):
        st.error(f"[!] {file_path} が見つかりません。")
        return collection

    try:
        # JSONLからすべてのテキストを読み込む
        with open(file_path, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]
        
        if not lines:
            return collection

        # 既存のDBに登録されているIDを取得
        existing_data = collection.get()
        existing_ids = set(existing_data['ids']) if existing_data and 'ids' in existing_data else set()

        # 新規登録が必要なデータだけを抽出
        new_docs = []
        new_ids = []
        for line in lines:
            data = json.loads(line)
            text_content = json.dumps(data, ensure_ascii=False)
            doc_id = generate_id(text_content)
            
            if doc_id not in existing_ids:
                new_docs.append(text_content)
                new_ids.append(doc_id)

        # 新規データがあれば、Geminiでベクトル化してChromaDBに追加
        if new_docs:
            st.info(f"☕ {len(new_docs)}件の新規論文データを検出。専用データベース(ChromaDB)に構築中...")
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for i, (doc, doc_id) in enumerate(zip(new_docs, new_ids)):
                status_text.text(f"インデックス構築中... ({i+1}/{len(new_docs)})")
                vector = get_embedding(doc)
                
                if vector:
                    collection.add(
                        documents=[doc],
                        embeddings=[vector],
                        metadatas=[{"source": "web_tea_knowledge"}],
                        ids=[doc_id]
                    )
                time.sleep(0.05) # APIレートリミット対策
                progress_bar.progress((i + 1) / len(new_docs))
                
            progress_bar.empty()
            status_text.empty()
            st.success("✅ データベースの更新が完了しました！")

        return collection

    except Exception as e:
        st.error(f"[!] データベースの初期化中にエラーが発生いたしました: {e}")
        return client.get_or_create_collection(name="tea_knowledge_collection")

# ==========================================
# 3. アプリケーションの初期化
# ==========================================

with st.spinner("超高速ベクトルデータベース(ChromaDB)を起動中..."):
    collection = init_chromadb()
    db_count = collection.count()

if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0

with st.sidebar:
    st.header("☕ システムステータス")
    st.success(f"✅ ChromaDB 稼働中 (インデックス: {db_count}件)")
    st.success("✅ Google検索連携 (グラウンディング) スタンバイ")
    st.success("✅ 視覚・画像解析 (マルチモーダル) スタンバイ")
    st.markdown("---")
    
    st.header("📸 視覚解析ツール")
    uploaded_file = st.file_uploader(
        "茶葉や茶殻の画像をアップロード", 
        type=["png", "jpg", "jpeg", "webp"], 
        key=f"uploader_{st.session_state.uploader_key}"
    )
    if uploaded_file is not None:
        st.image(uploaded_file, caption="解析待機中...", use_column_width=True)

st.title("☕ AI ティーコンシェルジュ")
st.write("最新鋭のベクトルDB「Chroma」を搭載し、深遠な専門知識から瞬時に答えを導き出す専用エージェントでございます。")

if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.api_history = []
    greeting = "いらっしゃいませ。データベースの本格稼働を確認いたしました。どのようなマニアックなご質問でも、瞬時にインデックスから抽出しお答えいたします。"
    st.session_state.messages.append({"role": "assistant", "content": {"text": greeting}})

for msg in st.session_state.messages:
    avatar = "🧑‍💻" if msg["role"] == "user" else "🤵"
    with st.chat_message(msg["role"], avatar=avatar):
        content = msg["content"]
        if "text" in content:
            st.markdown(content["text"])
        if "image_bytes" in content:
            st.image(content["image_bytes"], width=250)

# ==========================================
# 4. 対話とChromaDB推論プロセス
# ==========================================

if prompt := st.chat_input("専門的な質問を入力してください..."):
    
    with st.chat_message("user", avatar="🧑‍💻"):
        st.markdown(prompt)
        if uploaded_file is not None:
            st.image(uploaded_file, width=250)

    user_parts = [{"text": prompt}]
    message_content = {"text": prompt}

    if uploaded_file is not None:
        file_bytes = uploaded_file.getvalue()
        base64_data = base64.b64encode(file_bytes).decode("utf-8")
        user_parts.append({
            "inlineData": {"mimeType": uploaded_file.type, "data": base64_data}
        })
        message_content["image_bytes"] = file_bytes
        st.session_state.uploader_key += 1

    st.session_state.messages.append({"role": "user", "content": message_content})
    
    with st.chat_message("assistant", avatar="🤵"):
        with st.spinner("ChromaDBインデックスを検索し、最新情報を照合中..."):
            
            # --- ChromaDBを使った超高速検索 ---
            query_emb = get_embedding(prompt)
            relevant_knowledge = ""
            
            if query_emb and collection.count() > 0:
                # ChromaDBエンジンのクエリ機能を呼び出し、瞬時にTop3を抽出
                results = collection.query(
                    query_embeddings=[query_emb],
                    n_results=min(3, collection.count())
                )
                if results['documents'] and len(results['documents']) > 0:
                    best_matches = results['documents'][0]
                    relevant_knowledge = "\n".join(best_matches)

            # 動的システムプロンプトの構築
            dynamic_system_prompt = f"""
            あなたは世界トップクラスの知識を持つ、洗練された「お茶の専属コンシェルジュ（紳士）」です。
            お客様の質問に対し、以下の【関連する専門知識】を最優先の根拠として論理的に回答してください。
            もし不足している最新情報（価格や気象など）があれば、Google検索ツールを用いて補完してください。

            【関連する専門知識 (ChromaDB抽出データ)】
            {relevant_knowledge if relevant_knowledge else "（特に関連データは見つかりませんでした）"}

            【対話のルール】
            1. 品位のある紳士的な口調（「〜でございます」等）で、専門家として解像度の高い解説を行うこと。
            2. お客様に画像が提示された場合は、プロの視点で視覚解析を行うこと。
            3. 生化学メカニズム、土壌、製茶プロセス、市場経済の因果関係を繋ぎ合わせて推論すること。
            """

            current_api_history = st.session_state.api_history.copy()
            current_api_history.append({"role": "user", "parts": user_parts})

            payload = {
                "systemInstruction": {"parts": [{"text": dynamic_system_prompt}]},
                "contents": current_api_history,
                "tools": [{"googleSearch": {}}],
                "generationConfig": {"temperature": 0.6}
            }
            url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={API_KEY}"
            headers = {'Content-Type': 'application/json'}
            
            try:
                response = requests.post(url, headers=headers, json=payload, timeout=60)
                
                if response.status_code == 200:
                    result = response.json()
                    reply_text = ""
                    if 'candidates' in result and len(result['candidates']) > 0:
                        parts = result['candidates'][0]['content']['parts']
                        for part in parts:
                            if 'text' in part:
                                reply_text += part['text']
                    
                    if not reply_text:
                        reply_text = "（情報の取得に問題が発生いたしました。）"

                    st.markdown(reply_text)
                    
                    st.session_state.messages.append({"role": "assistant", "content": {"text": reply_text}})
                    st.session_state.api_history.append({"role": "user", "parts": [{"text": prompt}]}) 
                    st.session_state.api_history.append({"role": "model", "parts": [{"text": reply_text}]})
                    st.rerun()
                    
                else:
                    st.error(f"APIエラー: HTTP {response.status_code}")
                    st.session_state.messages.pop()
                    
            except requests.exceptions.RequestException as e:
                st.error(f"通信エラー: {e}")
                st.session_state.messages.pop()