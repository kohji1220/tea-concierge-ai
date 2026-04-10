# クラウド環境(Streamlit Cloud)の古いSQLiteバージョン対策ハック
try:
    __import__('pysqlite3')
    import sys
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    pass

import json
import os
import requests
import base64
import time
import hashlib
import streamlit as st
import chromadb

# ==========================================
# 1. 初期設定 & UI (CSS)
# ==========================================
st.set_page_config(page_title="AI Tea Concierge & Analyzer", page_icon="🍵", layout="centered")

# ダークモード/ライトモードの両方に適応するよう、背景色・文字色の強制指定を解除
st.markdown("""
<style>
    .stApp { font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif; }
    h1 { font-weight: 300; letter-spacing: 2px; border-bottom: 1px solid #D5DBDB; padding-bottom: 10px; }
    .stChatInputContainer { border-radius: 20px !important; }
    /* サクラチェッカー用の強調表示 */
    .risk-high { color: #E74C3C; font-weight: bold; font-size: 1.2em; }
    .risk-medium { color: #F39C12; font-weight: bold; font-size: 1.2em; }
    .risk-low { color: #27AE60; font-weight: bold; font-size: 1.2em; }
</style>
""", unsafe_allow_html=True)

if "GEMINI_API_KEY" in st.secrets:
    API_KEY = st.secrets["GEMINI_API_KEY"]
else:
    API_KEY = "あなたのAPIキーをここに入力してください"

CHROMA_DB_PATH = "./tea_chroma_db"

# ==========================================
# 2. ベクトルデータベース (ChromaDB) エンジン
# ==========================================

def get_embedding(text):
    url = f"https://generativelanguage.googleapis.com/v1beta/models/text-embedding-004:embedContent?key={API_KEY}"
    headers = {'Content-Type': 'application/json'}
    payload = {"model": "models/text-embedding-004", "content": {"parts": [{"text": text}]}}
    try:
        res = requests.post(url, headers=headers, json=payload, timeout=10)
        if res.status_code == 200:
            return res.json()['embedding']['values']
        return None
    except Exception:
        return None

def generate_id(text):
    return hashlib.md5(text.encode('utf-8')).hexdigest()

@st.cache_resource(show_spinner=False)
def init_chromadb():
    file_path = "web_tea_knowledge.jsonl"
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    collection = client.get_or_create_collection(name="tea_knowledge_collection")
    
    if not os.path.exists(file_path):
        return collection

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]
        if not lines:
            return collection

        existing_data = collection.get()
        existing_ids = set(existing_data['ids']) if existing_data and 'ids' in existing_data else set()

        new_docs = []
        new_ids = []
        for line in lines:
            data = json.loads(line)
            text_content = json.dumps(data, ensure_ascii=False)
            doc_id = generate_id(text_content)
            if doc_id not in existing_ids:
                new_docs.append(text_content)
                new_ids.append(doc_id)

        if new_docs:
            st.info(f"🍵 {len(new_docs)}件の新規データを検出。インデックスを構築中...")
            progress_bar = st.progress(0)
            for i, (doc, doc_id) in enumerate(zip(new_docs, new_ids)):
                vector = get_embedding(doc)
                if vector:
                    collection.add(
                        documents=[doc],
                        embeddings=[vector],
                        metadatas=[{"source": "web_tea_knowledge"}],
                        ids=[doc_id]
                    )
                time.sleep(0.05)
                progress_bar.progress((i + 1) / len(new_docs))
            progress_bar.empty()
        return collection
    except Exception as e:
        return client.get_or_create_collection(name="tea_knowledge_collection")

# ==========================================
# 3. アプリケーションの初期化
# ==========================================

with st.spinner("Initializing Vector Database..."):
    collection = init_chromadb()
    db_count = collection.count()

with st.sidebar:
    st.markdown("### 🍵 System Status")
    st.success(f"ChromaDB Active ({db_count} records)")
    st.success("Google Search Grounding: Ready")
    st.success("Multimodal Vision: Ready")
    st.markdown("---")

st.title("AI Tea Concierge & Analyzer")

# ==========================================
# 4. タブによる機能切り替え (StreamlitのUI機能)
# ==========================================

tab1, tab2 = st.tabs(["💬 コンシェルジュ (通常対話)", "🚨 査定チェッカー (怪しさ判定)"])

# ------------------------------------------
# タブ1: 通常のコンシェルジュチャット
# ------------------------------------------
with tab1:
    st.markdown("*Advanced Reasoning Engine for Tea Science & Market Economy*")
    
    if "uploader_key" not in st.session_state:
        st.session_state.uploader_key = 0
        
    uploaded_file = st.file_uploader(
        "茶葉や水色の画像をアップロード (任意)", 
        type=["png", "jpg", "jpeg", "webp"], 
        key=f"uploader_{st.session_state.uploader_key}"
    )

    if "messages" not in st.session_state:
        st.session_state.messages = []
        st.session_state.api_history = []
        greeting = "いらっしゃいませ。最新鋭の推論エンジンと専用インデックスが稼働しております。どのようなご質問でもお聞かせください。"
        st.session_state.messages.append({"role": "assistant", "content": {"text": greeting}})

    for msg in st.session_state.messages:
        avatar = "🧑‍💻" if msg["role"] == "user" else "🤵"
        with st.chat_message(msg["role"], avatar=avatar):
            content = msg["content"]
            if "text" in content:
                st.markdown(content["text"])
            if "image_bytes" in content:
                st.image(content["image_bytes"], width=250)

    if prompt := st.chat_input("ご質問を入力してください..."):
        with st.chat_message("user", avatar="🧑‍💻"):
            st.markdown(prompt)
            if uploaded_file is not None:
                st.image(uploaded_file, width=250)

        user_parts = [{"text": prompt}]
        message_content = {"text": prompt}

        if uploaded_file is not None:
            file_bytes = uploaded_file.getvalue()
            base64_data = base64.b64encode(file_bytes).decode("utf-8")
            user_parts.append({"inlineData": {"mimeType": uploaded_file.type, "data": base64_data}})
            message_content["image_bytes"] = file_bytes
            st.session_state.uploader_key += 1

        st.session_state.messages.append({"role": "user", "content": message_content})
        
        with st.chat_message("assistant", avatar="🤵"):
            with st.spinner("Analyzing knowledge index and current web data..."):
                query_emb = get_embedding(prompt)
                relevant_knowledge = ""
                
                if query_emb and collection.count() > 0:
                    results = collection.query(query_embeddings=[query_emb], n_results=min(3, collection.count()))
                    if results['documents'] and len(results['documents']) > 0:
                        relevant_knowledge = "\n".join(results['documents'][0])

                dynamic_system_prompt = f"""
                あなたは世界トップクラスの知識を持つ「お茶の専属コンシェルジュ」です。
                以下の【関連する専門知識 (ChromaDB抽出)】を最優先の根拠とし、不足分はGoogle検索で補って回答してください。
                【関連する専門知識】\n{relevant_knowledge if relevant_knowledge else "なし"}
                """

                current_api_history = st.session_state.api_history.copy()
                current_api_history.append({"role": "user", "parts": user_parts})

                payload = {
                    "systemInstruction": {"parts": [{"text": dynamic_system_prompt}]},
                    "contents": current_api_history,
                    "tools": [{"google_search": {}}], # ★400エラー修正箇所：アンダーバーに変更！
                    "generationConfig": {"temperature": 0.6}
                }
                url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={API_KEY}"
                
                try:
                    res = requests.post(url, headers={'Content-Type': 'application/json'}, json=payload, timeout=60)
                    if res.status_code == 200:
                        reply_text = res.json().get('candidates', [{}])[0].get('content', {}).get('parts', [{}])[0].get('text', 'エラー')
                        st.markdown(reply_text)
                        st.session_state.messages.append({"role": "assistant", "content": {"text": reply_text}})
                        st.session_state.api_history.extend([
                            {"role": "user", "parts": [{"text": prompt}]},
                            {"role": "model", "parts": [{"text": reply_text}]}
                        ])
                        st.rerun()
                    else:
                        st.error(f"APIエラー: HTTP {res.status_code} - {res.text}")
                except Exception as e:
                    st.error(f"通信エラー: {e}")

# ------------------------------------------
# タブ2: 査定チェッカー (進化したプロンプトによるJSON出力とUI描画)
# ------------------------------------------
with tab2:
    st.markdown("### 🔍 商品の怪しさ・適正価格を鑑定します")
    st.write("商品の説明文（コピペ）や、スクショ画像をアップロードしてください。")
    
    if "checker_uploader_key" not in st.session_state:
        st.session_state.checker_uploader_key = 1000

    checker_image = st.file_uploader(
        "商品のスクショ画像 (任意)", 
        type=["png", "jpg", "jpeg", "webp"], 
        key=f"uploader_{st.session_state.checker_uploader_key}"
    )
    checker_text = st.text_area("商品説明やキャッチコピーを貼り付け", height=150)
    
    if st.button("鑑定開始", type="primary"):
        if not checker_text and not checker_image:
            st.warning("画像か説明文のどちらかを入力してください。")
        else:
            with st.spinner("RAGデータベースと照合し、科学的・市場的観点から査定中..."):
                
                # ユーザー入力を構築
                user_parts = []
                prompt_text = "以下の商品情報（画像・テキスト）を査定してください。\n\n"
                if checker_text:
                    prompt_text += f"【商品説明】\n{checker_text}\n"
                user_parts.append({"text": prompt_text})
                
                if checker_image is not None:
                    file_bytes = checker_image.getvalue()
                    base64_data = base64.b64encode(file_bytes).decode("utf-8")
                    user_parts.append({"inlineData": {"mimeType": checker_image.type, "data": base64_data}})
                
                # ChromaDBから類似知識を検索 (テキストがある場合のみ)
                relevant_knowledge = ""
                if checker_text and collection.count() > 0:
                    query_emb = get_embedding(checker_text)
                    if query_emb:
                        results = collection.query(query_embeddings=[query_emb], n_results=min(3, collection.count()))
                        if results['documents'] and len(results['documents']) > 0:
                            relevant_knowledge = "\n".join(results['documents'][0])

                # 【進化した査定用システムプロンプト】(JSONを出力させる)
                checker_system_prompt = f"""
                あなたは冷徹で論理的な「お茶の専門鑑定士」です。
                ユーザーが提示したお茶の商品情報（キャッチコピー、成分、価格等）を、以下の【RAG知識】と【Google検索】を用いて厳しく査定してください。
                特に「非科学的な健康効果」「不当な高価格」「原産地の偽装・誤認」を厳しくチェックしてください。
                
                【RAG知識 (あなたの専門知識データベース)】
                {relevant_knowledge if relevant_knowledge else "（特になし）"}
                
                【出力形式の厳守】
                回答は必ず以下のJSON形式（マークダウン不要）のみで出力してください。
                {{
                    "risk_score": 0〜100の整数 (100が最も怪しい/サクラ度高い),
                    "risk_level": "低", "中", "高" のいずれか,
                    "fair_price_estimate": "適正価格の推測（例：500円〜1000円）",
                    "scientific_analysis": "生化学的・農学的観点からの矛盾点や指摘",
                    "marketing_analysis": "マーケティング手法や価格設定についての辛口指摘"
                }}
                """

                payload = {
                    "systemInstruction": {"parts": [{"text": checker_system_prompt}]},
                    "contents": [{"role": "user", "parts": user_parts}],
                    "tools": [{"google_search": {}}], # ★400エラー修正箇所：アンダーバーに変更！
                    "generationConfig": {"temperature": 0.2, "responseMimeType": "application/json"}
                }
                
                url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={API_KEY}"
                
                try:
                    res = requests.post(url, headers={'Content-Type': 'application/json'}, json=payload, timeout=60)
                    if res.status_code == 200:
                        result_text = res.json().get('candidates', [{}])[0].get('content', {}).get('parts', [{}])[0].get('text', '{}')
                        
                        try:
                            # AIが返してきたJSONをPythonで解釈する
                            assessment = json.loads(result_text)
                            
                            # --- UIの描画 (Streamlitの魔法) ---
                            st.markdown("### 📊 鑑定結果")
                            
                            # 危険度スコアをメーターと色で表示
                            score = assessment.get("risk_score", 0)
                            level = assessment.get("risk_level", "不明")
                            
                            st.write(f"**危険度（サクラ度）スコア:** {score}/100")
                            # Streamlitのプログレスバー機能
                            st.progress(score / 100.0) 
                            
                            if level == "高":
                                st.error(f"🚨 【危険度：高】 この商品は誇大広告や不当価格の可能性が極めて高いです。")
                            elif level == "中":
                                st.warning(f"⚠️ 【危険度：中】 いくつか疑わしい点があります。購入は慎重に。")
                            else:
                                st.success(f"✅ 【危険度：低】 特に怪しい点・非科学的な記述は見当たりません。")
                                
                            st.markdown("---")
                            
                            col1, col2 = st.columns(2)
                            with col1:
                                st.info(f"💰 **適正価格の推測:**\n\n{assessment.get('fair_price_estimate', '判定不能')}")
                            
                            st.markdown("#### 🔬 科学的分析")
                            st.write(assessment.get("scientific_analysis", "なし"))
                            
                            st.markdown("#### 🛒 マーケティング分析 (辛口)")
                            st.write(assessment.get("marketing_analysis", "なし"))
                            
                            # 次の査定のためにファイルアップローダーのキーを更新
                            st.session_state.checker_uploader_key += 1
                            
                        except json.JSONDecodeError:
                            st.error("AIの判定結果のフォーマットエラーです。もう一度お試しください。")
                            st.write("生データ:", result_text)
                            
                    else:
                        st.error(f"APIエラー: HTTP {res.status_code} - {res.text}")
                except Exception as e:
                    st.error(f"通信エラー: {e}")
