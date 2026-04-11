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

st.markdown("""
<style>
    .stApp { font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif; }
    h1 { font-weight: 300; letter-spacing: 2px; border-bottom: 1px solid #D5DBDB; padding-bottom: 10px; }
    .stChatInputContainer { border-radius: 20px !important; }
</style>
""", unsafe_allow_html=True)

if "GEMINI_API_KEY" in st.secrets:
    API_KEY = st.secrets["GEMINI_API_KEY"]
else:
    API_KEY = "あなたのAPIキーをここに入力してください"

CHROMA_DB_PATH = "./tea_chroma_db"

# ==========================================
# 2. 【最重要】API安全呼び出し・見えないゴミ除去関数
# ==========================================
def build_safe_url(base_url, api_key):
    clean_key = str(api_key).strip().replace('"', '').replace("'", "").replace('\n', '').replace('\r', '')
    raw_url = f"{base_url}?key={clean_key}"
    safe_url = raw_url.encode('ascii', 'ignore').decode('ascii').strip()
    return safe_url

def call_api_with_retry(url, headers, payload):
    delays = [1, 2, 4, 8, 16]
    for attempt in range(len(delays) + 1):
        try:
            res = requests.post(url, headers=headers, json=payload, timeout=60)
            if res.status_code == 200:
                return res
            elif res.status_code in [429, 503] and attempt < len(delays):
                time.sleep(delays[attempt])
                continue
            else:
                return res
        except requests.exceptions.RequestException as e:
            if attempt < len(delays):
                time.sleep(delays[attempt])
                continue
            raise e
    return None

def get_embedding(text):
    base_url = "https://generativelanguage.googleapis.com/v1beta/models/text-embedding-004:embedContent"
    url = build_safe_url(base_url, API_KEY)
    headers = {'Content-Type': 'application/json'}
    payload = {"model": "models/text-embedding-004", "content": {"parts": [{"text": text[:10000]}]}} # 制限回避のため文字数をカット
    try:
        res = call_api_with_retry(url, headers, payload)
        if res and res.status_code == 200:
            return res.json()['embedding']['values']
        return None
    except Exception:
        return None

def generate_id(text):
    return hashlib.md5(text.encode('utf-8')).hexdigest()

def fetch_text_from_url(url):
    """Jina Readerを使ってURLからテキストをスクレイピングする関数"""
    try:
        # URLの先頭に https://r.jina.ai/ をつけるだけ
        jina_url = f"https://r.jina.ai/{url.strip()}"
        headers = {
            "Accept": "text/plain", # 余計なHTMLを弾く
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }
        response = requests.get(jina_url, headers=headers, timeout=20)
        if response.status_code == 200:
            return response.text
        else:
            return f"[エラー] URLからの情報取得に失敗しました (HTTP {response.status_code})"
    except Exception as e:
        return f"[エラー] URL通信中に問題が発生しました: {e}"

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
    st.success("Jina URL Scraper: Ready")
    st.markdown("---")

st.title("AI Tea Concierge & Analyzer")

# ==========================================
# 4. タブによる機能切り替え
# ==========================================

tab1, tab2 = st.tabs(["💬 コンシェルジュ (通常対話)", "🚨 査定チェッカー (怪しさ判定)"])

# ------------------------------------------
# タブ1: 通常のコンシェルジュチャット (省略なし)
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
                    "tools": [{"google_search": {}}],
                    "generationConfig": {"temperature": 0.6}
                }
                
                base_url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent"
                url = build_safe_url(base_url, API_KEY)
                
                try:
                    res = call_api_with_retry(url, headers={'Content-Type': 'application/json'}, payload=payload)
                    if res and res.status_code == 200:
                        reply_text = res.json().get('candidates', [{}])[0].get('content', {}).get('parts', [{}])[0].get('text', 'エラー')
                        st.markdown(reply_text)
                        st.session_state.messages.append({"role": "assistant", "content": {"text": reply_text}})
                        st.session_state.api_history.extend([
                            {"role": "user", "parts": [{"text": prompt}]},
                            {"role": "model", "parts": [{"text": reply_text}]}
                        ])
                        st.rerun()
                    elif res:
                        if res.status_code == 503:
                            st.error("現在AIサーバーが大変混み合っております。何度か自動再試行しましたが接続できませんでした。")
                        else:
                            st.error(f"APIエラー: HTTP {res.status_code} - {res.text}")
                    else:
                        st.error("APIリクエストが失敗しました。")
                except Exception as e:
                    st.error(f"通信エラー: {e}")

# ------------------------------------------
# タブ2: 査定チェッカー (URLスクレイピング機能追加)
# ------------------------------------------
with tab2:
    st.markdown("### 🔍 商品の怪しさ・適正価格を鑑定します")
    st.write("商品のURLを貼り付けるか、説明文・画像を直接入力してください。")
    
    # 🌟 新機能：URL入力欄
    checker_url = st.text_input("🔗 商品ページのURLを入力 (Amazon, 楽天, 専門店など)", placeholder="https://...")
    
    if "checker_uploader_key" not in st.session_state:
        st.session_state.checker_uploader_key = 1000

    checker_text = st.text_area("商品説明やキャッチコピーを手動で追加 (任意)", height=150)
    checker_image = st.file_uploader(
        "商品のスクショ画像 (任意)", 
        type=["png", "jpg", "jpeg", "webp"], 
        key=f"uploader_{st.session_state.checker_uploader_key}"
    )
    
    if st.button("🔥 鑑定開始", type="primary"):
        if not checker_url and not checker_text and not checker_image:
            st.warning("URL、画像、説明文のいずれかを入力してください。")
        else:
            with st.spinner("情報を収集中..."):
                scraped_text = ""
                # URLが入力されていればスクレイピングを実行
                if checker_url:
                    with st.spinner("URLから商品情報を抽出しています (数秒かかります)..."):
                        scraped_text = fetch_text_from_url(checker_url)
                        if "[エラー]" in scraped_text:
                            st.error(scraped_text)
                            scraped_text = ""
                        else:
                            st.success("✅ URLからの情報抽出に成功しました！")
                            # 抽出したテキストをチラ見せ（長すぎるので1000文字でカットして表示）
                            with st.expander("抽出されたテキストデータを確認する"):
                                st.text(scraped_text[:1000] + "\n...(以下省略)")

                # 査定用テキストの構築
                combined_text = ""
                if scraped_text:
                    combined_text += f"【URLから抽出された商品説明】\n{scraped_text}\n\n"
                if checker_text:
                    combined_text += f"【入力された商品説明】\n{checker_text}\n\n"

                user_parts = []
                prompt_text = "以下の商品情報（画像・テキスト）を査定してください。\n\n" + combined_text
                user_parts.append({"text": prompt_text})
                
                if checker_image is not None:
                    file_bytes = checker_image.getvalue()
                    base64_data = base64.b64encode(file_bytes).decode("utf-8")
                    user_parts.append({"inlineData": {"mimeType": checker_image.type, "data": base64_data}})
                
            with st.spinner("RAGデータベースと照合し、科学的・市場的観点から査定中..."):
                relevant_knowledge = ""
                # 文字数が多すぎるとChromaDBがエラーを吐くので、最初の1000文字だけでベクトル検索
                search_query_text = combined_text[:1000]
                if search_query_text and collection.count() > 0:
                    query_emb = get_embedding(search_query_text)
                    if query_emb:
                        results = collection.query(query_embeddings=[query_emb], n_results=min(3, collection.count()))
                        if results['documents'] and len(results['documents']) > 0:
                            relevant_knowledge = "\n".join(results['documents'][0])

                checker_system_prompt = f"""
                あなたは、茶葉の販売サイトや商品情報から「サクラ・偽装・粗悪品」を冷徹に見抜くデータ照合マシーンです。
                ユーザーから提供される【商品情報】（レビュー含む）と、システムから提供される【RAG相場・知識データ】を照合し、以下の厳密なスコアリングロジックに基づいて「サクラ度（0〜100）」を算出してください。
                感情や情緒には一切流されず、事実とデータのみに基づいて冷酷に判定を下してください。

                【RAG知識 (あなたの専門知識データベース)】
                {relevant_knowledge if relevant_knowledge else "（特になし）"}

                【査定ロジック：加点・減点方式】
                基準スコアを「50点（判断保留）」とし、以下の①〜③の基準で加点（信頼度アップ＝サクラ度低下）・減点（怪しい＝サクラ度上昇）を行います。最終的なサクラ度は0（完全に安全）〜100（極めて怪しい・詐欺）で出力してください。

                ### ① 情報の「解像度」と「客観的証拠」の評価（サクラ度を下げる要素）
                以下の情報が具体的であるほど、サクラ度を下げてください。（目安: 優れた情報1つにつき -5〜-10点）
                * 産地・地理的表示: 単なる国名や地域名ではなく、具体的な農園名、区画、ロット番号、標高の数値があるか。
                * 品種・栽培: 学術的な正式名称、具体的な摘採時期、摘採方法（手摘み等）。
                * 第三者認証: JAS有機、EU有機、フェアトレード等の客観的認証があるか。
                * 製法・生産者: 萎凋や焙煎の具体的な工程説明、製茶師の名前・経歴が明記されているか。
                * 鮮度管理: 賞味期限だけでなく、製造年月日や具体的な保存方法の指定があるか。

                ### ② 「価格」と「主張」の整合性評価（RAGデータとの照合）
                謳い文句と、RAG相場データを比較し矛盾を突きます。
                * 安すぎる矛盾（目安: +20〜+30点）: 「最高級」「手摘み」と謳っているのに相場より著しく安い場合。
                * 高すぎる矛盾（目安: +15〜+25点）: ①の「具体的な事実」がスッカスカであるにもかかわらず高価格な場合。
                * 正当な高価格（目安: -10〜-20点）: ①の客観的証拠が網羅的に提示されており、RAG相場と一致する場合。

                ### ③ 「ごまかし」と「不誠実さ」の検知（サクラ度を急上昇させる要素）
                * 情緒的ワードの多用（目安: +5〜+15点）
                * 健康効果の過剰・違法な主張（目安: +25〜+30点）: 薬機法に抵触する表現。
                * 煽り文句（目安: +10〜+15点）
                * 不自然なレビュー（目安: +15〜+25点）
                * 販売者の透明性欠如（目安: +10〜+20点）

                【出力フォーマット（JSON形式）】
                必ず以下のJSON形式のみで出力してください。Markdownのコードブロックは使用せず、純粋なJSON文字列のみを出力すること。
                {{
                  "sakura_score": [最終的なサクラ度を0〜100の整数で出力。100が最も怪しい],
                  "risk_level": "[安全 / 注意 / 危険 のいずれかを出力]",
                  "analysis_details": {{
                    "resolution_check": "[①情報の解像度に関する冷徹な分析コメント]",
                    "price_consistency": "[②価格と主張の整合性に関するRAGデータとの比較分析]",
                    "deception_signals": "[③ごまかしシグナルの検知結果]"
                  }},
                  "conclusion": "[一般の購入検討者に向けた、最終的な冷酷かつ的確なアドバイス]"
                }}
                """

                payload = {
                    "systemInstruction": {"parts": [{"text": checker_system_prompt}]},
                    "contents": [{"role": "user", "parts": user_parts}],
                    "generationConfig": {"temperature": 0.2, "responseMimeType": "application/json"}
                }
                
                base_url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent"
                url = build_safe_url(base_url, API_KEY)
                
                try:
                    res = call_api_with_retry(url, headers={'Content-Type': 'application/json'}, payload=payload)
                    if res and res.status_code == 200:
                        result_text = res.json().get('candidates', [{}])[0].get('content', {}).get('parts', [{}])[0].get('text', '{}')
                        
                        try:
                            assessment = json.loads(result_text)
                            
                            st.markdown("### 📊 鑑定結果")
                            score = assessment.get("sakura_score", 0)
                            level = assessment.get("risk_level", "不明")
                            
                            st.write(f"**危険度（サクラ度）スコア:** {score}/100")
                            st.progress(score / 100.0) 
                            
                            if level == "危険":
                                st.error(f"🚨 【危険度：高】 この商品は誇大広告や不当価格の可能性が極めて高いです。")
                            elif level == "注意":
                                st.warning(f"⚠️ 【危険度：中】 いくつか疑わしい点があります。購入は慎重に。")
                            else:
                                st.success(f"✅ 【危険度：低】 特に怪しい点・非科学的な記述は見当たりません。")
                                
                            st.markdown("---")
                            details = assessment.get("analysis_details", {})
                            
                            st.markdown("#### 🔎 ① 情報解像度チェック")
                            st.info(details.get("resolution_check", "情報なし"))
                            
                            st.markdown("#### 💰 ② 価格と主張の整合性")
                            st.warning(details.get("price_consistency", "情報なし"))
                            
                            st.markdown("#### 🎭 ③ ごまかしシグナル検知")
                            st.error(details.get("deception_signals", "情報なし"))
                            
                            st.markdown("#### 👨‍⚖️ 最終結論")
                            st.write(f"**{assessment.get('conclusion', '結論なし')}**")
                            
                            st.session_state.checker_uploader_key += 1
                            
                        except json.JSONDecodeError:
                            st.error("AIの判定結果のフォーマットエラーです。もう一度お試しください。")
                            st.write("生データ:", result_text)
                            
                    elif res:
                        if res.status_code == 503:
                            st.error("現在AIサーバーが大変混み合っております。")
                        else:
                            st.error(f"APIエラー: HTTP {res.status_code} - {res.text}")
                    else:
                        st.error("APIリクエストが失敗しました。")
                except Exception as e:
                    st.error(f"通信エラー: {e}")
